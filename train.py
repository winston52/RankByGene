import os
import torch
import torch.nn as nn
import argparse
import timm
import yaml
import numpy as np
import pytorch_lightning as pl
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# user define modules
from model.encoder import RankByGeneEncoder, ProjectionHeadCL
from transforms.transform import TrainTransform, TestTransform
from dataset.dataloader import FineTune
from utils.scheduler import cosine_schedule
from model.utils import update_momentum
from loss.infonce import InfoNCE
from loss.ranking import RankingLoss, calculate_rank_accuracy

def get_args_parser():
    parser = argparse.ArgumentParser(description="Training settings")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--gene_dir", type=str, default=None,
                        help="Override gene directory. If set, ignores the config "
                             "DATASET.task/smoothing path construction.")
    parser.add_argument("--devices", type=str, default=None,
                        help="Override config DEVICE (comma-separated GPU ids).")
    return parser

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class FineTuning(pl.LightningModule):
    def __init__(self, image_encoder, gene_encoder, image_learning_rate, gene_learning_rate, batch_size, **kwargs):
        super(FineTuning, self).__init__()
        self.image_encoder = image_encoder
        self.gene_encoder = gene_encoder
        self.image_learning_rate = image_learning_rate
        self.gene_learning_rate = gene_learning_rate
        self.batch_size = batch_size

        # RankByGene (cross-modal ranking consistency, teacher-student)
        self.alpha = kwargs.get('alpha')
        self.beta = kwargs.get('beta')
        self.ema_momentum = kwargs.get('ema_momentum')
        self.gene_image_flag = kwargs.get('gene_image_flag')

        self.validation_step_rank_pearson = []
        self.validation_step_rank_acc = []

        self.criterion1 = InfoNCE()
        self.criterion2 = RankingLoss()

    def forward_ts(self, images, genes):
        student_pt_feat, teacher_pt_feat, student_cl_feat, teacher_cl_feat = self.image_encoder(images)
        gene_cl_feat = self.gene_encoder(genes)
        return student_cl_feat, teacher_cl_feat, gene_cl_feat
    
    def training_step(self, batch):
        images, genes, *extra, slide_ids, spot_ids = batch

        student_cl_feat, teacher_cl_feat, gene_cl_feat = self.forward_ts(images, genes)
        # gene-image contrastive loss (InfoNCE)
        gene_image_loss = self.criterion1(student_cl_feat, gene_cl_feat)
        # cross-modal ranking consistency loss
        rank_loss = self.criterion2(gene_cl_feat, student_cl_feat)
        # intra-modal distillation (teacher-student consistency)
        consistency_loss = self.criterion1(student_cl_feat, teacher_cl_feat)
        if self.gene_image_flag:
            loss = gene_image_loss + self.alpha * rank_loss + self.beta * consistency_loss
        else:
            loss = self.alpha * rank_loss + self.beta * consistency_loss

        self.log('train_loss_gene_image', gene_image_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_rank', rank_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_consistency', consistency_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch):
        images, genes, slide_ids, spot_ids = batch

        # use the teacher-branch image feature for evaluation
        _, image_cl_feat, gene_cl_feat = self.forward_ts(images, genes)

        # Only evaluate within-slide rank consistency when the batch is a single slide.
        if all(x == slide_ids[0] for x in slide_ids):
            # normalize and compute within-modal similarity matrices
            gene_cl_feat = gene_cl_feat / gene_cl_feat.norm(dim=1, keepdim=True)
            image_cl_feat = image_cl_feat / image_cl_feat.norm(dim=1, keepdim=True)
            gene_gene_similarity = (gene_cl_feat @ gene_cl_feat.T).cpu().detach().numpy()
            image_image_similarity = (image_cl_feat @ image_cl_feat.T).cpu().detach().numpy()

            # per-row Pearson correlation and rank accuracy between the two similarity structures
            pearson_corr, rank_accuracy = [], []
            for i in range(gene_gene_similarity.shape[0]):
                pearson_corr.append(pearsonr(gene_gene_similarity[i], image_image_similarity[i])[0])
                rank_accuracy.append(
                    calculate_rank_accuracy(gene_gene_similarity[i], image_image_similarity[i], n_pairs=4))
            self.validation_step_rank_pearson.append(np.mean(pearson_corr))
            self.validation_step_rank_acc.append(np.mean(rank_accuracy))

    def on_validation_epoch_end(self):
        avg_rank_pearson = np.mean(self.validation_step_rank_pearson)
        avg_rank_acc = np.mean(self.validation_step_rank_acc)
        self.log('rank_pearson_avg', float(avg_rank_pearson), prog_bar=True)
        self.log('rank_acc_avg', float(avg_rank_acc), prog_bar=True)
        self.validation_step_rank_pearson.clear()
        self.validation_step_rank_acc.clear()    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.image_encoder.parameters(), 'lr': self.image_learning_rate},
            {'params': self.gene_encoder.parameters(), 'lr': self.gene_learning_rate}
        ])
        return optimizer

    def ema_update(self):
        # EMA update for teacher model
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, self.ema_momentum, 1)
        update_momentum(self.image_encoder.student_backbone, self.image_encoder.teacher_backbone, m=momentum)
        update_momentum(self.image_encoder.student_head, self.image_encoder.teacher_head, m=momentum)
    
    def on_train_epoch_end(self):
        self.ema_update()
        
        
if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()

    # Load config from file
    config = load_config(args.config)

    if config['MODEL']['pretrained_model'] == "UNI":
        uni = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model_path = config['MODEL']['pretrained_model_path']
        uni.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"), strict=True)

        # RankByGene: teacher-student UNI encoder
        image_encoder = RankByGeneEncoder(uni)
        print("UNI teacher-student encoder loaded successfully!")

        # dimension for contrastive learning
        cl_dim = uni.embed_dim

    else:
        raise ValueError(f"Pretrained model {config['MODEL']['pretrained_model']} not supported")

    # Initialize the Geneprojection head
    gene_encoder = ProjectionHeadCL(embedding_dim=config['MODEL']['gene_input_dim'], projection_dim = cl_dim)
    
    # experiment directory
    experiment_dir_parts = [
        config['DATASET']['train_dataset'],
        config['MODEL']['pretrained_model'],
        config['TRAINING']['loss'],
        config['TRAINING']['image_lr'],
        config['TRAINING']['gene_lr'],
        config['TRAINING']['batch_size']
    ]

    rank_kwargs = {
        'alpha': config['TRAINING']['alpha'],
        'beta': config['TRAINING']['beta'],
        'ema_momentum': config['TRAINING']['ema_momentum'],
        'gene_image_flag': config['TRAINING']['gene_image_flag'],
    }
    experiment_dir_parts.extend([
        config['TRAINING']['alpha'], config['TRAINING']['beta'], config['TRAINING']['ema_momentum'],
        config['DATASET']['task'], config['DATASET']['smoothing'],
    ])

    # Create the experiment directory
    experiment_dir = os.path.join(config['EXPERIMENT_DIR'], "_".join(map(str, experiment_dir_parts)))
    os.makedirs(experiment_dir, exist_ok=True)

    base_batch_size = 8
    batch_size = min(base_batch_size, config['TRAINING']['batch_size'])

    # Initialize the FineTuning framework
    model = FineTuning(
                       image_encoder=image_encoder,
                       gene_encoder=gene_encoder,
                       image_learning_rate=config['TRAINING']['image_lr'],
                       gene_learning_rate=config['TRAINING']['gene_lr'],
                       batch_size = batch_size,
                       **rank_kwargs
                       )
    
    # Define the dataset and transforms
    train_transform = TrainTransform()
    test_transform = TestTransform()

    # Resolve gene path. Prefer the explicit --gene_dir override; otherwise use
    # the per-spot expression directory configured under DATASET.gene_path.
    gene_path = args.gene_dir if args.gene_dir is not None else config['DATASET']['gene_path']
    print(f"gene_path: {gene_path}")

    patch_path = os.path.join(config['DATASET']['train_dataset_path'], "ST-patches")
    train_dataset = FineTune(config['DATASET']['train_dataset'], patch_path=patch_path,
                             gene_path=gene_path, transform=train_transform)
    test_dataset = FineTune(config['DATASET']['train_dataset'], patch_path=patch_path,
                            gene_path=gene_path, transform=test_transform)
    
    # Create the DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['DATASET']['num_workers'], drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config['DATASET']['num_workers'], drop_last=True)
    

    # Configure the logger and trainer
    logger = TensorBoardLogger(experiment_dir, name="tb_logs")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # save a checkpoint every 5 epochs (plus the final one)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{train_loss:.4f}',
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True,
    )
    
    # Calculate accumulate_grad_batches based on batch size
    accumulate_grad_batches = config['TRAINING']['batch_size'] // base_batch_size
    
    devices = config['DEVICE']
    if args.devices is not None:
        devices = [int(x) for x in args.devices.split(',')]

    trainer = pl.Trainer(
                         max_epochs=config['TRAINING']['epochs'],
                         devices=devices,
                         accelerator=accelerator,
                         logger=logger,
                         accumulate_grad_batches=accumulate_grad_batches,
                         log_every_n_steps = 1,
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps = 2
                         )
    
    
    # Save all hyperparameters
    logger.log_hyperparams(config)
    
    if args.checkpoint is not None:

        model = FineTuning.load_from_checkpoint(
            args.checkpoint,
            image_encoder=image_encoder,
            gene_encoder=gene_encoder,
            image_learning_rate=config['TRAINING']['image_lr'],
            gene_learning_rate=config['TRAINING']['gene_lr'],
            batch_size=batch_size,
            **rank_kwargs
        )
        
        print(f"Loaded model from checkpoint: {args.checkpoint}")
        
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=args.checkpoint)
        
    else:
        
        # Train the model
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        

    print("Model training completed!")