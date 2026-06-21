import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from dataset.dataloader import Prediction
from model.predictor import GenePredictionMLP


class GenePrediction(pl.LightningModule):
    def __init__(self, model_name, experiment_dir, input_dim, hidden_dim, output_dim,
                 lr=1e-4, dropout_prob=0.1, ignore_index=None):
        super(GenePrediction, self).__init__()
        self.mlp = GenePredictionMLP(input_dim, hidden_dim, output_dim, dropout_prob)
        self.lr = lr
        self.loss = nn.MSELoss()
        self.model_name = model_name
        self.output_dir = experiment_dir
        self.ignore_index = ignore_index

        self.test_step_mse_loss = []
        self.test_step_mae_loss = []
        self.test_step_pearson = []

        self.predicted_genes = {}

    def min_max_normalize(self, features):
        min_vals = features.min(dim=1, keepdim=True)[0]
        max_vals = features.max(dim=1, keepdim=True)[0]
        normalized_features = (features - min_vals) / (max_vals - min_vals + 1e-8)
        return normalized_features

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch):
        features, genes, _, _ = batch
        features = self.min_max_normalize(features)
        preds = self(features)
        loss = self.loss(preds, genes)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        features, genes, _, _ = batch
        features = self.min_max_normalize(features)
        preds = self(features)
        loss = self.loss(preds, genes)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch):
        features, genes, _, spot_id = batch
        features = self.min_max_normalize(features)

        preds = self(features)
        if self.ignore_index is not None:
            for i in self.ignore_index:
                preds = torch.cat((preds[:, :i], preds[:, i + 1:]), 1)
                genes = torch.cat((genes[:, :i], genes[:, i + 1:]), 1)

        loss = self.loss(preds, genes)
        mae = F.l1_loss(preds, genes)

        preds = preds.cpu().detach().numpy().astype(np.float16)
        exps = genes.cpu().detach().numpy()
        pearson_corr = []

        for i in range(len(spot_id)):
            self.predicted_genes[spot_id[i]] = preds[i]

        for g in range(preds.shape[1]):
            pearson_corr.append(pearsonr(preds[:, g], exps[:, g])[0])

        avg_pearson = np.nanmean(pearson_corr)
        loss = loss.cpu().detach().numpy()
        mae = mae.cpu().detach().numpy()

        self.test_step_mse_loss.append(loss)
        self.test_step_mae_loss.append(mae)
        self.test_step_pearson.append(avg_pearson)

    def on_test_epoch_end(self):
        avg_loss = float(np.mean(self.test_step_mse_loss))
        avg_mae = float(np.mean(self.test_step_mae_loss))
        avg_pearson = float(np.mean(self.test_step_pearson))
        print(f"Test MSE Loss: {avg_loss}")
        print(f"Test MAE Loss: {avg_mae}")
        print(f"Test Pearson Correlation: {avg_pearson}")
        metrics = {
            'test_mse_loss': avg_loss,
            'test_mae_loss': avg_mae,
            'test_pearson_corr': avg_pearson,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'predicted_genes.csv'), 'w') as f:
            for key, value in self.predicted_genes.items():
                f.write(f"{key}, {','.join(map(str, value))}\n")

        with open(os.path.join(self.output_dir, 'test_metrics.txt'), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        self._final_metrics = metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.mlp.parameters(), lr=self.lr)


def aggregate_fold_metrics(fold_metrics, output_dir):
    """Compute mean and std across all folds and save a summary file."""
    os.makedirs(output_dir, exist_ok=True)

    n_folds = len(fold_metrics)
    metric_names = ['test_mse_loss', 'test_mae_loss', 'test_pearson_corr']
    summary = {}
    for name in metric_names:
        values = np.array([m[name] for m in fold_metrics if m is not None and name in m])
        if len(values) == 0:
            continue
        summary[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'per_fold': values.tolist(),
        }

    print(f"\n===== {n_folds}-Fold Cross-Validation Summary =====")
    for name, stats in summary.items():
        print(f"{name}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
        print(f"  per-fold: {stats['per_fold']}")

    summary_path = os.path.join(output_dir, 'cv_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{n_folds}-Fold Cross-Validation Summary\n")
        f.write("=" * 40 + "\n")
        for name, stats in summary.items():
            f.write(f"{name}:\n")
            f.write(f"  mean: {stats['mean']}\n")
            f.write(f"  std:  {stats['std']}\n")
            f.write(f"  per-fold: {stats['per_fold']}\n")
    print(f"\nSummary saved to {summary_path}")

    csv_path = os.path.join(output_dir, 'cv_summary.csv')
    rows = []
    for name, stats in summary.items():
        row = {'metric': name, 'mean': stats['mean'], 'std': stats['std']}
        for i, v in enumerate(stats['per_fold']):
            row[f'fold_{i + 1}'] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="K-fold cross-validation for downstream gene prediction")

    parser.add_argument('--train_dataset_name', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--train_gene_path', type=str, required=True)
    parser.add_argument('--test_gene_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='rankbygene',
                        help='Label used for naming the output / feature directories.')
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--output_dim', type=int, default=258)
    parser.add_argument('--ignore_index', type=int, nargs='+')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--feature_save_dir', type=str, default='./features')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--test_split_name', type=str, default='test',
                        help="Subdir name under <feature_save_dir>/<model_name>/ holding the fixed external test features.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # K-fold CV over the TRAIN features (train/val split per fold). The external
    # TEST set is FIXED and evaluated at the end of every fold.
    train_feature_path = os.path.join(args.feature_save_dir, args.model_name, 'train')
    test_feature_path = os.path.join(args.feature_save_dir, args.model_name, args.test_split_name)

    print(f"Train feature path: {train_feature_path}")
    print(f"Test feature path : {test_feature_path}")

    train_dataset_full = Prediction(args.train_dataset_name, train_feature_path, args.train_gene_path)
    test_dataset = Prediction(args.test_dataset_name, test_feature_path, args.test_gene_path)
    print(f"Train dataset size: {len(train_dataset_full)}  Test size: {len(test_dataset)}")

    run_root = os.path.join(
        args.output_path,
        args.model_name + f'_lr={args.learning_rate}_dropout={args.dropout_prob}'
                          f'_epochs={args.epochs}_seed={args.seed}_cv',
    )
    os.makedirs(run_root, exist_ok=True)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_indices = np.arange(len(train_dataset_full))

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        fold_dir = os.path.join(run_root, f'fold_{fold + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n===== Fold {fold + 1}/{args.n_splits} =====")
        print(f"  train: {len(train_idx)}  val: {len(val_idx)}  test (fixed): {len(test_dataset)}")

        train_subset = Subset(train_dataset_full, train_idx.tolist())
        val_subset = Subset(train_dataset_full, val_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = GenePrediction(
            args.model_name, fold_dir,
            args.input_dim, args.hidden_dim, args.output_dim,
            args.learning_rate, args.dropout_prob, args.ignore_index,
        )

        logger = TensorBoardLogger(fold_dir, name="tb_logs")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename='gene_prediction_{epoch:02d}_{val_loss:.4f}',
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=args.devices,
            logger=logger,
            log_every_n_steps=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=2,
        )

        hparams = {
            'model_name': args.model_name,
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'devices': args.devices,
            'fold': fold + 1,
            'n_splits': args.n_splits,
            'seed': args.seed,
        }
        logger.log_hyperparams(hparams)

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path='best')

        fold_metrics.append(getattr(model, '_final_metrics', None))

    aggregate_fold_metrics(fold_metrics, run_root)


if __name__ == "__main__":
    main()