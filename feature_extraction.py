"""Extract gene-informed image features with a trained RankByGene encoder.

Loads a RankByGene checkpoint (the teacher-student UNI encoder), runs it over the
patches of each slide, and saves one CSV per slide containing
``(spot_id, comma-separated-feature)`` rows — the on-disk layout consumed by the
downstream gene-prediction step (``gene_prediction.py``).

Example
-------
    python feature_extraction.py \
        --train_dataset_name breast --test_dataset_name breast \
        --train_patch_path ./data/HEST/Breast/ST-patches \
        --train_gene_path  ./data/HEST/Breast/ST-expression-survival/8n \
        --test_patch_path  ./data/HEST/Breast/test/ST-patches \
        --test_gene_path   ./data/HEST/Breast/test/ST-expression-survival/8n \
        --checkpoint ./checkpoints/rankbygene_breast.ckpt \
        --feature_save_dir ./features --model_name rankbygene
"""
import os
import argparse

import torch
import timm
import pandas as pd
from torch.utils.data import DataLoader

from dataset.dataloader import FineTune
from transforms.transform import TestTransform
from model.encoder import RankByGeneEncoder


def get_encoder(checkpoint_path):
    """Load the trained RankByGene image encoder from a checkpoint."""
    uni = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    )
    print(f"Loading RankByGene encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    image_encoder_state_dict = {
        k.replace("image_encoder.", ""): v
        for k, v in state.items() if k.startswith("image_encoder.")
    }
    image_encoder = RankByGeneEncoder(uni)
    image_encoder.load_state_dict(image_encoder_state_dict, strict=True)
    print("RankByGene encoder loaded successfully!")
    return image_encoder


def save_features_by_slide(encoder, dataset, save_dir, batch_size=64, num_workers=8):
    os.makedirs(save_dir, exist_ok=True)
    encoder = encoder.cuda()
    encoder.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features_by_slide = {}

    with torch.no_grad():
        for i, (patches, _, slide_ids, spot_ids) in enumerate(data_loader):
            # RankByGene uses a teacher-student encoder; the dataloader yields a
            # pair of augmented views and we keep the teacher-branch feature.
            patches[0] = patches[0].cuda()
            patches[1] = patches[1].cuda()
            _, features, _, _ = encoder(patches)

            features = features.cpu().numpy()

            for j, slide_id in enumerate(slide_ids):
                if slide_id not in features_by_slide:
                    if features_by_slide:
                        prev_slide_id = list(features_by_slide.keys())[0]
                        feature_list = features_by_slide.pop(prev_slide_id)
                        feature_save_path = os.path.join(save_dir, f"{prev_slide_id}.csv")
                        feature_df = pd.DataFrame(feature_list, columns=["spot_id", "feature"])
                        feature_df.to_csv(feature_save_path, index=False)
                        print(f"Features for slide {prev_slide_id} saved to {feature_save_path}")

                    features_by_slide[slide_id] = []

                spot_id = spot_ids[j]
                feature_str = ','.join(map(str, features[j]))
                features_by_slide[slide_id].append([spot_id, feature_str])

    if features_by_slide:
        for slide_id, feature_list in features_by_slide.items():
            feature_save_path = os.path.join(save_dir, f"{slide_id}.csv")
            feature_df = pd.DataFrame(feature_list, columns=["spot_id", "feature"])
            feature_df.to_csv(feature_save_path, index=False)
            print(f"Features for slide {slide_id} saved to {feature_save_path}")

    print(f"All features saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="RankByGene feature extraction")
    parser.add_argument('--train_dataset_name', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--train_patch_path', type=str, required=True)
    parser.add_argument('--train_gene_path', type=str, required=True)
    parser.add_argument('--test_patch_path', type=str, required=True)
    parser.add_argument('--test_gene_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained RankByGene checkpoint.')
    parser.add_argument('--feature_save_dir', type=str, default='./features',
                        help='Features are saved under <feature_save_dir>/<model_name>/{train,<test_split_name>}/.')
    parser.add_argument('--model_name', type=str, default='rankbygene',
                        help='Label used for the feature sub-directory.')
    parser.add_argument('--test_split_name', type=str, default='test',
                        help='Sub-directory name holding the external test features.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    encoder = get_encoder(args.checkpoint)
    feature_save_dir = os.path.join(args.feature_save_dir, args.model_name)
    os.makedirs(feature_save_dir, exist_ok=True)

    train_dataset = FineTune(args.train_dataset_name, args.train_patch_path, args.train_gene_path)
    test_dataset = FineTune(args.test_dataset_name, args.test_patch_path, args.test_gene_path)
    train_dataset.transform = TestTransform()
    test_dataset.transform = TestTransform()

    save_features_by_slide(encoder, train_dataset, os.path.join(feature_save_dir, 'train'), args.batch_size, args.num_workers)
    save_features_by_slide(encoder, test_dataset, os.path.join(feature_save_dir, args.test_split_name), args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
