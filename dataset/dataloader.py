import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FineTune(Dataset):
    """Fine-tuning / training stage: returns an (image patch, gene vector) pair
    for each spot, read from raw patches and per-spot gene expression."""

    def __init__(self, name, patch_path, gene_path, transform=None):
        self.patch_path = patch_path
        self.transform = transform
        self.all_samples = []

        for gene_file in os.listdir(gene_path):
            if gene_file.endswith("_expression.csv"):
                # Determine slide_id
                if name in ["her2st", "hest_gbm", "breast", "lung"]:
                    slide_id = gene_file.split("_")[0]
                elif name == "gbm":
                    slide_id = "_".join(gene_file.split("_")[:-1])
                else:
                    slide_id = gene_file.split("_")[0]

                gene_df = pd.read_csv(os.path.join(gene_path, gene_file))
                for _, row in gene_df.iterrows():
                    spot_id = str(row.iloc[0])
                    gene_vector = np.array(row.iloc[1:].values, dtype=np.float32)
                    self.all_samples.append({
                        'slide_id': slide_id,
                        'spot_id': spot_id,
                        'gene': gene_vector,
                    })

    def __len__(self):
        return len(self.all_samples)

    def _load_patch(self, slide_id, spot_id):
        patch_file = os.path.join(self.patch_path, slide_id, f"{spot_id}.png")
        patch = Image.open(patch_file).convert('RGB')
        if self.transform:
            patch = self.transform(patch)
        return patch

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        patch = self._load_patch(sample['slide_id'], sample['spot_id'])
        gene = torch.tensor(sample['gene'])
        return patch, gene, sample['slide_id'], sample['spot_id']


class Prediction(Dataset):
    """Downstream prediction stage: returns a (pre-extracted feature, gene vector)
    pair for each spot, matching features saved by feature_extraction.py against
    the per-spot gene expression."""

    def __init__(self, name, feature_path, gene_path):
        self.feature_path = feature_path

        self.gene_data = []
        self.spot_ids = []
        self.slide_ids = []
        self.features = []

        for gene_file in os.listdir(gene_path):
            if gene_file.endswith("_expression.csv"):
                if name in ["her2st", "hest_gbm", "breast", "lung"]:
                    slide_id = gene_file.split("_")[0]
                elif name == "gbm":
                    slide_id = "_".join(gene_file.split("_")[:-1])
                else:
                    slide_id = gene_file.split("_")[0]

                gene_df = pd.read_csv(os.path.join(gene_path, gene_file))
                feature_df = pd.read_csv(os.path.join(self.feature_path, f"{slide_id}.csv"))
                feat_map = dict(zip(feature_df['spot_id'].astype(str), feature_df.iloc[:, 1]))

                for _, row in gene_df.iterrows():
                    spot_id = row.iloc[0]
                    gene_vector = np.array(row.iloc[1:].values, dtype=np.float32)
                    self.gene_data.append(gene_vector)
                    self.spot_ids.append(spot_id)
                    self.slide_ids.append(slide_id)

                    feature_string = feat_map.get(str(spot_id))
                    if feature_string is not None:
                        feature_vector = np.array([float(x) for x in feature_string.split(',')], dtype=np.float32)
                        self.features.append(feature_vector)
                    else:
                        raise ValueError(f"Feature for spot_id {spot_id} not found in {slide_id}.csv")

    def __len__(self):
        return len(self.gene_data)

    def __getitem__(self, idx):
        return self.features[idx], self.gene_data[idx], self.slide_ids[idx], self.spot_ids[idx]
