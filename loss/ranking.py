import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ranking_consistency(values, targets, iterations):
    """Sampling-based ranking consistency penalty (O(N^2)).

    For each of `iterations` cyclic shifts, penalizes pairs whose ordering in the
    predicted similarities (``values``) is inconsistent with the target
    similarities (``targets``). This is the specialized sampling strategy used by
    RankByGene, avoiding the O(N^3) cost of enumerating all triplets.
    """
    permuted_indices = torch.randperm(len(values))
    total_loss = 0

    values = values[permuted_indices]
    targets = targets[permuted_indices]

    for i in range(iterations):
        rolled_values = torch.roll(values, i + 1)
        rolled_targets = torch.roll(targets, i + 1)

        rank_difference = torch.nn.functional.relu(
            -torch.sign(rolled_targets - targets) * (rolled_targets - targets)
            + torch.abs(rolled_values - values)
        )
        total_loss += torch.sum(rank_difference) / len(values)

    return total_loss / iterations


class RankingLoss(nn.Module):
    """Cross-modal ranking consistency loss. Enforces that the within-modal
    similarity ordering of the image features matches that of the gene features."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, gene_embeddings, image_embeddings):
        # normalize the embeddings
        gene_embeddings = gene_embeddings / gene_embeddings.norm(dim=1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

        # within-modal cosine similarity matrices
        gene_gene_similarity = gene_embeddings @ gene_embeddings.T
        image_image_similarity = image_embeddings @ image_embeddings.T

        # softmax over similarities, then enforce ranking consistency
        gene_gene_similarity_softmax = F.softmax(gene_gene_similarity / self.temperature, dim=-1)
        image_image_similarity_softmax = F.softmax(image_image_similarity / self.temperature, dim=-1)
        rankloss = ranking_consistency(image_image_similarity_softmax, gene_gene_similarity_softmax, 1)

        return rankloss


def calculate_rank_accuracy(gene_similarities_row, image_similarities_row, n_pairs=4):
    """Rank accuracy metric: randomly sample `n_pairs` index pairs and check
    whether the relative ordering of the gene-feature similarities agrees with
    that of the image-feature similarities. Returns the fraction of agreeing pairs."""
    pair_indices = np.random.choice(len(gene_similarities_row), (n_pairs, 2), replace=False)

    correct_rank_count = 0
    for idx1, idx2 in pair_indices:
        image_sim_1, image_sim_2 = image_similarities_row[idx1], image_similarities_row[idx2]
        gene_sim_1, gene_sim_2 = gene_similarities_row[idx1], gene_similarities_row[idx2]
        if (image_sim_1 > image_sim_2 and gene_sim_1 > gene_sim_2) or \
           (image_sim_1 < image_sim_2 and gene_sim_1 < gene_sim_2):
            correct_rank_count += 1

    return correct_rank_count / n_pairs
