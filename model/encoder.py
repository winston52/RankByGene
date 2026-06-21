import torch.nn as nn
import copy

from model.utils import deactivate_requires_grad


class ProjectionHeadCL(nn.Module):
    """Projection head mapping backbone features into the contrastive space used
    by the gene-image alignment (a residual MLP with GELU + LayerNorm)."""

    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class RankByGeneEncoder(nn.Module):
    """The RankByGene image encoder: a teacher-student wrapper around a UNI ViT
    backbone. The teacher branch is an EMA copy of the student; both project to a
    shared contrastive space via ProjectionHeadCL."""

    def __init__(self, backbone):
        super().__init__()
        input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone)

        self.student_head = ProjectionHeadCL(embedding_dim=input_dim, projection_dim=1024)
        self.teacher_head = ProjectionHeadCL(embedding_dim=input_dim, projection_dim=1024)

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, images):
        # images[0] -> student (strongly augmented), images[1] -> teacher (weakly augmented)
        student_pt_feat = self.student_backbone(images[0])
        teacher_pt_feat = self.teacher_backbone(images[1])
        student_cl_feat = self.student_head(student_pt_feat)
        teacher_cl_feat = self.teacher_head(teacher_pt_feat)
        return student_pt_feat, teacher_pt_feat, student_cl_feat, teacher_cl_feat
