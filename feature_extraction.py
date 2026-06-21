"""Feature extraction with a trained RankByGene encoder.

All features are the **teacher-backbone** embedding (1024-dim) -- the branch the
RankByGene paper / original code uses for downstream tasks. This single script
covers three use cases:

1. Shared API
   ``get_encoder`` / ``get_transform`` / ``extract_features`` -- load the encoder
   and embed any batch of patches.

2. Gene-prediction features (``--mode gene``)
   Pairs each spot patch with its gene expression and writes one CSV per slide
   (``spot_id, comma-separated-feature``) -- the on-disk layout consumed by
   ``gene_prediction.py``.

3. Whole-slide features for MIL (``--mode patch``)
   Embeds every patch of a whole-slide image (gene-free) and writes one ``.h5``
   per WSI (``features`` [N, 1024], ``patch_ids`` [N], and ``coords`` [N, 2] when
   the patch names encode ``x_y``) for downstream MIL / slide-level frameworks.
   Patches are named ``x_y.png`` / ``x_y.jpeg`` (any trailing field like ``x_y_0``
   is ignored when parsing coords); both ``.jpeg`` and ``.png`` are supported.

Minimal single-patch usage (UNI-style)
--------------------------------------
    import torch
    from PIL import Image
    from feature_extraction import get_encoder, get_transform, extract_features

    encoder = get_encoder("./checkpoints/rankbygene_breast.ckpt").cuda().eval()
    transform = get_transform()

    image = Image.open("patch.png").convert("RGB")
    image = transform(image).unsqueeze(0).cuda()        # [1, 3, 224, 224]
    with torch.inference_mode():
        feature_emb = extract_features(encoder, image)   # [1, 1024]

Gene-prediction features (one CSV per slide; run once per split)
---------------------------------------------------------------
    python feature_extraction.py --mode gene \
        --dataset_name breast \
        --patch_path ./data/HEST/Breast/ST-patches \
        --gene_path  ./data/HEST/Breast/ST-expression/survival/8n \
        --checkpoint path/to/encoder \
        --feature_save_dir ./features --model_name rankbygene \
        --split_name train

Whole-slide features for MIL (one .h5 per WSI)
----------------------------------------------
    python feature_extraction.py --mode patch \
        --patch_dir ./data/WSI/single_b20_t20 \
        --checkpoint ./checkpoints/rankbygene_breast.ckpt \
        --output_dir ./features_h5

``--patch_dir`` may be a single WSI folder (patch images directly) or a parent
directory holding one sub-folder of patches per WSI; one ``<wsi_id>.h5`` is
written per WSI.
"""
import os
import argparse

import h5py
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dataset.dataloader import FineTune
from transforms.transform import TestTransform
from model.encoder import RankByGeneEncoder

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# Shared API
# --------------------------------------------------------------------------- #
def get_encoder(checkpoint_path):
    """Load the trained RankByGene image encoder from a checkpoint."""
    uni = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    )
    print(f"Loading RankByGene encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    image_encoder_state_dict = {
        k.replace("image_encoder.", ""): v
        for k, v in state.items() if k.startswith("image_encoder.")
    }
    image_encoder = RankByGeneEncoder(uni)
    image_encoder.load_state_dict(image_encoder_state_dict, strict=True)
    print("RankByGene encoder loaded successfully!")
    return image_encoder


def get_transform(input_image_size=224):
    """Deterministic inference transform (resize + ImageNet normalization),
    matching the per-view transform used during training/evaluation."""
    return transforms.Compose([
        transforms.Resize((input_image_size, input_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.inference_mode()
def extract_features(encoder, images):
    """Return the RankByGene feature for a batch of transformed patches.

    Args:
        encoder: a model returned by ``get_encoder`` (on the same device as ``images``).
        images: tensor of shape [B, 3, 224, 224].
    Returns:
        tensor of shape [B, 1024] -- the teacher-backbone embedding (``teacher_pt_feat``).
    """
    return encoder.teacher_backbone(images)


# --------------------------------------------------------------------------- #
# Mode 1: gene-prediction features (per-spot CSV)
# --------------------------------------------------------------------------- #
def save_features_by_slide(encoder, dataset, save_dir, batch_size=64, num_workers=8):
    """Embed each spot patch and write one CSV per slide for gene_prediction.py."""
    os.makedirs(save_dir, exist_ok=True)
    encoder = encoder.cuda()
    encoder.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features_by_slide = {}

    with torch.no_grad():
        for patches, _, slide_ids, spot_ids in data_loader:
            # TestTransform yields a pair of identical deterministic views;
            # the teacher branch consumes the second view.
            teacher_view = patches[1].cuda()
            features = extract_features(encoder, teacher_view).cpu().numpy()

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


def run_gene_mode(args):
    encoder = get_encoder(args.checkpoint)
    feature_save_dir = os.path.join(args.feature_save_dir, args.model_name)
    os.makedirs(feature_save_dir, exist_ok=True)

    dataset = FineTune(args.dataset_name, args.patch_path, args.gene_path)
    dataset.transform = TestTransform()
    save_features_by_slide(encoder, dataset, os.path.join(feature_save_dir, args.split_name),
                           args.batch_size, args.num_workers)


# --------------------------------------------------------------------------- #
# Mode 2: patch-level features for MIL (per-slide .h5)
# --------------------------------------------------------------------------- #
class PatchFolder(Dataset):
    """All patch images inside a single directory."""

    def __init__(self, patch_dir, transform):
        self.patch_dir = patch_dir
        self.transform = transform
        self.files = sorted(f for f in os.listdir(patch_dir)
                            if f.lower().endswith(IMG_EXTS))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = Image.open(os.path.join(self.patch_dir, fname)).convert("RGB")
        return self.transform(image), os.path.splitext(fname)[0]


def _parse_coords(patch_ids):
    """Parse the (x, y) coordinate from patch names.

    Patches are expected to be named ``x_y`` (e.g. ``12_34.png``); only the first
    two integer fields are used as the coordinate, so any trailing field such as
    the ``_0`` in ``12_34_0`` is ignored. Returns an [N, 2] int array, or None if
    any name does not start with two integers."""
    coords = []
    for b in patch_ids:
        parts = b.split("_")
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
            coords.append((int(parts[0]), int(parts[1])))
        else:
            return None
    return np.array(coords, dtype=np.int32)


def extract_slide(encoder, patch_dir, out_h5, transform, batch_size=64, num_workers=8):
    """Extract features for every patch of one WSI in ``patch_dir`` and save to ``out_h5``."""
    dataset = PatchFolder(patch_dir, transform)
    if len(dataset) == 0:
        print(f"No patch images in {patch_dir}, skipping.")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats, patch_ids = [], []
    with torch.inference_mode():
        for images, names in loader:
            images = images.cuda()
            features = extract_features(encoder, images)
            feats.append(features.cpu().numpy())
            patch_ids.extend(names)
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    coords = _parse_coords(patch_ids)

    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("features", data=feats)
        h.create_dataset("patch_ids", data=np.array(patch_ids, dtype="S"))
        if coords is not None:
            h.create_dataset("coords", data=coords)
    print(f"Saved {feats.shape[0]} x {feats.shape[1]} features to {out_h5}"
          + ("" if coords is not None else "  (no coords: patch names are not row_col)"))


def run_patch_mode(args):
    encoder = get_encoder(args.checkpoint).cuda().eval()
    os.makedirs(args.output_dir, exist_ok=True)

    subdirs = sorted(d for d in os.listdir(args.patch_dir)
                     if os.path.isdir(os.path.join(args.patch_dir, d)))
    if subdirs:
        slides = [(d, os.path.join(args.patch_dir, d)) for d in subdirs]
    else:
        slides = [(os.path.basename(os.path.normpath(args.patch_dir)), args.patch_dir)]

    for slide_id, slide_dir in slides:
        extract_slide(encoder, slide_dir, os.path.join(args.output_dir, f"{slide_id}.h5"),
                      get_transform(), args.batch_size, args.num_workers)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="RankByGene feature extraction")
    parser.add_argument('--mode', choices=['gene', 'patch'], required=True,
                        help="'gene': per-spot CSVs for gene_prediction.py; "
                             "'patch': per-slide .h5 of patch features for MIL.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained RankByGene checkpoint.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    # --- gene mode (run once per split, e.g. train then test) ---
    g = parser.add_argument_group('gene mode')
    g.add_argument('--dataset_name', type=str,
                   help='Dataset key used for slide_id parsing (e.g. breast, lung).')
    g.add_argument('--patch_path', type=str)
    g.add_argument('--gene_path', type=str)
    g.add_argument('--split_name', type=str, default='train',
                   help='Sub-directory name for this split (e.g. train, test).')
    g.add_argument('--feature_save_dir', type=str, default='./features',
                   help='Features are saved under <feature_save_dir>/<model_name>/<split_name>/.')
    g.add_argument('--model_name', type=str, default='rankbygene',
                   help='Label used for the feature sub-directory.')

    # --- patch mode ---
    p = parser.add_argument_group('patch mode')
    p.add_argument('--patch_dir', type=str,
                   help='Patch images directly (single slide) or one sub-directory of patches per slide.')
    p.add_argument('--output_dir', type=str, default='./features_h5',
                   help='Output directory; one <slide_id>.h5 is written per slide.')

    args = parser.parse_args()

    if args.mode == 'gene':
        required = ['dataset_name', 'patch_path', 'gene_path']
        missing = [f"--{r}" for r in required if getattr(args, r) is None]
        if missing:
            parser.error(f"--mode gene requires: {', '.join(missing)}")
        run_gene_mode(args)
    else:
        if args.patch_dir is None:
            parser.error("--mode patch requires: --patch_dir")
        run_patch_mode(args)


if __name__ == "__main__":
    main()
