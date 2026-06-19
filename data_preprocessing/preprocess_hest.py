"""HEST-1k preprocessing for RankByGene (breast / lung).

This script converts raw HEST-1k spatial-transcriptomics data into the layout
consumed by the training and evaluation pipeline, and builds the gene panels
used as the cross-modal supervisory signal.

Pipeline
--------
1. ``h5ad_to_csv``        : per-slide AnnData (.h5ad) -> expression CSV
                            (genes x spots) + spotfile (spot metadata).
2. ``patches_h5_to_png``  : HEST ST-patches (.h5) -> per-spot PNG patches.
3. ``build_survival_panel``: merge the Human Protein Atlas favorable +
                            unfavorable prognostic gene lists into one panel.
4. ``build_top_panel``    : rank genes by total expression across slides and
                            keep the top-N highly expressed genes.
5. ``compute_ignore_index``: indices of survival-panel genes that are NOT in
                            the top-N panel (used to mask non-prognostic genes
                            when evaluating on the survival panel).

8-neighborhood spatial smoothing is applied separately via ``smooth.py``.

Example
-------
    python preprocess_hest.py --raw_h5ad   ./data/HEST/Breast/ST-expression-raw \
                              --raw_patches ./data/HEST/Breast/ST-patches-original \
                              --expr_out    ./data/HEST/Breast/ST-expression-original \
                              --spot_out    ./data/HEST/Breast/ST-spotfiles \
                              --patch_out   ./data/HEST/Breast/ST-patches
"""
import os
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from PIL import Image


# ----------------------------------------------------------------------------
# 1. h5ad -> expression CSV (genes x spots) + spotfile
# ----------------------------------------------------------------------------
def h5ad_to_csv(raw_h5ad_dir, expr_out_dir, spot_out_dir):
    """Convert every ``.h5ad`` slide to an expression CSV (genes x spots) and a
    spotfile holding the spot metadata (in_tissue, array/pixel coordinates)."""
    os.makedirs(expr_out_dir, exist_ok=True)
    os.makedirs(spot_out_dir, exist_ok=True)
    for filename in os.listdir(raw_h5ad_dir):
        if not filename.endswith(".h5ad"):
            continue
        slide = filename.split(".")[0]
        adata = sc.read_h5ad(os.path.join(raw_h5ad_dir, filename))
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        # genes x spots
        expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names).T
        expr.to_csv(os.path.join(expr_out_dir, f"{slide}_expression.csv"))

        meta = adata.obs[["in_tissue", "array_row", "array_col",
                          "pxl_row_in_fullres", "pxl_col_in_fullres"]]
        meta.to_csv(os.path.join(spot_out_dir, f"{slide}.csv"))
        print(f"  [h5ad] {slide}: {expr.shape[0]} genes x {expr.shape[1]} spots")


# ----------------------------------------------------------------------------
# 2. ST-patches .h5 -> per-spot PNG
# ----------------------------------------------------------------------------
def patches_h5_to_png(raw_patch_dir, patch_out_dir):
    """Extract every patch image from the HEST ``.h5`` files and save it as
    ``{patch_out_dir}/{slide}/{barcode}.png``."""
    for filename in os.listdir(raw_patch_dir):
        if not filename.endswith(".h5"):
            continue
        slide = filename.split(".")[0]
        out_dir = os.path.join(patch_out_dir, slide)
        os.makedirs(out_dir, exist_ok=True)
        with h5py.File(os.path.join(raw_patch_dir, filename), "r") as f:
            barcodes = f["barcode"][:]
            imgs = f["img"][:]
        for i, img in enumerate(imgs):
            barcode = barcodes[i][0].decode("utf-8")
            Image.fromarray(np.uint8(img)).save(os.path.join(out_dir, f"{barcode}.png"))
        print(f"  [patch] {slide}: {len(imgs)} PNGs")


# ----------------------------------------------------------------------------
# 3. survival (prognostic) gene panel from the Human Protein Atlas
# ----------------------------------------------------------------------------
def build_survival_panel(favorable_tsv, unfavorable_tsv, out_csv):
    """Merge the HPA favorable + unfavorable prognostic gene lists into a single
    survival gene panel (one gene per line, no header)."""
    favorable = pd.read_csv(favorable_tsv, sep="\t")["Gene"]
    unfavorable = pd.read_csv(unfavorable_tsv, sep="\t")["Gene"]
    genes = pd.concat([favorable, unfavorable], axis=0)
    genes.to_csv(out_csv, index=False, header=False)
    print(f"  [survival panel] {len(genes)} genes -> {out_csv}")
    return genes.tolist()


# ----------------------------------------------------------------------------
# 4. top-N highly expressed gene panel
# ----------------------------------------------------------------------------
def build_top_panel(expr_dir, save_dir, top_n=250):
    """Rank genes by total expression summed across all slides (using only
    genes common to every slide) and save the top-N panel."""
    os.makedirs(save_dir, exist_ok=True)
    totals, common = {}, None
    for fname in os.listdir(expr_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(expr_dir, fname), index_col=0)
        slide_totals = df.sum(axis=1)  # sum across spots, per gene
        for gene, val in slide_totals.items():
            totals[gene] = totals.get(gene, 0) + val
        genes = set(slide_totals.index)
        common = genes if common is None else (common & genes)
    ranked = sorted(((g, totals[g]) for g in common), key=lambda x: x[1], reverse=True)
    pd.DataFrame(ranked).to_csv(os.path.join(save_dir, "full_gene_list.csv"),
                                index=False, header=False)
    top = ranked[:top_n]
    pd.DataFrame(top).to_csv(os.path.join(save_dir, f"top{top_n}_genes.csv"),
                             index=False, header=False)
    print(f"  [top panel] top-{top_n} of {len(ranked)} genes -> {save_dir}")
    return [g for g, _ in top]


# ----------------------------------------------------------------------------
# 5. ignore_index for masking non-prognostic genes within the survival panel
# ----------------------------------------------------------------------------
def compute_ignore_index(survival_csv, keep_csv):
    """Return the indices (within ``survival_csv``) of genes that are NOT in
    ``keep_csv``. These indices are passed to the predictor via ``--ignore_index``
    so that only the kept (e.g. top-250 prognostic) genes contribute to the loss."""
    with open(survival_csv) as f:
        survival_genes = [line.strip() for line in f]
    keep = set(pd.read_csv(keep_csv, header=None)[0])
    ignore_index = [i for i, g in enumerate(survival_genes) if g not in keep]
    print(f"  [ignore_index] {len(ignore_index)} / {len(survival_genes)} genes ignored")
    return ignore_index


def main():
    p = argparse.ArgumentParser(description="HEST-1k preprocessing for RankByGene")
    p.add_argument("--raw_h5ad", help="dir of raw per-slide .h5ad files")
    p.add_argument("--raw_patches", help="dir of raw HEST ST-patches .h5 files")
    p.add_argument("--expr_out", help="output dir for expression CSVs")
    p.add_argument("--spot_out", help="output dir for spotfiles")
    p.add_argument("--patch_out", help="output dir for PNG patches")
    p.add_argument("--favorable_tsv", help="HPA favorable prognostic gene TSV")
    p.add_argument("--unfavorable_tsv", help="HPA unfavorable prognostic gene TSV")
    p.add_argument("--survival_csv", help="output CSV for the merged survival panel")
    p.add_argument("--top_n", type=int, default=250)
    args = p.parse_args()

    if args.raw_h5ad and args.expr_out and args.spot_out:
        print("== h5ad -> CSV ==")
        h5ad_to_csv(args.raw_h5ad, args.expr_out, args.spot_out)
    if args.raw_patches and args.patch_out:
        print("== patches -> PNG ==")
        patches_h5_to_png(args.raw_patches, args.patch_out)
    if args.favorable_tsv and args.unfavorable_tsv and args.survival_csv:
        print("== survival gene panel ==")
        build_survival_panel(args.favorable_tsv, args.unfavorable_tsv, args.survival_csv)
    if args.expr_out:
        print("== top-N gene panel ==")
        build_top_panel(args.expr_out, os.path.dirname(args.survival_csv) if args.survival_csv else args.expr_out, args.top_n)


if __name__ == "__main__":
    main()
