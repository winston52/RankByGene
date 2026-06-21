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
6. ``build_panel_expression``: subset the full expression to a gene panel,
                            transpose to (spots x genes), apply 8-neighborhood
                            smoothing, and save the per-spot CSVs the training
                            dataloader consumes. This is the final, training-ready
                            output.

Example (raw -> training-ready survival top-250 expression, end to end)
-----------------------------------------------------------------------
    python preprocess_hest.py \
        --raw_h5ad        ./data/HEST/Breast/ST-expression-raw \
        --raw_patches     ./data/HEST/Breast/ST-patches-original \
        --expr_out        ./data/HEST/Breast/ST-expression-original \
        --spot_out        ./data/HEST/Breast/ST-spotfiles \
        --patch_out       ./data/HEST/Breast/ST-patches \
        --favorable_tsv   ./data/HEST/Breast/genelist/prognostic_breast_favorable.tsv \
        --unfavorable_tsv ./data/HEST/Breast/genelist/prognostic_breast_unfavorable.tsv \
        --survival_csv    ./data/HEST/Breast/genelist/survival_panel.csv \
        --panel_expr_out  ./data/HEST/Breast/ST-expression/survival250/8n \
        --panel_type survival_top --top_n 250 --smoothing 8n
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
def _common_genes(expr_dir):
    """Genes measured in every slide (intersection of the gene index across all
    per-slide genes x spots expression CSVs in ``expr_dir``)."""
    common = None
    for fname in os.listdir(expr_dir):
        if not fname.endswith("_expression.csv"):
            continue
        idx = set(pd.read_csv(os.path.join(expr_dir, fname), index_col=0, usecols=[0]).index)
        common = idx if common is None else (common & idx)
    return common or set()


def build_survival_panel(favorable_tsv, unfavorable_tsv, out_csv, expr_dir=None):
    """Merge the HPA favorable + unfavorable prognostic gene lists into a single
    survival gene panel (deduplicated, one gene per line, no header). When
    ``expr_dir`` is given, the panel is restricted to genes measured in every
    slide so that it matches the expression actually available for training."""
    favorable = pd.read_csv(favorable_tsv, sep="\t")["Gene"]
    unfavorable = pd.read_csv(unfavorable_tsv, sep="\t")["Gene"]
    genes = pd.concat([favorable, unfavorable], axis=0).drop_duplicates()
    if expr_dir is not None:
        common = _common_genes(expr_dir)
        genes = genes[genes.isin(common)]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
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


# ----------------------------------------------------------------------------
# 6. per-slide panel expression: subset to a gene panel, transpose to
#    (spots x genes), apply 8-neighborhood smoothing, and save in the layout
#    the training dataloader consumes.
# ----------------------------------------------------------------------------
def _barcode_to_coord(spot_csv):
    """Map spot barcodes to zero-padded ``RxC`` coordinate ids (e.g. row 59,
    col 19 -> ``059x019``) from a spotfile's ``array_row`` / ``array_col``. Some
    HEST slides index spots by barcode rather than coordinate; the coordinate id
    is what the patches are named by and what the spatial smoothing parses."""
    meta = pd.read_csv(spot_csv, index_col=0)
    return {bc: f"{int(r):03d}x{int(c):03d}"
            for bc, r, c in zip(meta.index, meta["array_row"], meta["array_col"])}


def _normalize_total_log1p(spots, target_sum=1e4):
    """Library-size normalize each spot to ``target_sum`` total counts, then log1p.
    Applied over the full gene set (before panel subsetting) so the per-spot
    library size reflects all measured genes, matching the scanpy convention
    ``normalize_total(target_sum=1e4)`` + ``log1p``."""
    lib = spots.sum(axis=1).replace(0, 1.0)
    return np.log1p(spots.div(lib, axis=0) * target_sum)


def build_panel_expression(expr_dir, panel_genes, out_dir, smoothing="8n",
                           spot_dir=None, patch_dir=None, normalize=True):
    """Turn the per-slide full expression CSVs (genes x spots, from ``h5ad_to_csv``)
    into the training-ready per-spot panel expression.

    For each slide it (i) transposes to spots x genes, (ii) when ``spot_dir`` is
    given, relabels barcode-indexed spots to ``RxC`` coordinate ids via the
    spotfile, (iii) when ``patch_dir`` is given, keeps only spots that have an
    extracted patch image, (iv) when ``normalize`` is set, library-size normalizes
    each spot and log1p-transforms (over the full gene set), (v) restricts to
    ``panel_genes`` (in the given order; genes absent from a slide are filled with
    0), (vi) applies 8-neighborhood spatial smoothing when ``smoothing == "8n"``,
    and (vii) writes ``{slide}_expression.csv`` (spots x genes) to ``out_dir``.
    """
    from smooth import smooth_exp

    os.makedirs(out_dir, exist_ok=True)
    panel_genes = list(panel_genes)
    for fname in sorted(os.listdir(expr_dir)):
        if not fname.endswith("_expression.csv"):
            continue
        slide = fname.split("_")[0]
        # genes x spots -> spots x genes
        spots = pd.read_csv(os.path.join(expr_dir, fname), index_col=0).T
        if spot_dir is not None:
            coord = _barcode_to_coord(os.path.join(spot_dir, f"{slide}.csv"))
            spots.index = [coord.get(b, b) for b in spots.index]
        if patch_dir is not None:
            have_patch = {os.path.splitext(p)[0] for p in os.listdir(os.path.join(patch_dir, slide))}
            spots = spots[spots.index.isin(have_patch)]
        if normalize:
            spots = _normalize_total_log1p(spots)
        # subset to the panel (panel order; genes absent from this slide -> 0)
        spots = spots.reindex(columns=panel_genes, fill_value=0.0)
        if smoothing == "8n":
            spots = smooth_exp(spots)
        spots.to_csv(os.path.join(out_dir, f"{slide}_expression.csv"))
        print(f"  [panel expr] {slide}: {spots.shape[0]} spots x {spots.shape[1]} genes -> {out_dir}")


def select_top_expressed(expr_dir, candidate_genes, top_n):
    """Rank genes by total expression summed across all slides (from the genes x
    spots full expression CSVs) and return the top-N gene names. ``candidate_genes``
    restricts the ranking to a panel (e.g. the survival panel, to derive the
    survival top-N subset); pass ``None`` to rank over all genes."""
    candidate = set(candidate_genes) if candidate_genes is not None else None
    totals = {}
    for fname in os.listdir(expr_dir):
        if not fname.endswith("_expression.csv"):
            continue
        df = pd.read_csv(os.path.join(expr_dir, fname), index_col=0)  # genes x spots
        slide_totals = df.sum(axis=1)
        for gene, val in slide_totals.items():
            if candidate is None or gene in candidate:
                totals[gene] = totals.get(gene, 0) + val
    ranked = sorted(totals, key=lambda g: totals[g], reverse=True)
    return ranked[:top_n]


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
    p.add_argument("--build_top", action="store_true",
                   help="also rank all genes by expression and save a top-N panel")
    p.add_argument("--panel_expr_out",
                   help="output dir for per-slide panel expression (spots x genes, training-ready)")
    p.add_argument("--panel_type", choices=["survival", "top", "survival_top"], default="survival_top",
                   help="gene panel used to build --panel_expr_out (ignored if --panel_genes_csv is given)")
    p.add_argument("--panel_genes_csv",
                   help="explicit gene-list CSV (one gene per line) to use as the panel for "
                        "--panel_expr_out, instead of deriving it from --panel_type")
    p.add_argument("--spot_dir",
                   help="spotfile dir; when given, barcode-indexed spots are relabeled to RxC "
                        "coordinate ids via array_row/array_col before subsetting and smoothing")
    p.add_argument("--patch_dir",
                   help="patch dir; when given, --panel_expr_out keeps only spots that have an "
                        "extracted patch image (matches the expression spot set to the patches)")
    p.add_argument("--smoothing", default="8n", help='spatial smoothing for panel expression ("8n" or "none")')
    p.add_argument("--no_normalize", action="store_true",
                   help="skip per-spot library-size normalization + log1p for --panel_expr_out")
    args = p.parse_args()

    if args.raw_h5ad and args.expr_out and args.spot_out:
        print("== h5ad -> CSV ==")
        h5ad_to_csv(args.raw_h5ad, args.expr_out, args.spot_out)
    if args.raw_patches and args.patch_out:
        print("== patches -> PNG ==")
        patches_h5_to_png(args.raw_patches, args.patch_out)
    if args.favorable_tsv and args.unfavorable_tsv and args.survival_csv:
        print("== survival gene panel ==")
        build_survival_panel(args.favorable_tsv, args.unfavorable_tsv, args.survival_csv, expr_dir=args.expr_out)
    if args.build_top and args.expr_out:
        print("== top-N gene panel ==")
        build_top_panel(args.expr_out, os.path.dirname(args.survival_csv) if args.survival_csv else args.expr_out, args.top_n)
    if args.panel_expr_out and args.expr_out:
        if args.panel_genes_csv:
            panel_genes = [g.strip() for g in open(args.panel_genes_csv) if g.strip()]
            print(f"== panel per-spot expression ({len(panel_genes)} genes from {os.path.basename(args.panel_genes_csv)}, {args.smoothing}) ==")
        else:
            survival_genes = [g.strip() for g in open(args.survival_csv) if g.strip()] if args.survival_csv else None
            if args.panel_type == "survival":
                panel_genes = survival_genes
            elif args.panel_type == "top":
                panel_genes = select_top_expressed(args.expr_out, None, args.top_n)
            else:  # survival_top: the top-N highest-expressed genes within the survival panel
                panel_genes = select_top_expressed(args.expr_out, survival_genes, args.top_n)
            print(f"== panel per-spot expression ({args.panel_type}, {args.smoothing}) ==")
        build_panel_expression(args.expr_out, panel_genes, args.panel_expr_out, args.smoothing,
                               spot_dir=args.spot_dir, patch_dir=args.patch_dir, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
