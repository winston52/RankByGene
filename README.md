# RankByGene

Official implementation of **_RankByGene_: Gene-Guided Histopathology Representation Learning Through Cross-Modal Ranking Consistency**.

RankByGene learns gene-informed histopathology image representations by aligning image and gene features through a **cross-modal ranking-consistency loss** that preserves the relative ordering of pairwise similarities across modalities, together with an **intra-modal teacher–student distillation** that stabilizes the image-branch representation. The learned features improve downstream gene expression prediction, slide-level classification, and survival analysis.

![Framework](figures/framework.png)

## Data preparation

We use the publicly available **HEST-1k** dataset ([jaume2024hest](https://github.com/mahmoodlab/HEST)) for spatial transcriptomics, and TCGA / BCNB cohorts for downstream classification and survival analysis.

After downloading the raw HEST-1k data (per-slide `.h5ad` expression and ST-patch `.h5` files), run the preprocessing script `data_preprocessing/preprocess_hest.py`, which (i) converts each `.h5ad` slide to an expression CSV (genes × spots) and a spotfile, (ii) extracts per-spot PNG patches, (iii) builds the gene panels used as the supervisory signal, and (iv) computes the `ignore_index` for masking non-prognostic genes:

```bash
python data_preprocessing/preprocess_hest.py \
    --raw_h5ad      ./data/HEST/Breast/ST-expression-raw \
    --raw_patches   ./data/HEST/Breast/ST-patches-original \
    --expr_out      ./data/HEST/Breast/ST-expression-original \
    --spot_out      ./data/HEST/Breast/ST-spotfiles \
    --patch_out     ./data/HEST/Breast/ST-patches \
    --favorable_tsv   ./data/HEST/Breast/genelist/prognostic_breast_favorable.tsv \
    --unfavorable_tsv ./data/HEST/Breast/genelist/prognostic_breast_unfavorable.tsv \
    --survival_csv    ./data/HEST/Breast/genelist/prognostic_breast_all.csv \
    --top_n 250
```

The survival (prognostic) gene panels are derived from the [Human Protein Atlas](https://www.proteinatlas.org/) favorable/unfavorable prognostic gene lists. The resulting panels used in our experiments are provided under `data_preprocessing/genelist/` (`breast_survival.csv`, `lung_survival.csv`).

Finally, apply 8-neighborhood spatial smoothing to the expression matrices via `smooth_exp` in `data_preprocessing/smooth.py`.

## Training

Train the gene-guided image encoder with `train.py`, which optimizes the cross-modal ranking-consistency loss together with the gene–image contrastive loss and the intra-modal teacher–student distillation:

```bash
python train.py --config configs/breast_rankinfots_survival.yaml
```

The config file specifies the dataset and hyperparameters. Before training, set the following fields in `configs/*.yaml`:

- `MODEL.pretrained_model_path`: path to the UNI backbone weights (`pytorch_model.bin`).
- `DATASET.train_dataset_path`: root of the preprocessed dataset (containing `ST-patches/`).
- `DATASET.gene_path`: directory of the per-spot panel expression (spots × genes) produced in the data-preparation step.
- `EXPERIMENT_DIR`: where checkpoints and TensorBoard logs are written.

Useful overrides:

```bash
python train.py --config configs/lung_rankinfots_survival.yaml \
    --devices 0          # comma-separated GPU ids, overrides config DEVICE
    --gene_dir <path>    # override DATASET.gene_path
    --checkpoint <ckpt>  # resume training from a checkpoint
```

Checkpoints are saved every 5 epochs (plus `last.ckpt`) under `EXPERIMENT_DIR/<run>/tb_logs/version_0/checkpoints/`. Configs for the two datasets are provided: `configs/breast_rankinfots_survival.yaml` and `configs/lung_rankinfots_survival.yaml`.
