# SparK-cDDPM Anomaly Detection

**First-place project, Summer Undergraduate Research Fellowship (SURF)**  
Academy of Life and Natural Sciences, Xi'an Jiaotong-Liverpool University.

This repository collects a cDDPM-SparK source snapshot and figures from the SURF brain MRI anomaly-detection project. A healthy-image diffusion model reconstructs an input image; reconstruction residuals provide pixel-level anomaly scores.

```mermaid
flowchart LR
    A["Healthy IXI MRI"] --> B["SparK representation learning"]
    A --> C["Conditioned diffusion training"]
    B --> C
    D["BraTS2020 T2 MRI"] --> E["Diffusion reconstruction"]
    C --> E
    E --> F["Absolute residual anomaly map"]
    F --> G["Pixel-level evaluation against tumour masks"]
```

## Recorded benchmark results

The figures show the **cDDPM-SparK five-fold ensemble, seed 2026**, evaluated on 369 BraTS2020 patients in the SURF benchmark. Scores are threshold-free pixel-level measures within the evaluation mask. Macro metrics average patient-level scores; pooled metrics aggregate pixels across patients. They answer different questions and are not interchangeable.

![Macro and pooled benchmark metrics](figures/poster_01_key_metrics.png)

![Pooled precision-recall and ROC curves](figures/poster_02_pr_roc_curves.png)

![Patient-level performance distribution](figures/poster_03_patient_distribution.png)

The recorded patient-macro AUPRC is **0.595**, macro AUROC **0.867**, pooled AUPRC **0.471**, and pooled AUROC **0.857**. The results are descriptive for this specific evaluation protocol; these figures do not establish clinical performance. No threshold-tuned Dice is claimed. The figure manifest states that metrics were not recomputed when the poster figures were produced.

## Qualitative predictions

![Three BraTS T2 examples showing inputs, reconstructions, anomaly maps and tumour reference contours](figures/poster_04_qualitative_cases.png)

The three cases were selected at the 20th, 50th and 80th percentiles of patient-level residual AUPRC, respectively, to show a range of performance. Each row shows a T2 input slice, its reconstruction, the residual and native anomaly maps, and the tumour reference contour. Red contours are ground-truth annotations shown for visualisation after evaluation, not model predictions. Brighter map values indicate larger anomaly scores; they are not thresholded segmentation masks. The displayed AUPRC is patient-level, not a score calculated solely on the pictured slice.

## Repository structure and provenance

- `source/`: tracked source snapshot of [Conditioned-Diffusion-Models-UAD](https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD), commit `c5904a7`, including the shared implementation and configuration used by the SparK code.
- `analysis/`: SURF poster figure-generation script.
- `figures/`: recorded benchmark summaries.

The source snapshot is a detached clean worktree. **The exact relationship between this snapshot and the historical ensemble checkpoints has not been independently verified.** The figures represent the documented historical run and should not be claimed as results reproduced from this snapshot without checking run manifests and checkpoints.

## Reproducibility

Datasets, trained checkpoints and per-patient predictions are not included. The source environment files provide starting points but may require site-specific dependencies, model resources and path changes. The HPC results used a healthy IXI training workflow and BraTS2020 T2 evaluation; consult the original source documentation and SURF run manifests for preprocessing, split, inference and evaluation details before attempting a rerun.

The source is third-party material; check upstream attribution and licence terms before any public redistribution. This private repository is a research snapshot, not a validated clinical tool.

