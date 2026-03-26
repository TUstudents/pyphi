# Spec: pyphi Tutorial Notebook Series

**Date:** 2026-03-26
**Status:** Approved

## Overview

A series of 15 Jupyter notebooks serving as a practitioner-level tutorial for the `pyphi` library. Audience: users who know latent variable methods (PCA, PLS) and want to learn the pyphi API — no conceptual hand-holding needed. Notebooks form a progressive series (later notebooks may reference datasets/concepts from earlier ones) rather than being fully self-contained.

## Folder Structure

```
notebooks/
├── data/                          # all example datasets (copied from examples/)
│   ├── Automobiles PLS.xlsx
│   ├── Automobiles PCA w MD.xlsx
│   ├── NIR.xlsx
│   ├── MBDataset.xlsx
│   ├── lpls_dataset.xlsx
│   ├── jrpls_tpls_dataset.xlsx
│   ├── OPLS Test Data.xlsx
│   ├── chemical_experiments_dataset.xlsx
│   ├── data.xlsx
│   ├── Batch Film Coating.xlsx
│   └── Batch Dryer Case Study.xlsx
│
├── 01_core/
│   ├── 01_pca.ipynb
│   ├── 02_pls.ipynb
│   ├── 03_prediction_and_monitoring.ipynb
│   └── 04_missing_data.ipynb
│
├── 02_spectra/
│   └── 01_nir_calibration.ipynb
│
├── 03_advanced/
│   ├── 01_multiblock_pls.ipynb
│   ├── 02_opls_cca.ipynb
│   ├── 03_lpls.ipynb
│   ├── 04_jrpls_tpls.ipynb
│   ├── 05_varimax_rotation.ipynb
│   ├── 06_lwpls.ipynb
│   └── 07_bootstrap_and_polynomial.ipynb
│
└── 04_batch/
    ├── 01_batch_alignment.ipynb
    ├── 02_mpca_mpls.ipynb
    └── 03_batch_monitoring.ipynb
```

## Shared Conventions

All notebooks follow these conventions:

- **First cell:** imports + `output_notebook()` for inline Bokeh rendering
- **Data paths:** relative `../data/<filename>` from each notebook's subfolder
- **Opening markdown cell:** 1–3 sentences — what the method is, when to use it
- **Section flow:** Load data → Build model → Interpret outputs → Predict on new data (where applicable)
- **No conceptual primers:** assume the reader knows what PCA/PLS is; focus on API usage and output interpretation

## Notebook Content

### `01_core/`

**`01_pca.ipynb`** — Dataset: `Automobiles PCA w MD.xlsx`
- `phi.pca()` with `mcs` options (`True`, `'center'`, `'autoscale'`, `False`)
- `cross_val` parameter
- Model dict keys: `T`, `P`, `r2x`, `r2xpv`, `speX`, `T2`, control limits
- Plots: `pp.r2pv()`, `pp.score_scatter()`, `pp.loadings()`, `pp.weighted_loadings()`, `pp.loadings_map()`, `pp.diagnostics()`
- Contribution plots: mean-to-observation, observation-to-observation

**`02_pls.ipynb`** — Dataset: `Automobiles PLS.xlsx`
- `phi.pls()` with `cross_val` and `cross_val_X`
- Model dict keys: adds `Q`, `W`, `Ws`, `r2y`, `r2ypv`, `speY`, `q2y`
- Plots: `pp.vip()`, `pp.predvsobs()`, class-colored score scatter
- VIP interpretation for variable selection

**`03_prediction_and_monitoring.ipynb`** — Dataset: `Automobiles PLS.xlsx`
- `phi.pca_pred()` and `phi.pls_pred()` on held-out observations
- Accessing SPE, T² on new observations
- Interpreting control limit violations
- `phi.hott2()` for monitoring new data

**`04_missing_data.ipynb`** — Dataset: `Automobiles PCA w MD.xlsx`
- Representing missing data as `np.nan`
- NIPALS (`md_algorithm='nipals'`) vs NLP (`md_algorithm='nlp'`) — when to use each
- Effect of missing variables on score estimates and SPE
- Known limitation: `pca_pred` divergence for NaN-containing rows (xfail documented)

### `02_spectra/`

**`01_nir_calibration.ipynb`** — Dataset: `NIR.xlsx`
- `pp.plot_spectra()` for raw visualization
- Preprocessing pipeline: `phi.spectra_snv()` → `phi.spectra_savgol(window, deriv, poly)` → `phi.spectra_msc()`
- `phi.pls()` with `mcsX='center'` for spectral data
- Comparing calibration models before/after preprocessing via `pp.predvsobs()`
- Class-colored pred-vs-obs by tablet type and scale

### `03_advanced/`

**`01_multiblock_pls.ipynb`** — Dataset: `MBDataset.xlsx`
- `XMB` as a dict of DataFrames: `{'X1': df1, 'X2': df2, ...}`
- `phi.mbpls()`, `phi.pls_pred()` on multi-block input
- Block-level plots: `pp.mb_weights()`, `pp.mb_r2pb()`, `pp.mb_vip()`

**`02_opls_cca.ipynb`** — Dataset: `OPLS Test Data.xlsx`
- `cca=True` flag in `phi.pls()`
- Additional keys: `Tcv` (covariant scores), `Pcv` (covariant/predictive loadings), `Wcv`
- Visual comparison to reference OPLS scores/loadings from SIMCA-P

**`03_lpls.ipynb`** — Dataset: `lpls_dataset.xlsx`
- L-shaped data concept: linked X, R, Y matrices
- `phi.lpls()`, `phi.lpls_pred()`
- Score and loadings interpretation in L-shaped space

**`04_jrpls_tpls.ipynb`** — Dataset: `jrpls_tpls_dataset.xlsx`
- `phi.reconcile_rows()`, `phi.parse_materials()` for data alignment
- `phi.jrpls()`, `phi.tpls()` and their `_pred()` counterparts
- Per-material scores and loadings

**`05_varimax_rotation.ipynb`** — Dataset: `chemical_experiments_dataset.xlsx`
- `phi.varimax_rotation()` post-fit
- Comparing rotated vs unrotated loadings and R²pv
- When rotation aids interpretability

**`06_lwpls.ipynb`** — Dataset: `NIRdata_tablets.MAT` (scipy.io.loadmat)
- `phi.lwpls()`: locally weighted PLS for nonlinear prediction
- Local parameter (`loc_par`) tuning
- Comparison of LWPLS vs global PLS prediction error

**`07_bootstrap_and_polynomial.ipynb`** — Dataset: `data.xlsx`
- `phi.bootstrap_pls()`: uncertainty bands on PLS coefficients
- `phi.build_polynomial()`: polynomial term specification, PLS-assisted variable selection
- Interpreting the returned equation string and coefficient vector

### `04_batch/`

**`01_batch_alignment.ipynb`** — Dataset: `Batch Film Coating.xlsx`
- Batch DataFrame format: col 1 = batch ID, col 2 = `PHASE`
- `phibatch.simple_align()`, `phase_simple_align()`, `phase_iv_align()`
- `phibatch.plot_var_all_batches()`, `phibatch.plot_batch()`
- `phibatch.phase_sampling_dist()` for choosing samples-per-phase

**`02_mpca_mpls.ipynb`** — Dataset: `Batch Film Coating.xlsx`
- `phibatch.unfold_horizontal()` — returns `(df, clbl, bid)` 3-tuple
- `phibatch.mpca()`, `phibatch.mpls()`
- `phibatch.r2pv()`, `phibatch.loadings()`, `phibatch.loadings_abs_integral()`
- Removing abnormal batches and refitting normal operating condition model

**`03_batch_monitoring.ipynb`** — Dataset: `Batch Film Coating.xlsx` + `Batch Dryer Case Study.xlsx`
- `phibatch.monitor()`: real-time monitoring with instantaneous SPE/T² charts
- `phibatch.predict()`: end-of-batch quality prediction
- `phibatch.contributions()`: static and dynamic (`dyn_conts=True`) contribution plots
- `phibatch.descriptors()`: extracting batch landmarks (min/max/mean per phase)
- `phibatch.build_rel_time()`: computing relative time from timestamps

## Implementation Notes

- The `notebooks/data/` folder will contain copies (not symlinks) of all needed datasets from `examples/`
- `NIRdata_tablets.MAT` is a MATLAB file — loaded with `scipy.io.loadmat()` in notebook 06
- Notebooks in `03_advanced/` and `04_batch/` assume the reader has completed `01_core/`
- All Bokeh plots use `output_notebook()` (inline mode); matplotlib plots (batch) render inline automatically
