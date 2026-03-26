# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`pyphi` (published as `pyphi-mvda` on PyPI) is a Python toolbox for multivariate latent variable analysis. It implements PCA, PLS, and several specialized variants for industrial/chemometric use cases.

## Installation & Setup

```bash
pip install -r requirements.txt
# or install in editable mode for development:
pip install -e .
```

Dependencies: `bokeh`, `matplotlib`, `numpy`, `openpyxl`, `pandas`, `pyomo`, `scipy`, `statsmodels`.

Optional: IPOPT solver (for NLP-based missing data handling). Without it, pyphi submits to the NEOS server (requires `NEOS_EMAIL` env var).

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

133 tests across three files (`tests/test_calc.py`, `tests/test_plots.py`, `tests/test_batch.py`). Always use `uv run python` — plain `python` may not resolve the virtualenv.

Session-scoped fixtures in `tests/conftest.py` build all models once per run. `matplotlib.use('Agg')` is the first line of conftest — must stay there.

To verify NLP/IPOPT missing data handling (not covered by the test suite):

```bash
cd "examples/Basic calculations PCA and PLS"
uv run python Example_Script_testing_MD_by_NLP.py
```

## Publishing

Releases publish to PyPI automatically via GitHub Actions on a GitHub Release. Version must be bumped manually in both `pyphi/__init__.py` (`__version__`) and `setup.py`.

## Architecture

The package has three modules, all importable via `import pyphi`:

### `pyphi.calc` (aliased as `phi` in examples)
The computational core. All model-building functions return plain Python **dicts** — there are no model classes. Key functions:

- **`pca(X, A, **kwargs)`** — PCA via NIPALS (supports missing data). Returns dict with `T`, `P`, `r2x`, `r2xpv`, `speX`, `T2`, control limits, etc.
- **`pls(X, Y, A, **kwargs)`** — PLS regression. Adds `Q`, `W`, `Ws`, `r2y`, `r2ypv`, `speY`. Optional `cca=True` adds OPLS-equivalent covariant components `Tcv`, `Pcv`, `Wcv`.
- **`pca_pred(Xnew, pcaobj)`** / **`pls_pred(Xnew, plsobj)`** — project new observations.
- **`mbpls(XMB, YMB, A, **kwargs)`** — multi-block PLS where `XMB` is a list of DataFrames.
- **`lpls(X, R, Y, A)`** / **`jrpls(Xi, Ri, Y, A)`** / **`tpls(Xi, Ri, Z, Y, A)`** — L-shaped, JR-PLS, and T-PLS for linked datasets.
- **`jypls(Xi, Yi, A)`** — Joint Y PLS.
- **`lwpls(xnew, loc_par, mvmobj, X, Y)`** — locally weighted PLS for prediction.
- **`varimax_rotation(mvm_obj, X)`** — Varimax rotation post-fit.
- **`bootstrap_pls(X, Y, num_latents, num_samples)`** — bootstrap uncertainty for PLS.
- **`build_polynomial(data, factors, response)`** — polynomial regression with PLS-assisted variable selection.
- Spectra preprocessing: `spectra_snv`, `spectra_savgol`, `spectra_msc`, `spectra_mean_center`, `spectra_autoscale`, `spectra_baseline_correction`.
- Utilities: `clean_empty_rows`, `clean_low_variances`, `cat_2_matrix`, `reconcile_rows`, `parse_materials`, `adapt_pls_4_pyomo`, `export_2_gproms`.

**DataFrame input convention**: when X or Y are DataFrames, the **first column must be observation IDs** (string row labels). Variable names come from remaining column headers. The model dict then includes `obsidX`/`varidX`/`obsidY`/`varidY` lists.

**Missing data**: represented as `np.nan`. NIPALS handles it natively; NLP method (`md_algorithm='nlp'`) uses Pyomo + IPOPT.

**`mcs` parameter** (mean-center/scale): `True` or `'autoscale'` = autoscale; `'center'` = mean-center only; `False` = no preprocessing.

### `pyphi.plots` (aliased as `pp` in examples)
Interactive plots using **Bokeh** (outputs `.html` files to working directory, timestamped). Also uses matplotlib for some batch plots. Key functions:

- `r2pv(mvmobj)` — R² per variable per component bar chart.
- `score_scatter(mvmobj, [lv1, lv2], **kwargs)` — 2D score plot with optional class coloring.
- `score_line(mvmobj, lv, **kwargs)` — score line plot (useful for time-series).
- `loadings(mvmobj)` / `weighted_loadings(mvmobj)` / `loadings_map(mvmobj, [lv1,lv2])`.
- `diagnostics(mvmobj, **kwargs)` — Hotelling's T² and SPE control charts.
- `contributions_plot(mvmobj, X, cont_type, **kwargs)` — contribution bar charts.
- `predvsobs(plsobj, X, Y, **kwargs)` — predicted vs observed for PLS.
- `vip(plsobj)` — VIP scores.
- `mb_weights`, `mb_r2pb`, `mb_vip` — multi-block model plots.

### `pyphi.batch`
Batch analysis toolbox. Operates on batch DataFrames where the first column is batch ID and optionally a second column `PHASE`/`Phase`/`phase`. Key functions:

- `simple_align` / `phase_simple_align` / `phase_iv_align` — batch trajectory alignment.
- `unfold_horizontal(bdata)` / `refold_horizontal(xuf, nvars, nsamples)` — batch unfolding/refolding for multi-way analysis.
- `mpca(xbatch, a, **kwargs)` — Multi-way PCA on batch data.
- `mpls(xbatch, y, a, **kwargs)` — Multi-way PLS on batch data.
- `monitor(mmvm_obj, bdata, **kwargs)` — real-time batch monitoring with control charts.
- `predict(xbatch, mmvm_obj)` — predict end-of-batch quality from trajectory.
- `contributions(mmvmobj, X, cont_type, **kwargs)` — batch contribution plots.
- `descriptors(bdata, which_var, desc, **kwargs)` — extract batch landmarks (min/max/mean per phase).
- `build_rel_time(bdata)` — compute relative time from timestamp column.

## Typical Usage Pattern

```python
import pyphi.calc as phi
import pyphi.plots as pp

# DataFrames: first column = observation IDs
pcaobj = phi.pca(X_df, 3, cross_val=5)   # 5% element-wise CV
pp.score_scatter(pcaobj, [1, 2])

plsobj = phi.pls(X_df, Y_df, 3, cross_val=5, cross_val_X=True)
Xnew_pred = phi.pls_pred(Xnew_df, plsobj)
```
