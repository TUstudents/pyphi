# pyphi Tutorial Notebook Series — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 15 Jupyter notebooks under `notebooks/` as a practitioner tutorial covering every pyphi method.

**Architecture:** Four topic groups (`01_core`, `02_spectra`, `03_advanced`, `04_batch`) under a shared `notebooks/` root with a `data/` subfolder containing all datasets. Notebooks are a progressive series; later ones can reference earlier context. All Bokeh plots render inline via `output_notebook()`.

**Tech Stack:** Python, `jupyter`, `nbformat`, `pyphi`, `bokeh`, `matplotlib`, `scipy`, `pandas`, `numpy`

---

## File Map

```
notebooks/
├── data/                         # copies of all datasets
├── 01_core/
│   ├── 01_pca.ipynb
│   ├── 02_pls.ipynb
│   ├── 03_prediction_and_monitoring.ipynb
│   └── 04_missing_data.ipynb
├── 02_spectra/
│   └── 01_nir_calibration.ipynb
├── 03_advanced/
│   ├── 01_multiblock_pls.ipynb
│   ├── 02_opls_cca.ipynb
│   ├── 03_lpls.ipynb
│   ├── 04_jrpls_tpls.ipynb
│   ├── 05_varimax_rotation.ipynb
│   ├── 06_lwpls.ipynb
│   └── 07_bootstrap_and_polynomial.ipynb
└── 04_batch/
    ├── 01_batch_alignment.ipynb
    ├── 02_mpca_mpls.ipynb
    └── 03_batch_monitoring.ipynb
```

## Shared Notebook Conventions

Every notebook opens with this setup cell:
```python
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
# Prevent pyphi.plots from writing .html files; render inline instead
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
```

Batch notebooks additionally use matplotlib; add `%matplotlib inline` to the setup cell.

Data paths are always `'../data/<filename>'` (one level up from the group folder).

Execution command for any notebook (replace path as needed):
```bash
uv run jupyter nbconvert --to notebook --execute <path> --output <path> --ExecutePreprocessor.timeout=180
```

---

## Task 0: Scaffold folders, copy datasets, verify Jupyter

**Files:**
- Create: `notebooks/data/` (directory)
- Create: `notebooks/01_core/`, `notebooks/02_spectra/`, `notebooks/03_advanced/`, `notebooks/04_batch/`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p notebooks/data \
         notebooks/01_core \
         notebooks/02_spectra \
         notebooks/03_advanced \
         notebooks/04_batch
```

- [ ] **Step 2: Copy all datasets**

```bash
cp "examples/Basic calculations PCA and PLS/Automobiles PLS.xlsx"   notebooks/data/
cp "examples/Basic calculations PCA and PLS/Automobiles PCA w MD.xlsx" notebooks/data/
cp "examples/NIR Calibration/NIR.xlsx"                               notebooks/data/
cp "examples/Multi-block PLS/MBDataset.xlsx"                         notebooks/data/
cp "examples/LPLS/lpls_dataset.xlsx"                                 notebooks/data/
cp "examples/JRPLS and TPLS/jrpls_tpls_dataset.xlsx"                notebooks/data/
cp "examples/plscca vs opls/OPLS Test Data.xlsx"                     notebooks/data/
cp "examples/Varimax Rotation/chemical_experiments_dataset.xlsx"     notebooks/data/
cp "examples/Misc/data.xlsx"                                         notebooks/data/polynomial_data.xlsx
cp "examples/Batch analysis/Batch Film Coating.xlsx"                 notebooks/data/
cp "examples/Batch analysis/Batch Dryer Case Study.xlsx"             notebooks/data/
cp "examples/LWPLS/NIRdata_tablets.MAT"                              notebooks/data/
```

- [ ] **Step 3: Verify jupyter and nbformat are available**

```bash
uv run python -c "import nbformat; import jupyter; print('OK')"
```

If the command fails, install:
```bash
uv add notebook nbformat
```

- [ ] **Step 4: Commit scaffold**

```bash
git add notebooks/
git commit -m "feat: scaffold notebooks/ directory with datasets"
```

---

## Task 1: `01_core/01_pca.ipynb` — PCA

**Files:**
- Create: `notebooks/01_core/01_pca.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_01_pca.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# PCA with pyphi\n\nBuilds a PCA model via NIPALS, interprets scores/loadings, and uses control charts for outlier detection."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data"),
C("""\
features = pd.read_excel('../data/Automobiles PCA w MD.xlsx', 'Features',
                         na_values=np.nan, engine='openpyxl')
classid  = pd.read_excel('../data/Automobiles PCA w MD.xlsx', 'CLASSID',
                         na_values=np.nan, engine='openpyxl')
print(features.shape)
features.head()
"""),
M("## Build PCA Model\n\n`mcs=True` (default) autoscales. Options: `'center'`, `'autoscale'`, `False`. `cross_val=5` holds out 5% of elements per CV round."),
C("""\
pcaobj = phi.pca(features, 3, cross_val=5)
print('Keys:', list(pcaobj.keys()))
print('T shape:', pcaobj['T'].shape, '  P shape:', pcaobj['P'].shape)
print('r2x (cumulative):', pcaobj['r2x'])
print('q2x (cross-val) :', pcaobj['q2x'])
"""),
M("## Captured Variance per Variable per Component"),
C("pp.r2pv(pcaobj)"),
M("## Score Plots"),
C("""\
pp.score_scatter(pcaobj, [1, 2], CLASSID=classid, colorby='Origin')
pp.score_scatter(pcaobj, [1, 2], CLASSID=classid, colorby='Cylinders')
pp.score_line(pcaobj, 1, CLASSID=classid, colorby='Origin', add_ci=True, add_labels=True)
"""),
M("## Loadings"),
C("""\
pp.loadings(pcaobj)
pp.weighted_loadings(pcaobj)
pp.loadings_map(pcaobj, [1, 2])
"""),
M("## Diagnostics: Hotelling's T² and SPE\n\nPoints beyond `T2_lim99` or `speX_lim99` are statistical outliers."),
C("""\
pp.diagnostics(pcaobj, score_plot_xydim=[1, 2])
print('T2  limits (95/99):', pcaobj['T2_lim95'],  pcaobj['T2_lim99'])
print('SPE limits (95/99):', pcaobj['speX_lim95'], pcaobj['speX_lim99'])
"""),
M("## Contribution Plots\n\n`'scores'` mode shows which variables drive the difference between two observations."),
C("""\
pp.contributions_plot(pcaobj, features, 'scores', to_obs=['Car1'])
pp.contributions_plot(pcaobj, features, 'scores', to_obs=['Car1'], from_obs=['Car4'])
"""),
]
os.makedirs('notebooks/01_core', exist_ok=True)
with open('notebooks/01_core/01_pca.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save this as `/tmp/make_01_pca.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_01_pca.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_core/01_pca.ipynb --output notebooks/01_core/01_pca.ipynb \
  --ExecutePreprocessor.timeout=180
```

Expected: `[NbConvertApp] Writing ... bytes to notebooks/01_core/01_pca.ipynb`

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_core/01_pca.ipynb
git commit -m "feat: add 01_core/01_pca tutorial notebook"
```

---

## Task 2: `01_core/02_pls.ipynb` — PLS

**Files:**
- Create: `notebooks/01_core/02_pls.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_02_pls.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# PLS with pyphi\n\nPLS finds latent variables that maximise covariance between X and Y. Key outputs: scores T/U, X-loadings P, Y-loadings Q, weights W/Ws, VIP scores."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data"),
C("""\
features    = pd.read_excel('../data/Automobiles PLS.xlsx', 'Features',
                            na_values=np.nan, engine='openpyxl')
performance = pd.read_excel('../data/Automobiles PLS.xlsx', 'Performance',
                            na_values=np.nan, engine='openpyxl')
classid     = pd.read_excel('../data/Automobiles PLS.xlsx', 'CLASSID',
                            na_values=np.nan, engine='openpyxl')
print('X:', features.shape, '  Y:', performance.shape)
"""),
M("## Build PLS Model\n\n`cross_val_X=True` adds Q²X to assess X-predictability in addition to Q²Y."),
C("""\
plsobj = phi.pls(features, performance, 3, cross_val=5, cross_val_X=True)
print('Keys:', list(plsobj.keys()))
print('r2y (cumulative):', plsobj['r2y'])
print('q2y (cross-val) :', plsobj['q2y'])
print('T shape:', plsobj['T'].shape, '  Q shape:', plsobj['Q'].shape)
"""),
M("## Variance Captured"),
C("""\
pp.r2pv(plsobj)
"""),
M("## Score Plots"),
C("""\
pp.score_scatter(plsobj, [1, 2], CLASSID=classid, colorby='Cylinders')
pp.score_scatter(plsobj, [1, 2], CLASSID=classid, colorby='Origin', add_ci=True)
"""),
M("## Loadings and Weights"),
C("""\
pp.loadings(plsobj)
pp.weighted_loadings(plsobj)
pp.loadings_map(plsobj, [1, 2])
"""),
M("## VIP Scores\n\nVIP > 1 suggests a variable is influential in the model."),
C("pp.vip(plsobj)"),
M("## Predicted vs Observed"),
C("""\
pp.predvsobs(plsobj, features, performance)
pp.predvsobs(plsobj, features, performance, CLASSID=classid, colorby='Origin')
pp.predvsobs(plsobj, features, performance, CLASSID=classid, colorby='Origin', x_space=True)
"""),
M("## Diagnostics"),
C("""\
pp.diagnostics(plsobj, score_plot_xydim=[1, 2])
"""),
M("## Contribution Plots"),
C("""\
pp.contributions_plot(plsobj, features, 'scores', to_obs=['Car1'])
pp.contributions_plot(plsobj, features, 'scores', to_obs=['Car1'], from_obs=['Car4'])
"""),
]
os.makedirs('notebooks/01_core', exist_ok=True)
with open('notebooks/01_core/02_pls.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_02_pls.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_02_pls.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_core/02_pls.ipynb --output notebooks/01_core/02_pls.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_core/02_pls.ipynb
git commit -m "feat: add 01_core/02_pls tutorial notebook"
```

---

## Task 3: `01_core/03_prediction_and_monitoring.ipynb`

**Files:**
- Create: `notebooks/01_core/03_prediction_and_monitoring.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_03_pred.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Prediction and Process Monitoring\n\nProjects new observations onto existing PCA/PLS models and evaluates them against training-set control limits."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Build Training Models"),
C("""\
features    = pd.read_excel('../data/Automobiles PLS.xlsx', 'Features',
                            na_values=np.nan, engine='openpyxl')
performance = pd.read_excel('../data/Automobiles PLS.xlsx', 'Performance',
                            na_values=np.nan, engine='openpyxl')

# Hold out last 10 rows as 'new' observations
X_train = features.iloc[:-10].reset_index(drop=True)
X_new   = features.iloc[-10:].reset_index(drop=True)
Y_train = performance.iloc[:-10].reset_index(drop=True)
Y_new   = performance.iloc[-10:].reset_index(drop=True)

pcaobj = phi.pca(X_train, 3)
plsobj = phi.pls(X_train, Y_train, 3)
print('PCA model built on', X_train.shape[0], 'observations')
print('PLS model built on', X_train.shape[0], 'observations')
"""),
M("## PCA Prediction on New Observations\n\n`pca_pred` returns a dict with `Tnew`, `speXnew`, `T2new` and their control limit flags."),
C("""\
pca_pred = phi.pca_pred(X_new, pcaobj)
print('Keys:', list(pca_pred.keys()))
print('Tnew shape:', pca_pred['Tnew'].shape)
print('speXnew:', pca_pred['speXnew'].ravel())
"""),
M("## PLS Prediction on New Observations\n\n`pls_pred` returns `Yhat` (predicted Y) plus the same SPE/T² diagnostics."),
C("""\
pls_pred = phi.pls_pred(X_new, plsobj)
print('Keys:', list(pls_pred.keys()))
print('Yhat shape:', pls_pred['Yhat'].shape)
Y_actual = np.array(Y_new.values[:, 1:]).astype(float)
resid = Y_actual - pls_pred['Yhat']
print('Prediction residuals (first 5 rows):\\n', resid[:5])
"""),
M("## Hotelling's T² for New Observations\n\n`hott2` returns `T2new` and flags observations outside the 95/99% limits."),
C("""\
t2_result = phi.hott2(pcaobj, Xnew=X_new)
print('T2new:', t2_result['T2new'].ravel())
print('T2_lim95:', pcaobj['T2_lim95'], '  T2_lim99:', pcaobj['T2_lim99'])
above_99 = t2_result['T2new'].ravel() > pcaobj['T2_lim99']
print('Observations above 99% limit:', X_new.iloc[:, 0].values[above_99])
"""),
M("## Interpreting SPE and T² Violations\n\n- **T² high, SPE normal**: observation is within the model hyperplane but at an extreme score location.\n- **SPE high**: observation has structure not captured by the model (possible process fault or new condition).\n- **Both high**: severe outlier."),
C("""\
spe_new = pca_pred['speXnew'].ravel()
t2_new  = t2_result['T2new'].ravel()
for i, obs in enumerate(X_new.iloc[:, 0].values):
    spe_flag = '!' if spe_new[i] > pcaobj['speX_lim99'] else ' '
    t2_flag  = '!' if t2_new[i]  > pcaobj['T2_lim99']  else ' '
    print(f'{obs:20s}  SPE={spe_new[i]:.3f}{spe_flag}  T2={t2_new[i]:.3f}{t2_flag}')
"""),
]
os.makedirs('notebooks/01_core', exist_ok=True)
with open('notebooks/01_core/03_prediction_and_monitoring.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_03_pred.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_03_pred.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_core/03_prediction_and_monitoring.ipynb \
  --output notebooks/01_core/03_prediction_and_monitoring.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_core/03_prediction_and_monitoring.ipynb
git commit -m "feat: add 01_core/03_prediction_and_monitoring tutorial notebook"
```

---

## Task 4: `01_core/04_missing_data.ipynb`

**Files:**
- Create: `notebooks/01_core/04_missing_data.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_04_md.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Missing Data Handling\n\nNIPALS estimates scores iteratively, skipping missing elements. The NLP method solves an optimization problem — more accurate but requires IPOPT or falls back to the NEOS server."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data with Missing Values"),
C("""\
features_md = pd.read_excel('../data/Automobiles PCA w MD.xlsx', 'Features',
                            na_values=np.nan, engine='openpyxl')
classid     = pd.read_excel('../data/Automobiles PCA w MD.xlsx', 'CLASSID',
                            na_values=np.nan, engine='openpyxl')

# Count missing values per variable
data_cols = features_md.iloc[:, 1:]
print('Missing values per variable:')
print(data_cols.isna().sum()[data_cols.isna().sum() > 0])
"""),
M("## NIPALS (default) — handles missing data natively"),
C("""\
pcaobj_nipals = phi.pca(features_md, 3, md_algorithm='nipals')
print('NIPALS loadings (PC1):', pcaobj_nipals['P'][:, 0])
pp.score_scatter(pcaobj_nipals, [1, 2], CLASSID=classid, colorby='Cylinders')
"""),
M("## Complete-Data Baseline"),
C("""\
features_complete = pd.read_excel('../data/Automobiles PLS.xlsx', 'Features',
                                  na_values=np.nan, engine='openpyxl')
pcaobj_complete = phi.pca(features_complete, 3)
print('Complete-data loadings (PC1):', pcaobj_complete['P'][:, 0])
"""),
M("## Effect of Missing Variables on Predictions\n\nWhen `Xnew` contains `np.nan`, NIPALS-based prediction still works but SPE/T² values may diverge from the stored training values. This is a known limitation (see xfail tests)."),
C("""\
# Introduce a missing value in a new observation
X_new_with_nan = features_complete.iloc[-5:].copy().reset_index(drop=True)
X_new_with_nan.iloc[0, 2] = np.nan  # blank out one variable
pred = phi.pca_pred(X_new_with_nan, pcaobj_complete)
print('Tnew for rows with/without NaN:')
print(pred['Tnew'])
"""),
M("## NLP Algorithm (requires IPOPT or NEOS server)\n\nSet `md_algorithm='nlp'` to use the NLP-based estimator. Requires either IPOPT in your system PATH or a valid `NEOS_EMAIL` environment variable."),
C("""\
import os
ipopt_available = bool(__import__('shutil').which('ipopt'))
neos_configured = bool(os.environ.get('NEOS_EMAIL'))

if ipopt_available or neos_configured:
    pcaobj_nlp = phi.pca(features_md, 3, md_algorithm='nlp')
    print('NLP loadings (PC1):', pcaobj_nlp['P'][:, 0])
else:
    print('Skipping NLP: IPOPT not found and NEOS_EMAIL not set.')
    print('To use NLP: install IPOPT or set NEOS_EMAIL=your@email.com')
"""),
]
os.makedirs('notebooks/01_core', exist_ok=True)
with open('notebooks/01_core/04_missing_data.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_04_md.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_04_md.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_core/04_missing_data.ipynb \
  --output notebooks/01_core/04_missing_data.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_core/04_missing_data.ipynb
git commit -m "feat: add 01_core/04_missing_data tutorial notebook"
```

---

## Task 5: `02_spectra/01_nir_calibration.ipynb`

**Files:**
- Create: `notebooks/02_spectra/01_nir_calibration.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_05_nir.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# NIR Calibration with Spectral Preprocessing\n\nDemonstrates the effect of SNV and Savitzky-Golay preprocessing on PLS calibration quality for pharmaceutical tablets.\n\nData: Dyrby et al., *Applied Spectroscopy* 56(5), 2002."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data"),
C("""\
NIR_raw   = pd.read_excel('../data/NIR.xlsx', 'NIR',         na_values=np.nan)
API_conc  = pd.read_excel('../data/NIR.xlsx', 'Y',           na_values=np.nan)
tab_class = pd.read_excel('../data/NIR.xlsx', 'Categorical', na_values=np.nan)
print('Spectra:', NIR_raw.shape, '  Y:', API_conc.shape)
"""),
M("## Raw Spectra"),
C("pp.plot_spectra(NIR_raw, plot_title='Raw NIR Spectra', xaxis_label='Channel', yaxis_label='a.u.')"),
M("## Preprocessing Step 1: SNV (Standard Normal Variate)\n\nEliminates multiplicative scatter effects (particle size variation)."),
C("""\
NIR_snv = phi.spectra_snv(NIR_raw)
pp.plot_spectra(NIR_snv, plot_title='After SNV', xaxis_label='Channel', yaxis_label='a.u.')
"""),
M("## Preprocessing Step 2: MSC (Multiplicative Scatter Correction)\n\nAlternative to SNV; corrects for scatter by regressing each spectrum against a reference."),
C("""\
NIR_msc = phi.spectra_msc(NIR_raw)
pp.plot_spectra(NIR_msc, plot_title='After MSC', xaxis_label='Channel', yaxis_label='a.u.')
"""),
M("## Preprocessing Step 3: Savitzky-Golay Derivative\n\n`spectra_savgol(window, deriv_order, poly_order, data)` — returns `(transformed, filter_matrix)`."),
C("""\
NIR_snv_sg, _ = phi.spectra_savgol(10, 1, 2, NIR_snv)
pp.plot_spectra(NIR_snv_sg, plot_title='After SNV + SavGol(10,1,2)',
                xaxis_label='Channel', yaxis_label='a.u.')
"""),
M("## PLS Calibration: Raw Spectra"),
C("""\
pls_raw = phi.pls(NIR_raw, API_conc, 3, mcsX='center', mcsY='center')
print('Q²Y (raw):', pls_raw.get('q2y', 'no CV'))
pp.predvsobs(pls_raw, NIR_raw, API_conc)
"""),
M("## PLS Calibration: SNV Preprocessed (with CV)"),
C("""\
pls_snv = phi.pls(NIR_snv, API_conc, 3, mcsX='center', mcsY='center', cross_val=10)
print('Q²Y (SNV):', pls_snv['q2y'])
pp.predvsobs(pls_snv, NIR_snv, API_conc)
"""),
M("## PLS Calibration: SNV + SavGol (best model)"),
C("""\
pls_best = phi.pls(NIR_snv_sg, API_conc, 1, mcsX='center', mcsY='center', cross_val=10)
print('Q²Y (SNV+SavGol, 1 LV):', pls_best['q2y'])
pp.predvsobs(pls_best, NIR_snv_sg, API_conc, CLASSID=tab_class, colorby='Type')
pp.predvsobs(pls_best, NIR_snv_sg, API_conc, CLASSID=tab_class, colorby='Scale')
"""),
M("## Summary\n\nCompare Q²Y across the three models to see the benefit of preprocessing. One latent variable after SNV+SavGol typically outperforms three LVs on raw spectra."),
]
os.makedirs('notebooks/02_spectra', exist_ok=True)
with open('notebooks/02_spectra/01_nir_calibration.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_05_nir.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_05_nir.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/02_spectra/01_nir_calibration.ipynb \
  --output notebooks/02_spectra/01_nir_calibration.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_spectra/01_nir_calibration.ipynb
git commit -m "feat: add 02_spectra/01_nir_calibration tutorial notebook"
```

---

## Task 6: `03_advanced/01_multiblock_pls.ipynb`

**Files:**
- Create: `notebooks/03_advanced/01_multiblock_pls.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_06_mbpls.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Multi-Block PLS\n\nMB-PLS handles X data from multiple sources (blocks) simultaneously, assigning a weight to each block. Useful when variables come from fundamentally different measurement sources (e.g., NIR + process + lab)."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data\n\n`XMB` must be a **dict** mapping block names to DataFrames (each with obs-ID first column)."),
C("""\
xb = {
    f'X{i}': pd.read_excel('../data/MBDataset.xlsx', sheet_name=f'X{i}')
    for i in range(1, 7)
}
y_data = pd.read_excel('../data/MBDataset.xlsx', sheet_name='Y')
for k, v in xb.items():
    print(k, v.shape)
print('Y:', y_data.shape)
"""),
M("## Build MB-PLS Model"),
C("""\
mbpls_obj = phi.mbpls(xb, y_data, 2)
print('Keys:', list(mbpls_obj.keys()))
"""),
M("## Score and Loadings Plots"),
C("""\
pp.score_scatter(mbpls_obj, [1, 2])
pp.loadings(mbpls_obj)
pp.weighted_loadings(mbpls_obj)
"""),
M("## Block-Level Diagnostics"),
C("""\
pp.r2pv(mbpls_obj)
pp.mb_r2pb(mbpls_obj)   # R² per block
pp.mb_weights(mbpls_obj)  # block super-weights
pp.mb_vip(mbpls_obj)      # VIP per block
pp.vip(mbpls_obj)         # overall VIP
"""),
M("## Prediction on New Data\n\nPass the same dict structure to `pls_pred`."),
C("""\
preds = phi.pls_pred(xb, mbpls_obj)
print('Yhat shape:', preds['Yhat'].shape)
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/01_multiblock_pls.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_06_mbpls.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_06_mbpls.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/01_multiblock_pls.ipynb \
  --output notebooks/03_advanced/01_multiblock_pls.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/01_multiblock_pls.ipynb
git commit -m "feat: add 03_advanced/01_multiblock_pls tutorial notebook"
```

---

## Task 7: `03_advanced/02_opls_cca.ipynb`

**Files:**
- Create: `notebooks/03_advanced/02_opls_cca.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_07_opls.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# OPLS via PLS-CCA\n\nThe `cca=True` flag in `phi.pls()` computes covariant (predictive) scores and loadings equivalent to OPLS. The covariant scores `Tcv` and loadings `Pcv` isolate Y-relevant variation; orthogonal components capture the rest.\n\nReference: Yu & MacGregor (same dataset used to validate against SIMCA-P OPLS)."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
import matplotlib.pyplot as plt
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
%matplotlib inline
"""),
M("## Load Data"),
C("""\
X_df     = pd.read_excel('../data/OPLS Test Data.xlsx', 'X')
Y_df     = pd.read_excel('../data/OPLS Test Data.xlsx', 'Y')
print('X:', X_df.shape, '  Y:', Y_df.shape)
"""),
M("## Build PLS with CCA flag"),
C("""\
pls_cca = phi.pls(X_df, Y_df, 5, cca=True)
print('Extra keys added by cca=True:', [k for k in pls_cca if k not in ['T','P','Q','W','Ws','r2x','r2y']])
print('Tcv shape:', pls_cca['Tcv'].shape)
print('Pcv shape:', pls_cca['Pcv'].shape)
"""),
M("## Covariant (Predictive) Loadings\n\n`Pcv` is equivalent to OPLS predictive loadings. They isolate the X-variation correlated with Y."),
C("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pls_cca['Pcv'], 'b-')
axes[0].set_title('Pcv (Predictive Loadings)')
axes[0].set_xlabel('Variable')
axes[0].set_ylabel('Loading')

axes[1].plot(pls_cca['Tcv'], 'b-')
axes[1].set_title('Tcv (Predictive Scores)')
axes[1].set_xlabel('Observation')
axes[1].set_ylabel('Score')
plt.tight_layout()
plt.show()
"""),
M("## Score and Loadings Plots via pyphi.plots"),
C("""\
pp.score_scatter(pls_cca, [1, 2])
pp.loadings(pls_cca)
pp.r2pv(pls_cca)
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/02_opls_cca.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_07_opls.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_07_opls.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/02_opls_cca.ipynb \
  --output notebooks/03_advanced/02_opls_cca.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/02_opls_cca.ipynb
git commit -m "feat: add 03_advanced/02_opls_cca tutorial notebook"
```

---

## Task 8: `03_advanced/03_lpls.ipynb`

**Files:**
- Create: `notebooks/03_advanced/03_lpls.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_08_lpls.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# L-shaped PLS (LPLS)\n\nLPLS models three linked matrices: X (blends × process), R (materials × properties), and Y (blends × quality). R links material-level information to blend-level outcomes.\n\nArguments: `phi.lpls(X, R, Y, A)`"),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data"),
C("""\
X = pd.read_excel('../data/lpls_dataset.xlsx', sheet_name='X')
R = pd.read_excel('../data/lpls_dataset.xlsx', sheet_name='R')
Y = pd.read_excel('../data/lpls_dataset.xlsx', sheet_name='Y')
print('X (blends × process):', X.shape)
print('R (materials × props):', R.shape)
print('Y (blends × quality):', Y.shape)
"""),
M("## Build LPLS Model"),
C("""\
lpls_obj = phi.lpls(X, R, Y, 4)
print('Keys:', list(lpls_obj.keys()))
print('type:', lpls_obj['type'])
"""),
M("## Score Plots\n\n`rscores=True` shows scores in the material (R) space."),
C("""\
pp.score_scatter(lpls_obj, [1, 2], add_labels=True, addtitle='Blend Scores')
pp.score_scatter(lpls_obj, [1, 2], add_labels=True, rscores=True, addtitle='Material Scores')
"""),
M("## Loadings and VIP"),
C("""\
pp.loadings_map(lpls_obj, [1, 2], addtitle='LPLS Model')
pp.loadings(lpls_obj, addtitle='LPLS Model')
pp.weighted_loadings(lpls_obj)
pp.vip(lpls_obj)
pp.r2pv(lpls_obj)
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/03_lpls.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_08_lpls.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_08_lpls.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/03_lpls.ipynb \
  --output notebooks/03_advanced/03_lpls.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/03_lpls.ipynb
git commit -m "feat: add 03_advanced/03_lpls tutorial notebook"
```

---

## Task 9: `03_advanced/04_jrpls_tpls.ipynb`

**Files:**
- Create: `notebooks/03_advanced/04_jrpls_tpls.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_09_jrpls.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# JR-PLS and T-PLS\n\n**JR-PLS**: Joint R-PLS links multiple material matrices `Xi` (one per material type) and recipe matrices `Ri` to a common quality `Y`. **T-PLS** adds a process block `Z` on top of JR-PLS.\n\nData loading uses `parse_materials` (reads the material list sheet) and `reconcile_rows_to_columns` / `reconcile_rows` to align observations."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load and Reconcile Data"),
C("""\
jr, materials = phi.parse_materials('../data/jrpls_tpls_dataset.xlsx', 'Materials')
x = [pd.read_excel('../data/jrpls_tpls_dataset.xlsx', sheet_name=m) for m in materials]
xc, jrc = phi.reconcile_rows_to_columns(x, jr)

quality = pd.read_excel('../data/jrpls_tpls_dataset.xlsx', sheet_name='QUALITY')
process = pd.read_excel('../data/jrpls_tpls_dataset.xlsx', sheet_name='PROCESS')

jrc.append(process)
jrc.append(quality)
AUX     = phi.reconcile_rows(jrc)
JR_     = AUX[:-2]
process = AUX[-2]
quality = AUX[-1]

Ri = {m: j for m, j in zip(materials, JR_)}
Xi = {m: x_ for m, x_ in zip(materials, xc)}

print('Materials:', materials)
print('Quality shape:', quality.shape)
"""),
M("## Build JR-PLS Model"),
C("""\
jrpls_obj = phi.jrpls(Xi, Ri, quality, 4)
print('Keys:', list(jrpls_obj.keys()))
"""),
M("## JR-PLS Plots"),
C("""\
pp.r2pv(jrpls_obj, material='MAT4', addtitle='For MAT4')
pp.score_scatter(jrpls_obj, [1, 2])
pp.score_scatter(jrpls_obj, [1, 2], rscores=True)
pp.loadings(jrpls_obj, material='MAT4')
pp.vip(jrpls_obj, plotwidth=1000, material='MAT4', addtitle='MAT4')
"""),
M("## JR-PLS Prediction\n\nNew blends are specified as a dict of `{material: [(lot_id, fraction), ...]}`"),
C("""\
rnew = {
    'MAT1': [('A0129', 0.558), ('A0130', 0.442)],
    'MAT2': [('Lac0003', 1)],
    'MAT3': [('TLC018', 1)],
    'MAT4': [('M0012', 1)],
    'MAT5': [('CS0017', 1)],
}
preds = phi.jrpls_pred(rnew, jrpls_obj)
print('JR-PLS prediction:', preds)
"""),
M("## Build T-PLS Model\n\nT-PLS adds a process (Z) block on top of the JR-PLS material structure."),
C("""\
tpls_obj = phi.tpls(Xi, Ri, process, quality, 4)
pp.r2pv(tpls_obj)
pp.r2pv(tpls_obj, zspace=True)
pp.loadings(tpls_obj, zspace=True)
pp.weighted_loadings(tpls_obj, zspace=True)
pp.vip(tpls_obj, plotwidth=1000, zspace=True, addtitle='Z Space')
"""),
M("## T-PLS Prediction"),
C("""\
znew = process[process.iloc[:, 0] == 'L001'].values.reshape(-1)[1:].astype(float)
preds_t = phi.tpls_pred(rnew, znew, tpls_obj)
print('T-PLS prediction:', preds_t)
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/04_jrpls_tpls.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_09_jrpls.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_09_jrpls.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/04_jrpls_tpls.ipynb \
  --output notebooks/03_advanced/04_jrpls_tpls.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/04_jrpls_tpls.ipynb
git commit -m "feat: add 03_advanced/04_jrpls_tpls tutorial notebook"
```

---

## Task 10: `03_advanced/05_varimax_rotation.ipynb`

**Files:**
- Create: `notebooks/03_advanced/05_varimax_rotation.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_10_varimax.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Varimax Rotation\n\nVarimax rotation maximises the variance of squared loadings, producing a simpler loading structure where each variable loads heavily on as few components as possible. Useful when the unrotated PLS loadings are spread across all components.\n\n`phi.varimax_rotation(mvmobj, X, Y=None)` returns a new model dict with rotated T, P, W, Ws, and recalculated R²pv."),
C("""\
import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
"""),
M("## Load Data"),
C("""\
ra_data = pd.read_excel('../data/chemical_experiments_dataset.xlsx', sheet_name='data')
ra_data, removed_cols = phi.clean_low_variances(ra_data)
print(f'Removed {len(removed_cols)} near-zero variance columns:', removed_cols)

x_cols = ['Lot', 'SM (eq)', 'A (eq)', 'B (eq)', 'C (eq)', 'Solvent (Vol)',
          'drops C', 'drops D', 'Water (vol)', 'N addition order',
          'SM addition order', 'A addition order', 'B addition order',
          'C Addition Temp C', 'N Addition Temp C', 'Reaction Temp C',
          'number of portions N']
y_cols = ['Lot', 'IPC_N', 'IPC_H', 'IPC_RT_17min', 'IPC_RT_19min']

ra_X = ra_data[x_cols]
ra_Y = ra_data[y_cols]
print('X:', ra_X.shape, '  Y:', ra_Y.shape)
"""),
M("## Build PLS Model (unrotated)"),
C("""\
plsobj = phi.pls(ra_X, ra_Y, 5)
pp.weighted_loadings(plsobj)
pp.loadings_map(plsobj, [1, 2], plotwidth=600)
pp.score_scatter(plsobj, [1, 2], add_ci=True)
pp.r2pv(plsobj)
"""),
M("## Apply Varimax Rotation"),
C("""\
plsobj_rot = phi.varimax_rotation(plsobj, ra_X, Y=ra_Y)
print('Keys unchanged:', set(plsobj.keys()) == set(plsobj_rot.keys()))
"""),
M("## Rotated Loadings — Simpler Structure"),
C("""\
pp.weighted_loadings(plsobj_rot)
pp.loadings_map(plsobj_rot, [1, 2], plotwidth=600)
pp.score_scatter(plsobj_rot, [1, 2], add_ci=True)
pp.r2pv(plsobj_rot)
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/05_varimax_rotation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_10_varimax.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_10_varimax.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/05_varimax_rotation.ipynb \
  --output notebooks/03_advanced/05_varimax_rotation.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/05_varimax_rotation.ipynb
git commit -m "feat: add 03_advanced/05_varimax_rotation tutorial notebook"
```

---

## Task 11: `03_advanced/06_lwpls.ipynb`

**Files:**
- Create: `notebooks/03_advanced/06_lwpls.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_11_lwpls.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Locally Weighted PLS (LWPLS)\n\nLWPLS fits a local PLS model around each query point, weighting training samples by similarity to the query. Effective for nonlinear data where a global PLS model performs poorly.\n\n`phi.lwpls(xnew, loc_par, mvm_pls, X_train, Y_train)` returns the scalar prediction for `xnew`.\n\nData: same NIR tablet dataset as the spectra notebook (MATLAB format)."),
C("""\
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import pyphi.calc as phi
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
%matplotlib inline
"""),
M("## Load and Preprocess Data"),
C("""\
NIRData = spio.loadmat('../data/NIRdata_tablets.MAT')
X = np.array(NIRData['Matrix'][:, 3:])
Y = np.array(NIRData['Matrix'][:, 0])

# Even-indexed rows = calibration; odd = validation
Xcal = X[::2, :]
Xval = X[1:X.shape[0]:2, :]

Xcal = phi.spectra_snv(Xcal)
Xval = phi.spectra_snv(Xval)
Xcal, _ = phi.spectra_savgol(5, 1, 2, Xcal)
Xval, _ = phi.spectra_savgol(5, 1, 2, Xval)

Ycal = Y[::2].reshape(-1, 1)
Yval = Y[1:X.shape[0]:2].reshape(-1, 1)
print('Cal:', Xcal.shape, Ycal.shape, '  Val:', Xval.shape, Yval.shape)
"""),
M("## Global PLS Baseline"),
C("""\
mvm_pls = phi.pls(Xcal, Ycal, 1, mcsX='center', mcsY='center')
pred_cal = phi.pls_pred(Xcal, mvm_pls)
pred_val = phi.pls_pred(Xval, mvm_pls)
rmse_pls_cal = float(np.sqrt(np.mean((Ycal - pred_cal['Yhat'])**2)))
rmse_pls_val = float(np.sqrt(np.mean((Yval - pred_val['Yhat'])**2)))
print(f'Global PLS  RMSE — CAL: {rmse_pls_cal:.4f}  VAL: {rmse_pls_val:.4f}')
"""),
M("## LWPLS — Sweep Localization Parameter\n\nSmall `loc_par` = very local model; large = approaches global PLS. Tune by minimising validation RMSE."),
C("""\
loc_params = [5, 10, 15, 20, 25, 30, 40, 50, 100]
rmse_lw_cal, rmse_lw_val = [], []

for loc_par in loc_params:
    # Calibration predictions
    yhat_cal = np.array([
        phi.lwpls(Xcal[i, :], loc_par, mvm_pls, Xcal, Ycal, shush=True)[0]
        for i in range(Xcal.shape[0])
    ]).reshape(-1, 1)
    # Validation predictions
    yhat_val = np.array([
        phi.lwpls(Xval[i, :], loc_par, mvm_pls, Xcal, Ycal, shush=True)[0]
        for i in range(Xval.shape[0])
    ]).reshape(-1, 1)

    rmse_lw_cal.append(float(np.sqrt(np.mean((Ycal - yhat_cal)**2))))
    rmse_lw_val.append(float(np.sqrt(np.mean((Yval - yhat_val)**2))))
    print(f'loc_par={loc_par:4d}  RMSE_CAL={rmse_lw_cal[-1]:.4f}  RMSE_VAL={rmse_lw_val[-1]:.4f}')
"""),
M("## RMSE vs Localization Parameter"),
C("""\
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loc_params, rmse_lw_cal, 'o-b', label='LWPLS CAL')
ax.plot(loc_params, rmse_lw_val, 'o-r', label='LWPLS VAL')
ax.axhline(rmse_pls_cal, ls='--', color='b', alpha=0.4, label='Global PLS CAL')
ax.axhline(rmse_pls_val, ls='--', color='r', alpha=0.4, label='Global PLS VAL')
ax.set_xlabel('Localization Parameter')
ax.set_ylabel('RMSE')
ax.legend()
ax.set_title('LWPLS vs Global PLS')
plt.tight_layout()
plt.show()
best_idx = int(np.argmin(rmse_lw_val))
print(f'Best loc_par: {loc_params[best_idx]}  (Val RMSE: {rmse_lw_val[best_idx]:.4f})')
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/06_lwpls.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_11_lwpls.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_11_lwpls.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/06_lwpls.ipynb \
  --output notebooks/03_advanced/06_lwpls.ipynb \
  --ExecutePreprocessor.timeout=300
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/06_lwpls.ipynb
git commit -m "feat: add 03_advanced/06_lwpls tutorial notebook"
```

---

## Task 12: `03_advanced/07_bootstrap_and_polynomial.ipynb`

**Files:**
- Create: `notebooks/03_advanced/07_bootstrap_and_polynomial.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_12_boot.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Bootstrap PLS and Polynomial Regression\n\n**`bootstrap_pls`**: resamples the training data to estimate uncertainty in PLS coefficients.\n\n**`build_polynomial`**: constructs polynomial/interaction terms, uses PLS variable selection to identify significant terms, and returns regression coefficients and equation string."),
C("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pyphi.calc as phi
warnings.simplefilter('ignore')
%matplotlib inline
"""),
M("## Bootstrap PLS\n\n`phi.bootstrap_pls(X, Y, num_latents, num_samples)` — returns a dict with `Ws_std` (std dev of Ws across bootstrap samples), `T_std`, etc."),
C("""\
features    = pd.read_excel('../data/Automobiles PLS.xlsx', 'Features',
                            na_values=np.nan, engine='openpyxl')
performance = pd.read_excel('../data/Automobiles PLS.xlsx', 'Performance',
                            na_values=np.nan, engine='openpyxl')

boot = phi.bootstrap_pls(features, performance, 3, 100)
print('Bootstrap result keys:', list(boot.keys()))
"""),
C("""\
# Plot Ws ± 2*std for LV1
var_names = features.columns[1:].tolist()
ws1     = boot['Ws'][:, 0]
ws1_std = boot['Ws_std'][:, 0]

fig, ax = plt.subplots(figsize=(10, 4))
x_pos = np.arange(len(var_names))
ax.bar(x_pos, ws1, color='steelblue', label='Ws LV1')
ax.errorbar(x_pos, ws1, yerr=2*ws1_std, fmt='none', color='black', capsize=3)
ax.set_xticks(x_pos)
ax.set_xticklabels(var_names, rotation=45, ha='right')
ax.set_title('Bootstrap Ws (LV1) ± 2σ  (100 resamples)')
ax.set_ylabel('Weight')
plt.tight_layout()
plt.show()
"""),
M("## Polynomial Regression with PLS Variable Selection\n\n`build_polynomial(data, factors, response)` selects terms via PLS cross-validation and fits OLS on the selected subset. Returns `(betas, selected_factors, X_matrix, Y_vector, equation_string)`."),
C("""\
ex_data = pd.read_excel('../data/polynomial_data.xlsx')
print(ex_data.columns.tolist())
print(ex_data.shape)
"""),
C("""\
factors = [
    'Variable A', 'Var B', 'VarC',
    'Variable A*VarC', 'Var B^2', 'Variable A^2*VarC'
]
betas, selected, X_mat, Y_vec, eq_str = phi.build_polynomial(ex_data, factors, 'Response 1')
print('Selected factors:', selected)
print('Equation:', eq_str)
"""),
C("""\
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(Y_vec, X_mat @ betas, 'o', color='steelblue')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Polynomial Model — Response 1')
lims = [min(Y_vec.min(), (X_mat@betas).min()),
        max(Y_vec.max(), (X_mat@betas).max())]
ax.plot(lims, lims, 'k--', alpha=0.4)
plt.tight_layout()
plt.show()
"""),
M("## Reduced Model (fewer terms)"),
C("""\
factors_reduced = ['Variable A', 'VarC', 'Variable A*VarC', 'Variable A^2*VarC']
betas2, selected2, X2, Y2, eq2 = phi.build_polynomial(ex_data, factors_reduced, 'Response 1')
print('Selected factors:', selected2)
print('Equation:', eq2)
"""),
]
os.makedirs('notebooks/03_advanced', exist_ok=True)
with open('notebooks/03_advanced/07_bootstrap_and_polynomial.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_12_boot.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_12_boot.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/03_advanced/07_bootstrap_and_polynomial.ipynb \
  --output notebooks/03_advanced/07_bootstrap_and_polynomial.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_advanced/07_bootstrap_and_polynomial.ipynb
git commit -m "feat: add 03_advanced/07_bootstrap_and_polynomial tutorial notebook"
```

---

## Task 13: `04_batch/01_batch_alignment.ipynb`

**Files:**
- Create: `notebooks/04_batch/01_batch_alignment.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_13_align.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Batch Alignment\n\nBatch trajectories must be time-aligned before unfolding for multi-way modelling. Three alignment strategies: simple (fixed count), phase-based, and indicator-variable (IV) alignment.\n\nBatch DataFrame format: col 1 = batch ID (string), col 2 = `PHASE` (optional phase label)."),
C("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyphi.batch as phibatch
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
%matplotlib inline
"""),
M("## Load Raw Batch Data"),
C("""\
bdata = pd.read_excel('../data/Batch Film Coating.xlsx')
print('Columns:', bdata.columns.tolist())
print('Batches:', bdata.iloc[:, 0].unique().tolist())
phibatch.plot_var_all_batches(bdata)
"""),
M("## Simple Alignment\n\nResamples each batch to exactly N samples (interpolation). Ignores phase structure."),
C("""\
bdata_simple = phibatch.simple_align(bdata, 250)
phibatch.plot_var_all_batches(bdata_simple, plot_title='Simple alignment (250 samples/batch)')
"""),
M("## Phase Sampling Distribution\n\nCheck how many samples per phase each batch has before choosing `samples_per_phase`."),
C("phibatch.phase_sampling_dist(bdata)"),
M("## Phase-Based Alignment\n\nAligns each phase independently to a fixed count. Preserves phase boundaries."),
C("""\
samples_per_phase = {
    'STARTUP': 3, 'HEATING': 20, 'SPRAYING': 40,
    'DRYING': 40, 'DISCHARGING': 5
}
bdata_phase = phibatch.phase_simple_align(bdata, samples_per_phase)
phibatch.plot_var_all_batches(bdata_phase,
                              plot_title='Phase alignment',
                              phase_samples=samples_per_phase)
"""),
M("## Indicator Variable (IV) Alignment\n\nFor a given phase, use a process variable trajectory (e.g. temperature) to determine alignment rather than sample count.\n\nSyntax: replace the sample count for a phase with `[variable_name, n_samples, end_value]`."),
C("""\
samples_iv = {
    'STARTUP': 3,
    'HEATING': ['INLET_AIR_TEMP', 35, 67],  # 35 samples from start until 67°C
    'SPRAYING': 40,
    'DRYING': 40,
    'DISCHARGING': 5
}
bdata_iv = phibatch.phase_iv_align(bdata, samples_iv)
phibatch.plot_var_all_batches(bdata_iv,
                              plot_title='IV alignment (Inlet Temp during Heating)',
                              phase_samples=samples_iv)
"""),
M("## Plot Individual Batches"),
C("""\
phibatch.plot_batch(bdata_phase, which_batch='B1805', which_var='INLET_AIR_TEMP',
                   include_set=True, include_mean_exc=True,
                   phase_samples=samples_per_phase)
"""),
]
os.makedirs('notebooks/04_batch', exist_ok=True)
with open('notebooks/04_batch/01_batch_alignment.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_13_align.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_13_align.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/04_batch/01_batch_alignment.ipynb \
  --output notebooks/04_batch/01_batch_alignment.ipynb \
  --ExecutePreprocessor.timeout=180
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_batch/01_batch_alignment.ipynb
git commit -m "feat: add 04_batch/01_batch_alignment tutorial notebook"
```

---

## Task 14: `04_batch/02_mpca_mpls.ipynb`

**Files:**
- Create: `notebooks/04_batch/02_mpca_mpls.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_14_mpca.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Multi-Way PCA and PLS for Batch Data\n\nAfter alignment, batch data is unfolded into a 2D matrix (observations × time×variables) for MPCA/MPLS.\n\n`unfold_horizontal` returns a 3-tuple `(df, clbl, bid)` — not a plain DataFrame."),
C("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyphi.batch as phibatch
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
%matplotlib inline
"""),
M("## Align Batch Data (from previous notebook)"),
C("""\
bdata = pd.read_excel('../data/Batch Film Coating.xlsx')
samples_per_phase = {
    'STARTUP': 3, 'HEATING': 20, 'SPRAYING': 40,
    'DRYING': 40, 'DISCHARGING': 5
}
bdata_aligned = phibatch.phase_simple_align(bdata, samples_per_phase)

# Remove abnormal batches for NOC model
noc = bdata_aligned[~bdata_aligned.iloc[:, 0].isin(['B1905', 'B1805'])].copy()
dev = bdata_aligned[ bdata_aligned.iloc[:, 0].isin(['B1905', 'B1805'])].copy()
print('NOC batches:', noc.iloc[:, 0].unique().tolist())
"""),
M("## Build MPCA Model"),
C("""\
mpca_obj = phibatch.mpca(noc, 2, phase_samples=samples_per_phase, cross_val=5)
print('Keys:', list(mpca_obj.keys()))
"""),
M("## Score Plot and Diagnostics"),
C("""\
pp.score_scatter(mpca_obj, [1, 2], add_labels=True)
pp.diagnostics(mpca_obj)
"""),
M("## R² per Variable and Loadings"),
C("""\
phibatch.r2pv(mpca_obj)
pp.r2pv(mpca_obj, plotwidth=1500)
phibatch.loadings_abs_integral(mpca_obj)
phibatch.loadings(mpca_obj, 2,
                  which_var=['INLET_AIR_TEMP', 'EXHAUST_AIR_TEMP', 'TOTAL_SPRAY_USED'],
                  r2_weighted=True)
"""),
M("## Batch Contributions"),
C("""\
phibatch.contributions(mpca_obj, noc, 'scores', to_obs=['B1910'],
                       plot_title='Contributions to B1910')
phibatch.contributions(mpca_obj, noc, 'scores', to_obs=['B1910'],
                       dyn_conts=True, plot_title='Dynamic Contributions to B1910')
"""),
M("## Build MPLS Model (Dryer Dataset)\n\nMPLS predicts end-of-batch quality from trajectory data."),
C("""\
bdata_d = pd.read_excel('../data/Batch Dryer Case Study.xlsx', sheet_name='Trajectories')
cqa     = pd.read_excel('../data/Batch Dryer Case Study.xlsx', sheet_name='ProductQuality')
samples_d = {'Deagglomerate': 20, 'Heat': 30, 'Cooldown': 40}

bdata_d_aligned = phibatch.phase_simple_align(bdata_d, samples_d)
mpls_obj = phibatch.mpls(bdata_d_aligned, cqa, 3, phase_samples=samples_d)
print('MPLS Keys:', list(mpls_obj.keys()))
phibatch.r2pv(mpls_obj)
"""),
]
os.makedirs('notebooks/04_batch', exist_ok=True)
with open('notebooks/04_batch/02_mpca_mpls.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_14_mpca.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_14_mpca.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/04_batch/02_mpca_mpls.ipynb \
  --output notebooks/04_batch/02_mpca_mpls.ipynb \
  --ExecutePreprocessor.timeout=300
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_batch/02_mpca_mpls.ipynb
git commit -m "feat: add 04_batch/02_mpca_mpls tutorial notebook"
```

---

## Task 15: `04_batch/03_batch_monitoring.ipynb`

**Files:**
- Create: `notebooks/04_batch/03_batch_monitoring.ipynb`

- [ ] **Step 1: Create notebook**

```python
# Run: uv run python /tmp/make_15_mon.py
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell
nb.cells = [
M("# Batch Monitoring, Prediction, and Forecasting\n\n`phibatch.monitor()` projects a batch (or partial batch) onto an MPCA/MPLS model and returns instantaneous T², SPE, and a complete-trajectory forecast.\n\n`phibatch.predict()` gives end-of-batch quality predictions from a partial trajectory."),
C("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyphi.batch as phibatch
import pyphi.plots as pp
from bokeh.io import output_notebook
output_notebook(hide_banner=True)
import pyphi.plots as _ppmod
_ppmod.output_file = lambda *a, **kw: None
%matplotlib inline
"""),
M("## Prepare NOC Model (Film Coating)"),
C("""\
bdata = pd.read_excel('../data/Batch Film Coating.xlsx')
samples_per_phase = {
    'STARTUP': 3, 'HEATING': 20, 'SPRAYING': 40,
    'DRYING': 40, 'DISCHARGING': 5
}
bdata_aligned = phibatch.phase_simple_align(bdata, samples_per_phase)
noc = bdata_aligned[~bdata_aligned.iloc[:, 0].isin(['B1905', 'B1805'])].copy()
dev = bdata_aligned[ bdata_aligned.iloc[:, 0].isin(['B1905', 'B1805'])].copy()

mpca_obj = phibatch.mpca(noc, 2, phase_samples=samples_per_phase)
"""),
M("## Monitor All Normal Batches"),
C("""\
all_batches = noc.iloc[:, 0].unique().tolist()
mon_noc = phibatch.monitor(mpca_obj, noc, which_batch=all_batches)
"""),
M("## Monitor a Deviant Batch (B1905)"),
C("""\
mon_dev = phibatch.monitor(mpca_obj, dev, which_batch=['B1905'])
"""),
M("## SPE Contributions at a Specific Sample\n\nInstantaneous contributions at sample 5 for batch B1905."),
C("""\
sam_num = 5
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(mon_dev['cont_spei'].columns, mon_dev['cont_spei'].iloc[sam_num - 1])
ax.set_xticklabels(mon_dev['cont_spei'].columns, rotation=90)
ax.set_ylabel('Contribution to i-SPE')
ax.set_title(f'SPE Contributions — B1905 at sample {sam_num}')
plt.tight_layout()
plt.show()
"""),
M("## Trajectory Forecasting\n\nAt any point in time, `monitor` returns a `'forecast'` key with predicted future trajectories based on the mean NOC trajectory."),
C("""\
batch2forecast = 'B2510'
mon_fc = phibatch.monitor(mpca_obj, noc, which_batch=batch2forecast)
forecast = mon_fc['forecast']

var = 'INLET_AIR_TEMP'
mdata = noc[noc.iloc[:, 0] == batch2forecast]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, pit in zip(axes.ravel(), [5, 20, 50, 80]):
    f = forecast[pit - 1]
    x_obs  = np.arange(pit) + 1
    x_fore = np.arange(len(mdata)) + 1
    fore_vals = np.array([np.nan] * pit + f[var].values[pit:].tolist())
    ax.plot(x_obs, mdata[var].values[:pit], 'o', label='Measured')
    ax.plot(x_fore, fore_vals, '-', label='Forecast')
    ax.plot(np.arange(pit, len(mdata)) + 1, mdata[var].values[pit:],
            'o', alpha=0.3, label='Known')
    ax.set_title(f'At sample {pit}')
    ax.set_xlabel('Sample')
    ax.set_ylabel(var)
    ax.legend(fontsize=7)
plt.suptitle(f'Trajectory Forecast — {batch2forecast}')
plt.tight_layout()
plt.show()
"""),
M("## Batch Descriptors\n\n`phibatch.descriptors()` extracts landmarks (min/max/mean of a variable within a phase) useful as features for end-of-batch models."),
C("""\
desc = phibatch.descriptors(bdata_aligned, 'INLET_AIR_TEMP',
                            ['min', 'max', 'mean'],
                            which_phase='SPRAYING')
print(desc.head())
"""),
M("## MPLS End-of-Batch Quality Prediction (Dryer Dataset)"),
C("""\
bdata_d = pd.read_excel('../data/Batch Dryer Case Study.xlsx', sheet_name='Trajectories')
cqa     = pd.read_excel('../data/Batch Dryer Case Study.xlsx', sheet_name='ProductQuality')
cat     = pd.read_excel('../data/Batch Dryer Case Study.xlsx', sheet_name='classifiers')
samples_d = {'Deagglomerate': 20, 'Heat': 30, 'Cooldown': 40}

bdata_d_aligned = phibatch.phase_simple_align(bdata_d, samples_d)
mpls_obj = phibatch.mpls(bdata_d_aligned, cqa, 3, phase_samples=samples_d)

pp.score_scatter(mpls_obj, [1, 2], CLASSID=cat, colorby='Quality')
mon_mpls = phibatch.monitor(mpls_obj, bdata_d_aligned, which_batch='Batch 5')
"""),
]
os.makedirs('notebooks/04_batch', exist_ok=True)
with open('notebooks/04_batch/03_batch_monitoring.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Written.')
```

Save as `/tmp/make_15_mon.py` and run:
```bash
cd /home/tensor/projects/pyphi && uv run python /tmp/make_15_mon.py
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /home/tensor/projects/pyphi && uv run jupyter nbconvert --to notebook --execute \
  notebooks/04_batch/03_batch_monitoring.ipynb \
  --output notebooks/04_batch/03_batch_monitoring.ipynb \
  --ExecutePreprocessor.timeout=300
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_batch/03_batch_monitoring.ipynb
git commit -m "feat: add 04_batch/03_batch_monitoring tutorial notebook"
```
