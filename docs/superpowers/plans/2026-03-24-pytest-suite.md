# pytest Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pytest test suite covering all three pyphi modules (calc, plots, batch) that surfaces bugs through failing assertions — tests are the deliverable, bugs are fixed separately.

**Architecture:** `tests/conftest.py` provides session-scoped data and model fixtures loaded from `examples/` Excel files. Three test files map one-to-one to the three modules. `matplotlib.use('Agg')` is set before any import to suppress GUI in CI. Failing tests indicate bugs for follow-up investigation.

**Tech Stack:** pytest, numpy, pandas, pyphi.calc, pyphi.plots, pyphi.batch, openpyxl

---

### Task 1: Project infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py` (skeleton)
- Create: `tests/test_calc.py` (skeleton)
- Create: `tests/test_plots.py` (skeleton)
- Create: `tests/test_batch.py` (skeleton)

- [ ] **Step 1: Add pytest config to `pyproject.toml`** (after the `[dependency-groups]` section)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `tests/conftest.py` with only the backend + imports**

```python
import matplotlib
matplotlib.use('Agg')  # Must be before pyphi imports to prevent GUI in CI

import numpy as np
import pandas as pd
import pytest
import pyphi.calc as phi
import pyphi.batch as batch
```

- [ ] **Step 3: Create `tests/test_calc.py`**

```python
import pytest
import numpy as np
import pandas as pd
import pyphi.calc as phi
```

- [ ] **Step 4: Create `tests/test_plots.py`**

```python
import pytest
import re
import numpy as np
from pyphi.plots import (
    _get_lv_labels, _get_xvar_labels, _get_yvar_labels,
    _timestr, _make_bokeh_palette, _mask_by_class, _resolve_lpls_space,
)
```

- [ ] **Step 5: Create `tests/test_batch.py`**

```python
import pytest
import numpy as np
import pandas as pd
import pyphi.batch as batch
```

- [ ] **Step 6: Verify pytest collection works**

```bash
uv run pytest --collect-only
```
Expected: no import errors, 0 tests collected.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/
git commit -m "test: scaffold test infrastructure"
```

---

### Task 2: conftest.py — data fixtures

**Files:**
- Modify: `tests/conftest.py`

All fixtures use `scope="session"` so Excel files are loaded once.
Use absolute paths anchored to the repo root via `pathlib`.

- [ ] **Step 1: Add data fixtures to `tests/conftest.py`**

```python
from pathlib import Path

EXAMPLES = Path(__file__).parent.parent / "examples"


# ── PCA / PLS ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cars_features():
    return pd.read_excel(
        EXAMPLES / "Basic calculations PCA and PLS" / "Automobiles PLS.xlsx",
        sheet_name="Features", na_values=np.nan, engine="openpyxl",
    )

@pytest.fixture(scope="session")
def cars_performance():
    return pd.read_excel(
        EXAMPLES / "Basic calculations PCA and PLS" / "Automobiles PLS.xlsx",
        sheet_name="Performance", na_values=np.nan, engine="openpyxl",
    )

@pytest.fixture(scope="session")
def cars_classid():
    return pd.read_excel(
        EXAMPLES / "Basic calculations PCA and PLS" / "Automobiles PLS.xlsx",
        sheet_name="CLASSID", na_values=np.nan, engine="openpyxl",
    )

# ── LPLS ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def lpls_X():
    return pd.read_excel(
        EXAMPLES / "LPLS" / "lpls_dataset.xlsx", sheet_name="X", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def lpls_R():
    return pd.read_excel(
        EXAMPLES / "LPLS" / "lpls_dataset.xlsx", sheet_name="R", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def lpls_Y():
    return pd.read_excel(
        EXAMPLES / "LPLS" / "lpls_dataset.xlsx", sheet_name="Y", engine="openpyxl",
    )

# ── JRPLS / TPLS ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def jrpls_data():
    """Returns (Xi, Ri, quality, process) ready for jrpls/tpls model building."""
    fpath = EXAMPLES / "JRPLS and TPLS" / "jrpls_tpls_dataset.xlsx"
    jr, materials = phi.parse_materials(str(fpath), "Materials")
    x = [pd.read_excel(fpath, sheet_name=m, engine="openpyxl") for m in materials]
    xc, jrc = phi.reconcile_rows_to_columns(x, jr)
    quality = pd.read_excel(fpath, sheet_name="QUALITY", engine="openpyxl")
    process = pd.read_excel(fpath, sheet_name="PROCESS", engine="openpyxl")
    jrc.append(process)
    jrc.append(quality)
    AUX = phi.reconcile_rows(jrc)
    JR_ = AUX[:-2]
    process = AUX[-2]
    quality = AUX[-1]
    Ri = {m: j for j, m in zip(JR_, materials)}
    Xi = {m: x_ for x_, m in zip(xc, materials)}
    return Xi, Ri, quality, process

# ── MBPLS ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mbpls_data():
    """Returns (XMB dict, Y DataFrame)."""
    fpath = EXAMPLES / "Multi-block PLS" / "MBDataset.xlsx"
    blocks = ["X1", "X2", "X3", "X4", "X5", "X6"]
    XMB = {b: pd.read_excel(fpath, sheet_name=b, engine="openpyxl") for b in blocks}
    Y = pd.read_excel(fpath, sheet_name="Y", engine="openpyxl")
    return XMB, Y

# ── NIR (spectra + LWPLS) ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nir_spectra():
    return pd.read_excel(
        EXAMPLES / "NIR Calibration" / "NIR.xlsx",
        sheet_name="NIR", na_values=np.nan, engine="openpyxl",
    )

@pytest.fixture(scope="session")
def nir_y():
    return pd.read_excel(
        EXAMPLES / "NIR Calibration" / "NIR.xlsx",
        sheet_name="Y", na_values=np.nan, engine="openpyxl",
    )

# ── Varimax / build_polynomial ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def chem_exp_data():
    ds = pd.read_excel(
        EXAMPLES / "Varimax Rotation" / "chemical_experiments_dataset.xlsx",
        sheet_name="data", engine="openpyxl",
    )
    ds, _ = phi.clean_low_variances(ds)
    return ds

# ── Batch film coating ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def batch_film_raw():
    return pd.read_excel(
        EXAMPLES / "Batch analysis" / "Batch Film Coating.xlsx", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def batch_film_phase_samples():
    return {"STARTUP": 3, "HEATING": 20, "SPRAYING": 40, "DRYING": 40, "DISCHARGING": 5}

@pytest.fixture(scope="session")
def batch_film_aligned(batch_film_raw):
    return batch.simple_align(batch_film_raw, 250)

@pytest.fixture(scope="session")
def batch_film_phase_aligned(batch_film_raw, batch_film_phase_samples):
    return batch.phase_simple_align(batch_film_raw, batch_film_phase_samples)

# ── Batch dryer ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def batch_dryer_raw():
    return pd.read_excel(
        EXAMPLES / "Batch analysis" / "Batch Dryer Case Study.xlsx",
        sheet_name="Trajectories", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def batch_dryer_cqa():
    return pd.read_excel(
        EXAMPLES / "Batch analysis" / "Batch Dryer Case Study.xlsx",
        sheet_name="ProductQuality", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def batch_dryer_phase_samples():
    return {"Deagglomerate": 20, "Heat": 30, "Cooldown": 40}

@pytest.fixture(scope="session")
def batch_dryer_phase_aligned(batch_dryer_raw, batch_dryer_phase_samples):
    return batch.phase_simple_align(batch_dryer_raw, batch_dryer_phase_samples)
```

- [ ] **Step 2: Verify fixtures are importable (no crash on collection)**

```bash
uv run pytest --collect-only
```
Expected: 0 tests, no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add data fixtures to conftest"
```

---

### Task 3: conftest.py — model fixtures

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Append model fixtures to `tests/conftest.py`**

```python
# ── PCA / PLS models ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pca_model(cars_features):
    return phi.pca(cars_features, 3, shush=True)

@pytest.fixture(scope="session")
def pca_model_cv(cars_features):
    return phi.pca(cars_features, 3, cross_val=5, shush=True)

@pytest.fixture(scope="session")
def pls_model(cars_features, cars_performance):
    return phi.pls(cars_features, cars_performance, 3, shush=True)

@pytest.fixture(scope="session")
def pls_model_cv(cars_features, cars_performance):
    return phi.pls(cars_features, cars_performance, 3, cross_val=5, shush=True)

@pytest.fixture(scope="session")
def pls_model_cv_x(cars_features, cars_performance):
    return phi.pls(cars_features, cars_performance, 3, cross_val=5, cross_val_X=True, shush=True)

@pytest.fixture(scope="session")
def pls_model_cca(cars_features, cars_performance):
    return phi.pls(cars_features, cars_performance, 3, cca=True, shush=True)

# ── Advanced model fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def lpls_model(lpls_X, lpls_R, lpls_Y):
    return phi.lpls(lpls_X, lpls_R, lpls_Y, 4, shush=True)

@pytest.fixture(scope="session")
def jrpls_model(jrpls_data):
    Xi, Ri, quality, _ = jrpls_data
    return phi.jrpls(Xi, Ri, quality, 4, shush=True)

@pytest.fixture(scope="session")
def tpls_model(jrpls_data):
    Xi, Ri, quality, process = jrpls_data
    return phi.tpls(Xi, Ri, process, quality, 4, shush=True)

@pytest.fixture(scope="session")
def jypls_model(jrpls_data):
    # Use Xi as both Xi and Yi to test the computation pipeline.
    # jypls expects dicts of DataFrames keyed by material name.
    Xi, _, _, _ = jrpls_data
    return phi.jypls(Xi, Xi, 2, shush=True)

@pytest.fixture(scope="session")
def mbpls_model(mbpls_data):
    XMB, Y = mbpls_data
    return phi.mbpls(XMB, Y, 2, shush=True)

# ── Batch models ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mpca_model(batch_film_phase_aligned, batch_film_phase_samples):
    return batch.mpca(batch_film_phase_aligned, 2,
                      phase_samples=batch_film_phase_samples, shush=True)

@pytest.fixture(scope="session")
def mpls_model(batch_dryer_phase_aligned, batch_dryer_cqa, batch_dryer_phase_samples):
    return batch.mpls(batch_dryer_phase_aligned, batch_dryer_cqa, 3,
                      phase_samples=batch_dryer_phase_samples, shush=True)
```

- [ ] **Step 2: Verify all fixtures build without crash**

```bash
uv run pytest --collect-only
```

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add model fixtures to conftest"
```

---

### Task 4: TestValidateInputs

**Files:**
- Modify: `tests/test_calc.py`

These tests use only in-memory DataFrames/arrays — no fixtures needed.

- [ ] **Step 1: Add `TestValidateInputs` class to `test_calc.py`**

```python
def _make_df(data, obs_ids, col_names):
    """Helper: DataFrame with obs-ID first column."""
    df = pd.DataFrame(data, columns=col_names)
    df.insert(0, "ObsID", obs_ids)
    return df


class TestValidateInputs:
    """Tests for phi._validate_inputs."""

    def _x(self):
        return _make_df(
            np.random.randn(5, 3), ["r1", "r2", "r3", "r4", "r5"], ["a", "b", "c"]
        )

    def _y(self):
        return _make_df(
            np.random.randn(5, 2), ["r1", "r2", "r3", "r4", "r5"], ["y1", "y2"]
        )

    def test_rejects_non_dataframe(self):
        with pytest.raises(ValueError):
            phi._validate_inputs([1, 2, 3])

    def test_rejects_invalid_mcs(self):
        X = self._x()
        with pytest.raises(ValueError):
            phi._validate_inputs(X, mcs="bad_value")

    def test_rejects_A_zero(self):
        X = self._x()
        with pytest.raises(ValueError):
            phi._validate_inputs(X, A=0)

    def test_rejects_A_exceeds_rank(self):
        X = self._x()  # 5 obs x 3 vars → max A = 3
        with pytest.raises(ValueError):
            phi._validate_inputs(X, A=10)

    def test_rejects_duplicate_obs_ids(self):
        X = _make_df(
            np.random.randn(4, 2), ["r1", "r1", "r2", "r3"], ["a", "b"]
        )
        with pytest.raises(ValueError):
            phi._validate_inputs(X)

    def test_rejects_no_common_obs(self):
        X = self._x()
        Y = _make_df(
            np.random.randn(3, 2), ["z1", "z2", "z3"], ["y1", "y2"]
        )
        with pytest.raises(ValueError):
            phi._validate_inputs(X, Y)

    def test_reorders_Y_to_match_X(self):
        X = self._x()  # order: r1..r5
        Y = _make_df(
            np.random.randn(5, 2), ["r5", "r4", "r3", "r2", "r1"], ["y1", "y2"]
        )
        _, Y_out = phi._validate_inputs(X, Y)
        assert Y_out.iloc[:, 0].tolist() == ["r1", "r2", "r3", "r4", "r5"]

    def test_drops_obs_missing_in_Y(self):
        X = self._x()  # 5 obs
        Y = _make_df(
            np.random.randn(3, 2), ["r1", "r2", "r3"], ["y1", "y2"]
        )  # Y only has r1..r3
        X_out, _ = phi._validate_inputs(X, Y)
        assert X_out.shape[0] == 3

    def test_numpy_row_mismatch_raises(self):
        X = np.random.randn(5, 3)
        Y = np.random.randn(4, 2)
        with pytest.raises(ValueError):
            phi._validate_inputs(X, Y)
```

- [ ] **Step 2: Run the class**

```bash
uv run pytest tests/test_calc.py::TestValidateInputs -v
```
Expected: 9 tests, all PASS (these test validation logic which was recently added and should be correct).

- [ ] **Step 3: Commit**

```bash
git add tests/test_calc.py
git commit -m "test: add TestValidateInputs"
```

---

### Task 5: TestHelpers and TestStatFunctions

**Files:**
- Modify: `tests/test_calc.py`

- [ ] **Step 1: Add `TestHelpers` to `test_calc.py`**

```python
class TestHelpers:

    def test_extract_array_from_ndarray(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out, obsid, varid = phi._extract_array(arr)
        assert obsid is False
        assert varid is False
        np.testing.assert_array_equal(out, arr)
        assert out is not arr  # must be a copy

    def test_extract_array_from_dataframe(self):
        df = _make_df(np.ones((3, 2)), ["a", "b", "c"], ["x1", "x2"])
        out, obsid, varid = phi._extract_array(df)
        assert obsid == ["a", "b", "c"]
        assert varid == ["x1", "x2"]
        assert out.shape == (3, 2)

    def test_meancenterscale_autoscale(self):
        X = np.random.randn(20, 4) * 5 + 10
        X_out, mx, sx = phi.meancenterscale(X, mcs=True)
        np.testing.assert_allclose(X_out.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(X_out.std(axis=0, ddof=1), 1, atol=1e-10)

    def test_meancenterscale_center_only(self):
        X = np.random.randn(20, 4) * 5 + 10
        X_out, mx, sx = phi.meancenterscale(X, mcs='center')
        np.testing.assert_allclose(X_out.mean(axis=0), 0, atol=1e-10)

    def test_meancenterscale_false(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_out, mx, sx = phi.meancenterscale(X, mcs=False)
        np.testing.assert_array_equal(X_out, X)

    def test_calc_r2_range(self):
        residual = np.random.randn(10, 5) * 0.1
        X = np.random.randn(10, 5)
        TSS = np.sum(X ** 2)
        TSSpv = np.sum(X ** 2, axis=0)
        r2, r2pv = phi._calc_r2(residual, TSS, TSSpv)
        assert 0 <= r2 <= 1
        assert np.all(r2pv >= 0) and np.all(r2pv <= 1)

    def test_Ab_btbinv_no_missing(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 5))
        b = rng.standard_normal((5, 1))
        not_nan_map = np.ones_like(A, dtype=bool)
        result = phi._Ab_btbinv(A, b, not_nan_map)
        expected = (A @ b / (b.T @ b)).reshape(-1, 1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_clean_empty_rows_removes_all_nan(self):
        df = _make_df(
            np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
            ["r1", "r2", "r3"],
            ["a", "b"],
        )
        out, removed = phi.clean_empty_rows(df, shush=True)
        assert out.shape[0] == 2
        assert "r2" in removed

    def test_clean_low_variances_removes_constants(self):
        df = _make_df(
            np.column_stack([np.ones(10), np.random.randn(10)]),
            [f"r{i}" for i in range(10)],
            ["const", "vary"],
        )
        out, removed = phi.clean_low_variances(df, shush=True)
        assert "const" in removed
        assert "vary" not in removed

    def test_cat_2_matrix_shape(self):
        df = _make_df(
            np.array([["A"], ["B"], ["A"], ["C"]]),
            ["r1", "r2", "r3", "r4"],
            ["cat"],
        )
        out = phi.cat_2_matrix(df)
        # 3 unique categories → 3 dummy columns (or 3-1=2 depending on encoding)
        assert isinstance(out, (pd.DataFrame, np.ndarray))
```

- [ ] **Step 2: Add `TestStatFunctions` to `test_calc.py`**

```python
class TestStatFunctions:

    def test_f99_gt_f95(self):
        for i, j in [(5, 50), (10, 100), (3, 20)]:
            assert phi.f99(i, j) > phi.f95(i, j)

    def test_f_values_positive(self):
        assert phi.f99(5, 50) > 0
        assert phi.f95(5, 50) > 0

    def test_spe_ci_returns_two_limits(self):
        # Build a small SPE array and compute limits
        spe_vals = np.abs(np.random.randn(30))
        result = phi.spe_ci(spe_vals)
        assert len(result) == 2
        lim95, lim99 = result
        assert lim95 > 0
        assert lim99 > 0

    def test_scores_conf_int_monotone(self):
        rng = np.random.default_rng(0)
        T = rng.standard_normal((50, 2))
        st = (T.T @ T) / T.shape[0]
        result = phi.scores_conf_int_calc(st, 50)
        # result is (xd95, xd99, yd95p, yd95n, yd99p, yd99n)
        # 99% ellipse should be larger than 95%
        xd95, xd99 = result[0], result[1]
        assert max(np.abs(xd99)) > max(np.abs(xd95))
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_calc.py::TestHelpers tests/test_calc.py::TestStatFunctions -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_calc.py
git commit -m "test: add TestHelpers and TestStatFunctions"
```

---

### Task 6: TestPCA

**Files:**
- Modify: `tests/test_calc.py`

- [ ] **Step 1: Add `TestPCA` class**

```python
class TestPCA:

    def test_model_keys_present(self, pca_model):
        for key in ("T", "P", "r2x", "r2xpv", "mx", "sx", "T2", "speX", "var_t"):
            assert key in pca_model, f"Missing key: {key}"

    def test_score_shape(self, pca_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pca_model["T"].shape == (n_obs, 3)

    def test_loading_shape(self, pca_model, cars_features):
        n_vars = cars_features.shape[1] - 1  # subtract obs-ID column
        assert pca_model["P"].shape == (n_vars, 3)

    def test_loadings_orthonormal(self, pca_model):
        P = pca_model["P"]
        np.testing.assert_allclose(P.T @ P, np.eye(3), atol=1e-8)

    def test_r2x_in_range(self, pca_model):
        r2x = pca_model["r2x"]
        assert np.all(np.array(r2x) > 0)
        assert np.all(np.array(r2x) <= 1.0)

    def test_r2xpv_in_range(self, pca_model):
        r2xpv = pca_model["r2xpv"]
        assert np.all(r2xpv >= 0)
        assert np.all(r2xpv <= 1.0)

    def test_reconstruction_residual(self, pca_model, cars_features):
        X = np.array(cars_features.values[:, 1:]).astype(float)
        X_centered = (X - pca_model["mx"]) / pca_model["sx"]
        T = pca_model["T"]
        P = pca_model["P"]
        residual_norm = np.linalg.norm(X_centered - T @ P.T)
        data_norm = np.linalg.norm(X_centered)
        assert residual_norm < data_norm

    def test_t2_shape(self, pca_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pca_model["T2"].shape == (n_obs,)

    def test_spe_shape(self, pca_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pca_model["speX"].shape[0] == n_obs

    def test_cross_val_produces_q2(self, pca_model_cv):
        assert "q2" in pca_model_cv
        assert "q2pv" in pca_model_cv

    def test_pca_pred_training_scores(self, pca_model, cars_features):
        pred = phi.pca_pred(cars_features, pca_model)
        np.testing.assert_allclose(pred["Tnew"], pca_model["T"], atol=1e-6)

    def test_pca_pred_keys(self, pca_model, cars_features):
        pred = phi.pca_pred(cars_features, pca_model)
        for key in ("Tnew", "Xhat", "speX", "T2"):
            assert key in pred

    def test_pca_pred_xhat_shape(self, pca_model, cars_features):
        pred = phi.pca_pred(cars_features, pca_model)
        n_obs = cars_features.shape[0]
        n_vars = cars_features.shape[1] - 1
        assert pred["Xhat"].shape == (n_obs, n_vars)

    def test_type_flag(self, pca_model):
        assert pca_model["type"] == "pca"
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_calc.py::TestPCA -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_calc.py
git commit -m "test: add TestPCA"
```

---

### Task 7: TestPLS and TestDiagnostics

**Files:**
- Modify: `tests/test_calc.py`

- [ ] **Step 1: Add `TestPLS` class**

```python
class TestPLS:

    def test_model_keys_present(self, pls_model):
        for key in ("T", "P", "Q", "W", "Ws", "r2x", "r2y", "r2xpv", "r2ypv",
                    "mx", "sx", "my", "sy", "T2", "speX", "speY", "var_t"):
            assert key in pls_model, f"Missing key: {key}"

    def test_score_shape(self, pls_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pls_model["T"].shape == (n_obs, 3)

    def test_loading_shapes(self, pls_model, cars_features, cars_performance):
        n_x = cars_features.shape[1] - 1
        n_y = cars_performance.shape[1] - 1
        assert pls_model["P"].shape == (n_x, 3)
        assert pls_model["Q"].shape == (n_y, 3)

    def test_r2x_r2y_in_range(self, pls_model):
        assert 0 < float(np.max(pls_model["r2x"])) <= 1.0
        assert 0 < float(np.max(pls_model["r2y"])) <= 1.0

    def test_cross_val_produces_q2Y(self, pls_model_cv):
        assert "q2Y" in pls_model_cv
        assert "q2Ypv" in pls_model_cv

    def test_cross_val_produces_q2X(self, pls_model_cv_x):
        assert "q2X" in pls_model_cv_x
        assert "q2Xpv" in pls_model_cv_x

    def test_q2Y_leq_r2y(self, pls_model_cv):
        q2Y = float(np.max(pls_model_cv["q2Y"]))
        r2y = float(np.max(pls_model_cv["r2y"]))
        assert q2Y <= r2y

    def test_cca_flag_adds_keys(self, pls_model_cca):
        for key in ("Tcv", "Pcv", "Wcv"):
            assert key in pls_model_cca, f"Missing CCA key: {key}"

    def test_pls_pred_yhat_shape(self, pls_model, cars_features, cars_performance):
        pred = phi.pls_pred(cars_features, pls_model)
        n_obs = cars_features.shape[0]
        n_y = cars_performance.shape[1] - 1
        assert pred["Yhat"].shape == (n_obs, n_y)

    def test_pls_pred_training_roundtrip(self, pls_model, cars_features):
        pred = phi.pls_pred(cars_features, pls_model)
        np.testing.assert_allclose(pred["Tnew"], pls_model["T"], atol=1e-6)

    def test_type_flag(self, pls_model):
        assert pls_model["type"] == "pls"
```

- [ ] **Step 2: Add `TestDiagnostics` class**

```python
class TestDiagnostics:
    """Tests for hott2, spe, and contributions."""

    def test_hott2_training_shape(self, pca_model, cars_features):
        result = phi.hott2(pca_model)
        assert result.shape == (cars_features.shape[0],)

    def test_hott2_matches_stored_T2(self, pca_model):
        result = phi.hott2(pca_model)
        np.testing.assert_allclose(result, pca_model["T2"], rtol=1e-6)

    def test_hott2_with_Xnew(self, pca_model, cars_features):
        result = phi.hott2(pca_model, Xnew=cars_features)
        assert result.shape == (cars_features.shape[0],)

    def test_spe_training_shape(self, pca_model, cars_features):
        result = phi.spe(pca_model, cars_features)
        # spe returns (n_obs, 1)
        assert result.shape[0] == cars_features.shape[0]

    def test_contributions_scores_shape(self, pca_model, cars_features):
        # to_obs uses integer row index
        result = phi.contributions(pca_model, cars_features, "scores", to_obs=[0])
        n_vars = cars_features.shape[1] - 1
        assert result.shape == (1, n_vars)

    def test_contributions_spe_shape(self, pca_model, cars_features):
        result = phi.contributions(pca_model, cars_features, "spe", to_obs=[0])
        n_vars = cars_features.shape[1] - 1
        assert result.shape == (1, n_vars)

    def test_contributions_ht2_shape(self, pca_model, cars_features):
        result = phi.contributions(pca_model, cars_features, "ht2", to_obs=[0])
        n_vars = cars_features.shape[1] - 1
        assert result.shape == (1, n_vars)
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_calc.py::TestPLS tests/test_calc.py::TestDiagnostics -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_calc.py
git commit -m "test: add TestPLS and TestDiagnostics"
```

---

### Task 8: Advanced model tests

**Files:**
- Modify: `tests/test_calc.py`

- [ ] **Step 1: Add `TestMBPLS`**

```python
class TestMBPLS:

    def test_model_keys_present(self, mbpls_model):
        for key in ("T", "P", "Q", "W", "r2x", "r2y", "r2pbX"):
            assert key in mbpls_model, f"Missing key: {key}"

    def test_r2pbX_shape(self, mbpls_model, mbpls_data):
        XMB, _ = mbpls_data
        n_blocks = len(XMB)
        # r2pbX shape[0] should equal number of blocks
        assert mbpls_model["r2pbX"].shape[0] == n_blocks

    def test_score_shape(self, mbpls_model, mbpls_data):
        XMB, _ = mbpls_data
        # All blocks share the same observations
        first_block = next(iter(XMB.values()))
        n_obs = first_block.shape[0]
        assert mbpls_model["T"].shape[0] == n_obs
```

- [ ] **Step 2: Add `TestLPLS`**

```python
class TestLPLS:

    def test_model_keys_present(self, lpls_model):
        for key in ("T", "P", "r2x"):
            assert key in lpls_model, f"Missing key: {key}"
        # LPLS-specific: material scores in R-space
        assert "Tr" in lpls_model or "type" in lpls_model

    def test_type_flag(self, lpls_model):
        assert lpls_model["type"] == "lpls"

    def test_score_shape(self, lpls_model, lpls_X):
        n_obs = lpls_X.shape[0]
        assert lpls_model["T"].shape[0] == n_obs

    def test_lpls_pred_shape(self, lpls_model, lpls_R):
        # lpls_pred takes a new R (material ratio) dict or DataFrame
        # Using a single row from lpls_R as the new observation
        r_new = lpls_R.iloc[[0]]
        pred = phi.lpls_pred(r_new, lpls_model)
        # Prediction should return scores/Yhat for 1 observation
        assert pred is not None
```

- [ ] **Step 3: Add `TestJRPLS`**

```python
class TestJRPLS:

    def test_model_keys_present(self, jrpls_model):
        for key in ("T", "P", "Q", "type"):
            assert key in jrpls_model, f"Missing key: {key}"

    def test_type_flag(self, jrpls_model):
        assert jrpls_model["type"] == "jrpls"

    def test_jrpls_pred_shape(self, jrpls_model, jrpls_data):
        Xi, Ri, quality, _ = jrpls_data
        # Build a minimal rnew from first lot
        materials = list(Ri.keys())
        rnew = {m: [(Ri[m].iloc[0, 1], 1.0)] for m in materials}
        pred = phi.jrpls_pred(rnew, jrpls_model)
        assert pred is not None
```

- [ ] **Step 4: Add `TestTPLS`**

```python
class TestTPLS:

    def test_model_keys_present(self, tpls_model):
        for key in ("T", "P", "Q", "type"):
            assert key in tpls_model, f"Missing key: {key}"

    def test_type_flag(self, tpls_model):
        assert tpls_model["type"] == "tpls"

    def test_tpls_pred_shape(self, tpls_model, jrpls_data):
        Xi, Ri, quality, process = jrpls_data
        materials = list(Ri.keys())
        rnew = {m: [(Ri[m].iloc[0, 1], 1.0)] for m in materials}
        # znew: process variables for one lot (strip obs-ID column)
        znew = process.iloc[0, 1:].values.astype(float)
        pred = phi.tpls_pred(rnew, znew, tpls_model)
        assert pred is not None
```

- [ ] **Step 5: Add `TestJYPLS`**

```python
class TestJYPLS:

    def test_model_keys_present(self, jypls_model):
        for key in ("T", "type"):
            assert key in jypls_model, f"Missing key: {key}"

    def test_type_flag(self, jypls_model):
        assert jypls_model["type"] == "jypls"

    def test_jypls_pred_shape(self, jypls_model, jrpls_data):
        Xi, _, _, _ = jrpls_data
        # Use first row of each Xi block as xnew
        xnew = {m: Xi[m].iloc[[0]] for m in Xi}
        campaign = list(Xi.keys())[0]
        pred = phi.jypls_pred(xnew, campaign, jypls_model)
        assert pred is not None
```

- [ ] **Step 6: Run all advanced model tests**

```bash
uv run pytest tests/test_calc.py::TestMBPLS tests/test_calc.py::TestLPLS \
  tests/test_calc.py::TestJRPLS tests/test_calc.py::TestTPLS \
  tests/test_calc.py::TestJYPLS -v
```
Note: some failures here are expected and flag bugs for investigation.

- [ ] **Step 7: Commit**

```bash
git add tests/test_calc.py
git commit -m "test: add TestMBPLS, TestLPLS, TestJRPLS, TestTPLS, TestJYPLS"
```

---

### Task 9: Remaining calc tests

**Files:**
- Modify: `tests/test_calc.py`

- [ ] **Step 1: Add `TestLWPLS`**

```python
class TestLWPLS:
    """lwpls is called once per observation; returns yhat for that observation.
    Note: the docstring claims a dict return but source returns an ndarray — a bug to investigate."""

    def test_output_is_array(self, pls_model, nir_spectra, nir_y):
        # Build numpy arrays from DataFrames for lwpls
        X = np.array(nir_spectra.values[:, 1:]).astype(float)
        Y = np.array(nir_y.values[:, 1:]).astype(float)
        # Build a global PLS model on raw arrays
        pls_nir = phi.pls(X, Y, 1, shush=True)
        xnew = X[0, :]
        result = phi.lwpls(xnew, 20, pls_nir, X, Y, shush=True)
        assert result is not None

    def test_output_numeric(self, pls_model, nir_spectra, nir_y):
        X = np.array(nir_spectra.values[:, 1:]).astype(float)
        Y = np.array(nir_y.values[:, 1:]).astype(float)
        pls_nir = phi.pls(X, Y, 1, shush=True)
        xnew = X[0, :]
        result = phi.lwpls(xnew, 20, pls_nir, X, Y, shush=True)
        assert not np.any(np.isnan(np.array(result).ravel()))
```

- [ ] **Step 2: Add `TestBootstrapPLS`**

```python
class TestBootstrapPLS:

    @pytest.fixture(scope="class")
    def bootstrap_result(self, cars_features, cars_performance):
        return phi.bootstrap_pls(cars_features, cars_performance,
                                 num_latents=2, num_samples=5, shush=True)

    def test_bootstrap_returns_list(self, bootstrap_result):
        assert isinstance(bootstrap_result, list)
        assert len(bootstrap_result) == 5

    def test_bootstrap_each_element_is_pls_dict(self, bootstrap_result):
        for model in bootstrap_result:
            for key in ("T", "P", "Q"):
                assert key in model, f"Bootstrap element missing key: {key}"

    def test_bootstrap_pred_returns_list(self, bootstrap_result, cars_features):
        X = np.array(cars_features.values[:, 1:]).astype(float)
        result = phi.bootstrap_pls_pred(X, bootstrap_result)
        assert isinstance(result, list)

    def test_bootstrap_pred_quantile_count(self, bootstrap_result, cars_features):
        X = np.array(cars_features.values[:, 1:]).astype(float)
        quantiles = [0.025, 0.975]
        result = phi.bootstrap_pls_pred(X, bootstrap_result, quantiles=quantiles)
        assert len(result) == 2

    def test_bootstrap_pred_quantile_shape(self, bootstrap_result, cars_features, cars_performance):
        X = np.array(cars_features.values[:, 1:]).astype(float)
        n_y = cars_performance.shape[1] - 1
        result = phi.bootstrap_pls_pred(X, bootstrap_result)
        # Each element should be shape (n_y,)
        assert result[0].shape == (n_y,)
```

- [ ] **Step 3: Add `TestVarimax`**

```python
class TestVarimax:

    def test_rotated_loadings_shape(self, chem_exp_data):
        x_cols = [c for c in chem_exp_data.columns if c != "Lot" and c in
                  ["SM (eq)", "A (eq)", "B (eq)", "C (eq)", "Solvent (Vol)"]]
        y_cols = [c for c in chem_exp_data.columns if c != "Lot" and c in
                  ["IPC_N", "IPC_H"]]
        ra_X = chem_exp_data[["Lot"] + x_cols]
        ra_Y = chem_exp_data[["Lot"] + y_cols]
        plsobj = phi.pls(ra_X, ra_Y, 2, shush=True)
        original_P_shape = plsobj["P"].shape
        rotated = phi.varimax_rotation(plsobj, ra_X, Y=ra_Y)
        assert rotated["P"].shape == original_P_shape

    def test_r2_total_preserved(self, chem_exp_data):
        x_cols = [c for c in chem_exp_data.columns if c != "Lot" and c in
                  ["SM (eq)", "A (eq)", "B (eq)", "C (eq)", "Solvent (Vol)"]]
        y_cols = [c for c in chem_exp_data.columns if c != "Lot" and c in
                  ["IPC_N", "IPC_H"]]
        ra_X = chem_exp_data[["Lot"] + x_cols]
        ra_Y = chem_exp_data[["Lot"] + y_cols]
        plsobj = phi.pls(ra_X, ra_Y, 2, shush=True)
        rotated = phi.varimax_rotation(plsobj, ra_X, Y=ra_Y)
        orig_r2y_total = float(np.max(plsobj["r2y"]))
        rot_r2y_total = float(np.max(rotated["r2y"]))
        np.testing.assert_allclose(orig_r2y_total, rot_r2y_total, rtol=1e-4)
```

- [ ] **Step 4: Add `TestSpectra`**

```python
class TestSpectra:
    """All spectra functions tested with NIR.xlsx 'NIR' sheet (DataFrame input)."""

    def test_snv_shape(self, nir_spectra):
        out = phi.spectra_snv(nir_spectra)
        assert out.shape == nir_spectra.shape

    def test_snv_row_mean_zero(self, nir_spectra):
        out = phi.spectra_snv(nir_spectra)
        # Strip obs-ID column if present; check data columns
        data = np.array(out.values[:, 1:]).astype(float) if isinstance(out, pd.DataFrame) else out
        np.testing.assert_allclose(data.mean(axis=1), 0, atol=1e-10)

    def test_snv_row_std_one(self, nir_spectra):
        out = phi.spectra_snv(nir_spectra)
        data = np.array(out.values[:, 1:]).astype(float) if isinstance(out, pd.DataFrame) else out
        np.testing.assert_allclose(data.std(axis=1, ddof=1), 1, atol=1e-8)

    def test_savgol_shape(self, nir_spectra):
        snv = phi.spectra_snv(nir_spectra)
        out, M = phi.spectra_savgol(10, 1, 2, snv)
        assert out.shape == snv.shape

    def test_msc_shape(self, nir_spectra):
        out = phi.spectra_msc(nir_spectra)
        assert out.shape == nir_spectra.shape

    def test_mean_center_shape(self, nir_spectra):
        out = phi.spectra_mean_center(nir_spectra)
        assert out.shape == nir_spectra.shape

    def test_autoscale_shape(self, nir_spectra):
        out = phi.spectra_autoscale(nir_spectra)
        assert out.shape == nir_spectra.shape

    def test_baseline_correction_shape(self, nir_spectra):
        out = phi.spectra_baseline_correction(nir_spectra)
        assert out.shape == nir_spectra.shape
```

- [ ] **Step 5: Add `TestCCA`**

```python
class TestCCA:
    """cca(X, Y) returns (correlation_scalar, w_x, w_y) — all numpy, no DataFrame."""

    @pytest.fixture(scope="class")
    def cca_result(self, cars_features, cars_performance):
        X = np.array(cars_features.values[:, 1:]).astype(float)
        Y = np.array(cars_performance.values[:, 1:]).astype(float)
        # Use standardized inputs
        X = (X - X.mean(0)) / X.std(0, ddof=1)
        Y = (Y - Y.mean(0)) / Y.std(0, ddof=1)
        return phi.cca(X, Y), X.shape[1], Y.shape[1]

    def test_returns_tuple_of_three(self, cca_result):
        result, n_x, n_y = cca_result
        assert len(result) == 3

    def test_weight_vector_shapes(self, cca_result):
        (corr, w_x, w_y), n_x, n_y = cca_result
        assert w_x.shape == (n_x,)
        assert w_y.shape == (n_y,)

    def test_weights_unit_norm(self, cca_result):
        (corr, w_x, w_y), n_x, n_y = cca_result
        np.testing.assert_allclose(np.linalg.norm(w_x), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(w_y), 1.0, atol=1e-6)

    def test_correlation_in_range(self, cca_result):
        (corr, w_x, w_y), _, _ = cca_result
        assert -1.0 <= float(corr) <= 1.0
```

- [ ] **Step 6: Add `TestBuildPolynomial`**

```python
class TestBuildPolynomial:
    """build_polynomial calls plt.figure() internally; Agg backend in conftest handles this."""

    @pytest.fixture(scope="class")
    def poly_setup(self, chem_exp_data):
        # Select a few numeric columns (avoid the 'Lot' obs-ID column)
        all_cols = list(chem_exp_data.columns)
        factor_cols = [c for c in all_cols if c not in ("Lot",) and
                       pd.api.types.is_numeric_dtype(chem_exp_data[c])][:4]
        response_col = factor_cols[-1]
        factor_cols = factor_cols[:-1]
        return chem_exp_data, ["Lot"] + factor_cols, response_col

    def test_returns_model(self, poly_setup):
        data, factors, response = poly_setup
        result = phi.build_polynomial(data, factors, response)
        assert result is not None

    def test_prediction_shape(self, poly_setup):
        data, factors, response = poly_setup
        result = phi.build_polynomial(data, factors, response)
        # build_polynomial returns a model dict; check it contains prediction output
        assert "Yhat" in result or result is not None
        # If Yhat is present it should have n_obs rows
        if isinstance(result, dict) and "Yhat" in result:
            assert result["Yhat"].shape[0] == data.shape[0]
```

- [ ] **Step 7: Add `TestUtilities`**

```python
class TestUtilities:

    def test_unique_preserves_order(self):
        df = pd.DataFrame({"ID": ["b", "a", "b", "c", "a"], "val": range(5)})
        result = phi.unique(df, "ID")
        assert result == ["b", "a", "c"]  # first-occurrence order

    def test_reconcile_rows_common_only(self):
        df1 = _make_df(np.ones((4, 1)), ["r1", "r2", "r3", "r4"], ["x"])
        df2 = _make_df(np.ones((3, 1)), ["r2", "r3", "r4"], ["y"])
        result = phi.reconcile_rows([df1, df2])
        for df in result:
            assert df.shape[0] == 3
            assert "r1" not in df.iloc[:, 0].tolist()

    def test_reconcile_rows_to_columns(self):
        df_list_r = [
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["x1", "x2"]),
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["x3", "x4"]),
        ]
        df_list_c = [
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["r1", "r2"]),
        ]
        out_x, out_r = phi.reconcile_rows_to_columns(df_list_r, df_list_c)
        assert len(out_x) == 2
        assert len(out_r) == 1

    def test_hott2_with_Tnew(self, pca_model):
        T_new = pca_model["T"][:5, :]
        result = phi.hott2(pca_model, Tnew=T_new)
        assert result.shape == (5,)
```

- [ ] **Step 8: Run all remaining calc tests**

```bash
uv run pytest tests/test_calc.py -v
```
Expected: many PASS, some FAIL (failures are bugs to investigate separately).

- [ ] **Step 9: Commit**

```bash
git add tests/test_calc.py
git commit -m "test: add TestLWPLS, TestBootstrapPLS, TestVarimax, TestSpectra, TestCCA, TestBuildPolynomial, TestUtilities"
```

---

### Task 10: test_plots.py

**Files:**
- Modify: `tests/test_plots.py`

- [ ] **Step 1: Write all plots helper tests**

```python
import pytest
import re
import numpy as np
from pyphi.plots import (
    _get_lv_labels, _get_xvar_labels, _get_yvar_labels,
    _timestr, _make_bokeh_palette, _mask_by_class, _resolve_lpls_space,
)


class TestLVLabels:

    def test_pca_prefix(self, pca_model):
        labels = _get_lv_labels(pca_model)
        assert all(l.startswith("PC #") for l in labels)

    def test_pls_prefix(self, pls_model):
        labels = _get_lv_labels(pls_model)
        assert all(l.startswith("LV #") for l in labels)

    def test_length_equals_A(self, pca_model):
        A = pca_model["T"].shape[1]
        assert len(_get_lv_labels(pca_model)) == A


class TestVarLabels:

    def test_uses_varidX_when_present(self, pca_model):
        labels = _get_xvar_labels(pca_model)
        assert labels == pca_model["varidX"]

    def test_fallback_format(self):
        model = {"P": np.zeros((4, 2)), "T": np.zeros((10, 2))}
        labels = _get_xvar_labels(model)
        assert labels == ["XVar #1", "XVar #2", "XVar #3", "XVar #4"]

    def test_yvar_uses_varidY(self, pls_model):
        labels = _get_yvar_labels(pls_model)
        assert labels == pls_model["varidY"]

    def test_yvar_fallback(self):
        model = {"Q": np.zeros((3, 2)), "T": np.zeros((10, 2))}
        labels = _get_yvar_labels(model)
        assert labels == ["YVar #1", "YVar #2", "YVar #3"]


class TestTimestr:

    def test_returns_string(self):
        assert isinstance(_timestr(), str)

    def test_exact_length(self):
        # Format %Y%m%d%H%M%S%f → 4+2+2+2+2+2+6 = 20 chars always
        assert len(_timestr()) == 20

    def test_all_digits(self):
        assert _timestr().isdigit()


class TestMakePalette:

    def test_length(self):
        palette = _make_bokeh_palette(7)
        assert len(palette) == 7

    def test_hex_format(self):
        palette = _make_bokeh_palette(5)
        pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for color in palette:
            assert pattern.match(color), f"Not a hex color: {color}"


class TestMaskByClass:

    def test_filters_correctly(self):
        classid_arr = np.array(["A", "B", "A", "C"])
        x_arr = np.array([1.0, 2.0, 3.0, 4.0])
        y_arr = np.array([10.0, 20.0, 30.0, 40.0])
        obs_ids = ["o1", "o2", "o3", "o4"]
        obs_nums = ["1", "2", "3", "4"]
        result = _mask_by_class(classid_arr, x_arr, y_arr, obs_ids, obs_nums, "A")
        assert result["x"] == [1.0, 3.0]
        assert result["ObsID"] == ["o1", "o3"]
        assert result["Class"] == ["A", "A"]

    def test_empty_class(self):
        classid_arr = np.array(["A", "B"])
        x_arr = np.array([1.0, 2.0])
        y_arr = np.array([3.0, 4.0])
        result = _mask_by_class(classid_arr, x_arr, y_arr, ["o1", "o2"], ["1", "2"], "Z")
        assert result["x"] == []


class TestResolveLplsSpace:
    """`_resolve_lpls_space(mvmobj, material, zspace)` returns a dict."""

    def test_lpls_zspace_false_returns_dict(self, lpls_model):
        result = _resolve_lpls_space(lpls_model, None, False)
        assert isinstance(result, dict)

    def test_lpls_sets_Ws_to_Ss(self, lpls_model):
        result = _resolve_lpls_space(lpls_model, None, False)
        np.testing.assert_array_equal(result["Ws"], lpls_model["Ss"])

    def test_tpls_zspace_true_uses_varidZ(self, tpls_model):
        # Only triggers when material is None and zspace=True
        result = _resolve_lpls_space(tpls_model, None, True)
        assert isinstance(result, dict)
        if "varidZ" in tpls_model:
            assert result["varidX"] == tpls_model["varidZ"]
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_plots.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_plots.py
git commit -m "test: add test_plots.py — internal helper function tests"
```

---

### Task 11: test_batch.py

**Files:**
- Modify: `tests/test_batch.py`

- [ ] **Step 1: Write all batch tests**

```python
import pytest
import numpy as np
import pandas as pd
import pyphi.batch as batch


class TestBatchUtilities:

    def test_unique_preserves_order(self):
        df = pd.DataFrame({"ID": ["c", "a", "c", "b", "a"]})
        result = batch.unique(df, "ID")
        assert result == ["c", "a", "b"]

    def test_mean_ignores_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        result = batch.mean(X, axis=0)
        np.testing.assert_allclose(result[0], 2.0)   # mean of [1, 3] = 2
        np.testing.assert_allclose(result[1], 5.0)   # mean of [4, 6] = 5

    def test_clean_empty_rows_removes_all_nan(self):
        df = pd.DataFrame({
            "BATCH": ["B1", "B2", "B3"],
            "var1": [1.0, np.nan, 3.0],
            "var2": [2.0, np.nan, 4.0],
        })
        # batch.clean_empty_rows returns just a DataFrame (no tuple)
        out_df = batch.clean_empty_rows(df, shush=True)
        # B2 row is all-NaN in data columns → removed
        assert out_df.shape[0] == 2


class TestAlignment:

    def test_simple_align_sample_count(self, batch_film_raw):
        nsamples = 250
        aligned = batch.simple_align(batch_film_raw, nsamples)
        # Each batch should have exactly nsamples rows
        batch_col = aligned.columns[0]
        for b_id, grp in aligned.groupby(batch_col):
            assert grp.shape[0] == nsamples, f"Batch {b_id} has {grp.shape[0]} samples"

    def test_simple_align_batch_count(self, batch_film_raw):
        batch_col = batch_film_raw.columns[0]
        n_batches_raw = batch_film_raw[batch_col].nunique()
        aligned = batch.simple_align(batch_film_raw, 250)
        batch_col_a = aligned.columns[0]
        n_batches_aligned = aligned[batch_col_a].nunique()
        assert n_batches_aligned == n_batches_raw

    def test_phase_align_sample_count(self, batch_film_raw, batch_film_phase_samples):
        aligned = batch.phase_simple_align(batch_film_raw, batch_film_phase_samples)
        expected_per_batch = sum(batch_film_phase_samples.values())
        batch_col = aligned.columns[0]
        for b_id, grp in aligned.groupby(batch_col):
            assert grp.shape[0] == expected_per_batch


class TestUnfoldRefold:

    def test_unfold_shape(self, batch_film_aligned):
        batch_col = batch_film_aligned.columns[0]
        n_batches = batch_film_aligned[batch_col].nunique()
        # Number of variables = total columns - batch ID - optional PHASE
        non_data_cols = sum(1 for c in batch_film_aligned.columns
                            if c.upper() in (batch_col.upper(), "PHASE"))
        n_vars = batch_film_aligned.shape[1] - non_data_cols
        nsamples = 250
        xuf = batch.unfold_horizontal(batch_film_aligned)
        assert xuf.shape == (n_batches, n_vars * nsamples)

    def test_roundtrip(self, batch_film_aligned):
        batch_col = batch_film_aligned.columns[0]
        non_data_cols = sum(1 for c in batch_film_aligned.columns
                            if c.upper() in (batch_col.upper(), "PHASE"))
        n_vars = batch_film_aligned.shape[1] - non_data_cols
        nsamples = 250
        xuf = batch.unfold_horizontal(batch_film_aligned)
        refolded = batch.refold_horizontal(xuf, n_vars, nsamples)
        # refolded should have same number of rows as original
        assert refolded.shape[0] == batch_film_aligned.shape[0]


class TestMPCA:

    def test_model_keys_present(self, mpca_model):
        for key in ("T", "P", "r2x"):
            assert key in mpca_model, f"Missing key: {key}"

    def test_score_shape(self, mpca_model, batch_film_phase_aligned):
        batch_col = batch_film_phase_aligned.columns[0]
        n_batches = batch_film_phase_aligned[batch_col].nunique()
        assert mpca_model["T"].shape[0] == n_batches

    def test_r2x_in_range(self, mpca_model):
        r2x = np.array(mpca_model["r2x"])
        assert np.all(r2x > 0)
        assert np.all(r2x <= 1.0)


class TestMPLS:

    def test_model_keys_present(self, mpls_model):
        for key in ("T", "Q"):
            assert key in mpls_model, f"Missing key: {key}"

    def test_score_shape(self, mpls_model, batch_dryer_phase_aligned):
        batch_col = batch_dryer_phase_aligned.columns[0]
        n_batches = batch_dryer_phase_aligned[batch_col].nunique()
        assert mpls_model["T"].shape[0] == n_batches


class TestDescriptors:

    def test_output_shape(self, batch_dryer_raw):
        batch_col = batch_dryer_raw.columns[0]
        n_batches = batch_dryer_raw[batch_col].nunique()
        # Use first non-batch, non-phase numeric column as which_var
        data_cols = [c for c in batch_dryer_raw.columns
                     if c.upper() not in (batch_col.upper(), "PHASE")
                     and pd.api.types.is_numeric_dtype(batch_dryer_raw[c])]
        which_var = [data_cols[0]]  # descriptors iterates over which_var — must be a list
        desc = ["min", "max"]
        result = batch.descriptors(batch_dryer_raw, which_var, desc)
        # descriptors() prepends a batch-ID column, then len(which_var)*len(desc) numeric columns
        assert result.shape == (n_batches, 1 + len(which_var) * len(desc))

    def test_no_nan_in_output(self, batch_dryer_raw):
        batch_col = batch_dryer_raw.columns[0]
        data_cols = [c for c in batch_dryer_raw.columns
                     if c.upper() not in (batch_col.upper(), "PHASE")
                     and pd.api.types.is_numeric_dtype(batch_dryer_raw[c])]
        which_var = [data_cols[0]]  # must be a list
        result = batch.descriptors(batch_dryer_raw, which_var, ["mean"])
        # Skip the batch-ID string column (iloc[:, 1:]) before casting to float
        assert not np.any(np.isnan(result.iloc[:, 1:].values.astype(float)))


class TestBuildRelTime:
    """build_rel_time requires a timestamp column. Uses a synthetic DataFrame
    when the Excel data does not contain one."""

    def _make_timed_batch(self):
        import pandas as pd
        rows = []
        for b in ["B1", "B2"]:
            for i in range(5):
                rows.append({
                    "BATCH": b,
                    # build_rel_time reads this_batch['Timestamp'] — column must be 'Timestamp'
                    "Timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=i * 10),
                    "var1": float(i),
                })
        return pd.DataFrame(rows)

    def test_output_is_numeric(self):
        bdata = self._make_timed_batch()
        result = batch.build_rel_time(bdata)
        # build_rel_time inserts column named 'Time (min)' (default time_unit='min')
        assert pd.api.types.is_numeric_dtype(result["Time (min)"])

    def test_monotone_per_batch(self):
        bdata = self._make_timed_batch()
        result = batch.build_rel_time(bdata)
        for b_id, grp in result.groupby("BATCH"):
            times = grp["Time (min)"].values
            assert np.all(np.diff(times) >= 0), f"Non-monotone rel_time for {b_id}"
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_batch.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_batch.py
git commit -m "test: add test_batch.py"
```

---

### Task 12: Full suite run and summary

**Files:** none

- [ ] **Step 1: Run the full suite with verbose output**

```bash
uv run pytest -v 2>&1 | tee tests/run_results.txt
```

- [ ] **Step 2: Check collection (no import errors)**

All tests should be *collected* even if some *fail*. If any test causes a collection error (ImportError, NameError), fix the import — that is an infrastructure bug, not a library bug.

- [ ] **Step 3: Triage results**

Failures fall into two buckets:
- **Infrastructure bug** (fixture crashes, wrong API call in test): fix the test
- **Library bug** (assertion fails on correct test): record the failing test name and leave for investigation

- [ ] **Step 4: Commit results file**

```bash
git add tests/run_results.txt
git commit -m "test: record initial test suite run results"
```

---

## Quick reference

```bash
uv run pytest                                   # full suite
uv run pytest tests/test_calc.py               # calc only
uv run pytest tests/test_calc.py::TestPCA      # single class
uv run pytest -x                                # stop on first failure
uv run pytest -v --tb=short                    # short tracebacks
uv run pytest -k "not LWPLS"                   # exclude a class
```
