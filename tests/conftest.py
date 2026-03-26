import matplotlib
matplotlib.use('Agg')  # Must be before pyphi imports to prevent GUI in CI

import numpy as np
import pandas as pd
import pytest
import pyphi.calc as phi
import pyphi.batch as batch
from pathlib import Path

EXAMPLES = Path(__file__).parent.parent / "examples"


# ── PCA / PLS ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cars_features():
    return pd.read_excel(
        EXAMPLES / "Basic calculations PCA and PLS" / "Automobiles PLS.xlsx",
        sheet_name="Features", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def cars_performance():
    return pd.read_excel(
        EXAMPLES / "Basic calculations PCA and PLS" / "Automobiles PLS.xlsx",
        sheet_name="Performance", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def cars_classid():
    return pd.read_excel(
        EXAMPLES / "Basic calculations PCA and PLS" / "Automobiles PLS.xlsx",
        sheet_name="CLASSID", engine="openpyxl",
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
    if jr is False:
        pytest.fail("parse_materials failed for jrpls_tpls_dataset.xlsx — check Materials sheet")
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
        sheet_name="NIR", engine="openpyxl",
    )

@pytest.fixture(scope="session")
def nir_y():
    return pd.read_excel(
        EXAMPLES / "NIR Calibration" / "NIR.xlsx",
        sheet_name="Y", engine="openpyxl",
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
    Xi, _, _, _ = jrpls_data
    materials = list(Xi.keys())
    # jypls requires all Yi blocks to have identical column structure.
    # Build synthetic Yi: same 2-column Y for every material, obs IDs matching Xi.
    rng = np.random.default_rng(42)
    Yi = {}
    for m in materials:
        obs_col = Xi[m].columns[0]
        obs_ids = Xi[m][obs_col].tolist()
        n = len(obs_ids)
        yi_data = rng.standard_normal((n, 2))
        yi_df = pd.DataFrame(yi_data, columns=["Y1", "Y2"])
        yi_df.insert(0, obs_col, obs_ids)
        Yi[m] = yi_df
    return phi.jypls(Xi, Yi, 2, shush=True)

@pytest.fixture(scope="session")
def mbpls_model(mbpls_data):
    XMB, Y = mbpls_data
    return phi.mbpls(XMB, Y, 2, shush_=True)

# ── Batch models ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mpca_model(batch_film_phase_aligned, batch_film_phase_samples):
    return batch.mpca(batch_film_phase_aligned, 2,
                      phase_samples=batch_film_phase_samples)

@pytest.fixture(scope="session")
def mpls_model(batch_dryer_phase_aligned, batch_dryer_cqa, batch_dryer_phase_samples):
    return batch.mpls(batch_dryer_phase_aligned, batch_dryer_cqa, 3,
                      phase_samples=batch_dryer_phase_samples)
