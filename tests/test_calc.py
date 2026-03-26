import pytest
import numpy as np
import pandas as pd
import pyphi.calc as phi


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

    def test_rejects_list_input(self):
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
        X = self._x()  # 5 obs: r1..r5
        Y = _make_df(
            np.random.randn(3, 2), ["r1", "r2", "r3"], ["y1", "y2"]
        )  # Y only has r1..r3
        X_out, _ = phi._validate_inputs(X, Y)
        assert X_out.shape[0] == 3
        assert X_out.iloc[:, 0].tolist() == ["r1", "r2", "r3"]

    def test_numpy_row_mismatch_raises(self):
        X = np.random.randn(5, 3)
        Y = np.random.randn(4, 2)
        with pytest.raises(ValueError):
            phi._validate_inputs(X, Y)


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
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 5))
        TSS = np.sum(X ** 2)
        TSSpv = np.sum(X ** 2, axis=0)
        # Perfect reconstruction: residual = 0 → r2 = 1.0
        residual = np.zeros_like(X)
        r2, r2pv = phi._calc_r2(residual, TSS, TSSpv)
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)
        np.testing.assert_allclose(r2pv.flatten(), 1.0, atol=1e-10)

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
        out, _ = phi.cat_2_matrix(df)
        # 4 rows × 4 columns (ObsID + 3 dummy columns for A, B, C)
        assert out.shape == (4, 4)


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
        # 99% y-extent should be larger than 95% y-extent
        assert np.nanmax(np.abs(result[4])) > np.nanmax(np.abs(result[2]))


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
        assert float(np.sum(pca_model["r2x"])) <= 1.0

    def test_r2xpv_in_range(self, pca_model):
        r2xpv = pca_model["r2xpv"]
        assert np.all(r2xpv >= 0)
        assert np.all(r2xpv <= 1.0)
        assert np.any(r2xpv > 0.1)

    def test_reconstruction_residual(self, pca_model, cars_features):
        X = np.array(cars_features.values[:, 1:]).astype(float)
        X_scaled = (X - pca_model["mx"]) / pca_model["sx"]
        T = pca_model["T"]
        P = pca_model["P"]
        # Mask rows with any NaN (NIPALS handles them during fitting but norms require finite values)
        complete_rows = ~np.any(np.isnan(X_scaled), axis=1)
        X_c = X_scaled[complete_rows]
        T_c = T[complete_rows]
        residual_norm = np.linalg.norm(X_c - T_c @ P.T)
        data_norm = np.linalg.norm(X_c)
        assert residual_norm < data_norm

    def test_t2_shape(self, pca_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pca_model["T2"].shape == (n_obs,)

    def test_spe_shape(self, pca_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pca_model["speX"].shape == (n_obs, 1)

    def test_cross_val_produces_q2(self, pca_model_cv):
        assert "q2" in pca_model_cv
        assert "q2pv" in pca_model_cv
        # q2 values should be finite and in a plausible range
        q2 = np.array(pca_model_cv["q2"])
        assert np.all(np.isfinite(q2))
        assert np.all(q2 <= 1.0)

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

    def test_spex_spey_shapes(self, pls_model, cars_features):
        n_obs = cars_features.shape[0]
        assert pls_model["speX"].shape == (n_obs, 1)
        assert pls_model["speY"].shape == (n_obs, 1)

    def test_r2x_r2y_in_range(self, pls_model):
        r2x = np.array(pls_model["r2x"])
        r2y = np.array(pls_model["r2y"])
        assert np.all(r2x > 0) and np.all(r2x <= 1.0)
        assert np.all(r2y > 0) and np.all(r2y <= 1.0)
        assert float(np.sum(r2x)) <= 1.0
        assert float(np.sum(r2y)) <= 1.0

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
        # For complete rows, should match stored T2
        X = np.array(cars_features.values[:, 1:]).astype(float)
        X_scaled = (X - pca_model["mx"]) / pca_model["sx"]
        complete_rows = ~np.any(np.isnan(X_scaled), axis=1)
        np.testing.assert_allclose(
            result[complete_rows], pca_model["T2"][complete_rows], rtol=1e-5
        )

    def test_spe_training_shape(self, pca_model, cars_features):
        result = phi.spe(pca_model, cars_features)
        # spe returns (n_obs, 1)
        assert result.shape == (cars_features.shape[0], 1)

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


class TestLPLS:

    def test_model_keys_present(self, lpls_model):
        for key in ("T", "P", "r2x"):
            assert key in lpls_model, f"Missing key: {key}"
        # LPLS-specific: R-space projected scores (only present in LPLS models)
        assert "Rscores" in lpls_model

    def test_type_flag(self, lpls_model):
        assert lpls_model["type"] == "lpls"

    def test_score_shape(self, lpls_model, lpls_X, lpls_R):
        # T in LPLS corresponds to R-space (raw material) scores
        n_materials = lpls_R.shape[0]
        assert lpls_model["T"].shape[0] == n_materials

    def test_lpls_pred_shape(self, lpls_model, lpls_R):
        r_new = lpls_R.iloc[[0]]
        pred = phi.lpls_pred(r_new, lpls_model)
        assert isinstance(pred, dict), f"Expected dict, got: {type(pred).__name__}: {pred}"
        assert "Tnew" in pred


class TestJRPLS:

    def test_model_keys_present(self, jrpls_model):
        for key in ("T", "P", "Q", "type"):
            assert key in jrpls_model, f"Missing key: {key}"

    def test_type_flag(self, jrpls_model):
        assert jrpls_model["type"] == "jrpls"

    def test_jrpls_pred_shape(self, jrpls_model, jrpls_data):
        Xi, Ri, quality, _ = jrpls_data
        materials = list(Ri.keys())
        rnew = {m: [(Ri[m].columns[1], 1.0)] for m in materials}
        pred = phi.jrpls_pred(rnew, jrpls_model)
        assert isinstance(pred, dict), f"Expected dict, got: {type(pred).__name__}: {pred}"
        assert "Tnew" in pred


class TestTPLS:

    def test_model_keys_present(self, tpls_model):
        for key in ("T", "P", "Q", "type"):
            assert key in tpls_model, f"Missing key: {key}"

    def test_type_flag(self, tpls_model):
        assert tpls_model["type"] == "tpls"

    def test_tpls_pred_shape(self, tpls_model, jrpls_data):
        Xi, Ri, quality, process = jrpls_data
        materials = list(Ri.keys())
        rnew = {m: [(Ri[m].columns[1], 1.0)] for m in materials}
        znew = process.iloc[0, 1:].values.astype(float)
        pred = phi.tpls_pred(rnew, znew, tpls_model)
        assert isinstance(pred, dict), f"Expected dict, got: {type(pred).__name__}: {pred}"
        assert "Tnew" in pred


class TestJYPLS:

    def test_model_keys_present(self, jypls_model):
        for key in ("T", "type"):
            assert key in jypls_model, f"Missing key: {key}"

    def test_type_flag(self, jypls_model):
        assert jypls_model["type"] == "jypls"

    def test_jypls_pred_shape(self, jypls_model, jrpls_data):
        Xi, _, _, _ = jrpls_data
        campaign = list(Xi.keys())[0]
        xnew = Xi[campaign].iloc[[0]]
        pred = phi.jypls_pred(xnew, campaign, jypls_model)
        assert isinstance(pred, dict), f"Expected dict, got: {type(pred).__name__}: {pred}"
        assert "Tnew" in pred


class TestLWPLS:
    """lwpls is called once per observation; returns yhat for that observation.
    Note: the docstring claims a dict return but source returns an ndarray — a bug to investigate."""

    @pytest.fixture(scope="class")
    def lwpls_result(self, nir_spectra, nir_y):
        X = np.array(nir_spectra.values[:, 1:]).astype(float)
        Y = np.array(nir_y.values[:, 1:]).astype(float)
        pls_nir = phi.pls(X, Y, 1, shush=True)
        xnew = X[0, :]
        return phi.lwpls(xnew, 20, pls_nir, X, Y, shush=True)

    def test_output_is_array(self, lwpls_result):
        assert isinstance(lwpls_result, np.ndarray)

    def test_output_numeric(self, lwpls_result):
        assert not np.any(np.isnan(lwpls_result.ravel()))


class TestBootstrapPLS:

    @pytest.fixture(scope="class")
    def bootstrap_result(self, cars_features, cars_performance):
        # Note: bootstrap_pls already hardcodes shush=True internally; do not pass it
        # again via kwargs or pls() will receive duplicate keyword argument (source bug).
        return phi.bootstrap_pls(cars_features, cars_performance,
                                 num_latents=2, num_samples=5)

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


class TestVarimax:

    @pytest.fixture(scope="class")
    def varimax_setup(self, chem_exp_data):
        x_cols = [c for c in chem_exp_data.columns if c != "Lot" and c in
                  ["SM (eq)", "A (eq)", "B (eq)", "C (eq)", "Solvent (Vol)"]]
        y_cols = [c for c in chem_exp_data.columns if c != "Lot" and c in
                  ["IPC_N", "IPC_H"]]
        ra_X = chem_exp_data[["Lot"] + x_cols]
        ra_Y = chem_exp_data[["Lot"] + y_cols]
        plsobj = phi.pls(ra_X, ra_Y, 2, shush=True)
        rotated = phi.varimax_rotation(plsobj, ra_X, Y=ra_Y)
        return plsobj, rotated

    def test_rotated_loadings_shape(self, varimax_setup):
        plsobj, rotated = varimax_setup
        assert rotated["P"].shape == plsobj["P"].shape

    def test_r2_total_preserved(self, varimax_setup):
        plsobj, rotated = varimax_setup
        # Varimax rotation redistributes variance across components but preserves
        # the total (sum) of r2y across all components.
        orig_r2y_total = float(np.sum(plsobj["r2y"]))
        rot_r2y_total = float(np.sum(rotated["r2y"]))
        np.testing.assert_allclose(orig_r2y_total, rot_r2y_total, rtol=1e-4)


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
        out, _ = phi.spectra_savgol(10, 1, 2, snv)
        # Savitzky-Golay trims ws-1 columns from each edge; row count preserved.
        # Only check that row count matches and output has fewer/equal columns.
        assert out.shape[0] == snv.shape[0]
        assert out.shape[1] <= snv.shape[1]

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
        result, _, _ = cca_result
        assert len(result) == 3

    def test_weight_vector_shapes(self, cca_result):
        (_, w_x, w_y), n_x, n_y = cca_result
        assert w_x.shape == (n_x,)
        assert w_y.shape == (n_y,)

    def test_weights_unit_norm(self, cca_result):
        (_, w_x, w_y), _, _ = cca_result
        np.testing.assert_allclose(np.linalg.norm(w_x), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(w_y), 1.0, atol=1e-6)

    def test_correlation_in_range(self, cca_result):
        (corr, _, _), _, _ = cca_result
        assert -1.0 <= float(corr) <= 1.0


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
        # Note: build_polynomial takes data (full df) + factors (variable names only,
        # without the obs-ID column) + response (column name).
        return chem_exp_data, factor_cols, response_col

    def test_returns_model(self, poly_setup):
        data, factors, response = poly_setup
        result = phi.build_polynomial(data, factors, response)
        assert result is not None

    def test_prediction_shape(self, poly_setup):
        data, factors, response = poly_setup
        # build_polynomial returns (betasOLSlssq, factors_out, Xaug, Y_arr, eqstr)
        result = phi.build_polynomial(data, factors, response)
        assert result is not None
        # Unpack the 5-tuple returned by build_polynomial
        _, _, Xaug, _, _ = result
        # Xaug is (n_obs, n_factors + 1 bias) — verify row count
        assert Xaug.shape[0] == data.shape[0]


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
        # reconcile_rows_to_columns zips df_list_r and df_list_c in pairs;
        # output length == min(len(df_list_r), len(df_list_c)).
        # Use equal-length paired lists — each r-block is matched 1-to-1 with a c-block.
        df_list_r = [
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["x1", "x2"]),
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["x3", "x4"]),
        ]
        df_list_c = [
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["r1", "r2"]),
            _make_df(np.ones((3, 2)), ["a", "b", "c"], ["r3", "r4"]),
        ]
        out_x, out_r = phi.reconcile_rows_to_columns(df_list_r, df_list_c)
        assert len(out_x) == 2
        assert len(out_r) == 2

    def test_hott2_with_Tnew(self, pca_model):
        T_new = pca_model["T"][:5, :]
        result = phi.hott2(pca_model, Tnew=T_new)
        assert result.shape == (5,)
