# pytest Test Suite Design — pyphi

**Date:** 2026-03-24
**Scope:** All three pyphi modules — `calc`, `plots`, `batch`

---

## Goals

Add a pytest test suite that exercises every public function and critical internal helper across all three pyphi modules. Tests are the deliverable; when a test fails it flags a bug for separate investigation — no fixes are in scope here.

---

## Decisions

| Question | Decision |
|---|---|
| plots module | Internal helper functions only (no Bokeh rendering) |
| Test data | Load from existing Excel files in `examples/` |
| Model coverage | Full — all model types at the same depth |
| On failure | Tests are the deliverable; bugs fixed separately |
| Structure | One file per module + shared `conftest.py` |

---

## File Layout

```
tests/
├── conftest.py        # session-scoped fixtures: Excel data + built model objects
├── test_calc.py       # pyphi.calc — all computation
├── test_plots.py      # pyphi.plots — internal helpers only
└── test_batch.py      # pyphi.batch — pure functions only
```

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## conftest.py

All fixtures are `scope="session"` to load Excel files and build models once.

The top of `conftest.py` must set the matplotlib backend before any pyphi import to prevent GUI windows in CI:
```python
import matplotlib
matplotlib.use('Agg')
```

### Data fixtures

| Fixture | Source file | Used by |
|---|---|---|
| `cars_features` | `examples/Basic calculations PCA and PLS/Automobiles PLS.xlsx` → `Features` sheet | TestPCA, TestPLS |
| `cars_performance` | same file → `Performance` sheet | TestPLS |
| `cars_classid` | same file → `CLASSID` sheet | TestPlots helpers |
| `lpls_dataset` | `examples/LPLS/lpls_dataset.xlsx` | TestLPLS |
| `jrpls_dataset` | `examples/JRPLS and TPLS/jrpls_tpls_dataset.xlsx` | TestJRPLS, TestTPLS |
| `jypls_dataset` | `examples/JRPLS and TPLS/jrpls_tpls_dataset.xlsx` | TestJYPLS (same file, different sheets/structure) |
| `mbpls_dataset` | `examples/Multi-block PLS/MBDataset.xlsx` | TestMBPLS |
| `nir_dataset` | `examples/NIR Calibration/NIR.xlsx` | TestLWPLS, TestSpectra |
| `batch_film_dataset` | `examples/Batch analysis/Batch Film Coating.xlsx` | TestMPCA, TestMPLS, TestAlignment, TestUnfoldRefold |
| `batch_dryer_dataset` | `examples/Batch analysis/Batch Dryer Case Study.xlsx` | TestDescriptors, TestBuildRelTime |
| `chem_exp_dataset` | `examples/Varimax Rotation/chemical_experiments_dataset.xlsx` | TestVarimax, TestBuildPolynomial |
| `opls_dataset` | `examples/plscca vs opls/OPLS Test Data.xlsx` | TestCCA |

### Model fixtures

Built on top of data fixtures — expensive models built once and reused:

| Fixture | Builds |
|---|---|
| `pca_model` | `phi.pca(cars_features, 3)` |
| `pca_model_cv` | `phi.pca(cars_features, 3, cross_val=5)` |
| `pls_model` | `phi.pls(cars_features, cars_performance, 3)` |
| `pls_model_cv` | `phi.pls(cars_features, cars_performance, 3, cross_val=5)` — Y CV only |
| `pls_model_cv_x` | `phi.pls(cars_features, cars_performance, 3, cross_val=5, cross_val_X=True)` — both X and Y CV |
| `pls_model_cca` | `phi.pls(cars_features, cars_performance, 3, cca=True)` |
| `lpls_model` | `phi.lpls(X, R, Y, 2)` from lpls_dataset |
| `jrpls_model` | `phi.jrpls(Xi, Ri, Y, 2)` from jrpls_dataset |
| `tpls_model` | `phi.tpls(Xi, Ri, Z, Y, 2)` from jrpls_dataset |
| `jypls_model` | `phi.jypls(Xi, Yi, 2)` from jypls_dataset (Xi, Yi are dicts of DataFrames) |
| `mbpls_model` | `phi.mbpls(XMB, YMB, 3)` from mbpls_dataset |
| `mpca_model` | `batch.mpca(xbatch, 3)` from batch_film_dataset |
| `mpls_model` | `batch.mpls(xbatch, y, 3)` from batch_film_dataset |

---

## test_calc.py

### TestValidateInputs

Covers `_validate_inputs`. All tests use in-memory DataFrames/arrays (no fixtures needed).

| Test | What it checks |
|---|---|
| `test_rejects_non_dataframe` | Raises `ValueError` for list/dict/string X |
| `test_rejects_invalid_mcs` | Raises `ValueError` for unrecognized mcs string |
| `test_rejects_A_zero` | Raises `ValueError` for A=0 |
| `test_rejects_A_exceeds_rank` | Raises `ValueError` when A > min(n, p) |
| `test_rejects_duplicate_obs_ids` | Raises `ValueError` for duplicate row IDs |
| `test_rejects_no_common_obs` | Raises `ValueError` when X and Y share no obs IDs |
| `test_reorders_Y_to_match_X` | Y rows reordered when order differs from X |
| `test_drops_obs_missing_in_Y` | Obs in X but not Y are silently dropped from X |
| `test_numpy_row_mismatch_raises` | Raises `ValueError` when X.shape[0] != Y.shape[0] |

### TestHelpers

| Test | What it checks |
|---|---|
| `test_extract_array_from_ndarray` | Returns copy, obsid=False, varid=False |
| `test_extract_array_from_dataframe` | Returns correct array, obsid list, varid list |
| `test_meancenterscale_autoscale` | Output mean ≈ 0, std ≈ 1 per column |
| `test_meancenterscale_center_only` | Output mean ≈ 0, std unchanged |
| `test_meancenterscale_false` | Output identical to input |
| `test_calc_r2_range` | R² values ∈ [0, 1] |
| `test_Ab_btbinv_no_missing` | Result matches direct `A @ b / (b'b)` |
| `test_clean_empty_rows_removes_all_nan` | All-NaN rows removed, others kept |
| `test_clean_low_variances_removes_constants` | Zero-variance columns removed |
| `test_cat_2_matrix_shape` | Output has correct number of dummy columns |

### TestStatFunctions

| Test | What it checks |
|---|---|
| `test_f99_gt_f95` | f99(i,j) > f95(i,j) for several (i,j) |
| `test_f_values_positive` | f99, f95 return positive floats |
| `test_spe_ci_returns_two_limits` | Returns tuple of two positive values |
| `test_scores_conf_int_monotone` | 99% limit > 95% limit |

### TestPCA

| Test | What it checks |
|---|---|
| `test_model_keys_present` | T, P, r2x, r2xpv, mx, sx, T2, speX, var_t in model |
| `test_score_shape` | T.shape == (n_obs, A) |
| `test_loading_shape` | P.shape == (n_vars, A) |
| `test_loadings_orthonormal` | P.T @ P ≈ I (PCA loadings are orthonormal by construction) |
| `test_r2x_in_range` | All r2x values ∈ (0, 1] |
| `test_r2xpv_in_range` | All r2xpv values ∈ [0, 1] |
| `test_reconstruction_residual` | ‖X - T·Pᵀ‖ < ‖X‖ (model explains variance) |
| `test_t2_shape` | T2.shape == (n_obs,) |
| `test_spe_shape` | speX.shape[0] == n_obs |
| `test_cross_val_produces_q2` | `q2` and `q2pv` keys present when cross_val=5 (`pca_model_cv`) |
| `test_pca_pred_training_scores` | pca_pred(X_train) Tnew ≈ model T |
| `test_pca_pred_keys` | Tnew, Xhat, speX, T2 in prediction dict |
| `test_pca_pred_xhat_shape` | Xhat.shape == input X shape (excluding obs-ID column) |
| `test_type_flag` | model['type'] == 'pca' |

### TestPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | T, P, Q, W, Ws, r2x, r2y, r2xpv, r2ypv, mx, sx, my, sy, T2, speX, speY, var_t |
| `test_score_shape` | T.shape == (n_obs, A) |
| `test_loading_shapes` | P.shape == (n_x, A), Q.shape == (n_y, A) |
| `test_r2x_r2y_in_range` | r2x, r2y ∈ (0, 1] |
| `test_cross_val_produces_q2Y` | `q2Y` and `q2Ypv` keys present in `pls_model_cv` (Y CV always computed when cross_val > 0) |
| `test_cross_val_produces_q2X` | `q2X` and `q2Xpv` keys present in `pls_model_cv_x` (only when cross_val_X=True) |
| `test_q2Y_leq_r2y` | q2Y ≤ r2y (CV score ≤ training score) |
| `test_cca_flag_adds_keys` | Tcv, Pcv, Wcv present when cca=True (`pls_model_cca`) |
| `test_pls_pred_yhat_shape` | Yhat.shape == (n_obs, n_y) |
| `test_pls_pred_training_roundtrip` | pls_pred(X_train) Tnew ≈ model T |
| `test_type_flag` | model['type'] == 'pls' |

### TestDiagnostics

Covers `hott2`, `spe`, `contributions` using the trained PCA/PLS models.

| Test | What it checks |
|---|---|
| `test_hott2_training_shape` | hott2(pca_model) shape == (n_obs,) |
| `test_hott2_matches_stored_T2` | hott2 result ≈ pca_model['T2'] |
| `test_hott2_with_Xnew` | hott2(pca_model, Xnew=X) returns array of correct length |
| `test_spe_training_shape` | spe(pca_model, X) shape[0] == n_obs |
| `test_contributions_scores_shape` | contributions(pca_model, X, 'scores') returns correct shape |
| `test_contributions_obs_shape` | contributions(pca_model, X, 'obs', to_obs=['obs1']) returns correct shape |

### TestMBPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | T, P, Q, W, r2x, r2y, r2pbX present |
| `test_r2pbX_shape` | r2pbX shape[0] == number of blocks |
| `test_score_shape` | T.shape == (n_obs, A) |

### TestLPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | T, P, r2x and LPLS-specific keys present |
| `test_score_shape` | T.shape correct |
| `test_lpls_pred_shape` | lpls_pred output has correct row count |

### TestJRPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | Expected keys present |
| `test_jrpls_pred_shape` | jrpls_pred output rows match input |

### TestTPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | Expected keys present |
| `test_tpls_pred_shape` | tpls_pred output rows match input |

### TestJYPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | Expected keys present in `jypls_model` |
| `test_jypls_pred_shape` | jypls_pred output rows match Xi rows |

### TestLWPLS

| Test | What it checks |
|---|---|
| `test_output_shape` | Output rows == number of new observations |
| `test_output_is_numeric` | No NaN in prediction output |

### TestBootstrapPLS

`bootstrap_pls` returns a **list** of PLS model dicts (one per bootstrap sample), not a single dict.
`bootstrap_pls_pred` returns a **list of 1-D arrays**, one per quantile, each of shape `(n_y,)`.

| Test | What it checks |
|---|---|
| `test_bootstrap_returns_list` | Result is a list of length `num_samples` |
| `test_bootstrap_each_element_is_pls_dict` | Each element has T, P, Q keys |
| `test_bootstrap_pred_returns_list` | bootstrap_pls_pred returns a list |
| `test_bootstrap_pred_quantile_count` | List length == len(quantiles) |
| `test_bootstrap_pred_quantile_shape` | Each quantile array has shape `(n_y,)` |

### TestVarimax

| Test | What it checks |
|---|---|
| `test_rotated_loadings_shape` | Rotated P same shape as original P |
| `test_r2_unchanged_sum` | Total R² preserved after rotation |

### TestSpectra

Each preprocessing function tested with the NIR dataset (numpy array, no obs-ID column):

| Test | What it checks |
|---|---|
| `test_snv_shape` | Output shape == input shape |
| `test_snv_row_mean_zero` | Each row mean ≈ 0 after SNV |
| `test_snv_row_std_one` | Each row std ≈ 1 after SNV |
| `test_savgol_shape` | Output shape == input shape |
| `test_msc_shape` | Output shape == input shape |
| `test_mean_center_shape` | Output shape == input shape |
| `test_autoscale_shape` | Output shape == input shape |
| `test_baseline_correction_shape` | Output shape == input shape |

### TestCCA

`cca(X, Y)` returns a **tuple** `(correlation_scalar, w_x, w_y)`. Tests call it inline (no model fixture needed).

| Test | What it checks |
|---|---|
| `test_returns_tuple_of_three` | Return value is a 3-tuple |
| `test_weight_vector_shapes` | w_x.shape == (n_x,), w_y.shape == (n_y,) |
| `test_weights_unit_norm` | ‖w_x‖ ≈ 1, ‖w_y‖ ≈ 1 |
| `test_correlation_in_range` | correlation scalar ∈ [-1, 1] |

### TestBuildPolynomial

`build_polynomial` calls `plt.figure()` internally. The `matplotlib.use('Agg')` call at the top of `conftest.py` prevents GUI display.

| Test | What it checks |
|---|---|
| `test_returns_model_dict` | Returns dict with regression coefficients |
| `test_prediction_shape` | Yhat shape matches input rows |

### TestUtilities

| Test | What it checks |
|---|---|
| `test_unique_preserves_order` | unique() returns values in first-occurrence order |
| `test_reconcile_rows_common_only` | reconcile_rows keeps only obs present in all DataFrames |
| `test_reconcile_rows_to_columns` | Output list has consistent row/column alignment |
| `test_hott2_new_observations` | hott2 with Tnew kwarg returns correct shape |

---

## test_plots.py

No Bokeh rendering; all tests operate on model dicts directly.

### TestLVLabels

| Test | What it checks |
|---|---|
| `test_pca_prefix` | Returns strings starting with "PC #" |
| `test_pls_prefix` | Returns strings starting with "LV #" |
| `test_length_equals_A` | Length of returned list == model's A |

### TestVarLabels

| Test | What it checks |
|---|---|
| `test_uses_varidX_when_present` | Returns model['varidX'] when key exists |
| `test_fallback_format` | Returns "XVar #N" strings when varidX absent |
| `test_yvar_uses_varidY` | Returns model['varidY'] for PLS model |
| `test_yvar_fallback` | Returns "YVar #N" strings when varidY absent |

### TestTimestr

| Test | What it checks |
|---|---|
| `test_returns_string` | Return type is str |
| `test_exact_length` | len(result) == 20 (format `%Y%m%d%H%M%S%f` always produces 20 chars) |
| `test_all_digits` | Result contains only digit characters |

### TestMakePalette

| Test | What it checks |
|---|---|
| `test_length` | Returns list of requested length |
| `test_hex_format` | Each entry matches `#[0-9a-fA-F]{6}` |

### TestMaskByClass

| Test | What it checks |
|---|---|
| `test_filters_correctly` | Only obs matching class label returned |
| `test_empty_class` | Returns empty list for unknown class |

### TestResolveLplsSpace

`_resolve_lpls_space(mvmobj, material, zspace)` returns a **dict** (a copy of the model with adjusted keys for the requested space). The `zspace` bool selects between Z-space and standard space; there is no `x_space` parameter.

| Test | What it checks |
|---|---|
| `test_lpls_zspace_false_returns_dict` | Returns a dict when called with an lpls model and `zspace=False` |
| `test_tpls_zspace_true_uses_varidZ` | Given a tpls model and `zspace=True`, returned dict's `varidX` equals original `varidZ` |

---

## test_batch.py

No matplotlib rendering; plot functions excluded. The `matplotlib.use('Agg')` in `conftest.py` covers any transitive matplotlib imports.

### TestBatchUtilities

| Test | What it checks |
|---|---|
| `test_unique_preserves_order` | Matches insertion order, no duplicates |
| `test_mean_ignores_nan` | NaN not counted in mean computation |
| `test_clean_empty_rows` | All-NaN rows removed; partial rows kept |

### TestAlignment

| Test | What it checks |
|---|---|
| `test_simple_align_sample_count` | Each batch has exactly `nsamples` rows |
| `test_simple_align_batch_count` | Number of batches unchanged |
| `test_phase_align_sample_count` | Each batch has exactly `nsamples` rows per phase |

### TestUnfoldRefold

| Test | What it checks |
|---|---|
| `test_unfold_shape` | unfolded shape == (n_batches, n_vars * n_samples) |
| `test_roundtrip` | `refold(unfold(X), nvars, nsamples)` recovers original data |

### TestMPCA

| Test | What it checks |
|---|---|
| `test_model_keys_present` | T, P, r2x keys present |
| `test_score_shape` | T.shape == (n_batches, A) |
| `test_r2x_in_range` | r2x values ∈ (0, 1] |

### TestMPLS

| Test | What it checks |
|---|---|
| `test_model_keys_present` | T, Q keys present |
| `test_score_shape` | T.shape == (n_batches, A) |

### TestDescriptors

| Test | What it checks |
|---|---|
| `test_output_shape` | Shape == (n_batches, len(desc)) |
| `test_no_nan_in_output` | No NaN in result for complete data |

### TestBuildRelTime

| Test | What it checks |
|---|---|
| `test_output_is_numeric` | Result column is numeric dtype |
| `test_monotone_per_batch` | Time is non-decreasing within each batch |

---

## pytest configuration (pyproject.toml addition)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Running tests

```bash
uv run pytest                              # all tests
uv run pytest tests/test_calc.py          # calc module only
uv run pytest tests/test_calc.py::TestPCA # single class
uv run pytest -x                           # stop at first failure
uv run pytest -v                           # verbose output
```
