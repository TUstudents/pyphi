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
        aligned: pd.DataFrame = batch.phase_simple_align(batch_film_raw, batch_film_phase_samples)  # type: ignore[assignment]
        expected_per_batch = sum(batch_film_phase_samples.values())
        batch_col = aligned.columns[0]
        for _, grp in aligned.groupby(batch_col):
            assert grp.shape[0] == expected_per_batch


class TestUnfoldRefold:
    # NOTE: batch.unfold_horizontal returns a tuple (bdata_hor_df, clbl, bid)
    # not a plain DataFrame. The first element is the unfolded DataFrame with
    # a batch-ID column prepended.

    def test_unfold_shape(self, batch_film_aligned):
        batch_col = batch_film_aligned.columns[0]
        n_batches = batch_film_aligned[batch_col].nunique()
        # Number of variables = total columns - batch ID - optional PHASE
        non_data_cols = sum(1 for c in batch_film_aligned.columns
                            if c.upper() in (batch_col.upper(), "PHASE"))
        n_vars = batch_film_aligned.shape[1] - non_data_cols
        nsamples = 250
        # unfold_horizontal returns (df, clbl, bid) tuple
        xuf_df, _, _ = batch.unfold_horizontal(batch_film_aligned)
        # xuf_df has the batch-ID column prepended; data columns = n_vars * nsamples
        assert xuf_df.shape == (n_batches, 1 + n_vars * nsamples)

    def test_roundtrip(self, batch_film_aligned):
        batch_col = batch_film_aligned.columns[0]
        non_data_cols = sum(1 for c in batch_film_aligned.columns
                            if c.upper() in (batch_col.upper(), "PHASE"))
        n_vars = batch_film_aligned.shape[1] - non_data_cols
        nsamples = 250
        xuf_df, _, _ = batch.unfold_horizontal(batch_film_aligned)
        # refold_horizontal needs strictly numeric array — drop the batch-ID column
        xuf_numeric = xuf_df.iloc[:, 1:].values
        refolded = np.array(batch.refold_horizontal(xuf_numeric, n_vars, nsamples))
        # refolded is a numpy array with shape (n_batches * nsamples, n_vars)
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
        rows = []
        for b in ["B1", "B2"]:
            for i in range(5):
                rows.append({
                    "BATCH": b,
                    # build_rel_time reads this_batch['Timestamp'] — column must be 'Timestamp'
                    "Timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=int(i) * 10),  # type: ignore[operator]
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
            times = grp["Time (min)"].to_numpy(dtype=float)
            assert np.all(np.diff(times) >= 0), f"Non-monotone rel_time for {b_id}"
