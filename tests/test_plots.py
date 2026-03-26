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
        assert "Ws" in result, "resolved lpls space must contain Ws"

    def test_lpls_sets_Ws_to_Ss(self, lpls_model):
        result = _resolve_lpls_space(lpls_model, None, False)
        np.testing.assert_array_equal(result["Ws"], lpls_model["Ss"])

    def test_tpls_zspace_true_uses_varidZ(self, tpls_model):
        # tpls models always carry varidZ; zspace=True must copy it to varidX
        assert "varidZ" in tpls_model, "tpls_model missing varidZ key"
        result = _resolve_lpls_space(tpls_model, None, True)
        assert result["varidX"] == tpls_model["varidZ"]
