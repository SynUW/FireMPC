import numpy as np
import pytest

from wildfire.modis.to_tiles import _reduce_level1_blocks, pyramid_policy


def test_optical_pyramid_policy_is_per_band():
    assert pyramid_policy("MOD09GA_b1237", 0) == "mean"
    assert pyramid_policy("MOD09GA_b1237", 4) == "sample"
    assert pyramid_policy("MYD09GA_b1237", 4) == "sample"


def test_sample_pyramid_uses_upper_left_value_and_mask():
    blocks = np.array(
        [[[[11, 12], [13, 14]], [[21, 22], [23, 24]]]],
        dtype=np.float64,
    )
    assert _reduce_level1_blocks(blocks, 11, "sample").tolist() == [[11, 13]]


def test_mean_pyramid_ignores_fill_and_rounds_half_up():
    blocks = np.array(
        [[[[1, -999], [2, 3]], [[4, 5], [6, 7]]]],
        dtype=np.float64,
    )
    assert _reduce_level1_blocks(blocks, -999, "mean").tolist() == [[3, 5]]


def test_unknown_pyramid_policy_fails():
    with pytest.raises(ValueError, match="unsupported pyramid policy"):
        _reduce_level1_blocks(np.zeros((1, 2, 1, 2)), None, "mode")
