import numpy as np
import pytest

from light_pole_tracking.lens import LensRemap, sha256_file


def test_lens_remap_blacks_invalid_pixels(tmp_path) -> None:
    height, width = 5, 6
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    valid = np.ones((height, width), dtype=bool)
    valid[:, 0] = False
    path = tmp_path / "remap.npz"
    np.savez(path, map_x=map_x, map_y=map_y, valid_mask=valid)
    remap = LensRemap.load(path, expected_sha256=sha256_file(path))
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    result = remap.apply(image)
    assert not result[:, 0].any()
    assert np.all(result[:, 1:] == 120)


def test_lens_hash_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "remap.npz"
    array = np.zeros((2, 2), dtype=np.float32)
    np.savez(path, map_x=array, map_y=array)
    with pytest.raises(ValueError, match="hash mismatch"):
        LensRemap.load(path, expected_sha256="0" * 64)
