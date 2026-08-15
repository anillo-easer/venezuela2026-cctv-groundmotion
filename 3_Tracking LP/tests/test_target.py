import numpy as np
import pytest

from light_pole_tracking.target import (
    detect_mask,
    detections_from_masks,
    largest_component,
)


def test_largest_component_discards_smaller_regions() -> None:
    mask = np.zeros((8, 9), dtype=bool)
    mask[1:3, 1:3] = True
    mask[3:7, 4:8] = True
    result = largest_component(mask)
    assert int(result.sum()) == 16


def test_detection_uses_full_frame_coordinates() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:4] = True
    result = detect_mask(mask, [100, 200, 106, 204])
    assert result is not None
    assert result.x_px == 102.5
    assert result.y_px == 201.5


def test_empty_and_large_steps_remain_explicit() -> None:
    masks = np.zeros((3, 6, 8), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[2, 1:3, 6:8] = True
    rows = detections_from_masks(
        [10, 11, 12], masks, [0, 0, 8, 6], fps=25.0, maximum_step_px=2.0
    )
    assert rows[0]["valid"] is True
    assert rows[1]["valid"] is False
    assert rows[1]["reason"] == "empty_mask"
    assert rows[2]["valid"] is False
    assert rows[2]["reason"] == "centroid_step_above_limit"
    assert np.isnan(rows[2]["x_px"])


def test_time_uses_source_video_frame_index() -> None:
    masks = np.ones((1, 2, 2), dtype=bool)
    rows = detections_from_masks([500], masks, [0, 0, 2, 2], fps=25.0)
    assert rows[0]["time_s"] == 20.0


def test_mask_shape_is_checked_even_when_empty() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        detect_mask(np.zeros((3, 3), dtype=bool), [0, 0, 4, 4])


def test_step_limit_must_be_valid() -> None:
    masks = np.ones((1, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        detections_from_masks(
            [0], masks, [0, 0, 2, 2], fps=25.0, maximum_step_px=-1.0
        )
