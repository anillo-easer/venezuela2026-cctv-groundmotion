import numpy as np
import pytest

from light_pole_tracking.config import Sam2SegmentConfig
from light_pole_tracking.sam2_tracker import (
    _prompt_arrays,
    _segment_assignment,
    _validate_roi,
)


def test_prompt_coordinates_are_local_to_roi() -> None:
    points, labels = _prompt_arrays([[12, 23]], [[18, 27]], (10, 20, 30, 40))
    np.testing.assert_allclose(points, [[2, 3], [8, 7]])
    np.testing.assert_array_equal(labels, [1, 0])


def test_positive_prompt_is_required() -> None:
    with pytest.raises(ValueError, match="positive"):
        _prompt_arrays([], [[12, 23]], (10, 20, 30, 40))


def test_roi_must_be_inside_frame() -> None:
    with pytest.raises(ValueError, match="outside"):
        _validate_roi([-1, 0, 20, 20], 100, 80)


def _segment(
    identifier: str,
    start: int,
    end: int,
    anchor: int,
    direction: str,
) -> Sam2SegmentConfig:
    return Sam2SegmentConfig(
        id=identifier,
        start_frame=start,
        end_frame=end,
        anchor_frame=anchor,
        direction=direction,  # type: ignore[arg-type]
        positive_points_xy=((12.0, 23.0),),
        negative_points_xy=(),
    )


def test_segment_assignment_closes_adjacent_chunk_boundary() -> None:
    assignment = _segment_assignment(
        range(6),
        [
            _segment("early", 0, 2, 2, "reverse"),
            _segment("late", 3, 5, 3, "forward"),
        ],
        "error",
    )
    np.testing.assert_array_equal(assignment, [0, 0, 0, 1, 1, 1])


def test_segment_assignment_overlap_policy_is_deterministic() -> None:
    segments = [
        _segment("early", 0, 3, 0, "forward"),
        _segment("late", 2, 5, 5, "reverse"),
    ]
    np.testing.assert_array_equal(
        _segment_assignment(range(6), segments, "first"),
        [0, 0, 0, 0, 1, 1],
    )
    np.testing.assert_array_equal(
        _segment_assignment(range(6), segments, "last"),
        [0, 0, 1, 1, 1, 1],
    )
    with pytest.raises(ValueError, match="overlap"):
        _segment_assignment(range(6), segments, "error")


def test_segment_assignment_rejects_missing_frame_coverage() -> None:
    with pytest.raises(ValueError, match="do not cover processed frame 3"):
        _segment_assignment(
            range(6),
            [
                _segment("early", 0, 2, 2, "reverse"),
                _segment("late", 4, 5, 4, "forward"),
            ],
            "error",
        )
