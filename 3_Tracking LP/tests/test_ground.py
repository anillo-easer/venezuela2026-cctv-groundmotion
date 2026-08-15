from __future__ import annotations

import cv2
import numpy as np

from light_pole_tracking.ground import (
    GroundTrackingConfig,
    estimate_ground_translation,
    track_ground_translations,
)


def make_reference() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rng = np.random.default_rng(42)
    frame = np.zeros((90, 110), dtype=np.uint8)
    rois = [(12, 14, 12, 10), (48, 18, 11, 13), (76, 58, 14, 11)]
    for x, y, width, height in rois:
        frame[y : y + height, x : x + width] = rng.integers(
            10,
            246,
            size=(height, width),
            dtype=np.uint8,
        )
    return frame, rois


def test_estimates_known_translation() -> None:
    reference, rois = make_reference()
    matrix = np.float32([[1, 0, 4], [0, 1, -3]])
    current = cv2.warpAffine(reference, matrix, (110, 90))
    estimate = estimate_ground_translation(
        reference,
        current,
        rois,
        frame_index=8,
        config=GroundTrackingConfig(
            search_radius_px=(8, 6),
            consensus_threshold_px=0.25,
            min_inliers=3,
            min_correlation=0.95,
        ),
    )
    assert estimate.valid
    np.testing.assert_allclose((estimate.dx, estimate.dy), (4.0, -3.0), atol=0.1)
    assert estimate.inlier_count == 3


def test_median_consensus_rejects_one_outlier() -> None:
    reference, rois = make_reference()
    current = np.zeros_like(reference)
    shifts = [(3, -2), (3, -2), (-5, 4)]
    for (x, y, width, height), (dx, dy) in zip(rois, shifts, strict=True):
        current[
            y + dy : y + dy + height,
            x + dx : x + dx + width,
        ] = reference[y : y + height, x : x + width]
    estimate = estimate_ground_translation(
        reference,
        current,
        rois,
        frame_index=3,
        config=GroundTrackingConfig(
            search_radius_px=8,
            consensus_threshold_px=0.25,
            min_inliers=2,
            min_correlation=0.95,
        ),
    )
    assert estimate.valid
    np.testing.assert_allclose((estimate.dx, estimate.dy), (3.0, -2.0), atol=0.1)
    assert estimate.inlier_count == 2


def test_missing_frame_is_explicitly_invalid() -> None:
    reference, rois = make_reference()
    rows = track_ground_translations(
        reference,
        [(10, reference), (11, None), (12, reference)],
        rois,
        config=GroundTrackingConfig(min_inliers=2, min_correlation=0.95),
    )
    assert [row.frame_index for row in rows] == [10, 11, 12]
    assert [row.valid for row in rows] == [True, False, True]
    assert np.isnan(rows[1].dx)
    assert np.isnan(rows[1].dy)
    assert rows[1].reason == "frame_missing"


def test_scalar_search_radius_expands_to_two_axes() -> None:
    config = GroundTrackingConfig(search_radius_px=7)
    assert config.search_radius_px == (7, 7)


def test_fractional_translation_uses_subpixel_peak() -> None:
    reference, rois = make_reference()
    matrix = np.float32([[1, 0, 2.4], [0, 1, -1.7]])
    current = cv2.warpAffine(reference, matrix, (110, 90))
    estimate = estimate_ground_translation(
        reference,
        current,
        rois,
        frame_index=1,
        config=GroundTrackingConfig(
            search_radius_px=6,
            consensus_threshold_px=0.5,
            min_inliers=3,
            min_correlation=0.6,
        ),
    )
    assert estimate.valid
    np.testing.assert_allclose(
        (estimate.dx, estimate.dy),
        (2.4, -1.7),
        atol=0.3,
    )
