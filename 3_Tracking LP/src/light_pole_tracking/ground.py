"""Local ground translation from fixed reference regions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

ROIInput: TypeAlias = "ROI | tuple[int, int, int, int]"


@dataclass(frozen=True)
class ROI:
    """Axis-aligned region in ``x, y, width, height`` form."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI dimensions must be positive")

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class GroundTrackingConfig:
    """Parameters for template matching and robust consensus."""

    search_radius_px: int | tuple[int, int] = (12, 12)
    consensus_threshold_px: float = 1.5
    min_inliers: int = 2
    min_correlation: float = 0.5

    def __post_init__(self) -> None:
        radius = self.search_radius_px
        if isinstance(radius, int):
            radius = (radius, radius)
        if len(radius) != 2 or any(value < 0 for value in radius):
            raise ValueError("search_radius_px must contain two non-negative values")
        object.__setattr__(
            self,
            "search_radius_px",
            tuple(int(value) for value in radius),
        )
        if self.consensus_threshold_px < 0:
            raise ValueError("consensus_threshold_px must be non-negative")
        if self.min_inliers < 1:
            raise ValueError("min_inliers must be positive")
        if not -1.0 <= self.min_correlation <= 1.0:
            raise ValueError("min_correlation must be between -1 and 1")

    @property
    def residual_threshold_px(self) -> float:
        """Compatibility alias for the consensus threshold."""

        return self.consensus_threshold_px

    @property
    def min_match_score(self) -> float:
        """Compatibility alias for the correlation threshold."""

        return self.min_correlation


@dataclass(frozen=True)
class GroundTranslation:
    """Consensus translation for one frame."""

    frame_index: int
    dx: float
    dy: float
    valid: bool
    inlier_count: int
    reference_count: int
    median_correlation: float
    reason: str = ""

    @property
    def median_score(self) -> float:
        """Compatibility alias for the median correlation."""

        return self.median_correlation

    @classmethod
    def invalid(
        cls,
        frame_index: int,
        reference_count: int,
        reason: str,
    ) -> GroundTranslation:
        return cls(
            frame_index=frame_index,
            dx=float("nan"),
            dy=float("nan"),
            valid=False,
            inlier_count=0,
            reference_count=reference_count,
            median_correlation=float("nan"),
            reason=reason,
        )


def coerce_rois(rois: Sequence[ROIInput]) -> tuple[ROI, ...]:
    """Validate and normalize an ROI sequence."""

    normalized = tuple(roi if isinstance(roi, ROI) else ROI(*roi) for roi in rois)
    if not normalized:
        raise ValueError("At least one ground reference ROI is required")
    return normalized


def _gray(frame: NDArray[np.generic]) -> NDArray[np.generic]:
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError("A frame must have two or three dimensions")
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError("A color frame must have three or four channels")


def _validate_rois(
    rois: Sequence[ROI],
    frame_shape: tuple[int, int],
) -> None:
    height, width = frame_shape
    for roi in rois:
        if roi.x + roi.width > width or roi.y + roi.height > height:
            raise ValueError(f"ROI {roi.xywh} lies outside the reference frame")


def _quadratic_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1.0e-12:
        return 0.0
    value = 0.5 * (left - right) / denominator
    return float(np.clip(value, -1.0, 1.0))


def _subpixel_peak(
    response: NDArray[np.floating],
    location: tuple[int, int],
) -> tuple[float, float]:
    column, row = location
    dx = 0.0
    dy = 0.0
    if 0 < column < response.shape[1] - 1:
        dx = _quadratic_offset(
            float(response[row, column - 1]),
            float(response[row, column]),
            float(response[row, column + 1]),
        )
    if 0 < row < response.shape[0] - 1:
        dy = _quadratic_offset(
            float(response[row - 1, column]),
            float(response[row, column]),
            float(response[row + 1, column]),
        )
    return dx, dy


def _match_reference(
    reference: NDArray[np.generic],
    current: NDArray[np.generic],
    roi: ROI,
    search_radius_px: tuple[int, int],
) -> tuple[float, float, float] | None:
    template = reference[
        roi.y : roi.y + roi.height,
        roi.x : roi.x + roi.width,
    ]
    frame_height, frame_width = current.shape[:2]
    radius_x, radius_y = search_radius_px
    x0 = max(0, roi.x - radius_x)
    y0 = max(0, roi.y - radius_y)
    x1 = min(frame_width, roi.x + roi.width + radius_x)
    y1 = min(frame_height, roi.y + roi.height + radius_y)
    search = current[y0:y1, x0:x1]
    if search.shape[0] < roi.height or search.shape[1] < roi.width:
        return None
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    if not np.isfinite(score):
        return None
    subpixel_x, subpixel_y = _subpixel_peak(response, location)
    matched_x = x0 + location[0] + subpixel_x
    matched_y = y0 + location[1] + subpixel_y
    return float(matched_x - roi.x), float(matched_y - roi.y), float(score)


def estimate_ground_translation(
    reference_frame: NDArray[np.generic],
    frame: NDArray[np.generic] | None,
    reference_rois: Sequence[ROIInput],
    *,
    frame_index: int,
    config: GroundTrackingConfig | None = None,
) -> GroundTranslation:
    """Estimate one translation and mark unsupported frames invalid."""

    settings = config or GroundTrackingConfig()
    rois = coerce_rois(reference_rois)
    if settings.min_inliers > len(rois):
        raise ValueError("min_inliers exceeds the number of reference ROIs")
    reference = _gray(np.asarray(reference_frame))
    _validate_rois(rois, reference.shape[:2])
    if frame is None:
        return GroundTranslation.invalid(frame_index, len(rois), "frame_missing")
    current = _gray(np.asarray(frame))
    if current.shape[:2] != reference.shape[:2]:
        raise ValueError("Current and reference frames must have the same shape")

    candidates = []
    for roi in rois:
        match = _match_reference(
            reference,
            current,
            roi,
            settings.search_radius_px,
        )
        if match is not None and match[2] >= settings.min_correlation:
            candidates.append(match)
    if len(candidates) < settings.min_inliers:
        return GroundTranslation.invalid(
            frame_index,
            len(rois),
            "insufficient_matches",
        )

    values = np.asarray(candidates, dtype=np.float64)
    center = np.median(values[:, :2], axis=0)
    residuals = np.linalg.norm(values[:, :2] - center, axis=1)
    inliers = residuals <= settings.consensus_threshold_px
    if int(inliers.sum()) < settings.min_inliers:
        return GroundTranslation.invalid(
            frame_index,
            len(rois),
            "insufficient_consensus",
        )

    consensus = np.median(values[inliers, :2], axis=0)
    return GroundTranslation(
        frame_index=frame_index,
        dx=float(consensus[0]),
        dy=float(consensus[1]),
        valid=True,
        inlier_count=int(inliers.sum()),
        reference_count=len(rois),
        median_correlation=float(np.median(values[inliers, 2])),
    )


def track_ground_translations(
    reference_frame: NDArray[np.generic],
    frames: Iterable[tuple[int, NDArray[np.generic] | None]],
    reference_rois: Sequence[ROIInput],
    *,
    config: GroundTrackingConfig | None = None,
) -> list[GroundTranslation]:
    """Track all supplied frames without filling invalid estimates."""

    return [
        estimate_ground_translation(
            reference_frame,
            frame,
            reference_rois,
            frame_index=frame_index,
            config=config,
        )
        for frame_index, frame in frames
    ]
