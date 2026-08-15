"""Convert segmentation masks into a luminaire-centre track."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskDetection:
    """Largest connected component in one segmentation mask."""

    x_px: float
    y_px: float
    area_px: int
    bbox_xywh: tuple[int, int, int, int]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest foreground component of a binary mask."""

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array")
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros(binary.shape, dtype=bool)
    label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    return labels == label


def detect_mask(mask: np.ndarray, roi_xyxy: Iterable[int]) -> MaskDetection | None:
    """Measure the largest mask component in full-frame coordinates."""

    component = largest_component(mask)
    x0, y0, x1, y1 = (int(value) for value in roi_xyxy)
    if component.shape != (y1 - y0, x1 - x0):
        raise ValueError("Mask dimensions do not match the configured ROI")
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        return None
    bx, by, width, height = cv2.boundingRect(component.astype(np.uint8))
    return MaskDetection(
        x_px=float(xs.mean() + x0),
        y_px=float(ys.mean() + y0),
        area_px=int(xs.size),
        bbox_xywh=(bx + x0, by + y0, width, height),
    )


def detections_from_masks(
    frames: Iterable[int],
    masks: np.ndarray,
    roi_xyxy: Iterable[int],
    fps: float,
    minimum_area_px: int = 1,
    maximum_step_px: float | None = None,
) -> list[dict[str, object]]:
    """Build track rows while keeping rejected or empty frames explicit."""

    frame_numbers = [int(frame) for frame in frames]
    mask_array = np.asarray(masks, dtype=bool)
    if mask_array.ndim != 3 or mask_array.shape[0] != len(frame_numbers):
        raise ValueError("Masks must have shape (frames, height, width)")
    if fps <= 0:
        raise ValueError("Frame rate must be positive")
    if minimum_area_px < 1:
        raise ValueError("Minimum mask area must be at least one pixel")
    if maximum_step_px is not None and (
        not np.isfinite(maximum_step_px) or maximum_step_px < 0
    ):
        raise ValueError("Maximum centroid step must be finite and non-negative")

    rows: list[dict[str, object]] = []
    previous: tuple[float, float] | None = None
    for frame, mask in zip(frame_numbers, mask_array, strict=True):
        detection = detect_mask(mask, roi_xyxy)
        valid = detection is not None and detection.area_px >= minimum_area_px
        reason = ""
        if detection is None:
            reason = "empty_mask"
        elif detection.area_px < minimum_area_px:
            reason = "area_below_minimum"
        elif maximum_step_px is not None and previous is not None:
            step = float(
                np.hypot(
                    detection.x_px - previous[0],
                    detection.y_px - previous[1],
                )
            )
            if step > maximum_step_px:
                valid = False
                reason = "centroid_step_above_limit"

        row: dict[str, object] = {
            "frame": frame,
            "time_s": frame / fps,
            "x_px": (
                detection.x_px if valid and detection is not None else float("nan")
            ),
            "y_px": (
                detection.y_px if valid and detection is not None else float("nan")
            ),
            "valid": bool(valid),
            "source": "sam2",
            "area_px": detection.area_px if detection is not None else 0,
            "reason": reason,
        }
        rows.append(row)
        if valid and detection is not None:
            previous = (detection.x_px, detection.y_px)
    return rows
