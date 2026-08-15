"""Strict manual corrections for target and local-ground tracks."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .ground import GroundTranslation
from .records import parse_bool


@dataclass(frozen=True)
class TargetCorrection:
    """One absolute target-coordinate override."""

    frame: int
    x_px: float | None
    y_px: float | None
    valid: bool
    note: str = ""

    def __post_init__(self) -> None:
        _validate_correction_values(
            self.frame,
            self.x_px,
            self.y_px,
            self.valid,
            self.note,
            names=("x_px", "y_px"),
        )


@dataclass(frozen=True)
class GroundCorrection:
    """One absolute local-ground translation override."""

    frame: int
    dx_px: float | None
    dy_px: float | None
    valid: bool
    note: str = ""

    def __post_init__(self) -> None:
        _validate_correction_values(
            self.frame,
            self.dx_px,
            self.dy_px,
            self.valid,
            self.note,
            names=("dx_px", "dy_px"),
        )


Correction = TypeVar("Correction", TargetCorrection, GroundCorrection)


def _validate_correction_values(
    frame: int,
    first: float | None,
    second: float | None,
    valid: bool,
    note: str,
    *,
    names: tuple[str, str],
) -> None:
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError("frame must be a non-negative integer")
    if not isinstance(valid, bool):
        raise ValueError("valid must be a boolean")
    if not isinstance(note, str):
        raise ValueError("note must be text")
    if valid:
        if first is None or second is None:
            raise ValueError(f"Valid corrections require {names[0]} and {names[1]}")
        if not math.isfinite(float(first)) or not math.isfinite(float(second)):
            raise ValueError("Valid correction coordinates must be finite")
    elif first is not None or second is not None:
        raise ValueError("Invalid corrections must not contain coordinates")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _frame(raw: str, *, frame_count: int, row_number: int) -> int:
    text = raw.strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError(f"CSV row {row_number}: frame must be a non-negative integer")
    value = int(text)
    if value >= frame_count:
        raise ValueError(
            f"CSV row {row_number}: frame {value} is outside [0, {frame_count})"
        )
    return value


def _valid(raw: str, *, row_number: int) -> bool:
    if raw.strip() == "":
        raise ValueError(f"CSV row {row_number}: valid must not be blank")
    try:
        return parse_bool(raw, "valid")
    except ValueError as error:
        raise ValueError(f"CSV row {row_number}: {error}") from error


def _coordinates(
    row: Mapping[str, str | None],
    names: tuple[str, str],
    *,
    valid: bool,
    row_number: int,
) -> tuple[float | None, float | None]:
    raw = tuple((row.get(name) or "").strip() for name in names)
    if not valid:
        if any(raw):
            raise ValueError(
                f"CSV row {row_number}: invalid corrections require blank "
                f"{names[0]} and {names[1]}"
            )
        return None, None
    if not all(raw):
        raise ValueError(
            f"CSV row {row_number}: valid corrections require {names[0]} and {names[1]}"
        )
    try:
        values = float(raw[0]), float(raw[1])
    except ValueError as error:
        raise ValueError(
            f"CSV row {row_number}: correction coordinates must be numeric"
        ) from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"CSV row {row_number}: correction coordinates must be finite")
    return values


def _read_rows(
    path: str | Path,
    required_fields: tuple[str, ...],
) -> list[tuple[int, dict[str, str | None]]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        allowed = (required_fields, (*required_fields, "note"))
        if fields not in allowed:
            expected = ",".join(required_fields) + "[,note]"
            raise ValueError(
                f"{source} must have exactly these columns in order: {expected}"
            )
        result: list[tuple[int, dict[str, str | None]]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CSV row {row_number}: too many values")
            result.append((row_number, row))
    if not result:
        raise ValueError(f"{source} must contain at least one correction row")
    return result


def _check_unique_frames(corrections: Iterable[Correction]) -> list[Correction]:
    values = list(corrections)
    seen: set[int] = set()
    for correction in values:
        if correction.frame in seen:
            raise ValueError(f"Correction frames must be unique: {correction.frame}")
        seen.add(correction.frame)
    return values


def load_target_corrections(
    path: str | Path,
    frame_count: int,
    width: int,
    height: int,
    roi_xyxy: Sequence[int],
) -> list[TargetCorrection]:
    """Load target overrides and validate them against the active image and ROI."""

    frame_count = _positive_integer(frame_count, "frame_count")
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    if len(roi_xyxy) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in roi_xyxy
    ):
        raise ValueError("roi_xyxy must contain four integers")
    x0, y0, x1, y1 = roi_xyxy
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("roi_xyxy must lie inside the active image")

    corrections: list[TargetCorrection] = []
    for row_number, row in _read_rows(path, ("frame", "x_px", "y_px", "valid")):
        frame = _frame(
            row["frame"] or "",
            frame_count=frame_count,
            row_number=row_number,
        )
        valid = _valid(row["valid"] or "", row_number=row_number)
        x_px, y_px = _coordinates(
            row, ("x_px", "y_px"), valid=valid, row_number=row_number
        )
        if valid:
            assert x_px is not None and y_px is not None
            if not (0.0 <= x_px < width and 0.0 <= y_px < height):
                raise ValueError(
                    f"CSV row {row_number}: target coordinate lies outside "
                    "the active image"
                )
            if not (x0 <= x_px < x1 and y0 <= y_px < y1):
                raise ValueError(
                    f"CSV row {row_number}: target coordinate lies outside the ROI"
                )
        corrections.append(
            TargetCorrection(
                frame=frame,
                x_px=x_px,
                y_px=y_px,
                valid=valid,
                note=(row.get("note") or "").strip(),
            )
        )
    return _check_unique_frames(corrections)


def load_ground_corrections(
    path: str | Path,
    frame_count: int,
) -> list[GroundCorrection]:
    """Load absolute local-ground translation overrides."""

    frame_count = _positive_integer(frame_count, "frame_count")
    corrections: list[GroundCorrection] = []
    for row_number, row in _read_rows(path, ("frame", "dx_px", "dy_px", "valid")):
        frame = _frame(
            row["frame"] or "",
            frame_count=frame_count,
            row_number=row_number,
        )
        valid = _valid(row["valid"] or "", row_number=row_number)
        dx_px, dy_px = _coordinates(
            row, ("dx_px", "dy_px"), valid=valid, row_number=row_number
        )
        corrections.append(
            GroundCorrection(
                frame=frame,
                dx_px=dx_px,
                dy_px=dy_px,
                valid=valid,
                note=(row.get("note") or "").strip(),
            )
        )
    return _check_unique_frames(corrections)


def _mapping_frame(row: Mapping[str, Any]) -> int:
    try:
        value = row["frame"]
    except KeyError as error:
        raise ValueError("Every target row must contain frame") from error
    if isinstance(value, bool):
        raise ValueError("Target row frames must be non-negative integers")
    try:
        frame = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Target row frames must be non-negative integers") from error
    if frame < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError("Target row frames must be non-negative integers")
    return frame


def _finite_pair(row: Mapping[str, Any], first: str, second: str) -> bool:
    try:
        return math.isfinite(float(row[first])) and math.isfinite(float(row[second]))
    except (KeyError, TypeError, ValueError):
        return False


def _target_audit(
    original: Mapping[str, Any],
    final: Mapping[str, Any],
    correction: TargetCorrection,
) -> dict[str, Any]:
    original_valid = parse_bool(original.get("valid", False))
    final_valid = bool(final["valid"])
    comparable = (
        original_valid
        and final_valid
        and _finite_pair(original, "x_px", "y_px")
        and _finite_pair(final, "x_px", "y_px")
    )
    delta_x = float(final["x_px"]) - float(original["x_px"]) if comparable else None
    delta_y = float(final["y_px"]) - float(original["y_px"]) if comparable else None
    return {
        "frame": correction.frame,
        "note": correction.note,
        "original_x_px": original.get("x_px"),
        "original_y_px": original.get("y_px"),
        "original_valid": original_valid,
        "original_source": original.get("source", ""),
        "original_reason": original.get("reason", ""),
        "manual_x_px": correction.x_px,
        "manual_y_px": correction.y_px,
        "manual_valid": correction.valid,
        "final_x_px": final["x_px"],
        "final_y_px": final["y_px"],
        "final_valid": final_valid,
        "final_source": final["source"],
        "final_reason": final["reason"],
        "delta_x_px": delta_x,
        "delta_y_px": delta_y,
        "delta_distance_px": (
            math.hypot(delta_x, delta_y)
            if delta_x is not None and delta_y is not None
            else None
        ),
    }


def _summary(
    correction_count: int,
    valid_count: int,
    audit_rows: Iterable[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    distances = [
        float(row["delta_distance_px"])
        for row in audit_rows
        if row.get("delta_distance_px") is not None
    ]
    return {
        "correction_count": correction_count,
        "coordinate_override_count": valid_count,
        "invalidation_count": correction_count - valid_count,
        "max_delta_px": max(distances, default=None),
    }


def apply_target_corrections(
    rows: Iterable[Mapping[str, Any]],
    corrections: Iterable[TargetCorrection],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int | float | None],
]:
    """Apply sparse exact-frame target overrides without changing other rows."""

    originals = list(rows)
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in originals:
        frame = _mapping_frame(row)
        if frame in indexed:
            raise ValueError(f"Target row frames must be unique: {frame}")
        indexed[frame] = row
    values = _check_unique_frames(corrections)
    missing = [
        correction.frame for correction in values if correction.frame not in indexed
    ]
    if missing:
        raise ValueError(f"Correction frames are absent from target rows: {missing}")
    correction_by_frame = {correction.frame: correction for correction in values}

    final_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for original in originals:
        final = dict(original)
        correction = correction_by_frame.get(_mapping_frame(original))
        if correction is not None:
            final.update(
                {
                    "x_px": correction.x_px if correction.valid else math.nan,
                    "y_px": correction.y_px if correction.valid else math.nan,
                    "valid": correction.valid,
                    "source": "manual_correction",
                    "reason": (
                        "manual_coordinate_override"
                        if correction.valid
                        else "manual_invalidation"
                    ),
                }
            )
            audit_rows.append(_target_audit(original, final, correction))
        final_rows.append(final)
    return (
        final_rows,
        audit_rows,
        _summary(
            len(values),
            sum(correction.valid for correction in values),
            audit_rows,
        ),
    )


def align_target_to_anchor(
    rows: Iterable[Mapping[str, Any]],
    *,
    anchor_frame: int,
    anchor_target_xy: tuple[float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Translate valid target coordinates so one frame matches a known point.

    The same two-dimensional offset is applied to every valid row with finite
    coordinates. This preserves all frame-to-frame target increments while
    setting the absolute image position at ``anchor_frame``.
    """

    if (
        isinstance(anchor_frame, bool)
        or not isinstance(anchor_frame, int)
        or anchor_frame < 0
    ):
        raise ValueError("anchor_frame must be a non-negative integer")
    if not isinstance(anchor_target_xy, tuple) or len(anchor_target_xy) != 2:
        raise ValueError("anchor_target_xy must contain two finite coordinates")
    try:
        target_x, target_y = (float(value) for value in anchor_target_xy)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "anchor_target_xy must contain two finite coordinates"
        ) from error
    if not math.isfinite(target_x) or not math.isfinite(target_y):
        raise ValueError("anchor_target_xy must contain two finite coordinates")

    originals = list(rows)
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in originals:
        frame = _mapping_frame(row)
        if frame in indexed:
            raise ValueError(f"Target row frames must be unique: {frame}")
        indexed[frame] = row

    if anchor_frame not in indexed:
        raise ValueError(f"anchor_frame {anchor_frame} is absent from target rows")
    anchor = indexed[anchor_frame]
    try:
        anchor_valid = parse_bool(anchor["valid"])
    except KeyError as error:
        raise ValueError("The anchor target row must contain valid") from error
    if not anchor_valid or not _finite_pair(anchor, "x_px", "y_px"):
        raise ValueError("The anchor target row must be valid with finite coordinates")

    original_x = float(anchor["x_px"])
    original_y = float(anchor["y_px"])
    offset_x = target_x - original_x
    offset_y = target_y - original_y

    aligned: list[dict[str, Any]] = []
    aligned_valid_count = 0
    for original in originals:
        final = dict(original)
        try:
            valid = parse_bool(original["valid"])
        except KeyError as error:
            raise ValueError("Every target row must contain valid") from error
        if valid and _finite_pair(original, "x_px", "y_px"):
            final["x_px"] = float(original["x_px"]) + offset_x
            final["y_px"] = float(original["y_px"]) + offset_y
            aligned_valid_count += 1
        aligned.append(final)

    summary = {
        "anchor_frame": anchor_frame,
        "original_anchor_xy": (original_x, original_y),
        "target_anchor_xy": (target_x, target_y),
        "offset_xy": (offset_x, offset_y),
        "offset_distance_px": math.hypot(offset_x, offset_y),
        "aligned_valid_count": aligned_valid_count,
    }
    return aligned, summary


def _ground_audit(
    original: GroundTranslation,
    final: GroundTranslation,
    correction: GroundCorrection,
) -> dict[str, Any]:
    comparable = (
        original.valid
        and final.valid
        and math.isfinite(original.dx)
        and math.isfinite(original.dy)
        and math.isfinite(final.dx)
        and math.isfinite(final.dy)
    )
    delta_x = final.dx - original.dx if comparable else None
    delta_y = final.dy - original.dy if comparable else None
    return {
        "frame": correction.frame,
        "note": correction.note,
        "original_dx_px": original.dx,
        "original_dy_px": original.dy,
        "original_valid": original.valid,
        "original_inlier_count": original.inlier_count,
        "original_reference_count": original.reference_count,
        "original_median_correlation": original.median_correlation,
        "original_reason": original.reason,
        "manual_dx_px": correction.dx_px,
        "manual_dy_px": correction.dy_px,
        "manual_valid": correction.valid,
        "final_dx_px": final.dx,
        "final_dy_px": final.dy,
        "final_valid": final.valid,
        "final_source": "manual_correction",
        "final_inlier_count": final.inlier_count,
        "final_reference_count": final.reference_count,
        "final_median_correlation": final.median_correlation,
        "final_reason": final.reason,
        "delta_dx_px": delta_x,
        "delta_dy_px": delta_y,
        "delta_distance_px": (
            math.hypot(delta_x, delta_y)
            if delta_x is not None and delta_y is not None
            else None
        ),
    }


def apply_ground_corrections(
    rows: Iterable[GroundTranslation],
    corrections: Iterable[GroundCorrection],
) -> tuple[
    list[GroundTranslation],
    list[dict[str, Any]],
    dict[str, int | float | None],
]:
    """Apply sparse exact-frame overrides to local-ground translations."""

    originals = list(rows)
    indexed: dict[int, GroundTranslation] = {}
    for row in originals:
        if row.frame_index in indexed:
            raise ValueError(f"Ground row frames must be unique: {row.frame_index}")
        indexed[row.frame_index] = row
    values = _check_unique_frames(corrections)
    missing = [
        correction.frame for correction in values if correction.frame not in indexed
    ]
    if missing:
        raise ValueError(f"Correction frames are absent from ground rows: {missing}")
    correction_by_frame = {correction.frame: correction for correction in values}

    final_rows: list[GroundTranslation] = []
    audit_rows: list[dict[str, Any]] = []
    for original in originals:
        correction = correction_by_frame.get(original.frame_index)
        if correction is None:
            final = original
        elif correction.valid:
            assert correction.dx_px is not None and correction.dy_px is not None
            final = GroundTranslation(
                frame_index=original.frame_index,
                dx=float(correction.dx_px),
                dy=float(correction.dy_px),
                valid=True,
                inlier_count=0,
                reference_count=original.reference_count,
                median_correlation=math.nan,
                reason="manual_coordinate_override",
            )
            audit_rows.append(_ground_audit(original, final, correction))
        else:
            final = GroundTranslation.invalid(
                original.frame_index,
                original.reference_count,
                "manual_invalidation",
            )
            audit_rows.append(_ground_audit(original, final, correction))
        final_rows.append(final)
    return (
        final_rows,
        audit_rows,
        _summary(
            len(values),
            sum(correction.valid for correction in values),
            audit_rows,
        ),
    )
