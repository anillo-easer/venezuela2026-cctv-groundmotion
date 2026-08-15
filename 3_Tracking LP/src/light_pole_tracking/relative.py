"""Compute luminaire displacement relative to local ground translation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from .records import GroundRecord, RelativeRecord, TargetRecord


def _value(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    raise ValueError(f"Record is missing any of: {', '.join(names)}")


def _target(item: Any) -> TargetRecord:
    frame = int(_value(item, "frame", "frame_index"))
    time_s = float(_value(item, "time_s"))
    valid = bool(_value(item, "valid"))
    x = float(_value(item, "x_px", "x")) if valid else math.nan
    y = float(_value(item, "y_px", "y")) if valid else math.nan
    source = str(_value(item, "source")) if (
        isinstance(item, Mapping) and "source" in item
    ) or hasattr(item, "source") else "target"
    return TargetRecord(frame, time_s, x, y, valid, source)


def _ground(item: Any) -> GroundRecord:
    frame = int(_value(item, "frame", "frame_index"))
    time_s = float(_value(item, "time_s"))
    valid = bool(_value(item, "valid"))
    dx = float(_value(item, "dx_px", "dx")) if valid else math.nan
    dy = float(_value(item, "dy_px", "dy")) if valid else math.nan
    try:
        inliers = int(_value(item, "inlier_count"))
    except ValueError:
        inliers = 0
    try:
        correlation = float(_value(item, "median_correlation", "median_score"))
    except ValueError:
        correlation = math.nan
    return GroundRecord(frame, time_s, dx, dy, valid, inliers, correlation)


def compute_relative_displacement(
    target_records: Iterable[Any],
    ground_records: Iterable[Any],
    *,
    zero_frame_start: int,
    zero_frame_end: int,
) -> list[RelativeRecord]:
    """Join by frame, subtract ground translation, and apply a median zero."""

    if zero_frame_end < zero_frame_start:
        raise ValueError("zero_frame_end must not precede zero_frame_start")
    targets: dict[int, TargetRecord] = {}
    for item in target_records:
        record = _target(item)
        if record.frame in targets:
            raise ValueError(f"Duplicate target frame: {record.frame}")
        targets[record.frame] = record
    grounds: dict[int, GroundRecord] = {}
    for item in ground_records:
        record = _ground(item)
        if record.frame in grounds:
            raise ValueError(f"Duplicate ground frame: {record.frame}")
        grounds[record.frame] = record
    frames = sorted(set(targets) | set(grounds))
    stabilized: dict[int, tuple[float, float, float] | None] = {}
    for frame in frames:
        target = targets.get(frame)
        ground = grounds.get(frame)
        if target is None or ground is None or not target.valid or not ground.valid:
            stabilized[frame] = None
            continue
        if not math.isclose(target.time_s, ground.time_s, abs_tol=1e-9):
            raise ValueError(f"Target and ground times differ at frame {frame}")
        stabilized[frame] = (
            target.time_s,
            target.x_px - ground.dx_px,
            target.y_px - ground.dy_px,
        )
    zero_values = [
        value
        for frame, value in stabilized.items()
        if zero_frame_start <= frame <= zero_frame_end and value is not None
    ]
    if not zero_values:
        raise ValueError("The zero interval contains no jointly valid samples")
    zero_x = float(np.median([value[1] for value in zero_values]))
    zero_y = float(np.median([value[2] for value in zero_values]))
    result: list[RelativeRecord] = []
    for frame in frames:
        value = stabilized[frame]
        if value is None:
            time = (
                targets[frame].time_s if frame in targets else grounds[frame].time_s
            )
            result.append(RelativeRecord(frame, time, math.nan, math.nan, False))
        else:
            result.append(
                RelativeRecord(
                    frame,
                    value[0],
                    value[1] - zero_x,
                    value[2] - zero_y,
                    True,
                )
            )
    return result


def compute_relative_motion(
    target_records: Iterable[Any],
    ground_records: Iterable[Any],
    *,
    zero_frame_start: int,
    zero_frame_end: int,
) -> list[dict[str, Any]]:
    """Pipeline adapter returning CSV-ready mappings."""

    return [
        asdict(record)
        for record in compute_relative_displacement(
            target_records,
            ground_records,
            zero_frame_start=zero_frame_start,
            zero_frame_end=zero_frame_end,
        )
    ]
