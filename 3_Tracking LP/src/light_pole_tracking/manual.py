"""Normalize manually digitized luminaire tracks."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .records import TargetRecord, parse_bool


def _coordinate(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def normalize_manual_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    fps: float,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[TargetRecord]:
    """Return a contiguous track and insert explicit invalid rows for gaps."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    indexed: dict[int, TargetRecord] = {}
    for row in rows:
        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every manual row needs an integer frame") from error
        if frame < 0 or frame in indexed:
            raise ValueError("Manual frames must be unique and non-negative")
        x = _coordinate(row, "x_px")
        y = _coordinate(row, "y_px")
        finite = math.isfinite(x) and math.isfinite(y)
        valid = parse_bool(row["valid"]) if "valid" in row else finite
        if valid and not finite:
            raise ValueError(f"Frame {frame} is valid but has missing coordinates")
        if not valid:
            x = y = math.nan
        time_s = frame / fps
        source = str(row.get("source") or "manual").strip()
        indexed[frame] = TargetRecord(frame, time_s, x, y, valid, source)

    if not indexed and (start_frame is None or end_frame is None):
        raise ValueError("An empty manual track requires an explicit frame range")
    start = min(indexed) if start_frame is None else int(start_frame)
    end = max(indexed) if end_frame is None else int(end_frame)
    if start < 0 or end < start:
        raise ValueError("Invalid manual frame range")
    outside = [frame for frame in indexed if frame < start or frame > end]
    if outside:
        raise ValueError(f"Manual frames outside configured range: {outside}")
    return [
        indexed.get(
            frame,
            TargetRecord(
                frame,
                frame / fps,
                math.nan,
                math.nan,
                False,
                "manual_missing",
            ),
        )
        for frame in range(start, end + 1)
    ]


def normalize_manual_csv(
    path: str | Path,
    *,
    fps: float,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[TargetRecord]:
    """Read a manual CSV and preserve every missing frame explicitly."""

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = {"frame", "x_px", "y_px"}.difference(fields)
        if missing:
            raise ValueError(f"Manual CSV is missing columns: {sorted(missing)}")
        return normalize_manual_rows(
            list(reader),
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )


load_manual_track = normalize_manual_csv


def normalize_manual_track(
    path: str | Path | None,
    *,
    fps: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    """Pipeline adapter that expands a manual track to the full video."""

    if path is None:
        raise ValueError("A manual track path is required")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    records = normalize_manual_csv(
        path,
        fps=fps,
        start_frame=0,
        end_frame=frame_count - 1,
    )
    return [
        {
            **asdict(record),
            "area_px": 0,
            "reason": "" if record.valid else "not_observed",
        }
        for record in records
    ]
