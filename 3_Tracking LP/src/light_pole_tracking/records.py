"""CSV records used by the tracking pipeline."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class TargetRecord:
    frame: int
    time_s: float
    x_px: float
    y_px: float
    valid: bool
    source: str


@dataclass(frozen=True)
class GroundRecord:
    frame: int
    time_s: float
    dx_px: float
    dy_px: float
    valid: bool
    inlier_count: int
    median_correlation: float


@dataclass(frozen=True)
class RelativeRecord:
    frame: int
    time_s: float
    relative_x_px: float
    relative_y_px: float
    valid: bool


Record = TypeVar("Record", TargetRecord, GroundRecord, RelativeRecord)
TARGET_FIELDS = ("frame", "time_s", "x_px", "y_px", "valid", "source")
GROUND_FIELDS = (
    "frame",
    "time_s",
    "dx_px",
    "dy_px",
    "valid",
    "inlier_count",
    "median_correlation",
)
RELATIVE_FIELDS = (
    "frame",
    "time_s",
    "relative_x_px",
    "relative_y_px",
    "valid",
)


def parse_bool(value: Any, name: str = "valid") -> bool:
    """Parse a strict CSV boolean."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"{name} must be true or false")


def _float(value: Any, name: str, *, finite: bool) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if finite and not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _frame(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("frame must be an integer") from error
    if str(value).strip() not in {str(result), f"{result}.0"} or result < 0:
        raise ValueError("frame must be a non-negative integer")
    return result


def _rows(path: str | Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return list(reader)


def _validate_frames(records: Iterable[Any]) -> None:
    frames = [record.frame for record in records]
    if len(frames) != len(set(frames)):
        raise ValueError("frame values must be unique")
    if frames != sorted(frames):
        raise ValueError("records must be sorted by frame")


def read_target_csv(path: str | Path) -> list[TargetRecord]:
    records: list[TargetRecord] = []
    for row in _rows(path, TARGET_FIELDS):
        valid = parse_bool(row["valid"])
        x = _float(row["x_px"], "x_px", finite=valid)
        y = _float(row["y_px"], "y_px", finite=valid)
        if not valid:
            x = y = math.nan
        source = row["source"].strip()
        if not source:
            raise ValueError("source must not be empty")
        records.append(
            TargetRecord(
                _frame(row["frame"]),
                _float(row["time_s"], "time_s", finite=True),
                x,
                y,
                valid,
                source,
            )
        )
    _validate_frames(records)
    return records


def read_ground_csv(path: str | Path) -> list[GroundRecord]:
    records: list[GroundRecord] = []
    for row in _rows(path, GROUND_FIELDS):
        valid = parse_bool(row["valid"])
        dx = _float(row["dx_px"], "dx_px", finite=valid)
        dy = _float(row["dy_px"], "dy_px", finite=valid)
        correlation = _float(
            row["median_correlation"], "median_correlation", finite=valid
        )
        if not valid:
            dx = dy = correlation = math.nan
        records.append(
            GroundRecord(
                _frame(row["frame"]),
                _float(row["time_s"], "time_s", finite=True),
                dx,
                dy,
                valid,
                int(row["inlier_count"]),
                correlation,
            )
        )
    _validate_frames(records)
    return records


def read_relative_csv(path: str | Path) -> list[RelativeRecord]:
    records: list[RelativeRecord] = []
    for row in _rows(path, RELATIVE_FIELDS):
        valid = parse_bool(row["valid"])
        x = _float(row["relative_x_px"], "relative_x_px", finite=valid)
        y = _float(row["relative_y_px"], "relative_y_px", finite=valid)
        if not valid:
            x = y = math.nan
        records.append(
            RelativeRecord(
                _frame(row["frame"]),
                _float(row["time_s"], "time_s", finite=True),
                x,
                y,
                valid,
            )
        )
    _validate_frames(records)
    return records


def _write(path: str | Path, fields: tuple[str, ...], records: Iterable[Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            values: Mapping[str, Any]
            values = record if isinstance(record, Mapping) else asdict(record)
            writer.writerow({field: values[field] for field in fields})
    return output


def write_target_csv(path: str | Path, records: Iterable[TargetRecord]) -> Path:
    return _write(path, TARGET_FIELDS, records)


def write_ground_csv(path: str | Path, records: Iterable[GroundRecord]) -> Path:
    return _write(path, GROUND_FIELDS, records)


def write_relative_csv(path: str | Path, records: Iterable[RelativeRecord]) -> Path:
    return _write(path, RELATIVE_FIELDS, records)


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a pipeline CSV and convert standard fields to native types."""

    integer_fields = {"frame", "inlier_count", "reference_count", "area_px"}
    float_fields = {
        "time_s",
        "x_px",
        "y_px",
        "dx_px",
        "dy_px",
        "median_correlation",
        "relative_x_px",
        "relative_y_px",
    }
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in integer_fields.intersection(row):
                row[field] = int(row[field])
            for field in float_fields.intersection(row):
                row[field] = float(row[field])
            if "valid" in row:
                row["valid"] = parse_bool(row["valid"])
            rows.append(row)
    frames = [int(row["frame"]) for row in rows if "frame" in row]
    if len(frames) != len(set(frames)) or frames != sorted(frames):
        raise ValueError("CSV frames must be unique and sorted")
    return rows


def write_csv_records(path: str | Path, records: Iterable[Any]) -> Path:
    """Write mapping or dataclass rows without changing their schema."""

    values = list(records)
    if not values:
        raise ValueError("Cannot infer a CSV schema from an empty record set")
    first = values[0]
    if isinstance(first, Mapping):
        fields = tuple(first.keys())
    elif is_dataclass(first):
        fields = tuple(asdict(first).keys())
    else:
        raise TypeError("Records must be mappings or dataclass instances")
    return _write(path, fields, values)
