"""Hash input files without publishing machine-specific paths."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: str | Path, base_directory: str | Path) -> str:
    """Return a portable relative path without storing an absolute path."""

    source = Path(path).resolve()
    base = Path(base_directory).resolve()
    try:
        value = Path(os.path.relpath(source, base))
    except ValueError:
        return source.name
    if value.is_absolute():
        raise ValueError("Could not produce a relative input path")
    return value.as_posix()


def input_record(
    path: str | Path,
    *,
    label: str,
    base_directory: str | Path,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    clean_label = str(label).strip()
    if not clean_label:
        raise ValueError("Input label must not be empty")
    return {
        "label": clean_label,
        "path": relative_path(source, base_directory),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def build_provenance(
    inputs: Mapping[str, str | Path],
    *,
    base_directory: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = [
        input_record(path, label=label, base_directory=base_directory)
        for label, path in sorted(inputs.items())
    ]
    result: dict[str, Any] = {
        "schema_version": "light_pole_tracking_provenance_v1",
        "inputs": records,
    }
    if metadata:
        result["metadata"] = dict(metadata)
    return result


def write_provenance(
    path: str | Path,
    inputs: Mapping[str, str | Path],
    *,
    base_directory: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a source manifest containing only relative paths."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_provenance(
        inputs,
        base_directory=base_directory,
        metadata=metadata,
    )
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=output.name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return output


def write_manifest(
    path: str | Path,
    *,
    config: Any,
    inputs: Mapping[str, str | Path | None],
    artifacts: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> Path:
    """Write the pipeline manifest without machine-specific absolute paths."""

    base = Path(config.root)
    present_inputs = {
        label: value for label, value in inputs.items() if value is not None
    }
    payload = build_provenance(present_inputs, base_directory=base)
    payload["configuration"] = input_record(
        config.path,
        label="configuration",
        base_directory=base,
    )
    payload["artifacts"] = [
        {"label": label, "path": label, "sha256": digest}
        for label, digest in sorted(artifacts.items())
    ]
    payload["metadata"] = dict(metadata)
    payload["status"] = "completed_with_observation_gaps_preserved"
    payload["parameters"] = {
        "target_mode": config.target.mode,
        "target_corrections_enabled": (
            getattr(config.target, "corrections_path", None) is not None
        ),
        "target_anchor_target_xy": (
            list(config.target.anchor_target_xy)
            if getattr(config.target, "anchor_target_xy", None) is not None
            else None
        ),
        "target_roi_xyxy": list(config.target.roi_xyxy),
        "target_sam2_segment_overlap_policy": getattr(
            config.target, "segment_overlap_policy", "error"
        ),
        "target_sam2_segments": [
            {
                "id": segment.id,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "anchor_frame": segment.anchor_frame,
                "direction": segment.direction,
                "positive_points_xy": [
                    list(point) for point in segment.positive_points_xy
                ],
                "negative_points_xy": [
                    list(point) for point in segment.negative_points_xy
                ],
            }
            for segment in getattr(config.target, "segments", ())
        ],
        "ground_reference_frame": config.ground.reference_frame,
        "ground_rois_xywh": [list(roi) for roi in config.ground.rois_xywh],
        "ground_search_radius_px": list(config.ground.search_radius_px),
        "ground_min_correlation": config.ground.min_correlation,
        "ground_consensus_threshold_px": config.ground.consensus_threshold_px,
        "ground_min_inliers": config.ground.min_inliers,
        "ground_corrections_enabled": (
            getattr(config.ground, "corrections_path", None) is not None
        ),
        "zero_frame_start": config.relative.zero_frame_start,
        "zero_frame_end": config.relative.zero_frame_end,
    }
    metric = getattr(config, "metric", None)
    if metric is not None:
        payload["parameters"]["metric"] = {
            "calibration_path": metric.calibration_path.name,
            "calibration_sha256": metric.calibration_sha256,
            "validation_status": metric.validation_status,
            "physical_use_allowed": False,
            "coordinate_frame_id": metric.coordinate_frame_id,
            "motion_subspace": metric.motion_subspace,
            "ground_reference_frame": metric.ground_reference_frame,
            "image_model": metric.image_model,
            "image_width_px": metric.image_width_px,
            "image_height_px": metric.image_height_px,
            "remap_sha256": metric.remap_sha256,
            "material_point_world_xyz_m": list(
                metric.material_point_world_xyz_m
            ),
            "image_measurement_axis_xy": list(metric.image_measurement_axis_xy),
            "calibration_reference": {
                "description": metric.calibration_reference.description,
                "value_m": metric.calibration_reference.value_m,
                "uncertainty_m": metric.calibration_reference.uncertainty_m,
                "source": metric.calibration_reference.source,
                "relationship": metric.calibration_reference.relationship,
            },
        }
    payload["software"] = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    limitations = [
        "Ground regions are a proxy for the hidden pole-ground contact.",
    ]
    if config.target.mode == "sam2":
        limitations.append("SAM 2 masks require visual review.")
    has_manual_corrections = (
        getattr(config.target, "corrections_path", None) is not None
        or getattr(config.ground, "corrections_path", None) is not None
    )
    if has_manual_corrections:
        limitations.append(
            "The final series includes reviewed coordinate corrections."
        )
    if config.target.mode == "sam2" and config.target.corrections_path is not None:
        limitations.append("The final target track is not an unmodified SAM 2 result.")
    target_summary = metadata.get("target_corrections")
    if (
        isinstance(target_summary, Mapping)
        and target_summary.get("correction_count")
        == metadata.get("target_processed_frame_count")
    ):
        limitations.append(
            "Every target frame is controlled by the selected reviewed "
            "candidate; SAM 2 outputs are comparison evidence only."
        )
    ground_summary = metadata.get("ground_corrections")
    if (
        isinstance(ground_summary, Mapping)
        and ground_summary.get("correction_count")
        == metadata.get("ground_processed_frame_count")
    ):
        limitations.append(
            "Every ground frame is controlled by the selected reviewed candidate; "
            "the registration model is comparison evidence only."
        )
    if getattr(config.target, "anchor_target_xy", None) is not None:
        limitations.append(
            "A constant image offset aligns the SAM 2 mask centroid to the "
            "reviewed material point at the anchor frame."
        )
    if metric is None:
        limitations.append("No metric camera geometry was supplied for this run.")
    else:
        limitations.extend(
            [
                "Metric geometry is supplied, not independently estimated or "
                "validated by this run.",
                "One-camera metric output is a directional projection, not a "
                "two-dimensional trajectory.",
            ]
        )
    payload["limitations"] = limitations
    if metric is not None:
        payload["status"] = (
            "completed_with_conditional_metric_projection_and_manual_corrections"
            if has_manual_corrections
            else "completed_with_conditional_metric_projection"
        )
    elif has_manual_corrections:
        payload["status"] = "completed_with_manual_corrections"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=output.name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return output
