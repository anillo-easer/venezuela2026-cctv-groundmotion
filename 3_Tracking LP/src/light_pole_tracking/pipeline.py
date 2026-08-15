"""End-to-end orchestration for one video view."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig
from .corrections import (
    align_target_to_anchor,
    apply_ground_corrections,
    apply_target_corrections,
    load_ground_corrections,
    load_target_corrections,
)
from .ground import GroundTrackingConfig, GroundTranslation, track_ground_translations
from .lens import LensRemap
from .manual import normalize_manual_track
from .metric import (
    MetricProjection,
    build_metric_projection,
    project_relative_motion,
    projection_to_dict,
)
from .provenance import sha256_file, write_manifest
from .qa_video import render_tracking_qa_video
from .records import read_csv_records, write_csv_records
from .relative import compute_relative_motion
from .sam2_tracker import track_luminaire_with_sam2
from .video import VideoRemap, iter_video_frames, read_video_metadata


@dataclass(frozen=True)
class PipelineResult:
    output_directory: Path
    target_track: Path
    ground_track: Path
    relative_motion: Path
    arrays: Path
    manifest: Path
    qa_video: Path | None
    projected_motion: Path | None = None
    metric_projection: Path | None = None
    target_track_sam2: Path | None = None
    target_track_sam2_aligned: Path | None = None
    target_corrections_applied: Path | None = None
    ground_track_model: Path | None = None
    ground_corrections_applied: Path | None = None


def _load_remap(path: Path | None) -> VideoRemap | None:
    if path is None:
        return None
    lens = LensRemap.load(path)
    map_x = lens.map_x.copy()
    map_y = lens.map_y.copy()
    map_x[~lens.valid_mask] = -1.0
    map_y[~lens.valid_mask] = -1.0
    return VideoRemap(map_x, map_y)


def _read_frame(path: Path, index: int, remap: VideoRemap | None) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise OSError(f"Could not decode reference frame {index}")
    return remap.apply(frame) if remap is not None else frame


def _write_ground_track(
    path: Path,
    translations: list[GroundTranslation],
    fps: float,
) -> None:
    fields = [
        "frame",
        "time_s",
        "dx_px",
        "dy_px",
        "valid",
        "inlier_count",
        "reference_count",
        "median_correlation",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in translations:
            writer.writerow(
                {
                    "frame": item.frame_index,
                    "time_s": item.frame_index / fps,
                    "dx_px": item.dx,
                    "dy_px": item.dy,
                    "valid": item.valid,
                    "inlier_count": item.inlier_count,
                    "reference_count": item.reference_count,
                    "median_correlation": item.median_correlation,
                    "reason": item.reason,
                }
            )


def _target_points(rows: list[dict[str, Any]]) -> dict[int, tuple[float, float] | None]:
    points: dict[int, tuple[float, float] | None] = {}
    for row in rows:
        valid = bool(row["valid"])
        points[int(row["frame"])] = (
            (float(row["x_px"]), float(row["y_px"])) if valid else None
        )
    return points


def _save_arrays(
    path: Path,
    target_rows: list[dict[str, Any]],
    ground_rows: list[dict[str, Any]],
    relative_rows: list[dict[str, Any]],
    projected_rows: list[dict[str, Any]] | None = None,
) -> None:
    frames = [int(row["frame"]) for row in relative_rows]
    target = {int(row["frame"]): row for row in target_rows}
    ground = {int(row["frame"]): row for row in ground_rows}

    def aligned(
        rows: dict[int, dict[str, Any]],
        key: str,
        dtype: Any,
        missing: Any,
    ) -> np.ndarray:
        return np.asarray(
            [rows.get(frame, {}).get(key, missing) for frame in frames],
            dtype=dtype,
        )

    def values(key: str, dtype: Any) -> np.ndarray:
        return np.asarray([row[key] for row in relative_rows], dtype=dtype)

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray("light_pole_tracking_v1"),
        "frame": values("frame", np.int64),
        "time_s": values("time_s", np.float64),
        "target_x_px": aligned(target, "x_px", np.float64, np.nan),
        "target_y_px": aligned(target, "y_px", np.float64, np.nan),
        "target_valid": aligned(target, "valid", bool, False),
        "ground_dx_px": aligned(ground, "dx_px", np.float64, np.nan),
        "ground_dy_px": aligned(ground, "dy_px", np.float64, np.nan),
        "ground_valid": aligned(ground, "valid", bool, False),
        "relative_x_px": values("relative_x_px", np.float64),
        "relative_y_px": values("relative_y_px", np.float64),
        "relative_valid": values("valid", bool),
    }
    if projected_rows is not None:
        arrays["projected_displacement_px"] = np.asarray(
            [row["projected_displacement_px"] for row in projected_rows],
            dtype=np.float64,
        )
        arrays["projected_displacement_m"] = np.asarray(
            [row["projected_displacement_m"] for row in projected_rows],
            dtype=np.float64,
        )
        arrays["projected_valid"] = np.asarray(
            [row["valid"] for row in projected_rows],
            dtype=bool,
        )
        arrays["metric_validation_status"] = np.asarray(
            projected_rows[0]["metric_validation_status"]
        )
        arrays["metric_physical_use_allowed"] = np.asarray(False)
    np.savez_compressed(path, **arrays)


def _metric_projection(config: PipelineConfig) -> MetricProjection:
    assert config.metric is not None
    return build_metric_projection(
        config.metric.material_point_world_xyz_m,
        config.metric.camera_matrix_px,
        config.metric.rotation_world_to_camera,
        config.metric.optical_center_world_xyz_m,
        config.metric.image_measurement_axis_xy,
    )


def _projected_rows(
    relative_rows: list[dict[str, Any]],
    projection: MetricProjection,
    validation_status: str,
) -> list[dict[str, Any]]:
    vectors = np.asarray(
        [
            (
                float(row["relative_x_px"]),
                float(row["relative_y_px"]),
            )
            if bool(row["valid"])
            else (np.nan, np.nan)
            for row in relative_rows
        ],
        dtype=np.float64,
    )
    pixels, metres = project_relative_motion(vectors, projection)
    return [
        {
            "frame": int(row["frame"]),
            "time_s": float(row["time_s"]),
            "projected_displacement_px": float(pixels[index]),
            "projected_displacement_m": float(metres[index]),
            "valid": bool(row["valid"]),
            "metric_validation_status": validation_status,
            "physical_use_allowed": False,
        }
        for index, row in enumerate(relative_rows)
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _zero_reference_pixel(
    relative_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    ground_rows: list[dict[str, Any]],
    *,
    zero_frame_start: int,
    zero_frame_end: int,
) -> np.ndarray:
    targets = {int(row["frame"]): row for row in target_rows}
    grounds = {int(row["frame"]): row for row in ground_rows}
    values = []
    for row in relative_rows:
        frame = int(row["frame"])
        target = targets.get(frame)
        ground = grounds.get(frame)
        if (
            zero_frame_start <= frame <= zero_frame_end
            and target is not None
            and ground is not None
            and bool(target["valid"])
            and bool(ground["valid"])
        ):
            values.append(
                (
                    float(target["x_px"]) - float(ground["dx_px"]),
                    float(target["y_px"]) - float(ground["dy_px"]),
                )
            )
    if not values:
        raise ValueError("The zero interval has no valid target samples")
    return np.median(np.asarray(values, dtype=np.float64), axis=0)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run target tracking, ground registration, relative motion, and QA."""

    config.validate_inputs()
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {config.output_directory}"
        )
    config.output_directory.mkdir(parents=True, exist_ok=True)
    metadata = read_video_metadata(config.video_path)
    remap = _load_remap(config.remap_path)
    if remap is None:
        active_width, active_height = metadata.width, metadata.height
    else:
        active_height, active_width = remap.map_x.shape

    target_path = config.output_directory / "target_track.csv"
    target_sam2_path: Path | None = None
    target_aligned_path: Path | None = None
    target_audit_path: Path | None = None
    target_correction_summary: dict[str, Any] | None = None
    target_anchor_alignment: dict[str, Any] | None = None
    if config.target.mode == "manual":
        rows = normalize_manual_track(
            config.target.track_path,
            fps=metadata.fps,
            frame_count=metadata.frame_count,
        )
        write_csv_records(target_path, rows)
    else:
        assert config.sam2 is not None
        target_path = track_luminaire_with_sam2(
            video_path=config.video_path,
            output_directory=config.output_directory,
            checkpoint_path=config.sam2.checkpoint_path,
            model_config=config.sam2.model_config,
            device=config.sam2.device,
            roi_xyxy=config.target.roi_xyxy,
            anchor_frame=config.target.anchor_frame,
            positive_points_xy=config.target.positive_points_xy,
            negative_points_xy=config.target.negative_points_xy,
            start_frame=config.target.start_frame,
            end_frame=config.target.end_frame,
            minimum_area_px=config.target.minimum_area_px,
            maximum_step_px=config.target.maximum_step_px,
            remap=remap,
            segments=config.target.segments,
            segment_overlap_policy=config.target.segment_overlap_policy,
        )
        if (
            config.target.anchor_target_xy is not None
            or config.target.corrections_path is not None
        ):
            target_sam2_path = config.output_directory / "target_track_sam2.csv"
            target_path.replace(target_sam2_path)
            sam2_rows = read_csv_records(target_sam2_path)
            final_rows = sam2_rows
            if config.target.anchor_target_xy is not None:
                final_rows, target_anchor_alignment = align_target_to_anchor(
                    final_rows,
                    anchor_frame=config.target.anchor_frame,
                    anchor_target_xy=config.target.anchor_target_xy,
                )
                target_aligned_path = (
                    config.output_directory / "target_track_sam2_aligned.csv"
                )
                write_csv_records(target_aligned_path, final_rows)
        if config.target.corrections_path is not None:
            corrections = load_target_corrections(
                config.target.corrections_path,
                metadata.frame_count,
                active_width,
                active_height,
                config.target.roi_xyxy,
            )
            final_rows, audit_rows, target_correction_summary = (
                apply_target_corrections(final_rows, corrections)
            )
            target_audit_path = (
                config.output_directory / "target_corrections_applied.csv"
            )
            write_csv_records(target_audit_path, audit_rows)
        if (
            config.target.anchor_target_xy is not None
            or config.target.corrections_path is not None
        ):
            write_csv_records(target_path, final_rows)

    reference = _read_frame(
        config.video_path, config.ground.reference_frame, remap
    )
    frames = (
        (item.index, item.image)
        for item in iter_video_frames(config.video_path, remap=remap)
    )
    ground_settings = GroundTrackingConfig(
        search_radius_px=config.ground.search_radius_px,
        consensus_threshold_px=config.ground.consensus_threshold_px,
        min_inliers=config.ground.min_inliers,
        min_correlation=config.ground.min_correlation,
    )
    translations = track_ground_translations(
        reference,
        frames,
        config.ground.rois_xywh,
        config=ground_settings,
    )
    ground_path = config.output_directory / "ground_track.csv"
    ground_model_path: Path | None = None
    ground_audit_path: Path | None = None
    ground_correction_summary: dict[str, Any] | None = None
    if config.ground.corrections_path is not None:
        ground_model_path = config.output_directory / "ground_track_model.csv"
        _write_ground_track(ground_model_path, translations, metadata.fps)
        ground_corrections = load_ground_corrections(
            config.ground.corrections_path,
            metadata.frame_count,
        )
        translations, ground_audit_rows, ground_correction_summary = (
            apply_ground_corrections(translations, ground_corrections)
        )
        ground_audit_path = (
            config.output_directory / "ground_corrections_applied.csv"
        )
        write_csv_records(ground_audit_path, ground_audit_rows)
    _write_ground_track(ground_path, translations, metadata.fps)

    target_rows = read_csv_records(target_path)
    ground_rows = read_csv_records(ground_path)
    relative_rows = compute_relative_motion(
        target_rows,
        ground_rows,
        zero_frame_start=config.relative.zero_frame_start,
        zero_frame_end=config.relative.zero_frame_end,
    )
    relative_path = config.output_directory / "relative_motion.csv"
    write_csv_records(relative_path, relative_rows)

    projected_path: Path | None = None
    metric_path: Path | None = None
    projected_rows: list[dict[str, Any]] | None = None
    metric_scale_m_per_px: float | None = None
    metric_sensitivity_px_per_m: float | None = None
    metric_rest_reprojection_error_px: float | None = None
    if config.metric is not None:
        projection = _metric_projection(config)
        if (
            active_width != config.metric.image_width_px
            or active_height != config.metric.image_height_px
        ):
            raise ValueError(
                "Metric calibration image dimensions do not match the video"
            )
        zero_reference_pixel = _zero_reference_pixel(
            relative_rows,
            target_rows,
            ground_rows,
            zero_frame_start=config.relative.zero_frame_start,
            zero_frame_end=config.relative.zero_frame_end,
        )
        rest_residual = zero_reference_pixel - projection.reference_pixel_xy
        rest_error_px = float(np.linalg.norm(rest_residual))
        metric_rest_reprojection_error_px = rest_error_px
        if rest_error_px > config.metric.maximum_rest_reprojection_error_px:
            raise ValueError(
                "Metric calibration rest-point reprojection error exceeds "
                "the configured maximum"
            )
        metric_scale_m_per_px = projection.scale_m_per_px
        metric_sensitivity_px_per_m = float(
            np.linalg.norm(projection.sensitivity_px_per_m_xy)
        )
        projected_rows = _projected_rows(
            relative_rows,
            projection,
            config.metric.validation_status,
        )
        projected_path = config.output_directory / "projected_motion.csv"
        write_csv_records(projected_path, projected_rows)
        metric_path = config.output_directory / "metric_projection.json"
        _write_json(
            metric_path,
            {
                "schema_version": "light_pole_metric_projection_v1",
                "status": "conditional_metric_projection",
                "metric_validation_status": config.metric.validation_status,
                "physical_use_allowed": False,
                "calibration_physical_use_allowed": (
                    config.metric.calibration_physical_use_allowed
                ),
                "calibration": {
                    "path": config.metric.calibration_path.name,
                    "sha256": config.metric.calibration_sha256,
                },
                "coordinate_frame_id": config.metric.coordinate_frame_id,
                "motion_subspace": config.metric.motion_subspace,
                "ground_plane_z_m": config.metric.ground_plane_z_m,
                "ground_reference_frame": config.metric.ground_reference_frame,
                "image_geometry": {
                    "model": config.metric.image_model,
                    "width_px": config.metric.image_width_px,
                    "height_px": config.metric.image_height_px,
                    "remap_sha256": config.metric.remap_sha256,
                },
                "material_point_world_xyz_m": list(
                    config.metric.material_point_world_xyz_m
                ),
                "camera_matrix_px": [
                    list(row) for row in config.metric.camera_matrix_px
                ],
                "rotation_world_to_camera": [
                    list(row) for row in config.metric.rotation_world_to_camera
                ],
                "optical_center_world_xyz_m": list(
                    config.metric.optical_center_world_xyz_m
                ),
                "image_measurement_axis_xy": list(
                    config.metric.image_measurement_axis_xy
                ),
                "calibration_reference": {
                    "description": config.metric.calibration_reference.description,
                    "value_m": config.metric.calibration_reference.value_m,
                    "uncertainty_m": (
                        config.metric.calibration_reference.uncertainty_m
                    ),
                    "source": config.metric.calibration_reference.source,
                    "relationship": (
                        config.metric.calibration_reference.relationship
                    ),
                },
                "projection": projection_to_dict(projection),
                "validation_gates": {
                    "image_dimensions_match": True,
                    "rest_reference_pixel_xy": zero_reference_pixel.tolist(),
                    "predicted_reference_pixel_xy": (
                        projection.reference_pixel_xy.tolist()
                    ),
                    "rest_reprojection_residual_xy_px": rest_residual.tolist(),
                    "rest_reprojection_error_px": rest_error_px,
                    "maximum_rest_reprojection_error_px": (
                        config.metric.maximum_rest_reprojection_error_px
                    ),
                    "minimum_horizontal_sensitivity_px_per_m": (
                        config.metric.minimum_horizontal_sensitivity_px_per_m
                    ),
                },
                "interpretation": (
                    "One-camera ground-relative displacement projected on the "
                    "camera-sensitive horizontal direction."
                ),
            },
        )
    arrays_path = config.output_directory / "tracks.npz"
    _save_arrays(
        arrays_path,
        target_rows,
        ground_rows,
        relative_rows,
        projected_rows,
    )

    qa_path: Path | None = None
    if config.qa.enabled:
        qa_path = config.qa.output_path
        points = _target_points(target_rows)
        ground_map = {item.frame_index: item for item in translations}
        ground_by_frame = {int(row["frame"]): row for row in ground_rows}
        valid_zero = []
        for row in target_rows:
            frame = int(row["frame"])
            ground_row = ground_by_frame.get(frame)
            if (
                row["valid"]
                and ground_row is not None
                and ground_row["valid"]
                and config.relative.zero_frame_start
                <= frame
                <= config.relative.zero_frame_end
            ):
                valid_zero.append(
                    (
                        float(row["x_px"]) - float(ground_row["dx_px"]),
                        float(row["y_px"]) - float(ground_row["dy_px"]),
                    )
                )
        if not valid_zero:
            raise ValueError("The zero interval has no valid target samples")
        reference_target = (
            float(np.median([value[0] for value in valid_zero])),
            float(np.median([value[1] for value in valid_zero])),
        )
        render_tracking_qa_video(
            config.video_path,
            qa_path,
            target_points=points,
            ground_translations=ground_map,
            ground_reference_rois=config.ground.rois_xywh,
            reference_target_point=reference_target,
            remap=remap,
        )

    artifacts = [target_path, ground_path, relative_path, arrays_path]
    for correction_artifact in (
        target_sam2_path,
        target_aligned_path,
        target_audit_path,
        ground_model_path,
        ground_audit_path,
    ):
        if correction_artifact is not None:
            artifacts.append(correction_artifact)
    if projected_path is not None and metric_path is not None:
        artifacts.extend([projected_path, metric_path])
    target_masks = config.output_directory / "target_masks.npz"
    if target_masks.is_file():
        artifacts.append(target_masks)
    target_segments = config.output_directory / "target_segments.json"
    if target_segments.is_file():
        artifacts.append(target_segments)
    if qa_path is not None:
        artifacts.append(qa_path)
    full_target_replay = bool(
        target_correction_summary is not None
        and target_correction_summary["correction_count"] == len(target_rows)
    )
    if full_target_replay:
        target_track_source = "reviewed_candidate_replay_with_sam2_evidence"
    elif (
        config.target.corrections_path is not None
        and config.target.anchor_target_xy is not None
    ):
        target_track_source = (
            "sam2_segmented_aligned_with_manual_corrections"
            if config.target.segments
            else "sam2_aligned_with_manual_corrections"
        )
    elif config.target.corrections_path is not None:
        target_track_source = (
            "sam2_segmented_with_manual_corrections"
            if config.target.segments
            else "sam2_with_manual_corrections"
        )
    elif config.target.anchor_target_xy is not None:
        target_track_source = (
            "sam2_segmented_aligned_to_material_point"
            if config.target.segments
            else "sam2_aligned_to_material_point"
        )
    elif config.target.segments:
        target_track_source = "sam2_segmented"
    else:
        target_track_source = config.target.mode
    manifest_path = config.output_directory / "manifest.json"
    write_manifest(
        manifest_path,
        config=config,
        inputs={
            "video": config.video_path,
            "remap": config.remap_path,
            "metric_calibration": (
                config.metric.calibration_path
                if config.metric is not None
                else None
            ),
            "manual_track": config.target.track_path,
            "target_corrections": config.target.corrections_path,
            "ground_corrections": config.ground.corrections_path,
            "sam2_checkpoint": (
                config.sam2.checkpoint_path
                if config.target.mode == "sam2" and config.sam2 is not None
                else None
            ),
        },
        artifacts={path.name: sha256_file(path) for path in artifacts},
        metadata={
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "target_processed_frame_count": len(target_rows),
            "ground_processed_frame_count": len(ground_rows),
            "relative_valid_count": sum(bool(row["valid"]) for row in relative_rows),
            "target_track_source": target_track_source,
            "target_sam2_segments": [
                {
                    "id": segment.id,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "anchor_frame": segment.anchor_frame,
                    "direction": segment.direction,
                }
                for segment in config.target.segments
            ],
            "target_sam2_segment_overlap_policy": (
                config.target.segment_overlap_policy
            ),
            "target_anchor_alignment": target_anchor_alignment,
            "target_corrections": target_correction_summary,
            "target_correction_coverage_fraction": (
                target_correction_summary["correction_count"]
                / len(target_rows)
                if target_correction_summary is not None
                else 0.0
            ),
            "ground_corrections": ground_correction_summary,
            "ground_correction_coverage_fraction": (
                ground_correction_summary["correction_count"]
                / len(ground_rows)
                if ground_correction_summary is not None
                else 0.0
            ),
            "metric_projection_enabled": config.metric is not None,
            "metric_scale_m_per_px": metric_scale_m_per_px,
            "metric_physical_use_allowed": (
                False if config.metric is not None else None
            ),
            "metric_validation_status": (
                config.metric.validation_status
                if config.metric is not None
                else None
            ),
            "metric_calibration_physical_use_allowed": (
                config.metric.calibration_physical_use_allowed
                if config.metric is not None
                else None
            ),
            "metric_sensitivity_px_per_m": metric_sensitivity_px_per_m,
            "metric_rest_reprojection_error_px": (
                metric_rest_reprojection_error_px
            ),
            "metric_gates_passed": (
                True if config.metric is not None else None
            ),
        },
    )
    return PipelineResult(
        config.output_directory,
        target_path,
        ground_path,
        relative_path,
        arrays_path,
        manifest_path,
        qa_path,
        projected_path,
        metric_path,
        target_sam2_path,
        target_aligned_path,
        target_audit_path,
        ground_model_path,
        ground_audit_path,
    )
