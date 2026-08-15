"""Load and validate one-view tracking configurations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class Sam2SegmentConfig:
    """One independently prompted SAM 2 propagation interval."""

    id: str
    start_frame: int
    end_frame: int
    anchor_frame: int
    direction: Literal["forward", "reverse", "bidirectional"]
    positive_points_xy: tuple[tuple[float, float], ...]
    negative_points_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TargetConfig:
    """Compact-luminaire tracking settings."""

    mode: Literal["sam2", "manual"]
    track_path: Path | None
    roi_xyxy: tuple[int, int, int, int]
    anchor_frame: int
    positive_points_xy: tuple[tuple[float, float], ...]
    negative_points_xy: tuple[tuple[float, float], ...]
    anchor_target_xy: tuple[float, float] | None = None
    start_frame: int = 0
    end_frame: int | None = None
    minimum_area_px: int = 1
    maximum_step_px: float | None = None
    corrections_path: Path | None = None
    segments: tuple[Sam2SegmentConfig, ...] = ()
    segment_overlap_policy: Literal["error", "first", "last"] = "error"


@dataclass(frozen=True)
class GroundConfig:
    """Local-ground registration settings."""

    reference_frame: int
    rois_xywh: tuple[tuple[int, int, int, int], ...]
    search_radius_px: tuple[int, int]
    min_correlation: float
    consensus_threshold_px: float
    min_inliers: int
    corrections_path: Path | None = None


@dataclass(frozen=True)
class RelativeConfig:
    """Reference interval for relative displacement."""

    zero_frame_start: int
    zero_frame_end: int


@dataclass(frozen=True)
class Sam2Config:
    """External SAM 2 runtime settings."""

    checkpoint_path: Path
    model_config: str
    device: str


@dataclass(frozen=True)
class QaConfig:
    """QA-video settings."""

    enabled: bool
    output_path: Path


@dataclass(frozen=True)
class CalibrationReferenceConfig:
    """Physical reference encoded in the external calibration artifact."""

    description: str
    value_m: float
    uncertainty_m: float | None
    source: str
    relationship: Literal["world_z_above_ground_plane"]


@dataclass(frozen=True)
class MetricConfig:
    """Validated external local image-to-metric calibration."""

    calibration_path: Path
    calibration_sha256: str
    validation_status: str
    calibration_physical_use_allowed: bool
    coordinate_frame_id: str
    motion_subspace: Literal["horizontal_world_xy"]
    ground_plane_z_m: float
    ground_reference_frame: int
    image_model: Literal["undistorted_pinhole", "rectified_pinhole"]
    image_width_px: int
    image_height_px: int
    remap_sha256: str | None
    material_point_world_xyz_m: tuple[float, float, float]
    camera_matrix_px: tuple[tuple[float, float, float], ...]
    rotation_world_to_camera: tuple[tuple[float, float, float], ...]
    optical_center_world_xyz_m: tuple[float, float, float]
    image_measurement_axis_xy: tuple[float, float]
    calibration_reference: CalibrationReferenceConfig
    minimum_horizontal_sensitivity_px_per_m: float
    maximum_rest_reprojection_error_px: float


@dataclass(frozen=True)
class PipelineConfig:
    """Validated configuration with absolute runtime paths."""

    path: Path
    root: Path
    video_path: Path
    output_directory: Path
    remap_path: Path | None
    target: TargetConfig
    ground: GroundConfig
    relative: RelativeConfig
    sam2: Sam2Config | None
    qa: QaConfig
    metric: MetricConfig | None

    def validate_inputs(self) -> None:
        """Require every configured runtime input to exist."""

        required: list[tuple[str, Path | None]] = [("video", self.video_path)]
        if self.remap_path is not None:
            required.append(("lens remap", self.remap_path))
        if self.target.mode == "manual":
            required.append(("manual target track", self.target.track_path))
        if self.sam2 is not None and self.target.mode == "sam2":
            required.append(("SAM 2 checkpoint", self.sam2.checkpoint_path))
        if self.target.corrections_path is not None:
            required.append(("target corrections", self.target.corrections_path))
        if self.ground.corrections_path is not None:
            required.append(("ground corrections", self.ground.corrections_path))
        if self.metric is not None:
            required.append(("metric calibration", self.metric.calibration_path))
        for label, path in required:
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if self.metric is not None:
            if _sha256_file(self.metric.calibration_path) != (
                self.metric.calibration_sha256
            ):
                raise ValueError("Metric calibration changed after configuration load")
            if self.remap_path is not None and (
                self.metric.remap_sha256 is None
                or _sha256_file(self.remap_path) != self.metric.remap_sha256
            ):
                raise ValueError("Lens remap changed after configuration load")
        try:
            qa_relative = self.qa.output_path.relative_to(self.output_directory)
        except ValueError:
            qa_relative = None
        if self.qa.enabled and (
            qa_relative is None or len(qa_relative.parts) != 1
        ):
            raise ValueError("qa.output_path must be inside output_directory")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sha256_text(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root: Path, value: Any, name: str) -> Path:
    raw = Path(_text(value, name))
    return (raw if raw.is_absolute() else root / raw).resolve()


def _points(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    result: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{name}[{index}] must contain x and y")
        result.append(
            (
                _number(point[0], f"{name}[{index}][0]"),
                _number(point[1], f"{name}[{index}][1]"),
            )
        )
    return tuple(result)


def _number_tuple(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _matrix_3x3(value: Any, name: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three rows")
    rows = tuple(
        _number_tuple(row, f"{name}[{index}]", 3)
        for index, row in enumerate(value)
    )
    return rows  # type: ignore[return-value]


def _integer_tuple(
    value: Any,
    name: str,
    length: int,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} integers")
    return tuple(
        _integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _sam2_segments(
    value: Any,
    *,
    start_frame: int,
    end_frame: int | None,
    default_positive_points_xy: tuple[tuple[float, float], ...],
    default_negative_points_xy: tuple[tuple[float, float], ...],
    overlap_policy: str,
) -> tuple[Sam2SegmentConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("target.segments must be an array")
    if not value:
        return ()
    if end_frame is None:
        raise ValueError(
            "target.end_frame is required when target.segments are configured"
        )
    segments: list[Sam2SegmentConfig] = []
    identifiers: set[str] = set()
    for index, item in enumerate(value):
        name = f"target.segments[{index}]"
        raw = _mapping(item, name)
        identifier_value = raw.get("id", f"segment_{index + 1:03d}")
        identifier = _text(identifier_value, f"{name}.id")
        if identifier in identifiers:
            raise ValueError(f"{name}.id must be unique")
        identifiers.add(identifier)
        segment_start = _integer(raw.get("start_frame"), f"{name}.start_frame")
        segment_end = _integer(raw.get("end_frame"), f"{name}.end_frame")
        anchor = _integer(raw.get("anchor_frame"), f"{name}.anchor_frame")
        if segment_end < segment_start:
            raise ValueError(f"{name}.end_frame must not precede start_frame")
        if segment_start < start_frame or segment_end > end_frame:
            raise ValueError(f"{name} must lie inside the processed frame range")
        if not segment_start <= anchor <= segment_end:
            raise ValueError(f"{name}.anchor_frame must lie inside the segment")
        direction = _text(raw.get("direction"), f"{name}.direction")
        if direction not in {"forward", "reverse", "bidirectional"}:
            raise ValueError(
                f"{name}.direction must be 'forward', 'reverse', or 'bidirectional'"
            )
        if direction == "forward" and anchor != segment_start:
            raise ValueError(
                f"{name}.anchor_frame must equal start_frame for forward propagation"
            )
        if direction == "reverse" and anchor != segment_end:
            raise ValueError(
                f"{name}.anchor_frame must equal end_frame for reverse propagation"
            )
        positive = (
            default_positive_points_xy
            if "positive_points_xy" not in raw
            else _points(raw.get("positive_points_xy"), f"{name}.positive_points_xy")
        )
        negative = (
            default_negative_points_xy
            if "negative_points_xy" not in raw
            else _points(raw.get("negative_points_xy"), f"{name}.negative_points_xy")
        )
        if not positive:
            raise ValueError(f"{name}.positive_points_xy needs one point")
        segments.append(
            Sam2SegmentConfig(
                id=identifier,
                start_frame=segment_start,
                end_frame=segment_end,
                anchor_frame=anchor,
                direction=direction,  # type: ignore[arg-type]
                positive_points_xy=positive,
                negative_points_xy=negative,
            )
        )

    coverage = np.zeros(end_frame - start_frame + 1, dtype=np.int16)
    for segment in segments:
        first = segment.start_frame - start_frame
        stop = segment.end_frame - start_frame + 1
        coverage[first:stop] += 1
    missing = np.flatnonzero(coverage == 0)
    if missing.size:
        frame = int(missing[0] + start_frame)
        raise ValueError(f"target.segments do not cover processed frame {frame}")
    overlap = np.flatnonzero(coverage > 1)
    if overlap.size and overlap_policy == "error":
        frame = int(overlap[0] + start_frame)
        raise ValueError(
            f"target.segments overlap at frame {frame}; set "
            "target.segment_overlap_policy to 'first' or 'last'"
        )
    return tuple(segments)


def _target(root: Path, value: Any) -> TargetConfig:
    raw = _mapping(value, "target")
    mode = _text(raw.get("mode"), "target.mode")
    if mode not in {"sam2", "manual"}:
        raise ValueError("target.mode must be 'sam2' or 'manual'")
    roi = _integer_tuple(raw.get("roi_xyxy"), "target.roi_xyxy", 4)
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("target.roi_xyxy must have positive width and height")
    start = _integer(raw.get("start_frame", 0), "target.start_frame")
    end_value = raw.get("end_frame")
    end = None if end_value is None else _integer(end_value, "target.end_frame")
    if end is not None and end < start:
        raise ValueError("target.end_frame must not precede target.start_frame")
    segment_values = raw.get("segments")
    has_segments = isinstance(segment_values, list) and bool(segment_values)
    anchor_value = raw.get("anchor_frame")
    if anchor_value is None and has_segments:
        first_segment = _mapping(segment_values[0], "target.segments[0]")
        anchor_value = first_segment.get("anchor_frame")
    anchor = _integer(anchor_value, "target.anchor_frame")
    if anchor < start or (end is not None and anchor > end):
        raise ValueError("target.anchor_frame must be inside the processed range")
    anchor_target_value = raw.get("anchor_target_xy")
    anchor_target = (
        None
        if anchor_target_value is None
        else _number_tuple(
            anchor_target_value,
            "target.anchor_target_xy",
            2,
        )
    )
    if mode != "sam2" and anchor_target is not None:
        raise ValueError("target.anchor_target_xy is only supported in SAM 2 mode")
    if anchor_target is not None and not (
        roi[0] <= anchor_target[0] < roi[2]
        and roi[1] <= anchor_target[1] < roi[3]
    ):
        raise ValueError("target.anchor_target_xy must lie inside target.roi_xyxy")
    track_value = raw.get("track_path")
    track_path = (
        None
        if track_value is None
        else _path(root, track_value, "target.track_path")
    )
    if mode == "manual" and track_path is None:
        raise ValueError("target.track_path is required in manual mode")
    corrections_value = raw.get("corrections_path")
    corrections_path = (
        None
        if corrections_value is None
        else _path(root, corrections_value, "target.corrections_path")
    )
    if mode != "sam2" and corrections_path is not None:
        raise ValueError("target.corrections_path is only supported in SAM 2 mode")
    minimum_area = _integer(
        raw.get("minimum_area_px", 1),
        "target.minimum_area_px",
        minimum=1,
    )
    maximum_step_value = raw.get("maximum_step_px")
    maximum_step = (
        None
        if maximum_step_value is None
        else _number(maximum_step_value, "target.maximum_step_px")
    )
    if maximum_step is not None and maximum_step < 0:
        raise ValueError("target.maximum_step_px must be non-negative")
    positive_points = _points(
        raw.get("positive_points_xy", []), "target.positive_points_xy"
    )
    negative_points = _points(
        raw.get("negative_points_xy", []), "target.negative_points_xy"
    )
    overlap_policy = _text(
        raw.get("segment_overlap_policy", "error"),
        "target.segment_overlap_policy",
    )
    if overlap_policy not in {"error", "first", "last"}:
        raise ValueError(
            "target.segment_overlap_policy must be 'error', 'first', or 'last'"
        )
    if mode != "sam2" and has_segments:
        raise ValueError("target.segments are only supported in SAM 2 mode")
    segments = _sam2_segments(
        segment_values,
        start_frame=start,
        end_frame=end,
        default_positive_points_xy=positive_points,
        default_negative_points_xy=negative_points,
        overlap_policy=overlap_policy,
    )
    if mode == "sam2" and not positive_points and not segments:
        raise ValueError("target.positive_points_xy needs one point in SAM 2 mode")
    return TargetConfig(
        mode=mode,  # type: ignore[arg-type]
        track_path=track_path,
        roi_xyxy=roi,  # type: ignore[arg-type]
        anchor_frame=anchor,
        positive_points_xy=positive_points,
        negative_points_xy=negative_points,
        anchor_target_xy=(
            None
            if anchor_target is None
            else (float(anchor_target[0]), float(anchor_target[1]))
        ),
        start_frame=start,
        end_frame=end,
        minimum_area_px=minimum_area,
        maximum_step_px=maximum_step,
        corrections_path=corrections_path,
        segments=segments,
        segment_overlap_policy=overlap_policy,  # type: ignore[arg-type]
    )


def _ground(root: Path, value: Any) -> GroundConfig:
    raw = _mapping(value, "ground")
    roi_values = raw.get("rois_xywh")
    if not isinstance(roi_values, list) or not roi_values:
        raise ValueError("ground.rois_xywh must contain at least one ROI")
    rois: list[tuple[int, int, int, int]] = []
    for index, item in enumerate(roi_values):
        roi = _integer_tuple(item, f"ground.rois_xywh[{index}]", 4)
        if roi[2] <= 0 or roi[3] <= 0:
            raise ValueError("ground ROI width and height must be positive")
        rois.append(roi)  # type: ignore[arg-type]
    radius = _integer_tuple(
        raw.get("search_radius_px"), "ground.search_radius_px", 2
    )
    correlation = _number(raw.get("min_correlation"), "ground.min_correlation")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("ground.min_correlation must be between -1 and 1")
    threshold = _number(
        raw.get("consensus_threshold_px"), "ground.consensus_threshold_px"
    )
    if threshold < 0:
        raise ValueError("ground.consensus_threshold_px must be non-negative")
    min_inliers = _integer(raw.get("min_inliers"), "ground.min_inliers", minimum=1)
    if min_inliers > len(rois):
        raise ValueError("ground.min_inliers exceeds the number of ROIs")
    corrections_value = raw.get("corrections_path")
    corrections_path = (
        None
        if corrections_value is None
        else _path(root, corrections_value, "ground.corrections_path")
    )
    return GroundConfig(
        reference_frame=_integer(
            raw.get("reference_frame"), "ground.reference_frame"
        ),
        rois_xywh=tuple(rois),
        search_radius_px=radius,  # type: ignore[arg-type]
        min_correlation=correlation,
        consensus_threshold_px=threshold,
        min_inliers=min_inliers,
        corrections_path=corrections_path,
    )


def _relative(value: Any) -> RelativeConfig:
    raw = _mapping(value, "relative")
    start = _integer(raw.get("zero_frame_start"), "relative.zero_frame_start")
    end = _integer(raw.get("zero_frame_end"), "relative.zero_frame_end")
    if end < start:
        raise ValueError("relative.zero_frame_end must not precede zero_frame_start")
    return RelativeConfig(start, end)


def _sam2(root: Path, value: Any) -> Sam2Config:
    raw = _mapping(value, "sam2")
    return Sam2Config(
        checkpoint_path=_path(root, raw.get("checkpoint_path"), "sam2.checkpoint_path"),
        model_config=_text(raw.get("model_config"), "sam2.model_config"),
        device=_text(raw.get("device"), "sam2.device"),
    )


def _qa(root: Path, value: Any) -> QaConfig:
    raw = _mapping(value, "qa")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("qa.enabled must be a boolean")
    return QaConfig(
        enabled=enabled,
        output_path=_path(root, raw.get("output_path"), "qa.output_path"),
    )


def _calibration_reference(value: Any) -> CalibrationReferenceConfig:
    raw = _mapping(value, "calibration.physical_reference")
    physical_value = _number(
        raw.get("value_m"), "calibration.physical_reference.value_m"
    )
    if physical_value <= 0.0:
        raise ValueError("calibration.physical_reference.value_m must be positive")
    uncertainty_value = raw.get("uncertainty_m")
    uncertainty = (
        None
        if uncertainty_value is None
        else _number(
            uncertainty_value,
            "calibration.physical_reference.uncertainty_m",
        )
    )
    if uncertainty is not None and uncertainty < 0.0:
        raise ValueError(
            "calibration.physical_reference.uncertainty_m must be non-negative"
        )
    relationship = _text(
        raw.get("relationship"),
        "calibration.physical_reference.relationship",
    )
    if relationship != "world_z_above_ground_plane":
        raise ValueError(
            "calibration.physical_reference.relationship must be "
            "'world_z_above_ground_plane'"
        )
    source = _text(raw.get("source"), "calibration.physical_reference.source")
    if Path(source).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", source):
        raise ValueError("calibration.physical_reference.source must be a label")
    return CalibrationReferenceConfig(
        description=_text(
            raw.get("description"),
            "calibration.physical_reference.description",
        ),
        value_m=physical_value,
        uncertainty_m=uncertainty,
        source=source,
        relationship=relationship,  # type: ignore[arg-type]
    )


def _metric(root: Path, value: Any, remap_path: Path | None) -> MetricConfig:
    from .metric import build_metric_projection

    raw = _mapping(value, "metric")
    calibration_path = _path(
        root,
        raw.get("calibration_path"),
        "metric.calibration_path",
    )
    expected_sha256 = _sha256_text(
        raw.get("calibration_sha256"),
        "metric.calibration_sha256",
    )
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing metric calibration: {calibration_path}")
    actual_sha256 = _sha256_file(calibration_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("metric.calibration_sha256 does not match calibration_path")
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {calibration_path}: {error.msg}"
        ) from error
    calibration = _mapping(payload, "metric calibration")
    if calibration.get("schema_version") != "light_pole_metric_calibration_v1":
        raise ValueError(
            "calibration.schema_version must be "
            "'light_pole_metric_calibration_v1'"
        )
    contract = _mapping(
        calibration.get("coordinate_contract"),
        "calibration.coordinate_contract",
    )
    motion_subspace = _text(
        contract.get("motion_subspace"),
        "calibration.coordinate_contract.motion_subspace",
    )
    if motion_subspace != "horizontal_world_xy":
        raise ValueError(
            "calibration.coordinate_contract.motion_subspace must be "
            "'horizontal_world_xy'"
        )
    image = _mapping(calibration.get("image_geometry"), "calibration.image_geometry")
    image_model = _text(image.get("model"), "calibration.image_geometry.model")
    if image_model not in {"undistorted_pinhole", "rectified_pinhole"}:
        raise ValueError(
            "calibration.image_geometry.model must be 'undistorted_pinhole' "
            "or 'rectified_pinhole'"
        )
    remap_digest_value = image.get("remap_sha256")
    remap_digest = (
        None
        if remap_digest_value is None
        else _sha256_text(
            remap_digest_value,
            "calibration.image_geometry.remap_sha256",
        )
    )
    if image_model == "rectified_pinhole":
        if remap_path is None or remap_digest is None:
            raise ValueError(
                "rectified_pinhole calibration requires remap_path and "
                "remap_sha256"
            )
        if not remap_path.is_file():
            raise FileNotFoundError(f"Missing lens remap: {remap_path}")
        if _sha256_file(remap_path) != remap_digest:
            raise ValueError(
                "calibration image geometry does not match the lens remap"
            )
    elif remap_path is not None or remap_digest is not None:
        raise ValueError(
            "undistorted_pinhole calibration cannot be combined with remap_path"
        )
    material_point = _number_tuple(
        calibration.get("material_point_world_xyz_m"),
        "calibration.material_point_world_xyz_m",
        3,
    )
    ground_plane_z = _number(
        contract.get("ground_plane_z_m"),
        "calibration.coordinate_contract.ground_plane_z_m",
    )
    reference = _calibration_reference(calibration.get("physical_reference"))
    encoded_height = material_point[2] - ground_plane_z
    if not abs(encoded_height - reference.value_m) <= 1.0e-9:
        raise ValueError(
            "calibration physical reference must equal the material-point "
            "height above the ground plane"
        )
    metric = MetricConfig(
        calibration_path=calibration_path,
        calibration_sha256=actual_sha256,
        validation_status=_text(
            calibration.get("validation_status"),
            "calibration.validation_status",
        ),
        calibration_physical_use_allowed=_boolean(
            calibration.get("physical_use_allowed"),
            "calibration.physical_use_allowed",
        ),
        coordinate_frame_id=_text(
            contract.get("coordinate_frame_id"),
            "calibration.coordinate_contract.coordinate_frame_id",
        ),
        motion_subspace=motion_subspace,  # type: ignore[arg-type]
        ground_plane_z_m=ground_plane_z,
        ground_reference_frame=_integer(
            contract.get("ground_reference_frame"),
            "calibration.coordinate_contract.ground_reference_frame",
        ),
        image_model=image_model,  # type: ignore[arg-type]
        image_width_px=_integer(
            image.get("width_px"), "calibration.image_geometry.width_px", minimum=1
        ),
        image_height_px=_integer(
            image.get("height_px"),
            "calibration.image_geometry.height_px",
            minimum=1,
        ),
        remap_sha256=remap_digest,
        material_point_world_xyz_m=material_point,  # type: ignore[arg-type]
        camera_matrix_px=_matrix_3x3(
            calibration.get("camera_matrix_px"), "calibration.camera_matrix_px"
        ),
        rotation_world_to_camera=_matrix_3x3(
            calibration.get("rotation_world_to_camera"),
            "calibration.rotation_world_to_camera",
        ),
        optical_center_world_xyz_m=_number_tuple(
            calibration.get("optical_center_world_xyz_m"),
            "calibration.optical_center_world_xyz_m",
            3,
        ),  # type: ignore[arg-type]
        image_measurement_axis_xy=_number_tuple(
            calibration.get("image_measurement_axis_xy"),
            "calibration.image_measurement_axis_xy",
            2,
        ),  # type: ignore[arg-type]
        calibration_reference=reference,
        minimum_horizontal_sensitivity_px_per_m=_number(
            calibration.get("minimum_horizontal_sensitivity_px_per_m"),
            "calibration.minimum_horizontal_sensitivity_px_per_m",
        ),
        maximum_rest_reprojection_error_px=_number(
            calibration.get("maximum_rest_reprojection_error_px"),
            "calibration.maximum_rest_reprojection_error_px",
        ),
    )
    if metric.minimum_horizontal_sensitivity_px_per_m <= 0.0:
        raise ValueError(
            "calibration.minimum_horizontal_sensitivity_px_per_m must be positive"
        )
    if metric.maximum_rest_reprojection_error_px < 0.0:
        raise ValueError(
            "calibration.maximum_rest_reprojection_error_px must be non-negative"
        )
    projection = build_metric_projection(
        metric.material_point_world_xyz_m,
        metric.camera_matrix_px,
        metric.rotation_world_to_camera,
        metric.optical_center_world_xyz_m,
        metric.image_measurement_axis_xy,
    )
    sensitivity = float(np.linalg.norm(projection.sensitivity_px_per_m_xy))
    if sensitivity < metric.minimum_horizontal_sensitivity_px_per_m:
        raise ValueError(
            "calibration horizontal sensitivity is below the configured minimum"
        )
    return metric


def load_config(path: str | Path) -> PipelineConfig:
    """Load JSON and resolve all user paths from the configuration directory."""

    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {config_path}: {error.msg}") from error
    root = config_path.parent
    payload = _mapping(raw, "configuration")
    target = _target(root, payload.get("target"))
    sam2_value = payload.get("sam2")
    if target.mode == "sam2" and sam2_value is None:
        raise ValueError("sam2 settings are required in SAM 2 mode")
    remap_value = payload.get("remap_path")
    remap_path = (
        None if remap_value is None else _path(root, remap_value, "remap_path")
    )
    metric_value = payload.get("metric")
    metric = (
        None
        if metric_value is None
        else _metric(root, metric_value, remap_path)
    )
    ground = _ground(root, payload.get("ground"))
    if metric is not None and (
        metric.ground_reference_frame != ground.reference_frame
    ):
        raise ValueError(
            "Metric calibration ground_reference_frame must match the run"
        )
    return PipelineConfig(
        path=config_path,
        root=root,
        video_path=_path(root, payload.get("video_path"), "video_path"),
        output_directory=_path(
            root, payload.get("output_directory"), "output_directory"
        ),
        remap_path=remap_path,
        target=target,
        ground=ground,
        relative=_relative(payload.get("relative")),
        sam2=None if sam2_value is None else _sam2(root, sam2_value),
        qa=_qa(root, payload.get("qa")),
        metric=metric,
    )
