"""Convert image-plane relative motion to a local metric projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _camera_matrix(value: object) -> np.ndarray:
    matrix = _array(value, name="camera_matrix_px", shape=(3, 3))
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("camera_matrix_px must have positive focal lengths")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1.0e-10):
        raise ValueError("camera_matrix_px last row must be [0, 0, 1]")
    if abs(float(np.linalg.det(matrix))) <= np.finfo(np.float64).eps:
        raise ValueError("camera_matrix_px must be nonsingular")
    return matrix


def _rotation_matrix(value: object) -> np.ndarray:
    rotation = _array(
        value,
        name="rotation_world_to_camera",
        shape=(3, 3),
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7):
        raise ValueError("rotation_world_to_camera must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-7):
        raise ValueError("rotation_world_to_camera must have determinant +1")
    return rotation


@dataclass(frozen=True)
class MetricProjection:
    """Local scalar observation geometry at one tracked material point."""

    sensitivity_px_per_m_xy: np.ndarray
    scale_m_per_px: float
    measurement_axis_world_xy: np.ndarray
    image_measurement_axis_xy: np.ndarray
    reference_pixel_xy: np.ndarray
    depth_m: float

    def __post_init__(self) -> None:
        sensitivity = _array(
            self.sensitivity_px_per_m_xy,
            name="sensitivity_px_per_m_xy",
            shape=(2,),
        ).copy()
        axis = _array(
            self.measurement_axis_world_xy,
            name="measurement_axis_world_xy",
            shape=(2,),
        ).copy()
        image_axis = _array(
            self.image_measurement_axis_xy,
            name="image_measurement_axis_xy",
            shape=(2,),
        ).copy()
        reference_pixel = _array(
            self.reference_pixel_xy,
            name="reference_pixel_xy",
            shape=(2,),
        ).copy()
        norm = float(np.linalg.norm(sensitivity))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("sensitivity_px_per_m_xy must be nonzero")
        if not np.isfinite(self.scale_m_per_px) or self.scale_m_per_px <= 0.0:
            raise ValueError("scale_m_per_px must be finite and positive")
        if not np.isclose(self.scale_m_per_px * norm, 1.0, atol=1.0e-10):
            raise ValueError("scale_m_per_px is inconsistent with sensitivity")
        if not np.allclose(axis, sensitivity / norm, atol=1.0e-10):
            raise ValueError(
                "measurement_axis_world_xy is inconsistent with sensitivity"
            )
        if not np.isclose(np.linalg.norm(image_axis), 1.0, atol=1.0e-10):
            raise ValueError("image_measurement_axis_xy must be a unit vector")
        if not np.isfinite(self.depth_m) or self.depth_m <= 0.0:
            raise ValueError("depth_m must be finite and positive")
        sensitivity.setflags(write=False)
        axis.setflags(write=False)
        image_axis.setflags(write=False)
        reference_pixel.setflags(write=False)
        object.__setattr__(self, "sensitivity_px_per_m_xy", sensitivity)
        object.__setattr__(self, "measurement_axis_world_xy", axis)
        object.__setattr__(self, "image_measurement_axis_xy", image_axis)
        object.__setattr__(self, "reference_pixel_xy", reference_pixel)
        object.__setattr__(self, "scale_m_per_px", float(self.scale_m_per_px))
        object.__setattr__(self, "depth_m", float(self.depth_m))


def projection_jacobian_world(
    point_world_xyz_m: object,
    camera_matrix_px: object,
    rotation_world_to_camera: object,
    optical_center_world_xyz_m: object,
) -> tuple[np.ndarray, float]:
    """Return ``d(pixel_xy)/d(world_xyz)`` and camera depth at one point."""

    point = _array(
        point_world_xyz_m,
        name="point_world_xyz_m",
        shape=(3,),
    )
    matrix = _camera_matrix(camera_matrix_px)
    rotation = _rotation_matrix(rotation_world_to_camera)
    center = _array(
        optical_center_world_xyz_m,
        name="optical_center_world_xyz_m",
        shape=(3,),
    )
    camera_point = rotation @ (point - center)
    homogeneous = matrix @ camera_point
    denominator = float(homogeneous[2])
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("point_world_xyz_m must be in front of the camera")
    quotient = np.vstack(
        (
            (matrix[0] * denominator - homogeneous[0] * matrix[2])
            / denominator**2,
            (matrix[1] * denominator - homogeneous[1] * matrix[2])
            / denominator**2,
        )
    )
    return quotient @ rotation, float(camera_point[2])


def project_world_point(
    point_world_xyz_m: object,
    camera_matrix_px: object,
    rotation_world_to_camera: object,
    optical_center_world_xyz_m: object,
) -> tuple[np.ndarray, float]:
    """Project one world point and return its pixel coordinates and depth."""

    point = _array(point_world_xyz_m, name="point_world_xyz_m", shape=(3,))
    matrix = _camera_matrix(camera_matrix_px)
    rotation = _rotation_matrix(rotation_world_to_camera)
    center = _array(
        optical_center_world_xyz_m,
        name="optical_center_world_xyz_m",
        shape=(3,),
    )
    camera_point = rotation @ (point - center)
    homogeneous = matrix @ camera_point
    denominator = float(homogeneous[2])
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("point_world_xyz_m must be in front of the camera")
    return homogeneous[:2] / denominator, float(camera_point[2])


def build_metric_projection(
    point_world_xyz_m: object,
    camera_matrix_px: object,
    rotation_world_to_camera: object,
    optical_center_world_xyz_m: object,
    image_measurement_axis_xy: object = (1.0, 0.0),
) -> MetricProjection:
    """Build the local pixel-to-metre projection for one image direction."""

    image_axis = _array(
        image_measurement_axis_xy,
        name="image_measurement_axis_xy",
        shape=(2,),
    )
    image_axis_norm = float(np.linalg.norm(image_axis))
    if image_axis_norm <= np.finfo(np.float64).eps:
        raise ValueError("image_measurement_axis_xy must be nonzero")
    normalized_image_axis = image_axis / image_axis_norm
    jacobian, depth = projection_jacobian_world(
        point_world_xyz_m,
        camera_matrix_px,
        rotation_world_to_camera,
        optical_center_world_xyz_m,
    )
    reference_pixel, reference_depth = project_world_point(
        point_world_xyz_m,
        camera_matrix_px,
        rotation_world_to_camera,
        optical_center_world_xyz_m,
    )
    if not np.isclose(depth, reference_depth, atol=1.0e-12):
        raise RuntimeError("Projection depth mismatch")
    sensitivity = normalized_image_axis @ jacobian[:, :2]
    norm = float(np.linalg.norm(sensitivity))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("image measurement axis has zero horizontal sensitivity")
    return MetricProjection(
        sensitivity_px_per_m_xy=sensitivity,
        scale_m_per_px=1.0 / norm,
        measurement_axis_world_xy=sensitivity / norm,
        image_measurement_axis_xy=normalized_image_axis,
        reference_pixel_xy=reference_pixel,
        depth_m=depth,
    )


def project_relative_motion(
    relative_xy_px: object,
    projection: MetricProjection,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``(..., 2)`` pixel vectors and convert the scalar to metres."""

    values = np.asarray(relative_xy_px, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("relative_xy_px must have shape (..., 2)")
    if np.any(np.isinf(values)):
        raise ValueError("relative_xy_px must not contain infinite values")
    displacement_px = values @ projection.image_measurement_axis_xy
    displacement_m = displacement_px * projection.scale_m_per_px
    return displacement_px, displacement_m


def projection_to_dict(projection: MetricProjection) -> dict[str, Any]:
    """Return JSON-compatible projection parameters."""

    return {
        "sensitivity_px_per_m_xy": (
            projection.sensitivity_px_per_m_xy.tolist()
        ),
        "scale_m_per_px": projection.scale_m_per_px,
        "measurement_axis_world_xy": (
            projection.measurement_axis_world_xy.tolist()
        ),
        "image_measurement_axis_xy": (
            projection.image_measurement_axis_xy.tolist()
        ),
        "reference_pixel_xy": projection.reference_pixel_xy.tolist(),
        "depth_m": projection.depth_m,
    }
