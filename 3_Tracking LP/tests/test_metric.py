import numpy as np
import pytest

from light_pole_tracking.metric import (
    MetricProjection,
    build_metric_projection,
    project_relative_motion,
    projection_jacobian_world,
    projection_to_dict,
)


def _camera() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.eye(3)
    center = np.zeros(3)
    point = np.array([1.0, -0.5, 10.0])
    return matrix, rotation, center, point


def test_projection_jacobian_matches_finite_difference() -> None:
    matrix, rotation, center, point = _camera()
    analytic, depth = projection_jacobian_world(
        point,
        matrix,
        rotation,
        center,
    )
    numerical = np.empty((2, 3))
    step = 1.0e-6

    def project(world_point: np.ndarray) -> np.ndarray:
        camera_point = rotation @ (world_point - center)
        homogeneous = matrix @ camera_point
        return homogeneous[:2] / homogeneous[2]

    for column in range(3):
        delta = np.zeros(3)
        delta[column] = step
        numerical[:, column] = (
            project(point + delta) - project(point - delta)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-7, atol=2.0e-7)
    assert depth == pytest.approx(10.0)


def test_build_metric_projection_has_pinhole_scale() -> None:
    matrix, rotation, center, point = _camera()
    projection = build_metric_projection(point, matrix, rotation, center)
    np.testing.assert_allclose(projection.sensitivity_px_per_m_xy, [80.0, 0.0])
    np.testing.assert_allclose(projection.measurement_axis_world_xy, [1.0, 0.0])
    assert projection.scale_m_per_px == pytest.approx(10.0 / 800.0)


def test_arbitrary_image_axis_is_normalized() -> None:
    matrix, rotation, center, point = _camera()
    projection = build_metric_projection(
        point,
        matrix,
        rotation,
        center,
        image_measurement_axis_xy=(1.0, 1.0),
    )
    expected = np.array([80.0, 80.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(projection.sensitivity_px_per_m_xy, expected)
    np.testing.assert_allclose(
        projection.measurement_axis_world_xy,
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    )
    assert projection.scale_m_per_px == pytest.approx(1.0 / 80.0)


def test_project_relative_motion_preserves_nan_samples() -> None:
    projection = MetricProjection(
        sensitivity_px_per_m_xy=np.array([50.0, 0.0]),
        scale_m_per_px=0.02,
        measurement_axis_world_xy=np.array([1.0, 0.0]),
        image_measurement_axis_xy=np.array([1.0, 0.0]),
        reference_pixel_xy=np.array([50.0, 25.0]),
        depth_m=8.0,
    )
    relative = np.array([[2.0, 9.0], [np.nan, np.nan], [-3.0, 4.0]])
    pixels, metres = project_relative_motion(relative, projection)
    np.testing.assert_allclose(pixels[[0, 2]], [2.0, -3.0])
    np.testing.assert_allclose(metres[[0, 2]], [0.04, -0.06])
    assert np.isnan(pixels[1])
    assert np.isnan(metres[1])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("matrix", np.diag([-1.0, 1.0, 1.0]), "positive focal"),
        ("rotation", np.diag([1.0, 1.0, 2.0]), "orthonormal"),
        ("point", np.array([0.0, 0.0, -1.0]), "in front"),
    ],
)
def test_invalid_camera_geometry_is_rejected(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    matrix, rotation, center, point = _camera()
    arguments = {
        "point_world_xyz_m": point,
        "camera_matrix_px": matrix,
        "rotation_world_to_camera": rotation,
        "optical_center_world_xyz_m": center,
    }
    arguments[
        {
            "matrix": "camera_matrix_px",
            "rotation": "rotation_world_to_camera",
            "point": "point_world_xyz_m",
        }[field]
    ] = value
    with pytest.raises(ValueError, match=message):
        projection_jacobian_world(**arguments)


def test_zero_image_axis_and_infinite_motion_are_rejected() -> None:
    matrix, rotation, center, point = _camera()
    with pytest.raises(ValueError, match="must be nonzero"):
        build_metric_projection(
            point,
            matrix,
            rotation,
            center,
            image_measurement_axis_xy=(0.0, 0.0),
        )
    projection = build_metric_projection(point, matrix, rotation, center)
    with pytest.raises(ValueError, match="infinite"):
        project_relative_motion([[np.inf, 0.0]], projection)


def test_projection_to_dict_is_json_compatible() -> None:
    matrix, rotation, center, point = _camera()
    result = projection_to_dict(
        build_metric_projection(point, matrix, rotation, center)
    )
    assert result == {
        "sensitivity_px_per_m_xy": [80.0, 0.0],
        "scale_m_per_px": 0.0125,
        "measurement_axis_world_xy": [1.0, 0.0],
        "image_measurement_axis_xy": [1.0, 0.0],
        "reference_pixel_xy": [720.0, 320.0],
        "depth_m": 10.0,
    }
