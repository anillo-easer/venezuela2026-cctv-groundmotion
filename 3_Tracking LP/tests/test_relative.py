import numpy as np

from light_pole_tracking.relative import compute_relative_motion


def test_relative_motion_subtracts_ground_and_zero_median() -> None:
    target = [
        {"frame": i, "time_s": i / 10, "x_px": x, "y_px": 8.0, "valid": True}
        for i, x in enumerate([10.0, 12.0, 17.0])
    ]
    ground = [
        {"frame": i, "time_s": i / 10, "dx_px": dx, "dy_px": 1.0, "valid": True}
        for i, dx in enumerate([0.0, 2.0, 3.0])
    ]
    rows = compute_relative_motion(
        target, ground, zero_frame_start=0, zero_frame_end=1
    )
    np.testing.assert_allclose([row["relative_x_px"] for row in rows], [0, 0, 4])


def test_invalid_ground_remains_missing() -> None:
    target = [{"frame": 0, "time_s": 0.0, "x_px": 1, "y_px": 2, "valid": True}]
    ground = [{"frame": 0, "time_s": 0.0, "dx_px": 0, "dy_px": 0, "valid": False}]
    try:
        compute_relative_motion(target, ground, zero_frame_start=0, zero_frame_end=0)
    except ValueError as error:
        assert "zero interval" in str(error)
    else:
        raise AssertionError("Expected a missing-zero failure")
