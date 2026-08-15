import hashlib
import json

import pytest

from light_pole_tracking.config import load_config


def base_config() -> dict:
    return {
        "video_path": "video.mp4",
        "output_directory": "run",
        "target": {
            "mode": "manual",
            "track_path": "track.csv",
            "roi_xyxy": [1, 2, 10, 20],
            "anchor_frame": 0,
        },
        "ground": {
            "reference_frame": 0,
            "rois_xywh": [[2, 3, 5, 6], [9, 3, 5, 6]],
            "search_radius_px": [4, 5],
            "min_correlation": 0.5,
            "consensus_threshold_px": 2.0,
            "min_inliers": 2,
        },
        "relative": {"zero_frame_start": 0, "zero_frame_end": 4},
        "metric": None,
        "qa": {"enabled": False, "output_path": "run/qa.mp4"},
    }


def metric_config(tmp_path) -> dict:
    calibration = {
        "schema_version": "light_pole_metric_calibration_v1",
        "validation_status": "synthetic_test",
        "physical_use_allowed": False,
        "coordinate_contract": {
            "coordinate_frame_id": "local_xy",
            "motion_subspace": "horizontal_world_xy",
            "ground_plane_z_m": 0.0,
            "ground_reference_frame": 0,
        },
        "image_geometry": {
            "model": "undistorted_pinhole",
            "width_px": 100,
            "height_px": 70,
            "remap_sha256": None,
        },
        "material_point_world_xyz_m": [0.0, 0.0, 10.0],
        "camera_matrix_px": [
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        "rotation_world_to_camera": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "optical_center_world_xyz_m": [0.0, 0.0, 0.0],
        "image_measurement_axis_xy": [1.0, 0.0],
        "physical_reference": {
            "description": "Known tracked-point height",
            "value_m": 10.0,
            "uncertainty_m": None,
            "source": "Synthetic test",
            "relationship": "world_z_above_ground_plane",
        },
        "minimum_horizontal_sensitivity_px_per_m": 1.0,
        "maximum_rest_reprojection_error_px": 2.0,
    }
    path = tmp_path / "metric-calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"calibration_path": path.name, "calibration_sha256": digest}


def test_config_resolves_paths_from_its_directory(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base_config()), encoding="utf-8")
    result = load_config(path)
    assert result.video_path == (tmp_path / "video.mp4").resolve()
    assert result.ground.search_radius_px == (4, 5)
    assert result.target.start_frame == 0
    assert result.target.anchor_target_xy is None
    assert result.target.corrections_path is None
    assert result.target.segments == ()
    assert result.target.segment_overlap_policy == "error"
    assert result.ground.corrections_path is None


def test_config_loads_complete_sam2_segments(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "track_path": None,
            "start_frame": 0,
            "end_frame": 5,
            "anchor_frame": 2,
            "positive_points_xy": [[5.0, 6.0]],
            "segments": [
                {
                    "id": "early",
                    "start_frame": 0,
                    "end_frame": 2,
                    "anchor_frame": 2,
                    "direction": "reverse",
                },
                {
                    "id": "late",
                    "start_frame": 3,
                    "end_frame": 5,
                    "anchor_frame": 3,
                    "direction": "forward",
                    "positive_points_xy": [[6.0, 7.0]],
                },
            ],
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_config(path)

    assert [segment.id for segment in result.target.segments] == ["early", "late"]
    assert result.target.segments[0].positive_points_xy == ((5.0, 6.0),)
    assert result.target.segments[1].positive_points_xy == ((6.0, 7.0),)


def test_segment_prompts_can_replace_legacy_global_anchor_and_prompt(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "track_path": None,
            "start_frame": 0,
            "end_frame": 1,
            "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 1,
                    "anchor_frame": 1,
                    "direction": "reverse",
                    "positive_points_xy": [[5.0, 6.0]],
                }
            ],
        }
    )
    data["target"].pop("anchor_frame")
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "segments-only.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_config(path)

    assert result.target.anchor_frame == 1
    assert result.target.positive_points_xy == ()
    assert result.target.segments[0].positive_points_xy == ((5.0, 6.0),)


def test_config_rejects_missing_sam2_segment_coverage(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "track_path": None,
            "start_frame": 0,
            "end_frame": 5,
            "positive_points_xy": [[5.0, 6.0]],
            "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 2,
                    "anchor_frame": 2,
                    "direction": "reverse",
                },
                {
                    "start_frame": 4,
                    "end_frame": 5,
                    "anchor_frame": 4,
                    "direction": "forward",
                },
            ],
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "segments-gap.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="do not cover processed frame 3"):
        load_config(path)


def test_config_requires_explicit_sam2_overlap_policy(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "track_path": None,
            "start_frame": 0,
            "end_frame": 5,
            "positive_points_xy": [[5.0, 6.0]],
            "segments": [
                {
                    "id": "early",
                    "start_frame": 0,
                    "end_frame": 3,
                    "anchor_frame": 0,
                    "direction": "forward",
                },
                {
                    "id": "late",
                    "start_frame": 2,
                    "end_frame": 5,
                    "anchor_frame": 5,
                    "direction": "reverse",
                },
            ],
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "segments-overlap.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap at frame 2"):
        load_config(path)

    data["target"]["segment_overlap_policy"] = "last"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_config(path)
    assert result.target.segment_overlap_policy == "last"


def test_config_resolves_target_and_ground_corrections_paths(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "positive_points_xy": [[5.0, 6.0]],
            "corrections_path": "corrections/target.csv",
        }
    )
    data["ground"]["corrections_path"] = "corrections/ground.csv"
    data["sam2"] = {
        "checkpoint_path": "models/sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_config(path)

    assert result.target.corrections_path == (
        tmp_path / "corrections" / "target.csv"
    ).resolve()
    assert result.ground.corrections_path == (
        tmp_path / "corrections" / "ground.csv"
    ).resolve()


def test_manual_mode_requires_track_path(tmp_path) -> None:
    data = base_config()
    data["target"]["track_path"] = None
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="track_path"):
        load_config(path)


def test_target_corrections_are_not_allowed_in_manual_mode(tmp_path) -> None:
    data = base_config()
    data["target"]["corrections_path"] = "target-corrections.csv"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="only supported in SAM 2 mode"):
        load_config(path)


def test_config_loads_sam2_anchor_target_inside_roi(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "positive_points_xy": [[5.0, 6.0]],
            "anchor_target_xy": [1.0, 19.999],
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_config(path)

    assert result.target.anchor_target_xy == (1.0, 19.999)


def test_anchor_target_is_not_allowed_in_manual_mode(tmp_path) -> None:
    data = base_config()
    data["target"]["anchor_target_xy"] = [5.0, 6.0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="only supported in SAM 2 mode"):
        load_config(path)


@pytest.mark.parametrize(
    "anchor_target",
    [
        [1.0],
        [1.0, 2.0, 3.0],
        [float("nan"), 6.0],
        [5.0, float("inf")],
    ],
)
def test_anchor_target_requires_two_finite_numbers(
    tmp_path, anchor_target
) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "positive_points_xy": [[5.0, 6.0]],
            "anchor_target_xy": anchor_target,
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="target.anchor_target_xy"):
        load_config(path)


@pytest.mark.parametrize(
    "anchor_target",
    [
        [0.999, 6.0],
        [10.0, 6.0],
        [5.0, 1.999],
        [5.0, 20.0],
    ],
)
def test_anchor_target_must_lie_inside_target_roi(
    tmp_path, anchor_target
) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "positive_points_xy": [[5.0, 6.0]],
            "anchor_target_xy": anchor_target,
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="must lie inside"):
        load_config(path)


def test_validate_inputs_requires_target_corrections_file(tmp_path) -> None:
    data = base_config()
    data["target"].update(
        {
            "mode": "sam2",
            "positive_points_xy": [[5.0, 6.0]],
            "corrections_path": "target-corrections.csv",
        }
    )
    data["sam2"] = {
        "checkpoint_path": "sam2.pt",
        "model_config": "sam2_hiera_l.yaml",
        "device": "cpu",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "video.mp4").write_bytes(b"video")
    (tmp_path / "sam2.pt").write_bytes(b"checkpoint")
    config = load_config(path)

    with pytest.raises(FileNotFoundError, match="Missing target corrections"):
        config.validate_inputs()


def test_validate_inputs_requires_ground_corrections_file(tmp_path) -> None:
    data = base_config()
    data["ground"]["corrections_path"] = "ground-corrections.csv"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "video.mp4").write_bytes(b"video")
    (tmp_path / "track.csv").write_text(
        "frame,x_px,y_px\n", encoding="utf-8"
    )
    config = load_config(path)

    with pytest.raises(FileNotFoundError, match="Missing ground corrections"):
        config.validate_inputs()


def test_qa_output_must_be_inside_run_directory(tmp_path) -> None:
    data = base_config()
    data["qa"] = {"enabled": True, "output_path": "other/qa.mp4"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = load_config(path)
    (tmp_path / "video.mp4").write_bytes(b"video")
    (tmp_path / "track.csv").write_text("frame,x_px,y_px\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside output_directory"):
        config.validate_inputs()


def test_config_loads_optional_metric_projection(tmp_path) -> None:
    data = base_config()
    data["metric"] = metric_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = load_config(path)
    assert config.metric is not None
    assert config.metric.material_point_world_xyz_m == (0.0, 0.0, 10.0)
    assert config.metric.calibration_reference.uncertainty_m is None


def test_metric_geometry_must_match_remap_state(tmp_path) -> None:
    data = base_config()
    data["remap_path"] = "remap.npz"
    data["metric"] = metric_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be combined with remap_path"):
        load_config(path)


def test_metric_calibration_hash_is_required(tmp_path) -> None:
    data = base_config()
    data["metric"] = metric_config(tmp_path)
    data["metric"]["calibration_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_config(path)


def test_physical_reference_is_tied_to_material_point_height(tmp_path) -> None:
    data = base_config()
    data["metric"] = metric_config(tmp_path)
    calibration_path = tmp_path / "metric-calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["physical_reference"]["value_m"] = 9.0
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    data["metric"]["calibration_sha256"] = hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="height above the ground plane"):
        load_config(path)


def test_metric_ground_reference_frame_must_match_run(tmp_path) -> None:
    data = base_config()
    data["metric"] = metric_config(tmp_path)
    data["ground"]["reference_frame"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="ground_reference_frame"):
        load_config(path)
