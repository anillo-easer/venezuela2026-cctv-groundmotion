from __future__ import annotations

import csv
import hashlib
import json

import cv2
import numpy as np
import pytest

from light_pole_tracking.config import load_config
from light_pole_tracking.ground import GroundTranslation
from light_pole_tracking.pipeline import run_pipeline
from light_pole_tracking.records import read_relative_csv


def _write_video(path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def test_manual_pipeline_recovers_known_relative_motion(tmp_path) -> None:
    rng = np.random.default_rng(7)
    reference = np.zeros((70, 100, 3), dtype=np.uint8)
    rois = [(8, 8, 14, 14), (34, 10, 14, 14), (60, 8, 14, 14)]
    for x, y, width, height in rois:
        reference[y : y + height, x : x + width] = rng.integers(
            0,
            256,
            size=(height, width, 3),
            dtype=np.uint8,
        )
    ground_dx = [0.0, 1.0, 2.0]
    frames = [
        cv2.warpAffine(
            reference,
            np.float32([[1, 0, dx], [0, 1, 0]]),
            (100, 70),
        )
        for dx in ground_dx
    ]
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames, 10.0)

    target = tmp_path / "target.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "x_px", "y_px", "valid"])
        writer.writerow([0, 80.0, 40.0, True])
        writer.writerow([1, 81.0, 40.0, True])
        writer.writerow([2, 85.0, 40.0, True])

    config_path = tmp_path / "config.json"
    calibration_path = tmp_path / "metric-calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": "light_pole_metric_calibration_v1",
                "validation_status": "synthetic_test",
                "physical_use_allowed": False,
                "coordinate_contract": {
                    "coordinate_frame_id": "synthetic_local_xy",
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
                "material_point_world_xyz_m": [3.0, 0.5, 10.0],
                "camera_matrix_px": [
                    [100.0, 0.0, 50.0],
                    [0.0, 100.0, 35.0],
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
                "maximum_rest_reprojection_error_px": 1.0,
            }
        ),
        encoding="utf-8",
    )
    calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    config_path.write_text(
        json.dumps(
            {
                "video_path": video.name,
                "output_directory": "run",
                "remap_path": None,
                "target": {
                    "mode": "manual",
                    "track_path": target.name,
                    "roi_xyxy": [75, 30, 95, 55],
                    "anchor_frame": 0,
                },
                "ground": {
                    "reference_frame": 0,
                    "rois_xywh": rois,
                    "search_radius_px": [4, 4],
                    "min_correlation": 0.7,
                    "consensus_threshold_px": 0.5,
                    "min_inliers": 3,
                },
                "relative": {"zero_frame_start": 0, "zero_frame_end": 1},
                "metric": {
                    "calibration_path": calibration_path.name,
                    "calibration_sha256": calibration_sha256,
                },
                "qa": {"enabled": False, "output_path": "run/qa.mp4"},
            }
        ),
        encoding="utf-8",
    )
    result = run_pipeline(load_config(config_path))
    relative = read_relative_csv(result.relative_motion)

    assert all(row.valid for row in relative)
    np.testing.assert_allclose(
        [row.relative_x_px for row in relative],
        [0.0, 0.0, 3.0],
        atol=0.25,
    )
    assert result.arrays.is_file()
    assert result.manifest.is_file()
    assert result.projected_motion is not None
    assert result.metric_projection is not None
    with result.projected_motion.open(newline="", encoding="utf-8") as stream:
        projected = list(csv.DictReader(stream))
    np.testing.assert_allclose(
        [float(row["projected_displacement_m"]) for row in projected],
        [0.0, 0.0, 0.3],
        atol=0.025,
    )
    metric = json.loads(result.metric_projection.read_text(encoding="utf-8"))
    assert metric["projection"]["scale_m_per_px"] == pytest.approx(0.1)
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["metadata"]["metric_projection_enabled"] is True
    assert manifest["metadata"]["metric_scale_m_per_px"] == pytest.approx(0.1)
    assert manifest["metadata"]["metric_physical_use_allowed"] is False
    assert metric["physical_use_allowed"] is False
    assert metric["validation_gates"]["rest_reprojection_error_px"] < 1.0
    assert all(row["physical_use_allowed"] == "False" for row in projected)
    assert all(
        row["metric_validation_status"] == "synthetic_test"
        for row in projected
    )
    assert manifest["status"] == "completed_with_conditional_metric_projection"
    assert manifest["metadata"]["metric_gates_passed"] is True
    with np.load(result.arrays, allow_pickle=False) as arrays:
        np.testing.assert_allclose(
            arrays["projected_displacement_m"],
            [0.0, 0.0, 0.3],
            atol=0.025,
        )
        assert arrays["metric_validation_status"].item() == "synthetic_test"
        assert arrays["metric_physical_use_allowed"].item() is False

    with pytest.raises(FileExistsError, match="not empty"):
        run_pipeline(load_config(config_path))


def test_sam2_reviewed_pipeline_preserves_models_and_uses_corrections(
    tmp_path, monkeypatch
) -> None:
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(3)]
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames, 10.0)
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    target_corrections = tmp_path / "target-corrections.csv"
    target_corrections.write_text(
        "frame,x_px,y_px,valid,note\n"
        "1,27.0,10.0,true,reviewed centre\n",
        encoding="utf-8",
    )
    ground_corrections = tmp_path / "ground-corrections.csv"
    ground_corrections.write_text(
        "frame,dx_px,dy_px,valid,note\n"
        "1,3.0,0.5,true,reviewed ground\n",
        encoding="utf-8",
    )

    def fake_sam2(**kwargs):
        output = kwargs["output_directory"] / "target_track.csv"
        start = kwargs.get("start_frame", 0)
        stop = kwargs.get("end_frame")
        stop = 2 if stop is None else stop
        rows = [
            {
                "frame": frame,
                "time_s": frame / 10.0,
                "x_px": 20.0 + frame,
                "y_px": 10.0,
                "valid": True,
                "source": "sam2",
                "area_px": 20,
                "reason": "",
            }
            for frame in range(start, stop + 1)
        ]
        from light_pole_tracking.records import write_csv_records

        write_csv_records(output, rows)
        return output

    def fake_ground(*args, **kwargs):
        return [
            GroundTranslation(
                frame_index=frame,
                dx=float(frame),
                dy=0.0,
                valid=True,
                inlier_count=3,
                reference_count=3,
                median_correlation=0.9,
            )
            for frame in range(3)
        ]

    monkeypatch.setattr(
        "light_pole_tracking.pipeline.track_luminaire_with_sam2", fake_sam2
    )
    monkeypatch.setattr(
        "light_pole_tracking.pipeline.track_ground_translations", fake_ground
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_path": video.name,
                "output_directory": "run",
                "remap_path": None,
                "target": {
                    "mode": "sam2",
                    "track_path": None,
                    "roi_xyxy": [0, 0, 40, 30],
                    "anchor_frame": 0,
                    "anchor_target_xy": [22.0, 10.0],
                    "positive_points_xy": [[20.0, 10.0]],
                    "negative_points_xy": [],
                    "corrections_path": target_corrections.name,
                },
                "ground": {
                    "reference_frame": 0,
                    "rois_xywh": [[1, 1, 5, 5]],
                    "search_radius_px": [2, 2],
                    "min_correlation": 0.5,
                    "consensus_threshold_px": 1.0,
                    "min_inliers": 1,
                    "corrections_path": ground_corrections.name,
                },
                "relative": {"zero_frame_start": 0, "zero_frame_end": 0},
                "metric": None,
                "sam2": {
                    "checkpoint_path": checkpoint.name,
                    "model_config": "synthetic.yaml",
                    "device": "cpu",
                },
                "qa": {"enabled": False, "output_path": "run/qa.mp4"},
            }
        ),
        encoding="utf-8",
    )

    result = run_pipeline(load_config(config_path))

    assert result.target_track_sam2 is not None
    assert result.target_track_sam2_aligned is not None
    assert result.target_corrections_applied is not None
    assert result.ground_track_model is not None
    assert result.ground_corrections_applied is not None
    with result.target_track_sam2.open(newline="", encoding="utf-8") as stream:
        original_target = list(csv.DictReader(stream))
    with result.target_track.open(newline="", encoding="utf-8") as stream:
        final_target = list(csv.DictReader(stream))
    assert float(original_target[1]["x_px"]) == 21.0
    assert float(final_target[1]["x_px"]) == 27.0
    assert final_target[1]["source"] == "manual_correction"
    with result.ground_track_model.open(newline="", encoding="utf-8") as stream:
        original_ground = list(csv.DictReader(stream))
    with result.ground_track.open(newline="", encoding="utf-8") as stream:
        final_ground = list(csv.DictReader(stream))
    assert float(original_ground[1]["dx_px"]) == 1.0
    assert float(final_ground[1]["dx_px"]) == 3.0
    relative = read_relative_csv(result.relative_motion)
    np.testing.assert_allclose(
        [row.relative_x_px for row in relative], [0.0, 2.0, 0.0]
    )
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_manual_corrections"
    assert manifest["metadata"]["target_track_source"] == (
        "sam2_aligned_with_manual_corrections"
    )
    assert manifest["metadata"]["target_anchor_alignment"]["offset_xy"] == [
        2.0,
        0.0,
    ]
    assert manifest["metadata"]["target_corrections"]["correction_count"] == 1
    assert manifest["metadata"]["ground_corrections"]["correction_count"] == 1
    artifact_labels = {row["label"] for row in manifest["artifacts"]}
    assert {
        "target_track_sam2.csv",
        "target_track_sam2_aligned.csv",
        "target_corrections_applied.csv",
        "ground_track_model.csv",
        "ground_corrections_applied.csv",
    } <= artifact_labels

    full_corrections = tmp_path / "target-corrections-full-processed-range.csv"
    full_corrections.write_text(
        "frame,x_px,y_px,valid,note\n"
        "1,27.0,10.0,true,reviewed centre\n"
        "2,28.0,10.0,true,reviewed centre\n",
        encoding="utf-8",
    )
    partial_payload = json.loads(config_path.read_text(encoding="utf-8"))
    partial_payload["output_directory"] = "run-partial"
    partial_payload["target"].update(
        {
            "anchor_frame": 1,
            "anchor_target_xy": [23.0, 10.0],
            "start_frame": 1,
            "end_frame": 2,
            "corrections_path": full_corrections.name,
        }
    )
    partial_payload["relative"] = {
        "zero_frame_start": 1,
        "zero_frame_end": 1,
    }
    partial_payload["qa"]["output_path"] = "run-partial/qa.mp4"
    partial_config = tmp_path / "config-partial.json"
    partial_config.write_text(json.dumps(partial_payload), encoding="utf-8")

    partial = run_pipeline(load_config(partial_config))
    partial_manifest = json.loads(partial.manifest.read_text(encoding="utf-8"))
    assert partial_manifest["metadata"]["target_processed_frame_count"] == 2
    assert partial_manifest["metadata"]["ground_processed_frame_count"] == 3
    assert partial_manifest["metadata"]["target_correction_coverage_fraction"] == 1.0
    assert partial_manifest["metadata"]["target_track_source"] == (
        "reviewed_candidate_replay_with_sam2_evidence"
    )


def test_sam2_anchor_only_writes_canonical_aligned_track(
    tmp_path, monkeypatch
) -> None:
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames, 10.0)
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_sam2(**kwargs):
        output = kwargs["output_directory"] / "target_track.csv"
        from light_pole_tracking.records import write_csv_records

        write_csv_records(
            output,
            [
                {
                    "frame": frame,
                    "time_s": frame / 10.0,
                    "x_px": 10.0 + frame,
                    "y_px": 5.0,
                    "valid": True,
                    "source": "sam2",
                    "area_px": 10,
                    "reason": "",
                }
                for frame in range(2)
            ],
        )
        return output

    monkeypatch.setattr(
        "light_pole_tracking.pipeline.track_luminaire_with_sam2", fake_sam2
    )
    monkeypatch.setattr(
        "light_pole_tracking.pipeline.track_ground_translations",
        lambda *args, **kwargs: [
            GroundTranslation(
                frame_index=frame,
                dx=0.0,
                dy=0.0,
                valid=True,
                inlier_count=1,
                reference_count=1,
                median_correlation=1.0,
            )
            for frame in range(2)
        ],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_path": video.name,
                "output_directory": "run",
                "remap_path": None,
                "target": {
                    "mode": "sam2",
                    "track_path": None,
                    "roi_xyxy": [0, 0, 30, 20],
                    "anchor_frame": 0,
                    "anchor_target_xy": [13.0, 7.0],
                    "positive_points_xy": [[10.0, 5.0]],
                },
                "ground": {
                    "reference_frame": 0,
                    "rois_xywh": [[1, 1, 4, 4]],
                    "search_radius_px": [1, 1],
                    "min_correlation": 0.5,
                    "consensus_threshold_px": 1.0,
                    "min_inliers": 1,
                },
                "relative": {"zero_frame_start": 0, "zero_frame_end": 0},
                "metric": None,
                "sam2": {
                    "checkpoint_path": checkpoint.name,
                    "model_config": "synthetic.yaml",
                    "device": "cpu",
                },
                "qa": {"enabled": False, "output_path": "run/qa.mp4"},
            }
        ),
        encoding="utf-8",
    )

    result = run_pipeline(load_config(config_path))

    assert result.target_track.is_file()
    assert result.target_track_sam2 is not None
    assert result.target_track_sam2_aligned is not None
    with result.target_track.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_allclose(
        [[float(row["x_px"]), float(row["y_px"])] for row in rows],
        [[13.0, 7.0], [14.0, 7.0]],
    )
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["metadata"]["target_track_source"] == (
        "sam2_aligned_to_material_point"
    )
