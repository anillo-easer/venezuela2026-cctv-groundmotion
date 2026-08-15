from __future__ import annotations

import csv
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

from light_pole_tracking.config import Sam2SegmentConfig
from light_pole_tracking.sam2_tracker import track_luminaire_with_sam2


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakePredictor:
    def init_state(self, *, video_path: str, **kwargs):
        return {
            "frame_count": len(list(Path(video_path).glob("*.jpg"))),
            "anchor": None,
            "target_x": None,
        }

    def add_new_points_or_box(
        self, state, *, frame_idx, obj_id, points, labels
    ) -> None:
        state["anchor"] = int(frame_idx)
        state["target_x"] = int(round(float(points[0, 0])))

    def propagate_in_video(
        self,
        state,
        *,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
    ):
        if reverse:
            stop = max(-1, start_frame_idx - max_frame_num_to_track)
            indices = range(start_frame_idx, stop, -1)
        else:
            stop = min(
                state["frame_count"],
                start_frame_idx + max_frame_num_to_track,
            )
            indices = range(start_frame_idx, stop)
        for frame in indices:
            logits = np.full((10, 10), -1.0, dtype=np.float32)
            logits[4, state["target_x"]] = 1.0
            yield frame, [1], [_FakeTensor(logits)]

    def reset_state(self, state) -> None:
        state.clear()


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (20, 20),
    )
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((20, 20, 3), dtype=np.uint8))
    writer.release()


def _segment(
    identifier: str,
    start: int,
    end: int,
    anchor: int,
    direction: str,
    prompt_x: float,
) -> Sam2SegmentConfig:
    return Sam2SegmentConfig(
        id=identifier,
        start_frame=start,
        end_frame=end,
        anchor_frame=anchor,
        direction=direction,  # type: ignore[arg-type]
        positive_points_xy=((prompt_x, 4.0),),
        negative_points_xy=(),
    )


def test_segmented_tracker_fuses_masks_and_preserves_frame_sources(
    tmp_path, monkeypatch
) -> None:
    torch = types.ModuleType("torch")
    torch.inference_mode = nullcontext
    sam2 = types.ModuleType("sam2")
    build_sam = types.ModuleType("sam2.build_sam")
    build_sam.build_sam2_video_predictor = lambda *args, **kwargs: _FakePredictor()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sam2", sam2)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_sam)

    video = tmp_path / "video.avi"
    checkpoint = tmp_path / "sam2.pt"
    output = tmp_path / "run"
    _write_video(video)
    checkpoint.write_bytes(b"checkpoint")
    path = track_luminaire_with_sam2(
        video_path=video,
        output_directory=output,
        checkpoint_path=checkpoint,
        model_config="synthetic.yaml",
        device="cpu",
        roi_xyxy=(0, 0, 10, 10),
        anchor_frame=0,
        positive_points_xy=((1.0, 4.0),),
        start_frame=0,
        end_frame=5,
        segments=(
            _segment("first", 0, 0, 0, "reverse", 1.0),
            _segment("early", 1, 2, 2, "reverse", 2.0),
            _segment("late", 3, 5, 3, "forward", 7.0),
        ),
    )

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["source"] for row in rows] == [
        "sam2_segment:first",
        "sam2_segment:early",
        "sam2_segment:early",
        "sam2_segment:late",
        "sam2_segment:late",
        "sam2_segment:late",
    ]
    assert float(rows[0]["x_px"]) == 1.0
    np.testing.assert_allclose([float(row["x_px"]) for row in rows[1:3]], 2.0)
    np.testing.assert_allclose([float(row["x_px"]) for row in rows[3:]], 7.0)
    with np.load(output / "target_masks.npz", allow_pickle=False) as masks:
        assert masks["schema_version"].item() == "light_pole_target_masks_v2"
        np.testing.assert_array_equal(
            masks["selected_segment_id"],
            ["first", "early", "early", "late", "late", "late"],
        )
        np.testing.assert_array_equal(masks["mask_nonempty"], np.ones(6, dtype=bool))
    manifest = json.loads((output / "target_segments.json").read_text(encoding="utf-8"))
    assert manifest["assignment_coverage_complete"] is True
    assert manifest["nonempty_mask_count"] == 6
    assert manifest["nonempty_mask_coverage_complete"] is True
    assert manifest["overlap_policy"] == "error"
    assert [segment["selected_frame_count"] for segment in manifest["segments"]] == [
        1,
        2,
        3,
    ]
    assert [
        segment["selected_nonempty_mask_count"] for segment in manifest["segments"]
    ] == [1, 2, 3]
    assert [segment["track_source"] for segment in manifest["segments"]] == [
        "sam2_segment:first",
        "sam2_segment:early",
        "sam2_segment:late",
    ]
