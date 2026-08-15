"""Track the compact luminaire with Meta Segment Anything Model 2."""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .target import detections_from_masks
from .video import VideoRemap

if TYPE_CHECKING:
    from .config import Sam2SegmentConfig


def _validate_roi(
    roi_xyxy: Iterable[int], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in roi_xyxy)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Target ROI is outside the video frame")
    return x0, y0, x1, y1


def _write_jpeg(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise OSError(f"Could not encode frame: {path.name}")
    encoded.tofile(path)


def _prepare_frames(
    video_path: Path,
    frame_directory: Path,
    roi_xyxy: Iterable[int],
    start_frame: int,
    end_frame: int | None,
    remap: VideoRemap | None = None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Video metadata is incomplete")
    stop = frame_count - 1 if end_frame is None else int(end_frame)
    if not (0 <= start_frame <= stop < frame_count):
        capture.release()
        raise ValueError("Requested frame range is outside the video")
    x0, y0, x1, y1 = _validate_roi(roi_xyxy, width, height)
    if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
        capture.release()
        raise OSError(f"Could not seek to frame {start_frame}")
    frame_numbers: list[int] = []
    try:
        for frame_number in range(start_frame, stop + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OSError(f"Decode failed at frame {frame_number}")
            actual_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if actual_frame != frame_number:
                raise OSError(
                    f"Decoder returned frame {actual_frame}, expected {frame_number}"
                )
            if remap is not None:
                frame = remap.apply(frame)
            crop = frame[y0:y1, x0:x1]
            _write_jpeg(frame_directory / f"{len(frame_numbers):06d}.jpg", crop)
            frame_numbers.append(frame_number)
    finally:
        capture.release()
    return np.asarray(frame_numbers, dtype=np.int64), fps, (width, height)


def _prompt_arrays(
    positive_points_xy: Iterable[Iterable[float]],
    negative_points_xy: Iterable[Iterable[float]],
    roi_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = [tuple(map(float, point)) for point in positive_points_xy]
    labels = [1] * len(points)
    negative = [tuple(map(float, point)) for point in negative_points_xy]
    points.extend(negative)
    labels.extend([0] * len(negative))
    if not points or not any(labels):
        raise ValueError("At least one positive target point is required")
    x0, y0, x1, y1 = roi_xyxy
    local = np.asarray([[x - x0, y - y0] for x, y in points], dtype=np.float32)
    if np.any(local[:, 0] < 0) or np.any(local[:, 0] >= x1 - x0):
        raise ValueError("A target point is outside the ROI")
    if np.any(local[:, 1] < 0) or np.any(local[:, 1] >= y1 - y0):
        raise ValueError("A target point is outside the ROI")
    return local, np.asarray(labels, dtype=np.int32)


def _collect_direction(
    predictor: Any,
    state: Any,
    anchor_local: int,
    local_start: int,
    local_end: int,
    masks: np.ndarray,
    logits_mean: np.ndarray,
    reverse: bool,
) -> None:
    maximum = (
        anchor_local - local_start + 1 if reverse else local_end - anchor_local + 1
    )
    for frame_index, object_ids, logits in predictor.propagate_in_video(
        state,
        start_frame_idx=anchor_local,
        max_frame_num_to_track=maximum,
        reverse=reverse,
    ):
        frame_index = int(frame_index)
        if not local_start <= frame_index <= local_end:
            continue
        ids = [int(value) for value in object_ids]
        object_index = ids.index(1)
        current = logits[object_index]
        array = current.detach().float().cpu().numpy().squeeze()
        masks[frame_index] = array > 0.0
        logits_mean[frame_index] = float(array.mean())


def _segment_assignment(
    frames: Iterable[int],
    segments: Iterable[Sam2SegmentConfig],
    overlap_policy: str,
) -> np.ndarray:
    """Assign each processed frame to one segment deterministically."""

    frame_array = np.asarray([int(frame) for frame in frames], dtype=np.int64)
    segment_list = tuple(segments)
    if frame_array.ndim != 1 or frame_array.size == 0:
        raise ValueError("Processed frames must be a non-empty sequence")
    if overlap_policy not in {"error", "first", "last"}:
        raise ValueError("Segment overlap policy must be 'error', 'first', or 'last'")
    assignment = np.full(frame_array.size, -1, dtype=np.int32)
    for index, segment in enumerate(segment_list):
        selected = (frame_array >= segment.start_frame) & (
            frame_array <= segment.end_frame
        )
        if overlap_policy == "error" and np.any(selected & (assignment >= 0)):
            frame = int(frame_array[np.flatnonzero(selected & (assignment >= 0))[0]])
            raise ValueError(f"SAM 2 segments overlap at processed frame {frame}")
        if overlap_policy == "first":
            selected &= assignment < 0
        assignment[selected] = index
    missing = np.flatnonzero(assignment < 0)
    if missing.size:
        raise ValueError(
            f"SAM 2 segments do not cover processed frame "
            f"{int(frame_array[missing[0]])}"
        )
    return assignment


def _write_track(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "frame",
        "time_s",
        "x_px",
        "y_px",
        "valid",
        "source",
        "area_px",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def track_luminaire_with_sam2(
    *,
    video_path: Path,
    output_directory: Path,
    checkpoint_path: Path,
    model_config: str,
    device: str,
    roi_xyxy: Iterable[int],
    anchor_frame: int,
    positive_points_xy: Iterable[Iterable[float]],
    negative_points_xy: Iterable[Iterable[float]] = (),
    start_frame: int = 0,
    end_frame: int | None = None,
    minimum_area_px: int = 1,
    maximum_step_px: float | None = None,
    remap: VideoRemap | None = None,
    segments: Iterable[Sam2SegmentConfig] = (),
    segment_overlap_policy: str = "error",
) -> Path:
    """Run one or more SAM 2 propagations and write a fused centre track."""

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Follow the setup guide before using SAM2 mode."
        ) from exc

    video_path = Path(video_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_directory = Path(output_directory).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pole-track-") as temporary:
        frame_directory = Path(temporary)
        frames, fps, frame_size = _prepare_frames(
            video_path,
            frame_directory,
            roi_xyxy,
            start_frame,
            end_frame,
            remap,
        )
        roi = _validate_roi(roi_xyxy, *frame_size)
        predictor = build_sam2_video_predictor(
            model_config,
            str(checkpoint_path),
            device=device,
        )
        height = roi[3] - roi[1]
        width = roi[2] - roi[0]
        masks = np.zeros((len(frames), height, width), dtype=bool)
        logits_mean = np.full(len(frames), np.nan, dtype=np.float32)
        segment_list = tuple(segments)
        selected_segment_index: np.ndarray | None = None
        segment_metadata: list[dict[str, object]] = []

        if not segment_list:
            if anchor_frame not in frames:
                raise ValueError("Anchor frame is outside the processed frame range")
            run_specs = [
                {
                    "id": "single_anchor",
                    "start_frame": int(frames[0]),
                    "end_frame": int(frames[-1]),
                    "anchor_frame": anchor_frame,
                    "direction": "bidirectional",
                    "positive_points_xy": tuple(positive_points_xy),
                    "negative_points_xy": tuple(negative_points_xy),
                }
            ]
        else:
            selected_segment_index = _segment_assignment(
                frames,
                segment_list,
                segment_overlap_policy,
            )
            run_specs = [
                {
                    "id": segment.id,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "anchor_frame": segment.anchor_frame,
                    "direction": segment.direction,
                    "positive_points_xy": segment.positive_points_xy,
                    "negative_points_xy": segment.negative_points_xy,
                }
                for segment in segment_list
            ]

        for segment_index, spec in enumerate(run_specs):
            segment_start = int(spec["start_frame"])
            segment_end = int(spec["end_frame"])
            segment_anchor = int(spec["anchor_frame"])
            direction = str(spec["direction"])
            frame_set = set(int(frame) for frame in frames)
            if (
                not {
                    segment_start,
                    segment_end,
                    segment_anchor,
                }
                <= frame_set
            ):
                raise ValueError("A SAM 2 segment lies outside the processed frames")
            local_start = int(np.flatnonzero(frames == segment_start)[0])
            local_end = int(np.flatnonzero(frames == segment_end)[0])
            anchor_local = int(np.flatnonzero(frames == segment_anchor)[0])
            points, labels = _prompt_arrays(
                spec["positive_points_xy"],
                spec["negative_points_xy"],
                roi,
            )
            segment_masks = np.zeros_like(masks)
            segment_logits = np.full_like(logits_mean, np.nan)
            state = predictor.init_state(
                video_path=str(frame_directory),
                offload_video_to_cpu=True,
                offload_state_to_cpu=device.startswith("cuda"),
                async_loading_frames=False,
            )
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.startswith("cuda")
                else nullcontext()
            )
            try:
                with torch.inference_mode(), autocast:
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=anchor_local,
                        obj_id=1,
                        points=points,
                        labels=labels,
                    )
                    singleton = local_start == local_end
                    if direction in {"forward", "bidirectional"} or singleton:
                        _collect_direction(
                            predictor,
                            state,
                            anchor_local,
                            local_start,
                            local_end,
                            segment_masks,
                            segment_logits,
                            False,
                        )
                    if not singleton and direction in {"reverse", "bidirectional"}:
                        _collect_direction(
                            predictor,
                            state,
                            anchor_local,
                            local_start,
                            local_end,
                            segment_masks,
                            segment_logits,
                            True,
                        )
            finally:
                predictor.reset_state(state)

            if selected_segment_index is None:
                selected = np.ones(len(frames), dtype=bool)
            else:
                selected = selected_segment_index == segment_index
            masks[selected] = segment_masks[selected]
            logits_mean[selected] = segment_logits[selected]
            segment_metadata.append(
                {
                    "id": str(spec["id"]),
                    "start_frame": segment_start,
                    "end_frame": segment_end,
                    "anchor_frame": segment_anchor,
                    "direction": direction,
                    "positive_points_xy": [
                        [float(value) for value in point]
                        for point in spec["positive_points_xy"]
                    ],
                    "negative_points_xy": [
                        [float(value) for value in point]
                        for point in spec["negative_points_xy"]
                    ],
                }
            )

    rows = detections_from_masks(
        frames,
        masks,
        roi,
        fps,
        minimum_area_px,
        maximum_step_px,
    )
    if selected_segment_index is not None:
        for index, row in enumerate(rows):
            source = segment_list[int(selected_segment_index[index])].id
            row["source"] = f"sam2_segment:{source}"
    track_path = output_directory / "target_track.csv"
    _write_track(track_path, rows)
    mask_payload: dict[str, object] = {
        "schema_version": np.asarray(
            "light_pole_target_masks_v2"
            if selected_segment_index is not None
            else "light_pole_target_masks_v1"
        ),
        "frame": frames,
        "roi_xyxy": np.asarray(roi, dtype=np.int64),
        "mask_roi": masks,
        "mean_logit": logits_mean,
    }
    if selected_segment_index is not None:
        mask_nonempty = np.any(masks, axis=(1, 2))
        mask_payload.update(
            {
                "selected_segment_index": selected_segment_index,
                "selected_segment_id": np.asarray(
                    [segment_list[int(index)].id for index in selected_segment_index]
                ),
                "segment_overlap_policy": np.asarray(segment_overlap_policy),
                "segment_provenance_json": np.asarray(
                    json.dumps(segment_metadata, sort_keys=True)
                ),
                "mask_nonempty": mask_nonempty,
            }
        )
    np.savez_compressed(output_directory / "target_masks.npz", **mask_payload)
    if selected_segment_index is not None:
        assigned_counts = np.bincount(
            selected_segment_index,
            minlength=len(segment_list),
        )
        segment_manifest = {
            "schema_version": "light_pole_sam2_segments_v1",
            "overlap_policy": segment_overlap_policy,
            "processed_frame_start": int(frames[0]),
            "processed_frame_end": int(frames[-1]),
            "processed_frame_count": int(len(frames)),
            "assignment_coverage_complete": True,
            "nonempty_mask_count": int(np.count_nonzero(mask_nonempty)),
            "nonempty_mask_coverage_complete": bool(np.all(mask_nonempty)),
            "segments": [
                {
                    **metadata,
                    "selected_frame_count": int(assigned_counts[index]),
                    "selected_nonempty_mask_count": int(
                        np.count_nonzero(
                            mask_nonempty & (selected_segment_index == index)
                        )
                    ),
                    "track_source": f"sam2_segment:{metadata['id']}",
                }
                for index, metadata in enumerate(segment_metadata)
            ],
        }
        (output_directory / "target_segments.json").write_text(
            json.dumps(segment_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return track_path
