"""Render compact visual checks for target and ground tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from .ground import GroundTranslation, ROIInput, coerce_rois
from .video import PathLike, VideoRemap, iter_video_frames, read_video_metadata

Point: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class QAStyle:
    """Overlay colors use OpenCV BGR order."""

    target_color: tuple[int, int, int] = (0, 0, 255)
    ground_color: tuple[int, int, int] = (0, 200, 0)
    vector_color: tuple[int, int, int] = (255, 180, 0)
    target_radius: int = 5
    line_thickness: int = 2


@dataclass(frozen=True)
class QAVideoSummary:
    path: Path
    frame_count: int
    width: int
    height: int
    fps: float


def _finite_point(point: Point | None) -> bool:
    return point is not None and bool(np.isfinite(point).all())


def render_tracking_qa_video(
    input_video: PathLike,
    output_video: PathLike,
    *,
    target_points: Mapping[int, Point | None],
    ground_translations: Mapping[int, GroundTranslation],
    ground_reference_rois: Sequence[ROIInput],
    reference_target_point: Point,
    remap: VideoRemap | None = None,
    codec: str = "mp4v",
    style: QAStyle | None = None,
) -> QAVideoSummary:
    """Overlay tracked geometry and write a frame-aligned QA video."""

    if len(codec) != 4:
        raise ValueError("codec must contain four characters")
    if not _finite_point(reference_target_point):
        raise ValueError("reference_target_point must be finite")
    rois = coerce_rois(ground_reference_rois)
    metadata = read_video_metadata(input_video)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Could not open output video: {output_path}")

    draw = style or QAStyle()
    frame_count = 0
    try:
        for video_frame in iter_video_frames(input_video, remap=remap):
            image = video_frame.image.copy()
            point = target_points.get(video_frame.index)
            ground = ground_translations.get(video_frame.index)
            ground_valid = (
                ground is not None
                and ground.valid
                and np.isfinite((ground.dx, ground.dy)).all()
            )
            if ground_valid:
                dx, dy = ground.dx, ground.dy
                for roi in rois:
                    top_left = (
                        int(round(roi.x + dx)),
                        int(round(roi.y + dy)),
                    )
                    bottom_right = (
                        int(round(roi.x + roi.width + dx)),
                        int(round(roi.y + roi.height + dy)),
                    )
                    cv2.rectangle(
                        image,
                        top_left,
                        bottom_right,
                        draw.ground_color,
                        draw.line_thickness,
                        cv2.LINE_AA,
                    )
            if _finite_point(point):
                target_xy = (int(round(point[0])), int(round(point[1])))
                cv2.circle(
                    image,
                    target_xy,
                    draw.target_radius,
                    draw.target_color,
                    draw.line_thickness,
                    cv2.LINE_AA,
                )
                if ground_valid:
                    transported_reference = (
                        int(round(reference_target_point[0] + ground.dx)),
                        int(round(reference_target_point[1] + ground.dy)),
                    )
                    cv2.arrowedLine(
                        image,
                        transported_reference,
                        target_xy,
                        draw.vector_color,
                        draw.line_thickness,
                        cv2.LINE_AA,
                        tipLength=0.2,
                    )
            writer.write(image)
            frame_count += 1
    finally:
        writer.release()

    return QAVideoSummary(
        path=output_path,
        frame_count=frame_count,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
    )
