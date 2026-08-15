"""Video metadata and frame iteration utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

PathLike: TypeAlias = str | Path


@dataclass(frozen=True)
class VideoMetadata:
    """Basic properties reported by the video decoder."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        """Return the nominal duration in seconds."""

        return self.frame_count / self.fps


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame with its source index and nominal time."""

    index: int
    time_s: float
    image: NDArray[np.uint8]


@dataclass(frozen=True)
class VideoRemap:
    """OpenCV remap fields in source pixel coordinates."""

    map_x: NDArray[np.floating]
    map_y: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.map_x.ndim != 2 or self.map_y.ndim != 2:
            raise ValueError("Remap arrays must be two-dimensional")
        if self.map_x.shape != self.map_y.shape:
            raise ValueError("Remap arrays must have the same shape")
        if not np.issubdtype(self.map_x.dtype, np.floating):
            raise TypeError("map_x must use a floating-point dtype")
        if not np.issubdtype(self.map_y.dtype, np.floating):
            raise TypeError("map_y must use a floating-point dtype")

    @property
    def shape(self) -> tuple[int, int]:
        return self.map_x.shape

    def apply(
        self,
        frame: NDArray[np.uint8],
        *,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int | tuple[int, int, int] = 0,
    ) -> NDArray[np.uint8]:
        """Apply this remap to one frame."""

        return apply_remap(
            frame,
            self,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
        )


def _open_capture(path: PathLike) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {path}")
    return capture


def _capture_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions are unavailable")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video frame rate is unavailable")
    return VideoMetadata(width, height, fps, frame_count)


def read_video_metadata(path: PathLike) -> VideoMetadata:
    """Read video properties without decoding the full stream."""

    capture = _open_capture(path)
    try:
        return _capture_metadata(capture)
    finally:
        capture.release()


def apply_remap(
    frame: NDArray[np.uint8],
    remap: VideoRemap,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Apply a full-frame remap while preserving the frame dimensions."""

    if frame.ndim not in (2, 3):
        raise ValueError("A frame must have two or three dimensions")
    if remap.shape != frame.shape[:2]:
        raise ValueError(
            f"Remap shape {remap.shape} does not match frame shape "
            f"{frame.shape[:2]}"
        )
    return cv2.remap(
        frame,
        remap.map_x.astype(np.float32, copy=False),
        remap.map_y.astype(np.float32, copy=False),
        interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def iter_video_frames(
    path: PathLike,
    *,
    remap: VideoRemap | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | tuple[int, int, int] = 0,
) -> Iterator[VideoFrame]:
    """Yield every decoded frame, optionally applying one fixed remap."""

    capture = _open_capture(path)
    try:
        metadata = _capture_metadata(capture)
        if remap is not None and remap.shape != (
            metadata.height,
            metadata.width,
        ):
            raise ValueError(
                f"Remap shape {remap.shape} does not match video shape "
                f"{(metadata.height, metadata.width)}"
            )
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None:
                raise OSError(f"Decoder returned an empty frame at {frame_index}")
            if frame.shape[:2] != (metadata.height, metadata.width):
                raise ValueError(
                    f"Frame {frame_index} has shape {frame.shape[:2]}, expected "
                    f"{(metadata.height, metadata.width)}"
                )
            if remap is not None:
                frame = apply_remap(
                    frame,
                    remap,
                    interpolation=interpolation,
                    border_mode=border_mode,
                    border_value=border_value,
                )
            yield VideoFrame(frame_index, frame_index / metadata.fps, frame)
            frame_index += 1
    finally:
        capture.release()
