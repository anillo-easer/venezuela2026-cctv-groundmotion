from __future__ import annotations

import cv2
import numpy as np
import pytest

from light_pole_tracking import video


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 20.0) -> None:
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.cursor = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, key: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: self.frames[0].shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self.frames[0].shape[0],
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
        }
        return float(values[key])

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cursor == len(self.frames):
            return False, None
        frame = self.frames[self.cursor]
        self.cursor += 1
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


def test_metadata_and_frame_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [np.full((4, 5, 3), value, np.uint8) for value in (10, 20)]
    captures: list[FakeCapture] = []

    def make_capture(_: str) -> FakeCapture:
        capture = FakeCapture(frames)
        captures.append(capture)
        return capture

    monkeypatch.setattr(video.cv2, "VideoCapture", make_capture)
    metadata = video.read_video_metadata("input.mp4")
    assert metadata == video.VideoMetadata(5, 4, 20.0, 2)
    assert metadata.duration_s == pytest.approx(0.1)

    decoded = list(video.iter_video_frames("input.mp4"))
    assert [frame.index for frame in decoded] == [0, 1]
    assert [frame.time_s for frame in decoded] == pytest.approx([0.0, 0.05])
    assert np.array_equal(decoded[1].image, frames[1])
    assert all(capture.released for capture in captures)


def test_iteration_applies_identity_remap(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda _: FakeCapture([frame]))
    map_x, map_y = np.meshgrid(
        np.arange(5, dtype=np.float32),
        np.arange(4, dtype=np.float32),
    )
    remap = video.VideoRemap(map_x, map_y)
    decoded = list(video.iter_video_frames("input.mp4", remap=remap))
    assert np.array_equal(decoded[0].image, frame)
    assert np.array_equal(remap.apply(frame), frame)


def test_remap_shape_must_match_frame() -> None:
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    remap = video.VideoRemap(
        np.zeros((3, 5), dtype=np.float32),
        np.zeros((3, 5), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="does not match"):
        video.apply_remap(frame, remap)
