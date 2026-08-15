from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from light_pole_tracking import qa_video
from light_pole_tracking.ground import GroundTranslation
from light_pole_tracking.video import VideoFrame, VideoMetadata


class FakeWriter:
    def __init__(self, fps: float, size: tuple[int, int]) -> None:
        self.fps = fps
        self.size = size
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def test_qa_video_preserves_stream_and_accepts_missing_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = [
        VideoFrame(index, index / 25.0, np.zeros((40, 60, 3), dtype=np.uint8))
        for index in range(3)
    ]
    monkeypatch.setattr(
        qa_video,
        "read_video_metadata",
        lambda _: VideoMetadata(width=60, height=40, fps=25.0, frame_count=3),
    )
    remaps: list[object] = []

    def iter_frames(_path: object, *, remap: object = None):
        remaps.append(remap)
        return iter(frames)

    monkeypatch.setattr(qa_video, "iter_video_frames", iter_frames)
    monkeypatch.setattr(qa_video.cv2, "VideoWriter_fourcc", lambda *_: 0)
    writers: list[FakeWriter] = []

    def make_writer(
        _path: str,
        _fourcc: int,
        fps: float,
        size: tuple[int, int],
    ) -> FakeWriter:
        writer = FakeWriter(fps, size)
        writers.append(writer)
        return writer

    monkeypatch.setattr(qa_video.cv2, "VideoWriter", make_writer)
    ground = {
        0: GroundTranslation(0, 0.0, 0.0, True, 2, 2, 1.0),
        2: GroundTranslation(2, 2.0, 0.0, True, 2, 2, 0.9),
    }
    summary = qa_video.render_tracking_qa_video(
        "input.mp4",
        tmp_path / "qa.mp4",
        target_points={0: (30.0, 20.0), 1: None, 2: (34.0, 21.0)},
        ground_translations=ground,
        ground_reference_rois=[(5, 5, 10, 8)],
        reference_target_point=(30.0, 20.0),
    )

    writer = writers[0]
    assert summary.frame_count == 3
    assert (summary.width, summary.height, summary.fps) == (60, 40, 25.0)
    assert writer.size == (60, 40)
    assert writer.fps == 25.0
    assert len(writer.frames) == 3
    assert all(frame.shape == (40, 60, 3) for frame in writer.frames)
    assert np.any(writer.frames[0])
    assert not np.any(writer.frames[1])
    assert np.any(writer.frames[2])
    assert writer.released
    assert remaps == [None]
