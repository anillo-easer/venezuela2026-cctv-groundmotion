import json
from types import SimpleNamespace

from light_pole_tracking.provenance import sha256_file, write_manifest


def test_manifest_does_not_store_absolute_input_path(tmp_path) -> None:
    source = tmp_path / "private" / "video.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        root=tmp_path,
        path=config_path,
        target=SimpleNamespace(mode="manual", roi_xyxy=(0, 0, 1, 1)),
        ground=SimpleNamespace(
            reference_frame=0,
            rois_xywh=((0, 0, 1, 1),),
            search_radius_px=(1, 1),
            min_correlation=0.5,
            consensus_threshold_px=1.0,
            min_inliers=1,
        ),
        relative=SimpleNamespace(zero_frame_start=0, zero_frame_end=0),
    )
    output = tmp_path / "manifest.json"
    write_manifest(
        output,
        config=config,
        inputs={"video": source},
        artifacts={"result.csv": "abc"},
        metadata={"frame_count": 1},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    video = next(item for item in payload["inputs"] if item["label"] == "video")
    assert video["path"].endswith("video.mp4")
    assert str(source.parent) not in output.read_text(encoding="utf-8")
    assert video["sha256"] == sha256_file(source)
