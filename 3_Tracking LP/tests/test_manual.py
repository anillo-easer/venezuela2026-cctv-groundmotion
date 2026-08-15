from light_pole_tracking.manual import normalize_manual_track


def test_manual_normalization_keeps_missing_frames_explicit(tmp_path) -> None:
    source = tmp_path / "track.csv"
    source.write_text(
        "frame,x_px,y_px,valid\n0,10,20,true\n2,12,22,false\n",
        encoding="utf-8",
    )
    rows = normalize_manual_track(source, fps=25.0, frame_count=3)
    assert [row["valid"] for row in rows] == [True, False, False]
    assert rows[1]["reason"] == "not_observed"
    assert rows[2]["x_px"] != rows[2]["x_px"]
