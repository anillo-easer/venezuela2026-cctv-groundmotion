from __future__ import annotations

import math

import pytest

from light_pole_tracking.corrections import (
    TargetCorrection,
    align_target_to_anchor,
    apply_ground_corrections,
    apply_target_corrections,
    load_ground_corrections,
    load_target_corrections,
)
from light_pole_tracking.ground import GroundTranslation


def test_load_target_corrections_accepts_valid_and_invalidation(tmp_path) -> None:
    path = tmp_path / "target-corrections.csv"
    path.write_text(
        "frame,x_px,y_px,valid,note\n"
        "2,12.5,21.25,true,move centre\n"
        "4,,,false,occluded\n",
        encoding="utf-8",
    )

    rows = load_target_corrections(path, 5, 100, 80, (10, 20, 30, 40))

    assert rows == [
        TargetCorrection(2, 12.5, 21.25, True, "move centre"),
        TargetCorrection(4, None, None, False, "occluded"),
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("0,12,22,false", "require blank"),
        ("0,,22,true", "require x_px and y_px"),
        ("0,nan,22,true", "must be finite"),
        ("0,31,22,true", "outside the ROI"),
        ("5,12,22,true", r"outside \[0, 5\)"),
        (
            "0,12,22,true\n0,13,23,true",
            "must be unique",
        ),
    ],
)
def test_load_target_corrections_rejects_invalid_rows(
    tmp_path, body: str, message: str
) -> None:
    path = tmp_path / "target-corrections.csv"
    path.write_text(
        f"frame,x_px,y_px,valid\n{body}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_target_corrections(path, 5, 100, 80, (10, 20, 30, 40))


def test_load_target_corrections_enforces_exact_schema(tmp_path) -> None:
    path = tmp_path / "target-corrections.csv"
    path.write_text(
        "frame,x_px,y_px,valid,comment\n0,12,22,true,no\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly these columns"):
        load_target_corrections(path, 5, 100, 80, (10, 20, 30, 40))


def test_load_target_corrections_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "target-corrections.csv"
    path.write_text("frame,x_px,y_px,valid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one correction row"):
        load_target_corrections(path, 5, 100, 80, (10, 20, 30, 40))


def test_apply_target_corrections_changes_only_exact_frames() -> None:
    rows = [
        {
            "frame": 0,
            "time_s": 0.0,
            "x_px": 10.0,
            "y_px": 20.0,
            "valid": True,
            "source": "sam2",
            "area_px": 40,
            "reason": "",
        },
        {
            "frame": 1,
            "time_s": 0.04,
            "x_px": 11.0,
            "y_px": 21.0,
            "valid": True,
            "source": "sam2",
            "area_px": 41,
            "reason": "",
        },
        {
            "frame": 2,
            "time_s": 0.08,
            "x_px": 12.0,
            "y_px": 22.0,
            "valid": True,
            "source": "sam2",
            "area_px": 42,
            "reason": "",
        },
    ]
    corrections = [
        TargetCorrection(1, 13.0, 18.0, True, "centroid"),
        TargetCorrection(2, None, None, False, "hidden"),
    ]

    final, audit, summary = apply_target_corrections(rows, corrections)

    assert final[0] == rows[0]
    assert final[1]["time_s"] == rows[1]["time_s"]
    assert final[1]["area_px"] == rows[1]["area_px"]
    assert (final[1]["x_px"], final[1]["y_px"]) == (13.0, 18.0)
    assert final[1]["source"] == "manual_correction"
    assert final[1]["reason"] == "manual_coordinate_override"
    assert final[2]["valid"] is False
    assert math.isnan(final[2]["x_px"])
    assert math.isnan(final[2]["y_px"])
    assert final[2]["reason"] == "manual_invalidation"
    assert audit[0]["original_x_px"] == 11.0
    assert audit[0]["manual_x_px"] == 13.0
    assert audit[0]["final_x_px"] == 13.0
    assert audit[0]["delta_x_px"] == 2.0
    assert audit[0]["delta_y_px"] == -3.0
    assert audit[0]["delta_distance_px"] == pytest.approx(math.sqrt(13.0))
    assert audit[1]["delta_distance_px"] is None
    assert summary == {
        "correction_count": 2,
        "coordinate_override_count": 1,
        "invalidation_count": 1,
        "max_delta_px": pytest.approx(math.sqrt(13.0)),
    }


def test_apply_target_corrections_rejects_missing_frame() -> None:
    with pytest.raises(ValueError, match="absent from target rows"):
        apply_target_corrections(
            [{"frame": 0, "x_px": 1.0, "y_px": 2.0, "valid": True}],
            [TargetCorrection(1, 3.0, 4.0, True)],
        )


def test_align_target_to_anchor_preserves_relative_increments_and_invalid_rows(
) -> None:
    rows = [
        {
            "frame": 0,
            "x_px": 10.0,
            "y_px": 20.0,
            "valid": True,
            "source": "sam2",
            "reason": "",
        },
        {
            "frame": 1,
            "x_px": 13.0,
            "y_px": 24.0,
            "valid": True,
            "source": "sam2",
            "reason": "selected_component",
        },
        {
            "frame": 2,
            "x_px": math.nan,
            "y_px": math.nan,
            "valid": False,
            "source": "sam2",
            "reason": "no_mask",
        },
    ]

    aligned, summary = align_target_to_anchor(
        rows,
        anchor_frame=1,
        anchor_target_xy=(20.0, 30.0),
    )

    assert (aligned[1]["x_px"], aligned[1]["y_px"]) == (20.0, 30.0)
    assert aligned[1]["x_px"] - aligned[0]["x_px"] == pytest.approx(3.0)
    assert aligned[1]["y_px"] - aligned[0]["y_px"] == pytest.approx(4.0)
    assert aligned[0]["source"] == "sam2"
    assert aligned[1]["reason"] == "selected_component"
    assert aligned[2]["valid"] is False
    assert math.isnan(aligned[2]["x_px"])
    assert math.isnan(aligned[2]["y_px"])
    assert aligned[2]["source"] == "sam2"
    assert aligned[2]["reason"] == "no_mask"
    assert math.isnan(rows[2]["x_px"])
    assert summary == {
        "anchor_frame": 1,
        "original_anchor_xy": (13.0, 24.0),
        "target_anchor_xy": (20.0, 30.0),
        "offset_xy": (7.0, 6.0),
        "offset_distance_px": pytest.approx(math.hypot(7.0, 6.0)),
        "aligned_valid_count": 2,
    }


@pytest.mark.parametrize(
    ("anchor_frame", "anchor_target_xy", "message"),
    [
        (3, (1.0, 2.0), "absent"),
        (0, (math.nan, 2.0), "two finite"),
    ],
)
def test_align_target_to_anchor_rejects_missing_anchor_or_nonfinite_target(
    anchor_frame: int,
    anchor_target_xy: tuple[float, float],
    message: str,
) -> None:
    rows = [{"frame": 0, "x_px": 1.0, "y_px": 2.0, "valid": True}]

    with pytest.raises(ValueError, match=message):
        align_target_to_anchor(
            rows,
            anchor_frame=anchor_frame,
            anchor_target_xy=anchor_target_xy,
        )


def test_align_target_to_anchor_rejects_invalid_anchor_row() -> None:
    rows = [
        {
            "frame": 0,
            "x_px": math.nan,
            "y_px": math.nan,
            "valid": False,
        }
    ]

    with pytest.raises(ValueError, match="must be valid with finite coordinates"):
        align_target_to_anchor(
            rows,
            anchor_frame=0,
            anchor_target_xy=(1.0, 2.0),
        )


def test_load_and_apply_ground_corrections(tmp_path) -> None:
    path = tmp_path / "ground-corrections.csv"
    path.write_text(
        "frame,dx_px,dy_px,valid,note\n"
        "1,2.5,-1.25,true,local control\n"
        "2,,,false,car occlusion\n",
        encoding="utf-8",
    )
    corrections = load_ground_corrections(path, frame_count=3)
    original = [
        GroundTranslation(0, 0.0, 0.0, True, 3, 3, 0.99),
        GroundTranslation(1, 1.0, -1.0, True, 2, 3, 0.91),
        GroundTranslation(2, 2.0, -2.0, True, 2, 3, 0.90),
    ]

    final, audit, summary = apply_ground_corrections(original, corrections)

    assert final[0] is original[0]
    assert (final[1].dx, final[1].dy, final[1].valid) == (2.5, -1.25, True)
    assert final[1].inlier_count == 0
    assert final[1].reference_count == 3
    assert math.isnan(final[1].median_correlation)
    assert final[1].reason == "manual_coordinate_override"
    assert final[2].valid is False
    assert math.isnan(final[2].dx)
    assert final[2].reason == "manual_invalidation"
    assert audit[0]["delta_dx_px"] == 1.5
    assert audit[0]["delta_dy_px"] == -0.25
    assert audit[0]["final_source"] == "manual_correction"
    assert audit[1]["delta_distance_px"] is None
    assert summary == {
        "correction_count": 2,
        "coordinate_override_count": 1,
        "invalidation_count": 1,
        "max_delta_px": pytest.approx(math.hypot(1.5, -0.25)),
    }


def test_load_ground_corrections_requires_blank_invalid_coordinates(
    tmp_path,
) -> None:
    path = tmp_path / "ground-corrections.csv"
    path.write_text(
        "frame,dx_px,dy_px,valid\n0,1,,false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require blank"):
        load_ground_corrections(path, frame_count=1)
