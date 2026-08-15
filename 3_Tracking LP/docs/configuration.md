# Configuration

The pipeline reads one JSON file. Relative paths start from that file's directory.

## Top level

| Field | Required | Purpose |
|---|---:|---|
| `video_path` | yes | Input video. |
| `output_directory` | yes | Empty or new run directory. |
| `remap_path` | no | Lens-remap NPZ, or `null`. |
| `target` | yes | Target source and prompt. |
| `ground` | yes | Ground-registration settings. |
| `relative` | yes | Zero-window settings. |
| `sam2` | SAM 2 mode | Model checkpoint and device. |
| `metric` | no | Calibration JSON and SHA-256, or `null`. |
| `qa` | yes | QA-video settings. |

## Target

| Field | Purpose |
|---|---|
| `mode` | `sam2` or `manual`. |
| `track_path` | Manual target CSV. |
| `roi_xyxy` | Target ROI `[x0, y0, x1, y1]`; upper bounds are exclusive. |
| `start_frame`, `end_frame` | Inclusive processed range. |
| `anchor_frame` | Prompt frame for a single SAM 2 propagation. |
| `positive_points_xy` | Full-frame points inside the target. |
| `negative_points_xy` | Full-frame points outside the target. |
| `minimum_area_px` | Smallest accepted mask. |
| `maximum_step_px` | Largest accepted frame-to-frame centre movement, or `null`. |
| `anchor_target_xy` | Reviewed material point at `anchor_frame`, or `null`. |
| `corrections_path` | Target-correction CSV, or `null`. |
| `segments` | Optional independently prompted SAM 2 intervals. |
| `segment_overlap_policy` | `error`, `first`, or `last`. |

Use a tight ROI that contains the luminaire throughout the processed range. All
points use full-frame coordinates.

Each segment defines `start_frame`, `end_frame`, `anchor_frame`, and `direction`.
A forward segment is anchored at its first frame; a reverse segment at its last;
a bidirectional segment anywhere inside its interval. Segments must cover every
processed frame. Overlaps are rejected unless `first` or `last` is selected.

## Ground

| Field | Purpose |
|---|---|
| `reference_frame` | Fixed ground reference frame. |
| `rois_xywh` | Rigid regions `[x, y, width, height]`. |
| `search_radius_px` | Horizontal and vertical search radii `[x, y]`. |
| `min_correlation` | Minimum template correlation. |
| `consensus_threshold_px` | Maximum distance from the robust consensus. |
| `min_inliers` | Minimum agreeing regions. |
| `corrections_path` | Ground-correction CSV, or `null`. |

Choose stationary, textured regions near the pole. Avoid vehicles, vegetation,
reflections, and moving shadows.

## Relative motion

`zero_frame_start` and `zero_frame_end` define an inclusive interval. The median
valid ground-stabilized target position in that interval is set to zero.

## Corrections

Target corrections use:

```text
frame,x_px,y_px,valid,note
```

Ground corrections use:

```text
frame,dx_px,dy_px,valid,note
```

The `note` column is optional. Valid rows require two finite coordinates. Invalid
rows require empty coordinates. Corrections must use the run's raw or rectified
geometry. See [manual tracking and corrections](manual-tracking.md).

## SAM 2

| Field | Purpose |
|---|---|
| `checkpoint_path` | Local SAM 2 checkpoint. |
| `model_config` | SAM 2 model configuration. |
| `device` | Torch device, such as `cuda` or `cpu`. |

## QA

Set `enabled` to `true` to write an overlay video. `output_path` must be directly
inside `output_directory`.

## Metric projection

Set `metric` to `null` for pixels only. Otherwise provide `calibration_path` and
`calibration_sha256`. See [metric projection](metric-projection.md).

## Lens remap

The optional NPZ contains finite, equally sized `map_x` and `map_y` arrays using
the OpenCV inverse-remap convention.
