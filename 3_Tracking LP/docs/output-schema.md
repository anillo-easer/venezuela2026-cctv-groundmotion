# Output Files

Frame numbers are zero-based. Time is reported in seconds using the video frame
rate. Invalid coordinates are empty; missing samples are not interpolated.

## Main records

### `target_track.csv`

Final target coordinates used by the pipeline:

```text
frame,time_s,x_px,y_px,valid,source
```

SAM 2 runs also include mask area and rejection reason.

### `ground_track.csv`

Ground translation from the fixed reference frame:

```text
frame,time_s,dx_px,dy_px,valid,inlier_count,reference_count,median_correlation,reason
```

### `relative_motion.csv`

Zeroed target displacement relative to ground:

```text
frame,time_s,relative_x_px,relative_y_px,valid
```

Additional source coordinates and a rejection reason may follow these columns.

### `projected_motion.csv`

Written only when metric projection is enabled:

```text
frame,time_s,projected_displacement_px,projected_displacement_m,valid,metric_validation_status,physical_use_allowed
```

## SAM 2 files

- `target_masks.npz`: masks, ROI, frames, and logits.
- `target_segments.json`: segment intervals, anchors, prompts, and coverage.
- `target_track_sam2.csv`: unmodified mask-centroid track when review is enabled.
- `target_track_sam2_aligned.csv`: optional constant-offset material-point track.

For segmented runs, the mask NPZ also stores the selected segment for every frame.
An assigned segment may still return an empty mask; that frame remains invalid.

## Review files

- `target_corrections_applied.csv`: original, reviewed, and final target values.
- `ground_track_model.csv`: ground-registration result before corrections.
- `ground_corrections_applied.csv`: original, reviewed, and final ground values.

## Other files

- `tracks.npz`: typed arrays for the main numerical records.
- `metric_projection.json`: metric scale, direction, geometry, hashes, and gates.
- `qa_video.mp4`: optional visual overlay.
- `manifest.json`: effective configuration, input hashes, output hashes, versions,
  correction counts, run status, and limitations.

Editing an input after a run invalidates the hashes stored in the manifest.
