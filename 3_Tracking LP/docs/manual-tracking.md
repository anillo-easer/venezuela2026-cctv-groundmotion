# Manual Tracking and Corrections

All coordinates use full-frame pixels in the geometry processed by the run. If a
lens remap is enabled, measure coordinates on the rectified frame.

## Manual target track

Set `target.mode` to `manual` and provide `target.track_path`.

Required CSV columns:

```text
frame,x_px,y_px
```

Optional columns:

```text
valid,source
```

Rules:

- `frame` is a unique zero-based frame number.
- `(x_px, y_px)` is the centre of the luminaire housing.
- Invalid rows use `valid=false` and empty coordinates.
- Missing frames are exported as invalid.
- Time is calculated as `frame / fps`.

Example:

```csv
frame,x_px,y_px,valid,source
0,481.25,83.70,true,manual
1,481.41,83.76,true,manual
2,,,false,manual
```

## Review an automatic run

Keep `target.mode` set to `sam2`. Use `target.anchor_target_xy` when the desired
measurement point has a constant offset from the SAM 2 mask centroid.

Use `target.corrections_path` for target replacements:

```text
frame,x_px,y_px,valid,note
```

Use `ground.corrections_path` for ground replacements:

```text
frame,dx_px,dy_px,valid,note
```

Only listed frames are replaced. Corrections are not interpolated or propagated.
Invalid corrections have empty coordinate fields.

The pipeline preserves the automatic tracks and writes an audit CSV for every
replacement. If corrections cover every frame, the final track is the reviewed
coordinate history; the original SAM 2 masks and centroids remain available for
comparison.
