# Metric Projection

Set `metric` to `null` for pixel-only output. To convert one image component to
metres, provide a calibration JSON and its SHA-256:

```json
"metric": {
  "calibration_path": "../calibration/view_1.json",
  "calibration_sha256": "<64 hexadecimal characters>"
}
```

The calibration JSON must define:

- schema version `light_pole_metric_calibration_v1`;
- image size and pinhole geometry;
- camera matrix, pose, and optical centre;
- tracked three-dimensional material point;
- image measurement direction;
- ground plane and tracked-point height;
- minimum sensitivity and maximum rest-point reprojection error;
- exact remap SHA-256 for rectified images.

The run verifies the calibration hash, image geometry, physical-reference
relationship, sensitivity, and rest-point reprojection error. Raw distorted images
are not supported by the metric model; use a validated rectified image instead.

Outputs:

- `projected_motion.csv`: directional displacement in pixels and metres.
- `metric_projection.json`: scale, direction, geometry, hashes, and validation gates.

This is a local one-camera projection under a horizontal-motion assumption. It is
not a two-dimensional trajectory. Two-camera fusion and north/east registration
must be performed separately.

The supplied [calibration](../examples/calibration/metric.synthetic.example.json)
and [run configuration](../examples/configs/metric.synthetic.example.json) are
synthetic examples.
