# SAM 2 Setup

SAM 2 is required only when `target.mode` is `sam2`.

```powershell
python -m pip install -e ".[sam2]"
```

Download the `sam2.1_hiera_tiny` checkpoint from the official SAM 2 project and
keep it outside this repository. The tested upstream commit is
`2b90b9f5ceec907a1c18123530e92e794ad901a4`.

```json
"sam2": {
  "checkpoint_path": "C:/models/sam2.1_hiera_tiny.pt",
  "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
  "device": "cuda"
}
```

## Prompt

- Use a tight ROI that contains the luminaire throughout the run.
- Put a positive point inside the luminaire housing.
- Add negative points on the shaft, arm, cables, or background when needed.
- Use several independently anchored segments if one propagation loses the target.

See the [single-segment](../examples/configs/sam2.example.json) and
[multi-segment](../examples/configs/sam2-segments.example.json) examples.

## Review

Enable the QA video and inspect the full sequence. Reject masks that switch objects,
follow the target late, include background, or continue through an occlusion without
visible evidence.
