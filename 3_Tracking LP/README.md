# Light Pole Motion Tracking

Track a light-pole luminaire relative to nearby stationary regions in fixed-camera video.

The package supports Meta SAM 2, manual tracks, reviewed coordinate corrections,
optional lens remapping, and optional conversion from pixels to metres with an
external camera calibration.

## Install

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Install the optional SAM 2 dependency for automatic tracking:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[sam2]"
```

Download the SAM 2 checkpoint separately. Videos and model weights are not included.

## Run

Copy an example and set the video, target points, ground regions, and output directory:

```powershell
Copy-Item examples\configs\sam2.example.json run.json
.\.venv\Scripts\pole-track.exe validate-config --config run.json
.\.venv\Scripts\pole-track.exe run --config run.json
```

Use `.\.venv\Scripts\pole-track.exe --help` for all commands.

## Tracking modes

- **SAM 2:** segments the luminaire and propagates the mask through the video.
- **Reviewed SAM 2:** replaces selected target or ground coordinates after review.
- **Manual:** reads reviewed target coordinates from CSV.

The tracked point is the centre of the luminaire housing. The shaft, arm, cables,
and background are excluded.

## Main outputs

- `relative_motion.csv`: target displacement relative to the ground reference.
- `target_track.csv`: final luminaire coordinates.
- `ground_track.csv`: estimated ground translation.
- `qa_video.mp4`: optional tracking overlay.
- `manifest.json`: run settings and file hashes.

References: [configuration](docs/configuration.md),
[manual corrections](docs/manual-tracking.md),
[output files](docs/output-schema.md),
[metric conversion](docs/metric-projection.md), and
[SAM 2 setup](docs/sam2-setup.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m build
```
