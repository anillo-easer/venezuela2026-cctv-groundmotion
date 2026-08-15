# TDP (Traffic Delineator Post) displacement from CCTV footage

Recovers the ground displacement of an object (a Traffic Delineator Post, TDP) filmed by an oblique-view CCTV camera,
using per-frame homography, scale anchoring against a known object, and Monte Carlo
uncertainty. When the same event is filmed by two cameras, a second notebook fuses both
tracked traces with a BLUE (Best Linear Unbiased Estimator) per-frame combination.

## Pipeline

```
01_tdp_displacement.ipynb   (run once per view/target)
        │
        ▼
  output/{View}_{Target}/mc_result_{View}_{Target}.npz
        │
        ▼
02_fusion_BLUE.ipynb          (run once per target, combining 2 views)
        │
        ▼
  output/Fusion_BLUE/fusion_BLUE_target_{Target}.txt
```

- **`01_tdp_displacement.ipynb`** — digitizes ground control points (GCP) and the target on a
  sampled sequence of frames from a single camera view, computes a per-frame homography,
  anchors the scale, and propagates click/reference uncertainty through Monte Carlo to produce
  a filtered displacement trace with per-frame covariance.
- **`02_fusion_BLUE.ipynb`** — reads the `mc_result_*.npz` output of two views (of the same
  target), estimates the time lag between them by velocity cross-correlation, and combines
  both traces frame-by-frame with a covariance-weighted BLUE fusion.

Run `01_tdp_displacement.ipynb` once per (view, target) pair you need (e.g. Vista_1/R,
Vista_2/R), then run `02_fusion_BLUE.ipynb` once per target to fuse the two views.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

`ipympl` is required for the interactive digitizing widgets — if a figure renders as a static
image instead of an interactive one, install it and restart the kernel:

```bash
pip install ipympl
```

## Expected input data layout

Set `BASE_PATH` at the top of `01_tdp_displacement.ipynb` to a folder containing:

```
{BASE_PATH}/
├── Vista_1.mp4, Vista_2.mp4, ...        # one CCTV video per camera view
├── ref_points_v4.gpkg                   # GCP in lat/long (read with geopandas)
└── ref_points_dxf.txt                   # GCP in local (x, y) survey coordinates (m)
```

Everything the notebooks generate is written under `{BASE_PATH}/output/` and
`{BASE_PATH}/{View}_picks/`, so a single `BASE_PATH` edit is enough to point the whole
pipeline at a new dataset. No other cell should need editing besides the parameters cell.

## Workflow inside `01_tdp_displacement.ipynb`

The notebook is meant to be run top to bottom, in order, once per (view, target):

1. **Parameters** — set `BASE_PATH`, `VIEW_NAME`, `TARGET`, the frame window, and the number
   of GCP.
2. **Lens distortion correction** (optional but recommended) — digitize a handful of
   real-world-straight lines once; the fitted correction is cached to
   `<video>_distortion.json` and reused automatically on later runs.
3. **Frame explorer** (optional) — locate frame indices for step 1's parameters.
4. **GCP digitization** on the reference frame.
5. **Real-world GCP coordinates** — reconciles the satellite-derived (`.gpkg`) and
   survey-derived (`.txt`) control points into one local frame.
6. **Sanity-check measurements** and **automatic homography verification** — both must look
   reasonable before digitizing the full sequence.
7. **Per-frame digitization** of the target across the sampled frame range (interactive,
   resumable).
8. **Review/correction pass** (interactive, optional).
9. **Trajectory** — per-frame homography, RMSE per frame, raw trace saved to disk.
10. **Monte Carlo uncertainty** — the notebook's final product:
    `output/{View}_{Target}/mc_result_{View}_{Target}.npz`.

All interactive picking tools are keyboard/mouse driven; each class's docstring in the
notebook lists its exact keybindings.
