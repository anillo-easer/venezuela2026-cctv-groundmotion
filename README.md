# Caraballeda CCTV Ground-Motion Reconstruction

Data and code supporting:

> Bustos, J., N. Girmay, M. Bravo-Haro, R. Rojas, D. Montemayor, P. Heresi, E. Miranda, F. Pinto, G. Montalva, N. Bastías, J. Ojeda, F. Paz, F. Villena, and A. Villegas. (2026) **Ground motion intensity in Caraballeda during the Mw 7.5 June 24, 2026, Venezuela earthquake estimated from CCTV video.** *The Seismic Record* 6(4):411–422. https://doi.org/10.1785/0320260045.

## Summary

The Mw 7.5 June 24, 2026, Venezuela earthquake struck Caraballeda without any nearby strong-motion recordings. This study reconstructs the critical near-fault velocity pulse at the site from two fixed CCTV cameras that captured the event, by (1) tracking the frame-by-frame sliding response of a roadside traffic delineator post (TDP), (2) inverting that sliding history for the bidirectional ground acceleration using a rigid-block Coulomb-friction model constrained by in-situ friction tests, and (3) validating the reconstructed pulse against the independently tracked oscillation of a nearby light pole (LP). The resulting motion — PGV ≈ 202 cm/s, incremental velocity ≈ 366 cm/s in the north–south direction — is compared against NGA-West2 ground-motion models (adjusted for rupture directivity and directionality) and against the Venezuelan seismic code (COVENIN 1756-1:2019).

This repository contains everything referenced in the paper's *Data and code availability* statement: "*The original closed‐circuit television recordings, the reconstructed ground‐motion acceleration and velocity histories, the results of the friction‐coefficient tests, and the scripts used to track the traffic delineator post and light pole are available on GitHub.*"

## Repository structure

```
.
├── 1_Videos/                CCTV footage from cameras C1 and C2
├── 2_Tracking TDP/          Notebooks for digitizing and reconstructing TDP sliding motion
├── 3_Tracking LP/           Python package for tracking light-pole oscillation
├── 4_Friction tests/        In-situ friction coefficient measurements (TDP base vs. pavement)
└── 5_Reconstructed pulses/  Final reconstructed NS/EW acceleration and velocity time histories
```

Each folder corresponds to a stage of the workflow described in the manuscript's *Video Recordings and Field Measurements* and *Tracking and Pulse Reconstruction* sections, roughly in processing order: raw footage → object tracking → friction characterization → inverted ground-motion pulse.

### `1_Videos/`

Raw CCTV recordings from the two fixed cameras (Dahua DH-IPC-HFW2631TP-ZA5-S2, 1920×1080, mounted at 4.02 m on the La Guaira Bowling building) that captured the earthquake, referenced throughout the paper as C1 and C2 (see Figures 1d and 2).

| File | Description |
|---|---|
| `C1.mp4` | Footage from camera C1 |
| `C2.mp4` | Footage from camera C2 (used for LP tracking; only its transverse direction is approximately north–south) |

Video courtesy of José Elías Naassaneh, owner of the La Guaira Bowling building.

### `2_Tracking TDP/`

Jupyter notebooks used to digitize the traffic delineator post frame by frame in each camera view and to fuse the two resulting trajectories into a single displacement history (Figures 1d, 2, and the analyst traces in Figure 4c). See the folder's own [README](2_Tracking%20TDP/README.md) for the full pipeline, notebook-by-notebook workflow, and expected input data layout.

- `notebooks/01_tdp_displacement.ipynb` — per-view digitization, homography, and Monte Carlo uncertainty propagation.
- `notebooks/02_fusion_BLUE.ipynb` — best-linear-unbiased-estimate fusion of the two camera views.
- `ref_pionts_v4.gpkg`, `ref_points_dxf.txt` — ground control points (P1–P7) in geographic and local survey coordinates, respectively.
- `TDP_Displacement_Manual.pdf` — step-by-step digitizing manual for analysts.
- `requirements.txt` — Python dependencies.

### `3_Tracking LP/`

Standalone Python package (`light-pole-motion-tracking`) used to track the light-pole luminaire's oscillation relative to the ground (Figure 4d), supporting SAM 2 automatic segmentation, manual digitization, and reviewed corrections. See the package's own [README](3_Tracking%20LP/README.md) for installation, CLI usage, tracking modes, and output schema, and [`THIRD_PARTY_NOTICES.md`](3_Tracking%20LP/THIRD_PARTY_NOTICES.md) for third-party licensing (Meta SAM 2, OpenCV, NumPy).

### `4_Friction tests/`

Results of the 15 in-situ friction tests between the TDP base and the pavement, performed with a calibrated digital dynamometer during the July 28, 2026 field campaign (Figure 3). These constrain the Coulomb friction coefficient (μ = 0.42 ± 0.03) used in the sliding-block inversion.

`FrictionTests.csv` columns:

| Column | Description |
|---|---|
| `TestID` | Test number (1–15) |
| `Velocity [m/s]` | Sliding velocity during the test |
| `Force_dynamic [kgf]` | Measured dynamic friction force |
| `Force_static [kgf]` | Measured static friction force |

### `5_Reconstructed pulses/`

The final reconstructed ground-motion time histories obtained by inverting the TDP sliding response (Figure 4a–b), given for the north–south (NS) and east–west (EW) components and for three friction coefficients spanning the estimated uncertainty range (μ = 0.39, 0.42, 0.45).

| File | Description |
|---|---|
| `Acceleration_g.csv` | Ground acceleration time histories, in g |
| `Velocity_cm_s.csv` | Ground velocity time histories, in cm/s |

Both files share the same column layout:

| Column | Description |
|---|---|
| `t_s` | Time, in seconds, referenced to the first frame of the C2 recording (absolute start time ≈ 10:05:08.84 UTC on 2026-06-24, ±1 s, cameras not GPS-synchronized) |
| `*NS_mu0.39_*`, `*NS_mu0.42_*`, `*NS_mu0.45_*` | North–south component for μ = 0.39, 0.42, and 0.45 |
| `*EW_mu0.39_*`, `*EW_mu0.42_*`, `*EW_mu0.45_*` | East–west component for μ = 0.39, 0.42, and 0.45 |

μ = 0.42 is the mean reconstruction reported in the paper; μ = 0.39 and 0.45 bound the friction uncertainty (±0.03).

## Reproducing the analysis

1. Track the TDP: follow [`2_Tracking TDP/README.md`](2_Tracking%20TDP/README.md) to digitize each camera view and fuse the two traces.
2. Track the LP: follow [`3_Tracking LP/README.md`](3_Tracking%20LP/README.md) to obtain the light pole's relative motion.
3. Combine the fused TDP trajectory with the friction test results in `4_Friction tests/` to invert for the ground-motion pulse (sliding-block model, Section *Tracking and Pulse Reconstruction* of the paper).
4. Validate against the LP response, and compare the resulting pulse — provided directly in `5_Reconstructed pulses/` — with ground-motion models and the Venezuelan seismic code, as described in the paper's *Comparison with Ground-Motion Models and Seismic Codes* section.

## Citation

If you use this code or data, please cite the paper:

> Bustos, J., N. Girmay, M. Bravo-Haro, R. Rojas, D. Montemayor, P. Heresi, E. Miranda, F. Pinto, G. Montalva, N. Bastías, J. Ojeda, F. Paz, F. Villena, and A. Villegas (2026). Ground motion intensity in Caraballeda during the Mw 7.5 June 24, 2026, Venezuela earthquake estimated from CCTV video, *The Seismic Record* 6(4):411–422. https://doi.org/10.1785/0320260045.

## Acknowledgments

The post-earthquake field campaign was funded by the John A. Blume Earthquake Engineering Center and Universidad de Chile. The authors thank José Elías Naassaneh for providing the CCTV footage and site access, and Alexandra López, Aitor Munarriz, Ana Marcano, and Martín Arnal for assistance with the in-situ friction tests. This work was also supported by ANID through FONDECYT Iniciación en Investigación Grant No. 11260870, FONDECYT Regular Grant No. 1261147, and ANILLO EASER (Evolution Assessment of Seismic Risk) Project under Grant No. ACT240044.

## License

- **Code** (`2_Tracking TDP/`, `3_Tracking LP/`): released under the [MIT License](LICENSE).
- **Data** (`1_Videos/`, `4_Friction tests/`, `5_Reconstructed pulses/`): released under [CC BY 4.0](DATA_LICENSE), requiring attribution to the authors.

See `LICENSE` and `DATA_LICENSE` for full terms. Third-party components used by `3_Tracking LP/` (Meta SAM 2, OpenCV, NumPy) retain their own licenses — see [`THIRD_PARTY_NOTICES.md`](3_Tracking%20LP/THIRD_PARTY_NOTICES.md).

## Contact

For questions about this repository, please open an issue or contact the corresponding author: Pablo Heresi (pheresi@uchile.cl).
