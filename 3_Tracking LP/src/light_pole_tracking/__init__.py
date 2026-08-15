"""Light-pole motion tracking relative to local ground controls."""

from .config import PipelineConfig, load_config
from .corrections import (
    GroundCorrection,
    TargetCorrection,
    align_target_to_anchor,
    apply_ground_corrections,
    apply_target_corrections,
    load_ground_corrections,
    load_target_corrections,
)
from .lens import LensRemap, load_lens_remap
from .manual import load_manual_track, normalize_manual_csv
from .metric import (
    MetricProjection,
    build_metric_projection,
    project_relative_motion,
)
from .records import GroundRecord, RelativeRecord, TargetRecord
from .relative import compute_relative_displacement

__all__ = [
    "GroundRecord",
    "GroundCorrection",
    "LensRemap",
    "MetricProjection",
    "PipelineConfig",
    "RelativeRecord",
    "TargetRecord",
    "TargetCorrection",
    "align_target_to_anchor",
    "apply_ground_corrections",
    "apply_target_corrections",
    "build_metric_projection",
    "compute_relative_displacement",
    "load_config",
    "load_ground_corrections",
    "load_lens_remap",
    "load_manual_track",
    "load_target_corrections",
    "normalize_manual_csv",
    "project_relative_motion",
]

__version__ = "0.3.0"
