"""Apply a precomputed OpenCV inverse lens map."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LensRemap:
    """Immutable inverse map from output pixels to source pixels."""

    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        map_x = np.asarray(self.map_x)
        map_y = np.asarray(self.map_y)
        valid = np.asarray(self.valid_mask)
        if map_x.ndim != 2 or map_y.shape != map_x.shape:
            raise ValueError("map_x and map_y must have the same two-dimensional shape")
        if valid.shape != map_x.shape:
            raise ValueError("valid_mask must match the map shape")
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise ValueError("lens maps must contain only finite values")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> LensRemap:
        source = Path(path).resolve()
        digest = sha256_file(source)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError(
                f"Lens-map hash mismatch: expected {expected_sha256}, found {digest}"
            )
        with np.load(source, allow_pickle=False) as archive:
            missing = {"map_x", "map_y"}.difference(archive.files)
            if missing:
                raise ValueError(f"Lens map is missing arrays: {sorted(missing)}")
            map_x = np.asarray(archive["map_x"], dtype=np.float32).copy()
            map_y = np.asarray(archive["map_y"], dtype=np.float32).copy()
            valid = (
                np.asarray(archive["valid_mask"], dtype=bool).copy()
                if "valid_mask" in archive.files
                else np.ones(map_x.shape, dtype=bool)
            )
        return cls(map_x, map_y, valid, source, digest)

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.map_x.shape

    def apply(
        self,
        frame: np.ndarray,
        *,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Remap a frame and set invalid output pixels to black."""

        image = np.asarray(frame)
        if image.ndim not in {2, 3}:
            raise ValueError("frame must be grayscale or multi-channel")
        result = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        result[~self.valid_mask] = 0
        return result

    remap_image = apply


def load_lens_remap(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> LensRemap:
    return LensRemap.load(path, expected_sha256=expected_sha256)
