"""Physically consistent optical-frequency grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from isrs_scl.constants import C_M_PER_S


@dataclass(frozen=True)
class OpticalGrid:
    frequencies_hz: np.ndarray
    wavelengths_nm: np.ndarray
    bands: np.ndarray
    spacing_hz: float
    mode: str

    def __post_init__(self) -> None:
        n = len(self.frequencies_hz)
        if n < 1 or len(self.wavelengths_nm) != n or len(self.bands) != n:
            raise ValueError("Grid arrays must be non-empty and have equal lengths")
        if n > 1 and not np.allclose(
            np.diff(self.frequencies_hz), self.spacing_hz, rtol=0.0, atol=1.0
        ):
            raise ValueError("Frequencies are not on the requested exact grid")

    @property
    def n_channels(self) -> int:
        return self.frequencies_hz.size

    @property
    def bandwidth_hz(self) -> float:
        return float(self.frequencies_hz[-1] - self.frequencies_hz[0])

    @property
    def center_frequency_hz(self) -> float:
        return float(0.5 * (self.frequencies_hz[0] + self.frequencies_hz[-1]))

    def nearest_index_nm(self, wavelength_nm: float) -> int:
        return int(np.argmin(np.abs(self.wavelengths_nm - wavelength_nm)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "channel": np.arange(self.n_channels),
                "frequency_hz": self.frequencies_hz,
                "frequency_thz": self.frequencies_hz / 1e12,
                "wavelength_nm": self.wavelengths_nm,
                "band": self.bands,
            }
        )


def band_from_wavelength(wavelength_nm: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelength_nm)
    return np.select(
        [wl < 1530.0, wl < 1565.0, wl <= 1625.0],
        ["S", "C", "L"],
        default="OUT",
    ).astype("U3")


def _full_grid(
    lambda_min_nm: float,
    lambda_max_nm: float,
    spacing_hz: float,
) -> np.ndarray:
    # Frequency increases from the longest wavelength (L edge) to the shortest (S edge).
    f_low = C_M_PER_S / (lambda_max_nm * 1e-9)
    f_high = C_M_PER_S / (lambda_min_nm * 1e-9)
    count = int(np.floor((f_high - f_low) / spacing_hz)) + 1
    return f_low + spacing_hz * np.arange(count, dtype=float)


def build_grid(grid_cfg: dict) -> OpticalGrid:
    spacing_hz = float(grid_cfg["spacing_ghz"]) * 1e9
    full = _full_grid(
        float(grid_cfg["lambda_min_nm"]),
        float(grid_cfg["lambda_max_nm"]),
        spacing_hz,
    )
    mode = str(grid_cfg["mode"])
    if mode == "full_scl":
        frequencies = full
    elif mode == "paper_240_subset":
        n = int(grid_cfg["subset_channels"])
        if n > full.size:
            raise ValueError("Requested subset is larger than the full S+C+L grid")
        f_center = C_M_PER_S / (float(grid_cfg["subset_center_nm"]) * 1e-9)
        center_index = int(np.argmin(np.abs(full - f_center)))
        start = max(0, min(full.size - n, center_index - n // 2))
        frequencies = full[start : start + n]
    else:
        raise ValueError(f"Unsupported grid mode {mode!r}")

    wavelengths_nm = C_M_PER_S / frequencies * 1e9
    bands = band_from_wavelength(wavelengths_nm)
    if np.any(bands == "OUT"):
        raise ValueError("Generated grid contains wavelengths outside S+C+L limits")
    return OpticalGrid(frequencies, wavelengths_nm, bands, spacing_hz, mode)
