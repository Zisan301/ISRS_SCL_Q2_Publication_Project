"""Line-rate, AIR, and threshold-qualified capacity calculations.

The functions in this module deliberately keep three different quantities
separate:

* gross/net line rate: a transponder accounting quantity;
* achievable information rate (AIR): a GMI-derived information-theoretic rate;
* threshold-qualified net line rate: channels that meet the configured FEC gate.

Combining these quantities under one generic "capacity" label is a common source
of misleading manuscript claims, so every exported value has an explicit name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapacityResult:
    gross_bps: float
    net_bps: float
    gross_tbps: float
    net_tbps: float


@dataclass(frozen=True)
class CapacityMetrics:
    gross_line_bps: float
    net_line_bps: float
    air_bps: float
    thresholded_net_line_bps: float
    working_channels: int
    working_fraction: float
    minimum_ngmi: float
    mean_ngmi: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "gross_line_tbps": self.gross_line_bps / 1e12,
            "net_line_tbps": self.net_line_bps / 1e12,
            "air_tbps": self.air_bps / 1e12,
            "thresholded_net_capacity_tbps": self.thresholded_net_line_bps / 1e12,
            "working_channels": self.working_channels,
            "working_fraction": self.working_fraction,
            "minimum_ngmi": self.minimum_ngmi,
            "mean_ngmi": self.mean_ngmi,
        }


@dataclass(frozen=True)
class GMIThroughputResult:
    """Throughput summary produced directly from per-channel GMI values.

    This class restores the historical ``throughput_from_gmi`` API expected by
    the repository tests while keeping AIR and hard-FEC line rate separate.
    """

    air_bps: float
    fec_net_bps: float
    soft_fec_net_bps: float
    working_channels: int
    working_fraction: float
    mean_ngmi: float
    minimum_ngmi: float

    @property
    def air_tbps(self) -> float:
        return self.air_bps / 1e12

    @property
    def fec_net_tbps(self) -> float:
        return self.fec_net_bps / 1e12

    @property
    def soft_fec_net_tbps(self) -> float:
        return self.soft_fec_net_bps / 1e12

    def as_dict(self) -> dict[str, float | int]:
        return {
            "air_bps": self.air_bps,
            "fec_net_bps": self.fec_net_bps,
            "soft_fec_net_bps": self.soft_fec_net_bps,
            "air_tbps": self.air_tbps,
            "fec_net_tbps": self.fec_net_tbps,
            "soft_fec_net_tbps": self.soft_fec_net_tbps,
            "working_channels": self.working_channels,
            "working_fraction": self.working_fraction,
            "mean_ngmi": self.mean_ngmi,
            "minimum_ngmi": self.minimum_ngmi,
        }


# Backward-compatible alias used by older scripts/tests.
ThroughputFromGMIResult = GMIThroughputResult


def line_rate_capacity(
    n_channels: int,
    symbol_rate_baud: float,
    bits_per_symbol_per_pol: int,
    fec_overhead_fraction: float,
    polarizations: int = 2,
) -> CapacityResult:
    if n_channels < 0 or symbol_rate_baud <= 0 or bits_per_symbol_per_pol < 1:
        raise ValueError("Invalid capacity inputs")
    if fec_overhead_fraction < 0:
        raise ValueError("FEC overhead cannot be negative")
    if polarizations < 1:
        raise ValueError("polarizations must be positive")

    gross = (
        float(n_channels)
        * float(symbol_rate_baud)
        * int(bits_per_symbol_per_pol)
        * int(polarizations)
    )
    net = gross / (1.0 + float(fec_overhead_fraction))
    return CapacityResult(gross, net, gross / 1e12, net / 1e12)


def achievable_information_rate_bps(
    gmi_bits_per_2d_symbol_per_pol: np.ndarray,
    symbol_rate_baud: float,
    polarizations: int = 2,
    maximum_bits_per_symbol_per_pol: float | None = None,
) -> float:
    """Return AIR from per-polarization 2-D-symbol GMI values.

    GMI is already an achievable coded information rate. It must therefore not
    be divided by ``1 + FEC_overhead`` a second time. The optional upper bound is
    a numerical guard against malformed input, not a post-hoc data correction.
    """

    gmi = np.asarray(gmi_bits_per_2d_symbol_per_pol, dtype=float)
    if gmi.ndim != 1 or gmi.size == 0:
        raise ValueError("GMI must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(gmi)):
        raise ValueError("GMI contains non-finite values")
    if np.any(gmi < -1e-9):
        raise ValueError("GMI cannot be negative")
    if maximum_bits_per_symbol_per_pol is not None:
        maximum = float(maximum_bits_per_symbol_per_pol)
        if np.any(gmi > maximum + 1e-6):
            raise ValueError("GMI exceeds the modulation entropy")
        gmi = np.minimum(gmi, maximum)
    return float(np.sum(np.maximum(gmi, 0.0)) * float(symbol_rate_baud) * int(polarizations))


def thresholded_net_capacity_bps(
    metric: np.ndarray,
    threshold: float,
    symbol_rate_baud: float,
    bits_per_symbol_per_pol: int,
    fec_overhead_fraction: float,
    polarizations: int = 2,
) -> float:
    values = np.asarray(metric, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Threshold metric must be a finite one-dimensional array")
    working = int(np.count_nonzero(values >= float(threshold)))
    return line_rate_capacity(
        working,
        symbol_rate_baud,
        bits_per_symbol_per_pol,
        fec_overhead_fraction,
        polarizations,
    ).net_bps


def throughput_from_gmi(
    gmi_bits_per_2d_symbol_per_pol: np.ndarray,
    *,
    symbol_rate_baud: float,
    bits_per_symbol_per_pol: int,
    fec_overhead_fraction: float,
    ngmi_threshold: float = 0.90,
    soft_transition: float = 0.01,
    polarizations: int = 2,
) -> GMIThroughputResult:
    """Compute AIR plus hard/soft FEC-qualified throughput from GMI.

    The historical tests import this function directly.  It intentionally keeps
    AIR separate from hard-FEC throughput:

    * ``air_bps`` = sum(GMI) * symbol_rate * polarizations.
    * ``fec_net_bps`` = full net line rate only for channels whose NGMI meets
      the configured threshold.
    * ``soft_fec_net_bps`` = a smooth logistic utility around the threshold for
      optimization diagnostics, not a publishable hard-FEC claim.
    """

    gmi = np.asarray(gmi_bits_per_2d_symbol_per_pol, dtype=float)
    if gmi.ndim != 1 or gmi.size == 0:
        raise ValueError("GMI must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(gmi)):
        raise ValueError("GMI contains non-finite values")
    if bits_per_symbol_per_pol <= 0:
        raise ValueError("bits_per_symbol_per_pol must be positive")

    max_bits = float(bits_per_symbol_per_pol)
    gmi = np.clip(gmi, 0.0, max_bits)
    ngmi = gmi / max_bits
    working = ngmi >= float(ngmi_threshold)
    channel_net_bps = line_rate_capacity(
        1,
        symbol_rate_baud,
        bits_per_symbol_per_pol,
        fec_overhead_fraction,
        polarizations,
    ).net_bps
    if soft_transition <= 0:
        soft_weights = working.astype(float)
    else:
        x = np.clip((ngmi - float(ngmi_threshold)) / float(soft_transition), -700.0, 700.0)
        soft_weights = 1.0 / (1.0 + np.exp(-x))

    return GMIThroughputResult(
        air_bps=achievable_information_rate_bps(gmi, symbol_rate_baud, polarizations, max_bits),
        fec_net_bps=float(np.count_nonzero(working) * channel_net_bps),
        soft_fec_net_bps=float(np.sum(soft_weights) * channel_net_bps),
        working_channels=int(np.count_nonzero(working)),
        working_fraction=float(np.mean(working)),
        mean_ngmi=float(np.mean(ngmi)),
        minimum_ngmi=float(np.min(ngmi)),
    )


def summarize_capacity(
    gmi_bits_per_2d_symbol_per_pol: np.ndarray,
    ngmi: np.ndarray,
    ngmi_threshold: float,
    symbol_rate_baud: float,
    bits_per_symbol_per_pol: int,
    fec_overhead_fraction: float,
    polarizations: int = 2,
) -> CapacityMetrics:
    gmi = np.asarray(gmi_bits_per_2d_symbol_per_pol, dtype=float)
    normalized = np.asarray(ngmi, dtype=float)
    if gmi.shape != normalized.shape or gmi.ndim != 1:
        raise ValueError("GMI and NGMI arrays must be one-dimensional and shape matched")
    if not np.all(np.isfinite(gmi)) or not np.all(np.isfinite(normalized)):
        raise ValueError("GMI/NGMI arrays contain non-finite values")

    line = line_rate_capacity(
        gmi.size,
        symbol_rate_baud,
        bits_per_symbol_per_pol,
        fec_overhead_fraction,
        polarizations,
    )
    working = normalized >= float(ngmi_threshold)
    thresholded = line_rate_capacity(
        int(np.count_nonzero(working)),
        symbol_rate_baud,
        bits_per_symbol_per_pol,
        fec_overhead_fraction,
        polarizations,
    ).net_bps
    air = achievable_information_rate_bps(
        gmi,
        symbol_rate_baud,
        polarizations,
        maximum_bits_per_symbol_per_pol=float(bits_per_symbol_per_pol),
    )
    return CapacityMetrics(
        gross_line_bps=line.gross_bps,
        net_line_bps=line.net_bps,
        air_bps=air,
        thresholded_net_line_bps=thresholded,
        working_channels=int(np.count_nonzero(working)),
        working_fraction=float(np.mean(working)),
        minimum_ngmi=float(np.min(normalized)),
        mean_ngmi=float(np.mean(normalized)),
    )
