"""Multi-span link accumulation with explicit physical and calibrated metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from isrs_scl.dsp.metrics import analytical_ber_16qam, gmi_16qam_awgn_from_snr_db, normalized_evm_from_snr, q_factor_db_from_ber
from isrs_scl.fiber.amplification import dbm_to_w, reference_bandwidth_01nm_hz
from isrs_scl.fiber.span_model import SpanModel, SpanResult
from isrs_scl.system.capacity import achievable_information_rate_bps
from isrs_scl.system.grid import OpticalGrid


class ReceiverCalibration(Protocol):
    minimum_input_snr_db: float
    maximum_input_snr_db: float

    def predict_snr_db(self, input_snr_db: np.ndarray) -> np.ndarray: ...

    def predict_ngmi(self, input_snr_db: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class NoiseBudget:
    ase_psd_w_per_hz: np.ndarray
    ase_optical_w: np.ndarray
    ase_receiver_w: np.ndarray
    ase_01nm_w: np.ndarray
    nli_receiver_w: np.ndarray
    transceiver_receiver_w: np.ndarray
    total_receiver_w: np.ndarray
    optical_bandwidth_hz: float
    receiver_equivalent_bandwidth_hz: float


@dataclass(frozen=True)
class LinkResult:
    n_spans: int
    distance_km: float
    launch_power_w: np.ndarray
    ase_channel_w: np.ndarray
    ase_01nm_w: np.ndarray
    nli_w: np.ndarray
    transceiver_noise_w: np.ndarray
    total_noise_w: np.ndarray
    gsnr_linear: np.ndarray
    gsnr_db: np.ndarray
    osnr_01nm_db: np.ndarray
    ber: np.ndarray
    evm: np.ndarray
    q_factor_db: np.ndarray
    gmi: np.ndarray
    ngmi: np.ndarray
    air_per_channel_bps: np.ndarray
    air_total_bps: float
    span: SpanResult
    noise_budget: NoiseBudget
    calibrated_receiver_snr_db: np.ndarray | None = None
    calibrated_receiver_ngmi: np.ndarray | None = None
    metric_basis: str = "physical_gsnr_awgn"

    def to_frame(self, grid: OpticalGrid) -> pd.DataFrame:
        if grid.n_channels != self.launch_power_w.size:
            raise ValueError("Grid/result channel count mismatch")
        data: dict[str, Any] = {
            "channel": np.arange(grid.n_channels),
            "band": grid.bands,
            "frequency_thz": grid.frequencies_hz / 1e12,
            "wavelength_nm": grid.wavelengths_nm,
            "distance_km": self.distance_km,
            "launch_power_dbm": 10.0 * np.log10(self.launch_power_w / 1e-3),
            "ase_psd_w_per_hz": self.noise_budget.ase_psd_w_per_hz,
            "ase_channel_w": self.ase_channel_w,
            "ase_receiver_w": self.noise_budget.ase_receiver_w,
            "ase_01nm_w": self.ase_01nm_w,
            "nli_w": self.nli_w,
            "transceiver_noise_w": self.transceiver_noise_w,
            "total_noise_w": self.total_noise_w,
            "optical_noise_bandwidth_hz": self.noise_budget.optical_bandwidth_hz,
            "receiver_equivalent_bandwidth_hz": self.noise_budget.receiver_equivalent_bandwidth_hz,
            "gsnr_db": self.gsnr_db,
            "osnr_01nm_db": self.osnr_01nm_db,
            "ber": self.ber,
            "evm_percent": 100.0 * self.evm,
            "q_factor_db": self.q_factor_db,
            "gmi_bits_per_2d_symbol_per_pol": self.gmi,
            "ngmi": self.ngmi,
            "air_per_channel_gbps": self.air_per_channel_bps / 1e9,
            "metric_basis": self.metric_basis,
        }
        if self.calibrated_receiver_snr_db is not None:
            data["calibrated_receiver_snr_db"] = self.calibrated_receiver_snr_db
        if self.calibrated_receiver_ngmi is not None:
            data["calibrated_receiver_ngmi"] = self.calibrated_receiver_ngmi
        return pd.DataFrame(data)

    def band_summary(self, grid: OpticalGrid, threshold: float) -> pd.DataFrame:
        frame = self.to_frame(grid)
        rows = []
        metric = "calibrated_receiver_ngmi" if self.calibrated_receiver_ngmi is not None else "ngmi"
        for band, group in frame.groupby("band", sort=False):
            rows.append({
                "band": band,
                "channels": len(group),
                "minimum_gsnr_db": float(group["gsnr_db"].min()),
                "mean_gsnr_db": float(group["gsnr_db"].mean()),
                "minimum_ngmi": float(group[metric].min()),
                "mean_ngmi": float(group[metric].mean()),
                "working_fraction": float(np.mean(group[metric] >= threshold)),
                "metric_basis": self.metric_basis,
            })
        return pd.DataFrame(rows)


class LinkModel:
    def __init__(self, grid: OpticalGrid, cfg: dict, span_model: SpanModel | None = None, receiver_calibration: ReceiverCalibration | None = None):
        self.grid, self.cfg = grid, cfg
        self.span_model = span_model or SpanModel(grid, cfg)
        self.receiver_calibration = receiver_calibration

    def flat_launch_w(self, power_dbm: float | None = None) -> np.ndarray:
        value = float(power_dbm) if power_dbm is not None else float(self.cfg["launch"]["flat_power_dbm_per_channel"])
        return np.full(self.grid.n_channels, float(dbm_to_w(value)))

    def evaluate(self, launch_power_w: np.ndarray, n_spans: int, nli_model: str | None = None) -> LinkResult:
        if n_spans < 1:
            raise ValueError("n_spans must be positive")
        launch = np.asarray(launch_power_w, dtype=float)
        if launch.shape != self.grid.frequencies_hz.shape or np.any(launch <= 0) or not np.isfinite(launch).all():
            raise ValueError("Launch powers must be finite, positive, and match the grid")
        span = self.span_model.evaluate(launch, nli_model)
        epsilon = float(self.cfg["nli"]["coherence_epsilon"])
        if epsilon < 0:
            raise ValueError("nli.coherence_epsilon must be non-negative")
        amp = span.amplifier
        optical_bw = float(getattr(amp, "optical_noise_bandwidth_hz", self.span_model.amplifier.noise_bandwidth_hz))
        receiver_bw = float(getattr(amp, "receiver_equivalent_noise_bandwidth_hz", optical_bw))
        ase_psd = float(n_spans) * span.total_ase_psd_w_per_hz
        ase_optical = ase_psd * optical_bw
        ase_receiver = ase_psd * receiver_bw
        ase_01nm = ase_psd * reference_bandwidth_01nm_hz(self.grid.wavelengths_nm)
        nli = float(n_spans) ** (1.0 + epsilon) * span.nli.nli_power_w_per_span
        trx_snr = 10.0 ** (float(self.cfg["nli"]["transceiver_snr_db"]) / 10.0)
        trx_noise = launch / trx_snr
        total = ase_receiver + nli + trx_noise
        if np.any(total <= 0) or not np.isfinite(total).all():
            raise FloatingPointError("Non-positive or non-finite accumulated noise")
        gsnr = launch / total
        gsnr_db = 10.0 * np.log10(gsnr)
        calibrated_snr = calibrated_ngmi = None
        basis = "physical_gsnr_awgn"
        metric_snr_db = gsnr_db
        if self.receiver_calibration is not None:
            minimum = float(self.receiver_calibration.minimum_input_snr_db)
            maximum = float(self.receiver_calibration.maximum_input_snr_db)
            if np.any(gsnr_db < minimum) or np.any(gsnr_db > maximum):
                raise ValueError(f"Receiver calibration extrapolation requested outside {minimum:g}..{maximum:g} dB")
            calibrated_snr = np.asarray(self.receiver_calibration.predict_snr_db(gsnr_db), dtype=float)
            calibrated_ngmi = np.asarray(self.receiver_calibration.predict_ngmi(gsnr_db), dtype=float)
            metric_snr_db = calibrated_snr
            basis = "calibrated_receiver"
        metric_snr = 10.0 ** (metric_snr_db / 10.0)
        ber = analytical_ber_16qam(metric_snr)
        evm = normalized_evm_from_snr(metric_snr)
        q_db = q_factor_db_from_ber(ber)
        gmi = gmi_16qam_awgn_from_snr_db(metric_snr_db)
        bits = float(self.cfg["modulation"]["bits_per_symbol_per_pol"])
        ngmi = calibrated_ngmi if calibrated_ngmi is not None else gmi / bits
        symbol_rate = float(self.cfg["modulation"]["symbol_rate_gbaud"]) * 1e9
        air_channel = gmi * symbol_rate * 2.0
        budget = NoiseBudget(ase_psd, ase_optical, ase_receiver, ase_01nm, nli, trx_noise, total, optical_bw, receiver_bw)
        return LinkResult(
            int(n_spans), int(n_spans) * float(self.cfg["fiber"]["span_length_km"]), launch,
            ase_optical, ase_01nm, nli, trx_noise, total, gsnr, gsnr_db,
            10.0 * np.log10(launch / np.maximum(ase_01nm, 1e-30)), ber, evm, q_db,
            gmi, ngmi, air_channel,
            achievable_information_rate_bps(gmi, symbol_rate, polarizations=2, maximum_bits_per_symbol_per_pol=bits),
            span, budget, calibrated_snr, calibrated_ngmi, basis,
        )

    def sweep_spans(self, launch_power_w: np.ndarray, max_spans: int | None = None, nli_model: str | None = None) -> list[LinkResult]:
        count = int(max_spans or self.cfg["fiber"]["max_spans"])
        if count < 1:
            raise ValueError("max_spans must be positive")
        return [self.evaluate(launch_power_w, n, nli_model) for n in range(1, count + 1)]
