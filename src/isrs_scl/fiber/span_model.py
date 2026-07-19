"""One-span physical model joining Raman, amplification, ASE and NLI."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import warnings

import numpy as np

from isrs_scl.fiber.amplification import (
    AmplifierResult,
    LumpedAmplifier,
    equivalent_distributed_raman_ase_psd,
)
from isrs_scl.fiber.attenuation import (
    attenuation_db_per_km,
    beta2_beta3_si,
    db_per_km_to_np_per_m,
    gamma_per_w_m,
)
from isrs_scl.fiber.nonlinear_gn import NLIResult, PowerProfileGN, SemrauClosedFormGN
from isrs_scl.fiber.raman_solver import (
    RamanGainSpectrum,
    RamanResult,
    RamanSolver,
    pumps_from_config,
)
from isrs_scl.system.grid import OpticalGrid


@dataclass(frozen=True)
class SpanResult:
    launch_powers_w: np.ndarray
    raman_result: RamanResult
    passive_result: RamanResult
    pump_gain_linear: np.ndarray
    amplifier: AmplifierResult
    distributed_raman_ase_psd_w_per_hz: np.ndarray
    total_ase_psd_w_per_hz: np.ndarray
    total_ase_channel_w: np.ndarray
    nli: NLIResult


class SpanModel:
    def __init__(self, grid: OpticalGrid, cfg: dict, integration_step_m: float | None = None):
        self.grid = grid
        self.cfg = cfg
        fiber = cfg["fiber"]
        raman = cfg["raman"]
        modulation = cfg["modulation"]

        self.length_m = float(fiber["span_length_km"]) * 1000.0
        self.step_m = float(integration_step_m or raman["integration_step_m"])
        self.save_step_m = float(raman["save_step_m"])
        self.symbol_rate_hz = float(modulation["symbol_rate_gbaud"]) * 1e9
        self.roll_off = float(modulation["roll_off"])

        alpha_db = attenuation_db_per_km(grid.wavelengths_nm, fiber)
        self.alpha = db_per_km_to_np_per_m(alpha_db)
        self.beta2, self.beta3 = beta2_beta3_si(grid.wavelengths_nm, fiber)
        self.gamma = gamma_per_w_m(grid.wavelengths_nm, fiber)
        self.aeff_m2 = float(fiber["effective_area_um2"]) * 1e-12

        gain = RamanGainSpectrum(float(raman["gain_peak_m_per_w"]), raman.get("gain_csv"))
        self.solver = RamanSolver(
            grid.frequencies_hz,
            self.alpha,
            self.aeff_m2,
            gain,
            pumps_from_config(raman),
        )
        self.passive_solver = RamanSolver(
            grid.frequencies_hz,
            self.alpha,
            self.aeff_m2,
            gain,
            (),
        )
        self.amplifier = LumpedAmplifier(
            cfg["amplification"], self.symbol_rate_hz, self.roll_off
        )

        cr_si = float(raman["semrau_linear_slope_per_w_km_thz"]) / (1000.0 * 1e12)
        self.semrau = SemrauClosedFormGN(
            grid.frequencies_hz,
            self.symbol_rate_hz,
            self.alpha,
            self.beta2,
            self.beta3,
            self.gamma,
            cr_si,
            float(cfg["nli"]["semrau_valid_bandwidth_thz"]) * 1e12,
            float(cfg["nli"]["modulation_correction"]),
        )
        self.profile_gn = PowerProfileGN(
            grid.frequencies_hz,
            self.symbol_rate_hz,
            self.alpha,
            self.beta2,
            self.gamma,
            float(cfg["nli"]["modulation_correction"]),
        )
        self._cache: dict[str, SpanResult] = {}

    def _cache_key(self, launch_w: np.ndarray, nli_model: str) -> str:
        rounded = np.round(np.asarray(launch_w, dtype=float), 15)
        payload = rounded.tobytes() + nli_model.encode() + repr(self.step_m).encode()
        return sha1(payload).hexdigest()

    def evaluate(self, launch_powers_w: np.ndarray, nli_model: str | None = None) -> SpanResult:
        launch = np.asarray(launch_powers_w, dtype=float)
        if launch.shape != self.grid.frequencies_hz.shape or np.any(launch <= 0):
            raise ValueError("Launch profile must be positive and match the optical grid")
        model_name = nli_model or str(self.cfg["nli"]["primary_model"])
        key = self._cache_key(launch, model_name)
        if key in self._cache:
            return self._cache[key]

        raman_result = self.solver.solve(
            launch, self.length_m, self.step_m, self.save_step_m
        )
        passive_result = self.passive_solver.solve(
            launch, self.length_m, self.step_m, self.save_step_m
        )
        pump_gain = raman_result.output_powers_w / np.maximum(
            passive_result.output_powers_w, 1e-30
        )

        amp = self.amplifier.equalize(
            raman_result.output_powers_w,
            launch,
            self.grid.frequencies_hz,
            self.grid.wavelengths_nm,
        )
        tolerance = float(self.cfg["amplification"]["gain_flatness_tolerance_db"])
        if np.max(np.abs(amp.residual_db)) > tolerance:
            warnings.warn(
                "One or more channels cannot be restored to the requested launch profile "
                "within configured amplifier gain limits.",
                RuntimeWarning,
                stacklevel=2,
            )

        raman_ase_psd = equivalent_distributed_raman_ase_psd(
            self.grid.frequencies_hz,
            pump_gain,
            float(self.cfg["raman"]["equivalent_noise_figure_db"]),
        )
        total_ase_psd = amp.ase_psd_w_per_hz + raman_ase_psd
        total_ase_channel = total_ase_psd * self.amplifier.noise_bandwidth_hz

        if model_name == "power_profile_gn":
            nli = self.profile_gn.evaluate(
                launch, raman_result.z_m, raman_result.powers_w
            )
        elif model_name == "semrau_closed_form":
            nli = self.semrau.evaluate(launch)
        else:
            raise ValueError(f"Unknown NLI model {model_name!r}")

        result = SpanResult(
            launch,
            raman_result,
            passive_result,
            pump_gain,
            amp,
            raman_ase_psd,
            total_ase_psd,
            total_ase_channel,
            nli,
        )
        self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()
