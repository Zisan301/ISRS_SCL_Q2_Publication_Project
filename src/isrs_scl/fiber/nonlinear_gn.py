"""ISRS-aware Gaussian-noise models.

Two estimators are provided:

1. ``SemrauClosedFormGN`` implements the SPM/XPM approximation in Eqs. (10)
   and (11) of Semrau, Killey and Bayvel, JLT 37(9), 2019.
2. ``PowerProfileGN`` uses the numerically calculated channel power profiles to
   weight conventional SCI/XCI GN terms. It is the default for the full
   1460--1625 nm grid because that bandwidth exceeds the nominal 15-THz range
   of the first-order closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class NLIResult:
    eta_total_per_w2: np.ndarray
    eta_sci_per_w2: np.ndarray
    eta_xci_per_w2: np.ndarray
    nli_power_w_per_span: np.ndarray
    model_name: str


def _safe_positive(value: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.maximum(np.asarray(value, dtype=float), floor)


class SemrauClosedFormGN:
    """First-order ISRS GN closed-form SPM/XPM approximation.

    All inputs are SI. Channel frequencies are converted to offsets around the
    optical-grid center before evaluating the paper's equations.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        bandwidth_hz: float,
        alpha_np_per_m: np.ndarray,
        beta2_s2_per_m: np.ndarray,
        beta3_s3_per_m: np.ndarray,
        gamma_per_w_m: np.ndarray,
        raman_slope_per_w_m_hz: float,
        valid_bandwidth_hz: float = 15e12,
        modulation_correction: float = 1.0,
    ):
        self.f_abs = np.asarray(frequencies_hz, dtype=float)
        self.f = self.f_abs - 0.5 * (self.f_abs[0] + self.f_abs[-1])
        self.b = float(bandwidth_hz)
        self.alpha = _safe_positive(alpha_np_per_m)
        self.beta2 = np.asarray(beta2_s2_per_m, dtype=float)
        self.beta3 = np.asarray(beta3_s3_per_m, dtype=float)
        self.gamma = np.asarray(gamma_per_w_m, dtype=float)
        self.cr = float(raman_slope_per_w_m_hz)
        self.valid_bandwidth_hz = float(valid_bandwidth_hz)
        self.modulation_correction = float(modulation_correction)
        if self.f_abs[-1] - self.f_abs[0] > self.valid_bandwidth_hz:
            warnings.warn(
                "The optical bandwidth exceeds the configured validity range of the "
                "first-order Semrau closed form. Use PowerProfileGN as the primary model "
                "and treat this result as a benchmark only.",
                RuntimeWarning,
                stacklevel=2,
            )

    def evaluate(self, launch_power_w: np.ndarray) -> NLIResult:
        p = _safe_positive(launch_power_w)
        p_total = float(np.sum(p))
        n = p.size
        eta_sci = np.zeros(n)
        eta_xci = np.zeros(n)

        phi_i = 12.0 * np.pi**2 * (self.beta2 + 2.0 * np.pi * self.beta3 * self.f)
        t_i = 2.0 - self.f * p_total * self.cr / self.alpha
        abs_phi = _safe_positive(np.abs(phi_i))
        spm_bracket = (
            np.pi
            * (t_i**2 - 4.0 / 9.0)
            / (self.alpha * abs_phi)
            * np.arcsinh(self.b**2 * abs_phi / (16.0 * self.alpha))
            + self.b**2 / (9.0 * self.alpha**2)
        )
        eta_sci = 16.0 / 27.0 * self.gamma**2 / self.b**2 * spm_bracket

        for i in range(n):
            df = self.f - self.f[i]
            phi_ik = 2.0 * np.pi**2 * df * (
                self.beta2[i] + np.pi * self.beta3[i] * (self.f[i] + self.f)
            )
            mask = np.arange(n) != i
            ph = phi_ik[mask]
            # The atan/phi combinations are positive for either sign of ph.
            t_k = t_i[mask]
            term = (
                ((t_k**2 - 1.0) / 3.0)
                * np.arctan(self.b * ph / self.alpha[i])
                + ((4.0 - t_k**2) / 6.0)
                * np.arctan(self.b * ph / (2.0 * self.alpha[i]))
            )
            ratio = (p[mask] / p[i]) ** 2
            contribution = ratio * term / (self.b * ph)
            eta_xci[i] = (
                32.0
                / 27.0
                * self.gamma[i] ** 2
                / self.alpha[i]
                * np.sum(contribution)
            )

        eta_sci = np.maximum(eta_sci, 0.0)
        eta_xci = np.maximum(eta_xci, 0.0)
        eta_total = self.modulation_correction * (eta_sci + eta_xci)
        return NLIResult(
            eta_total,
            self.modulation_correction * eta_sci,
            self.modulation_correction * eta_xci,
            eta_total * p**3,
            "semrau_closed_form",
        )


class PowerProfileGN:
    """Power-profile-weighted SCI/XCI GN approximation.

    The cross-channel term is explicitly used:

    ``P_NLI,i = eta_SCI,i P_i^3 + sum_j eta_XCI,ij P_i P_j^2``.

    Numerical ISRS profiles weight each term through longitudinal overlap
    integrals. This avoids reducing the full S+C+L profile to one average gain.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        channel_bandwidth_hz: float,
        alpha_np_per_m: np.ndarray,
        beta2_s2_per_m: np.ndarray,
        gamma_per_w_m: np.ndarray,
        modulation_correction: float = 1.0,
    ):
        self.f = np.asarray(frequencies_hz, dtype=float)
        self.b = float(channel_bandwidth_hz)
        self.alpha = _safe_positive(alpha_np_per_m)
        self.beta2 = np.asarray(beta2_s2_per_m, dtype=float)
        self.gamma = np.asarray(gamma_per_w_m, dtype=float)
        self.modulation_correction = float(modulation_correction)

    def _base_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        abs_beta2 = _safe_positive(np.abs(self.beta2))
        # Standard high-dispersion rectangular-channel GN approximations.
        sci = (
            8.0
            / 27.0
            * self.gamma**2
            / (np.pi * self.alpha * abs_beta2 * self.b**2)
            * np.arcsinh(np.pi**2 * abs_beta2 * self.b**2 / (2.0 * self.alpha))
        )

        df = np.abs(self.f[:, None] - self.f[None, :])
        edge = self.b / 2.0
        ratio = (df + edge) / np.maximum(df - edge, edge * 1e-9)
        xci = (
            16.0
            / 27.0
            * self.gamma[:, None] ** 2
            / (2.0 * np.pi * self.alpha[:, None] * abs_beta2[:, None] * self.b**2)
            * np.log(np.maximum(ratio, 1.0))
        )
        np.fill_diagonal(xci, 0.0)
        return sci, xci

    def evaluate(
        self,
        launch_power_w: np.ndarray,
        z_m: np.ndarray,
        powers_w_z_channel: np.ndarray,
    ) -> NLIResult:
        p = _safe_positive(launch_power_w)
        z = np.asarray(z_m, dtype=float)
        profiles = _safe_positive(powers_w_z_channel)
        if profiles.shape != (z.size, p.size):
            raise ValueError("Power profile must have shape [z, channel]")
        if z.size < 2 or np.any(np.diff(z) <= 0):
            raise ValueError("z grid must be strictly increasing")

        rho = profiles / p[None, :]
        # Trapezoidal integration weights for arbitrary saved z positions.
        dz = np.diff(z)
        weights = np.zeros_like(z)
        weights[0] = dz[0] / 2.0
        weights[-1] = dz[-1] / 2.0
        if z.size > 2:
            weights[1:-1] = (dz[:-1] + dz[1:]) / 2.0

        passive = np.exp(-z[:, None] * self.alpha[None, :])
        sci_overlap = np.sum(weights[:, None] * rho**3, axis=0)
        sci_reference = np.sum(weights[:, None] * passive**3, axis=0)
        sci_weight = sci_overlap / _safe_positive(sci_reference)

        weighted_rho2 = weights[:, None] * rho**2
        cross_overlap = rho.T @ weighted_rho2
        weighted_passive2 = weights[:, None] * passive**2
        cross_reference = passive.T @ weighted_passive2
        cross_weight = cross_overlap / _safe_positive(cross_reference)

        base_sci, base_xci = self._base_coefficients()
        sci_power = base_sci * sci_weight * p**3
        xci_matrix_power = base_xci * cross_weight * p[:, None] * p[None, :] ** 2
        xci_power = np.sum(xci_matrix_power, axis=1)
        total_power = self.modulation_correction * (sci_power + xci_power)

        eta_sci = self.modulation_correction * sci_power / p**3
        eta_xci = self.modulation_correction * xci_power / p**3
        eta_total = total_power / p**3
        return NLIResult(
            eta_total,
            eta_sci,
            eta_xci,
            total_power,
            "power_profile_gn",
        )
