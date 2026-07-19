"""Current-run publication figures with uncertainty and provenance metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isrs_scl.visualization.style import apply_publication_style

_BAND_LIMITS = ((1460.0, 1530.0, "S"), (1530.0, 1565.0, "C"), (1565.0, 1625.0, "L"))
_WRITTEN: set[Path] = set()


def reset_figure_registry() -> None:
    _WRITTEN.clear()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns {sorted(missing)}")


_PDF_METADATA_KEYS = {"Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate", "Trapped"}


def _normalise_metadata(metadata: Mapping[str, object] | None) -> tuple[dict[str, str], dict[str, str]]:
    """Return full PNG/JSON metadata and a PDF-safe subset.

    Matplotlib's PDF backend rejects arbitrary info-dictionary keys such as
    ``run_id``.  We keep those fields in the sidecar JSON and PNG metadata,
    and encode the compact custom fields into the standard PDF ``Keywords``
    entry so no warning is emitted.
    """
    full = {"Creator": "ISRS_SCL_Q2_Publication_Project"}
    for key, value in dict(metadata or {}).items():
        full[str(key)] = str(value)
    pdf_meta = {key: value for key, value in full.items() if key in _PDF_METADATA_KEYS}
    custom = {key: value for key, value in full.items() if key not in _PDF_METADATA_KEYS}
    if custom:
        encoded = json.dumps(custom, sort_keys=True, separators=(",", ":"))
        existing = pdf_meta.get("Keywords", "")
        pdf_meta["Keywords"] = f"{existing}; {encoded}" if existing else encoded
    return full, pdf_meta


def _save(fig: plt.Figure, base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None, figure_data: pd.DataFrame | None = None) -> None:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    resolved = base.resolve()
    if resolved in _WRITTEN:
        raise RuntimeError(f"Duplicate figure output name: {base}")
    _WRITTEN.add(resolved)
    fig.tight_layout()
    full_meta, pdf_meta = _normalise_metadata(metadata)
    fig.savefig(base.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", metadata=full_meta)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata=pdf_meta)
    if figure_data is not None:
        figure_data.to_csv(base.with_name(base.name + "_data.csv"), index=False)
    base.with_name(base.name + "_metadata.json").write_text(json.dumps(full_meta, indent=2, sort_keys=True), encoding="utf-8")
    plt.close(fig)


def _shade_bands(ax: plt.Axes) -> None:
    for left, right, label in _BAND_LIMITS:
        ax.axvspan(left, right, alpha=0.035)
        ax.text((left + right) / 2, 0.985, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)


def plot_raman_validation(frame: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["z_km", "numerical_power_w", "analytical_power_w", "relative_error"], "Raman validation")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(frame["z_km"], 10 * np.log10(frame["numerical_power_w"] / 1e-3), label="RK4")
    ax.plot(frame["z_km"], 10 * np.log10(frame["analytical_power_w"] / 1e-3), "--", label="Analytical")
    ax.set(xlabel="Distance (km)", ylabel="Signal power (dBm)"); ax.legend()
    ax2 = ax.twinx(); ax2.plot(frame["z_km"], 100 * frame["relative_error"], ":"); ax2.set_ylabel("Relative error (%)")
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_spectral_tilt(frame: pd.DataFrame, output_base: Path, dpi: int, *, title: str = "One-span power evolution", metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["wavelength_nm", "launch_power_dbm", "span_output_power_dbm"], "Spectral tilt")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(frame["wavelength_nm"], frame["launch_power_dbm"], label="Span input")
    ax.plot(frame["wavelength_nm"], frame["span_output_power_dbm"], label="Before equalization")
    if "no_isrs_output_power_dbm" in frame: ax.plot(frame["wavelength_nm"], frame["no_isrs_output_power_dbm"], "--", label="No-ISRS")
    _shade_bands(ax); ax.set(xlabel="Wavelength (nm)", ylabel="Channel power (dBm)", title=title); ax.legend()
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_gsnr_profiles(frame: pd.DataFrame, output_base: Path, dpi: int, *, threshold_db: float | None = None, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["strategy", "wavelength_nm", "gsnr_db"], "GSNR profiles")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for strategy, group in frame.groupby("strategy", sort=False): ax.plot(group["wavelength_nm"], group["gsnr_db"], label=strategy)
    if threshold_db is not None: ax.axhline(float(threshold_db), linestyle="--", label="FEC-equivalent threshold")
    _shade_bands(ax); ax.set(xlabel="Wavelength (nm)", ylabel="GSNR (dB)"); ax.legend(ncol=2)
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_gsnr_distance_heatmap(frame: pd.DataFrame, output_base: Path, dpi: int, *, strategy: str = "adaptive", metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["strategy", "distance_km", "wavelength_nm", "gsnr_db"], "GSNR heatmap")
    data = frame[frame["strategy"] == strategy]
    pivot = data.pivot_table(index="distance_km", columns="wavelength_nm", values="gsnr_db")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(7.3, 4.3))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    ax.set(xlabel="Wavelength (nm)", ylabel="Distance (km)", title=f"{strategy.title()} GSNR"); fig.colorbar(image, ax=ax, label="GSNR (dB)")
    _save(fig, output_base, dpi, metadata=metadata, figure_data=data)


def plot_capacity_reach(summary: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(summary, ["strategy", "distance_km", "air_tbps", "thresholded_net_capacity_tbps"], "Capacity reach")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for strategy, group in summary.groupby("strategy", sort=False):
        ax.plot(group["distance_km"], group["thresholded_net_capacity_tbps"], marker="o", label=f"{strategy}: thresholded")
        ax.plot(group["distance_km"], group["air_tbps"], linestyle="--", label=f"{strategy}: AIR")
    ax.set(xlabel="Distance (km)", ylabel="Throughput (Tb/s)"); ax.legend(ncol=2, fontsize=8)
    _save(fig, output_base, dpi, metadata=metadata, figure_data=summary)


def plot_optimizer_history(history: pd.DataFrame, output_base: Path, dpi: int, *, multiseed: pd.DataFrame | None = None, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(history, ["iteration", "objective"], "Optimizer history")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(6.8, 4.0))
    groups = history.groupby([column for column in ("run", "restart") if column in history], sort=False) if any(column in history for column in ("run", "restart")) else [("optimizer", history)]
    for name, group in groups: ax.plot(group["iteration"], group["objective"], alpha=0.55, label=str(name))
    ax.set(xlabel="Iteration", ylabel="Band-aware robust objective"); ax.legend(fontsize=7, ncol=2)
    _save(fig, output_base, dpi, metadata=metadata, figure_data=history)


def plot_constellation(tx: np.ndarray, rx: np.ndarray, output_base: Path, dpi: int, *, title: str, annotation: str = "", metadata: Mapping[str, str] | None = None) -> None:
    transmitted, recovered = np.asarray(tx).reshape(-1), np.asarray(rx).reshape(-1)
    count = min(6000, recovered.size)
    apply_publication_style(); fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(recovered.real[:count], recovered.imag[:count], s=4, alpha=0.22, label="Recovered payload")
    unique = np.unique(np.round(transmitted.real, 8) + 1j * np.round(transmitted.imag, 8))
    ax.scatter(unique.real, unique.imag, marker="x", s=55, linewidths=1.5, label="Ideal")
    ax.set(xlabel="In-phase", ylabel="Quadrature", title=title); ax.set_aspect("equal", adjustable="box")
    if annotation: ax.text(0.02, 0.02, annotation, transform=ax.transAxes, va="bottom", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.legend(fontsize=8)
    frame = pd.DataFrame({"tx_real": transmitted[:count].real, "tx_imag": transmitted[:count].imag, "rx_real": recovered[:count].real, "rx_imag": recovered[:count].imag})
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_waveform_metrics_comparison(frame: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["band", "analytical_gsnr_db", "sample_snr_db", "acquisition_success"], "Waveform comparison")
    valid = frame[frame["acquisition_success"].astype(bool)]
    reduced = valid.groupby("band", sort=False).agg(analytical=("analytical_gsnr_db", "mean"), measured=("sample_snr_db", "mean"), measured_std=("sample_snr_db", "std")).reset_index()
    apply_publication_style(); fig, ax = plt.subplots(figsize=(6.6, 4.0)); x = np.arange(len(reduced)); width = 0.36
    ax.bar(x - width / 2, reduced["analytical"], width, label="Power-domain GSNR")
    ax.bar(x + width / 2, reduced["measured"], width, yerr=reduced["measured_std"].fillna(0), capsize=3, label="Held-out waveform SNR")
    failed = frame[~frame["acquisition_success"].astype(bool)]
    if not failed.empty:
        for band in failed["band"].unique(): ax.text(list(reduced["band"]).index(band) if band in set(reduced["band"]) else 0, 0, "acquisition failure", rotation=90, va="bottom", fontsize=7)
    ax.set_xticks(x, reduced["band"]); ax.set(xlabel="Band", ylabel="SNR / GSNR (dB)"); ax.legend()
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_launch_profiles(frame: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["wavelength_nm", "strategy", "launch_power_dbm"], "Launch profiles")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for strategy, group in frame.groupby("strategy", sort=False): ax.plot(group["wavelength_nm"], group["launch_power_dbm"], label=strategy)
    _shade_bands(ax); ax.set(xlabel="Wavelength (nm)", ylabel="Launch power/channel (dBm)"); ax.legend()
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_waveform_power_consistency(frame: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(frame, ["analytical_gsnr_db", "sample_snr_db", "acquisition_success"], "Power consistency")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(5.2, 4.6)); valid = frame[frame["acquisition_success"].astype(bool)]
    ax.errorbar(valid["analytical_gsnr_db"], valid["sample_snr_db"], yerr=np.vstack([valid["sample_snr_db"] - valid["snr_ci95_low_db"], valid["snr_ci95_high_db"] - valid["sample_snr_db"]]), fmt="o", capsize=3)
    low = min(frame["analytical_gsnr_db"].min(), valid["sample_snr_db"].min() if not valid.empty else frame["analytical_gsnr_db"].min()); high = max(frame["analytical_gsnr_db"].max(), valid["sample_snr_db"].max() if not valid.empty else frame["analytical_gsnr_db"].max())
    ax.plot([low, high], [low, high], "--", label="1:1"); ax.set(xlabel="Power-domain GSNR (dB)", ylabel="Waveform SNR (dB)"); ax.legend()
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_sensitivity(frame: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    apply_publication_style(); fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if {"parameter", "spearman_rho"}.issubset(frame.columns):
        data = frame.sort_values("abs_spearman_rho").tail(12); labels = data["strategy"].astype(str) + ": " + data["parameter"].astype(str)
        ax.barh(labels, data["spearman_rho"]); ax.set(xlabel="Spearman rank correlation", ylabel="Parameter")
    else:
        _require_columns(frame, ["launch_power_dbm", "strategy", "thresholded_net_capacity_tbps"], "Sensitivity")
        for strategy, group in frame.groupby("strategy", sort=False): ax.plot(group["launch_power_dbm"], group["thresholded_net_capacity_tbps"], marker="o", label=strategy)
        ax.legend(); ax.set(xlabel="Launch power (dBm/channel)", ylabel="Thresholded net capacity (Tb/s)")
    _save(fig, output_base, dpi, metadata=metadata, figure_data=frame)


def plot_external_validation(comparisons: pd.DataFrame, output_base: Path, dpi: int, *, metric: str = "gsnr_db", metadata: Mapping[str, str] | None = None) -> None:
    data = comparisons[(comparisons["metric"] == metric) & (comparisons["matched"] == 1)].dropna(subset=["model_value"])
    if data.empty: return
    apply_publication_style(); fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for source, group in data.groupby("source_id", sort=False): ax.scatter(group["reference_value"], group["model_value"], label=str(source))
    low = min(data["reference_value"].min(), data["model_value"].min()); high = max(data["reference_value"].max(), data["model_value"].max()); ax.plot([low, high], [low, high], "--")
    ax.set(xlabel=f"Independent reference: {metric}", ylabel=f"Model: {metric}"); ax.legend(fontsize=7)
    _save(fig, output_base, dpi, metadata=metadata, figure_data=data)


def plot_uncertainty_intervals(summary: pd.DataFrame, output_base: Path, dpi: int, *, metric: str = "thresholded_net_capacity_tbps", metadata: Mapping[str, str] | None = None) -> None:
    data = summary[summary["metric"] == metric].copy()
    if data.empty: return
    apply_publication_style(); fig, ax = plt.subplots(figsize=(6.2, 4.0)); x = np.arange(len(data))
    ax.errorbar(x, data["mean"], yerr=np.vstack([data["mean"] - data["ci95_low"], data["ci95_high"] - data["mean"]]), fmt="o", capsize=4)
    ax.set_xticks(x, data["strategy"]); ax.set(xlabel="Strategy", ylabel=metric.replace("_", " "))
    _save(fig, output_base, dpi, metadata=metadata, figure_data=data)


def plot_paired_robust_gain(paired: pd.DataFrame, output_base: Path, dpi: int, *, metadata: Mapping[str, str] | None = None) -> None:
    _require_columns(paired, ["baseline", "mean", "ci95_low", "ci95_high", "probability_positive"], "Paired robust gain")
    apply_publication_style(); fig, ax = plt.subplots(figsize=(6.2, 4.0)); x = np.arange(len(paired))
    ax.errorbar(x, paired["mean"], yerr=np.vstack([paired["mean"] - paired["ci95_low"], paired["ci95_high"] - paired["mean"]]), fmt="o", capsize=4)
    ax.axhline(0, linestyle="--"); ax.set_xticks(x, [f"adaptive - {name}" for name in paired["baseline"]]); ax.set_ylabel("Paired capacity gain (Tb/s)")
    for index, probability in enumerate(paired["probability_positive"]): ax.text(index, paired.iloc[index]["ci95_high"], f"P>0={probability:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, output_base, dpi, metadata=metadata, figure_data=paired)
