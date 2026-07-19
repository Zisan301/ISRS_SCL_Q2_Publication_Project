"""Robust-training scenario generation and paired performance statistics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json

import numpy as np
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class RobustScenarioBatch:
    role: str
    seed: int
    names: tuple[str, ...]
    z_scores: np.ndarray
    transformed: tuple[dict[str, float], ...]
    batch_hash: str


@dataclass(frozen=True)
class PairedGainSummary:
    samples: int
    mean: float
    median: float
    ci95_low: float
    ci95_high: float
    probability_positive: float
    cvar_lower: float
    worst_case: float


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=lambda x: np.asarray(x).tolist()).encode()
    return hashlib.sha256(encoded).hexdigest()


def cvar_lower(values: Sequence[float], alpha: float = 0.10) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(float(alpha) * array.size)))
    return float(np.mean(array[:count]))


def generate_training_batch(
    distributions: Mapping[str, Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    correlation: np.ndarray | None = None,
    role: str = "robust_training",
) -> RobustScenarioBatch:
    if samples < 4:
        raise ValueError("At least four robust scenarios are required")
    names = tuple(distributions)
    engine = qmc.LatinHypercube(d=len(names), seed=int(seed))
    uniforms = np.clip(engine.random(n=int(samples)), 1e-12, 1 - 1e-12)
    z = norm.ppf(uniforms)
    if correlation is not None:
        corr = np.asarray(correlation, dtype=float)
        if corr.shape != (len(names), len(names)):
            raise ValueError("Correlation matrix shape does not match distributions")
        eig = np.linalg.eigvalsh((corr + corr.T) / 2)
        if eig.min() < -1e-10:
            raise ValueError("Correlation matrix is not positive semidefinite")
        factor = np.linalg.cholesky(corr + np.eye(len(names)) * max(0.0, -eig.min() + 1e-12))
        z = z @ factor.T
    transformed: list[dict[str, float]] = []
    for row_index, row in enumerate(z):
        scenario: dict[str, float] = {"scenario": float(row_index)}
        for col, name in enumerate(names):
            spec = distributions[name]
            kind = str(spec.get("distribution", "normal")).lower()
            mean = float(spec.get("mean", 0.0))
            std = float(spec.get("std", 1.0))
            if kind == "normal":
                value = mean + std * row[col]
            elif kind == "lognormal":
                sigma2 = np.log(1.0 + (std / max(abs(mean), 1e-15)) ** 2)
                mu = np.log(max(mean, 1e-15)) - 0.5 * sigma2
                value = float(np.exp(mu + np.sqrt(sigma2) * row[col]))
            elif kind == "uniform":
                value = float(spec["minimum"]) + uniforms[row_index, col] * (float(spec["maximum"]) - float(spec["minimum"]))
            elif kind == "triangular":
                value = float(np.random.default_rng(seed + row_index * 7919 + col).triangular(float(spec["minimum"]), float(spec["mode"]), float(spec["maximum"])))
            elif kind == "empirical":
                values = np.sort(np.asarray(spec["values"], dtype=float))
                value = float(np.quantile(values, uniforms[row_index, col]))
            else:
                raise ValueError(f"Unsupported distribution {kind!r}")
            if "minimum" in spec:
                value = max(value, float(spec["minimum"]))
            if "maximum" in spec:
                value = min(value, float(spec["maximum"]))
            scenario[name] = float(value)
        transformed.append(scenario)
    payload = {"role": role, "seed": int(seed), "names": names, "z": z.tolist(), "transformed": transformed}
    return RobustScenarioBatch(role, int(seed), names, z, tuple(transformed), _hash_payload(payload))


def ensure_independent_batches(training: RobustScenarioBatch, holdout: RobustScenarioBatch) -> None:
    if training.seed == holdout.seed or training.batch_hash == holdout.batch_hash:
        raise ValueError("Robust-training and holdout scenario batches are not independent")
    if training.role == holdout.role:
        raise ValueError("Training and holdout batches must have different roles")


def aggregate_robust_objective(values: Sequence[float], *, nominal: float, cvar_alpha: float, nominal_weight: float = 1.0, cvar_weight: float = 1.0, worst_weight: float = 0.0) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return float("inf")
    return float(nominal_weight * nominal + cvar_weight * cvar_lower(array, cvar_alpha) + worst_weight * np.min(array))


def paired_gain_summary(adaptive: Sequence[float], baseline: Sequence[float], *, bootstrap_samples: int = 2000, seed: int = 0, cvar_alpha: float = 0.10) -> PairedGainSummary:
    adaptive_array = np.asarray(adaptive, dtype=float)
    baseline_array = np.asarray(baseline, dtype=float)
    if adaptive_array.shape != baseline_array.shape:
        raise ValueError("Paired arrays must have identical shapes")
    difference = adaptive_array - baseline_array
    difference = difference[np.isfinite(difference)]
    if difference.size < 4:
        raise ValueError("At least four paired scenarios are required")
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(bootstrap_samples), dtype=float)
    for index in range(boot.size):
        boot[index] = np.mean(rng.choice(difference, size=difference.size, replace=True))
    low, high = np.percentile(boot, [2.5, 97.5])
    return PairedGainSummary(
        samples=int(difference.size),
        mean=float(np.mean(difference)),
        median=float(np.median(difference)),
        ci95_low=float(low),
        ci95_high=float(high),
        probability_positive=float(np.mean(difference > 0)),
        cvar_lower=cvar_lower(difference, cvar_alpha),
        worst_case=float(np.min(difference)),
    )


def evaluate_common_scenarios(batch: RobustScenarioBatch, evaluator: Callable[[Mapping[str, float]], float]) -> np.ndarray:
    """Evaluate one callable on an immutable common-random-number scenario batch."""
    return np.asarray([float(evaluator(scenario)) for scenario in batch.transformed], dtype=float)
