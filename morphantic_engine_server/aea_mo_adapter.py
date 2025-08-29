"""
sdk/aea_mo_adapter.py

Multi-objective adapter around Morphantic AEA with universal, non-invasive module wrapping.

Design:
- Accept vector-valued objective f(x)->R^M (minimization). Adapter scalarizes for the optimizer
  (default: sum of objectives) but collects full vectors during optimization.
- Returns nondominated set and optional metrics (HV in 2D, IGD against a provided reference set).
- Supports SDK modules (safezone, drift_guard, turbo, diversity) without modifying the core engine.

Limitations:
- Hypervolume metric provided for M=2 via exact rectangle decomposition; for M>2, HV is omitted.
- IGD requires a reference front/set; if not provided, it is not computed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from scripts.core.complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution, cfg as _cfg
from sdk.morphantic_modules import attach_modules


@dataclass
class _Budget:
    max_evals: int
    nfe: int = 0
    def remaining(self) -> int: return self.max_evals - self.nfe
    def eval(self, f, x):
        if self.nfe + 1 > self.max_evals:
            raise StopIteration("budget exhausted")
        self.nfe += 1
        return f(x)
    def eval_batch(self, f, X):
        take = min(len(X), self.remaining())
        if take <= 0:
            return np.array([], float)
        vals = [f(x) for x in np.asarray(X[:take], float)]
        self.nfe += take
        return np.asarray(vals, float)


class _FitnessProxyMO:
    def __init__(self, f_vec: Callable[[np.ndarray], np.ndarray], budget: _Budget, bounds: Tuple[float, float],
                 scalarize: str = "sum"):
        self.f_vec = f_vec
        self._budget = budget
        self._bounds = (float(bounds[0]), float(bounds[1]))
        assert scalarize in ("sum", "mean")
        self.scalarize = scalarize
        self.archive_X: list[np.ndarray] = []
        self.archive_F: list[np.ndarray] = []

    def _to_scalar(self, F: np.ndarray) -> float:
        if self.scalarize == "mean":
            return float(np.mean(F))
        return float(np.sum(F))

    def __call__(self, x: np.ndarray) -> float:
        x = np.clip(np.asarray(x, float), self._bounds[0], self._bounds[1])
        try:
            F = np.asarray(self._budget.eval(self.f_vec, x), float).reshape(-1)
        except StopIteration:
            return float("inf")
        except Exception:
            return float("inf")
        if not np.all(np.isfinite(F)):
            return float("inf")
        self.archive_X.append(x.copy())
        self.archive_F.append(F.copy())
        return self._to_scalar(F)

    def batch(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, float), self._bounds[0], self._bounds[1])
        try:
            Fs = [np.asarray(self._budget.eval(self.f_vec, x), float).reshape(-1) for x in X]
        except StopIteration:
            return np.array([], float)
        vals = []
        for x, F in zip(X, Fs):
            if np.all(np.isfinite(F)):
                self.archive_X.append(x.copy())
                self.archive_F.append(F.copy())
                vals.append(self._to_scalar(F))
            else:
                vals.append(float("inf"))
        return np.asarray(vals, float)


def _nondominated(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, float)
    if F.size == 0:
        return F
    keep = []
    for i, p in enumerate(F):
        dom = False
        for j, q in enumerate(F):
            if j == i:
                continue
            if np.all(q <= p) and np.any(q < p):
                dom = True
                break
        if not dom:
            keep.append(i)
    return F[keep]


def _hv2d(F: np.ndarray, ref: Tuple[float, float]) -> float:
    # Minimization HV in 2D with axis-aligned rectangles
    if F.size == 0:
        return 0.0
    P = _nondominated(F)
    # Sort by first objective ascending
    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    prev_f1 = ref[0]
    for f1, f2 in P[::-1]:  # integrate from right to left
        w = prev_f1 - f1
        h = max(0.0, ref[1] - f2)
        if w > 0 and h > 0:
            hv += w * h
        prev_f1 = f1
    return float(max(0.0, hv))


def _igd(P: np.ndarray, R: np.ndarray) -> float:
    # Inverted GD: mean distance from each r in R to nearest p in P (Euclidean)
    if P.size == 0 or R.size == 0:
        return float("inf")
    dists = []
    for r in R:
        d = np.min(np.linalg.norm(P - r, axis=1))
        dists.append(d)
    return float(np.mean(dists))


def optimize_mo_with_modules(
    objective_vec: Callable[[np.ndarray], np.ndarray],
    dim: int,
    bounds: Tuple[float, float],
    budget_evals: int,
    M: int,
    seed: int = 123,
    modules: Optional[Sequence[str]] = None,
    problem_name: Optional[str] = None,
    warmup_frac: float = 0.2,
    scalarize: str = "sum",
    aea_kwargs: Optional[Dict[str, Any]] = None,
    ref_front: Optional[np.ndarray] = None,
    ref_point_2d: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    modules = list(modules or [])
    aea_kwargs = dict(aea_kwargs or {})

    n_islands = int(aea_kwargs.pop("n_islands", 3))
    pop_size = int(aea_kwargs.pop("pop_size", 24))
    max_generations = int(aea_kwargs.pop("max_generations", max(6, budget_evals // max(1, n_islands * pop_size))))
    _cfg.n_islands = n_islands
    _cfg.pop_size = pop_size
    _cfg.max_generations = max_generations

    budget = _Budget(int(budget_evals))
    fit = _FitnessProxyMO(objective_vec, budget, bounds, scalarize=scalarize)

    wrapped = attach_modules(fit, dim=dim, bounds=bounds, budget_max=budget.max_evals,
                             seed=seed, modules=modules, problem_name=problem_name,
                             warmup_frac=warmup_frac)

    aea = AdvancedArchipelagoEvolution(n_islands=n_islands, pop_size=pop_size,
                                       max_generations=max_generations, bounds=bounds,
                                       dimension=int(dim), **aea_kwargs)
    t0 = time.time()
    try:
        _ = aea.optimize(wrapped)
    except Exception:
        pass
    elapsed = time.time() - t0

    X = np.asarray(fit.archive_X, float) if fit.archive_X else np.empty((0, dim), float)
    F = np.asarray(fit.archive_F, float) if fit.archive_F else np.empty((0, M), float)
    ND = _nondominated(F)

    metrics: Dict[str, Any] = {"igd": None, "hv2d": None}
    if ref_front is not None and ref_front.size:
        metrics["igd"] = _igd(ND, np.asarray(ref_front, float))
    if M == 2 and ref_point_2d is not None:
        metrics["hv2d"] = _hv2d(ND, tuple(map(float, ref_point_2d)))

    return {
        "evaluations": int(budget.nfe),
        "elapsed_sec": float(elapsed),
        "config": {"n_islands": n_islands, "pop_size": pop_size, "max_generations": max_generations,
                    "modules": modules, "dim": int(dim), "bounds": (float(bounds[0]), float(bounds[1]))},
        "X": X,
        "F": F,
        "front": ND,
        "metrics": metrics,
    }

