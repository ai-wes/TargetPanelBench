"""
sdk/aea_core_adapter.py

Universal, easy-to-adopt adapter around the Morphantic Core AEA engine that:
- Accepts any Python callable as an objective (with optional .batch for speed)
- Optionally wraps the objective with SDK modules (SafeZone, DriftGuard, Turbo, Diversity)
- Runs `AdvancedArchipelagoEvolution` from `scripts.core.complete_teai_methods_slim_v2`
- Returns a compact result dictionary for quick integration

Usage:
    from sdk.aea_core_adapter import optimize_with_modules

    def sphere(x):
        return float((x*x).sum())

    res = optimize_with_modules(
        objective=sphere,
        dim=16,
        bounds=(-5.0, 5.0),
        budget_evals=2000,
        seed=123,
        modules=["turbo","diversity","safezone"],
        aea_kwargs={"n_islands": 3, "pop_size": 24}
    )

    print(res["best_f"], res["evaluations"], res["elapsed_sec"]) 
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
    def remaining(self) -> int:
        return self.max_evals - self.nfe
    def eval(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> float:
        if self.nfe + 1 > self.max_evals:
            raise StopIteration("budget exhausted")
        self.nfe += 1
        return float(f(x))
    def eval_batch(self, f: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
        take = min(len(X), self.remaining())
        if take <= 0:
            return np.array([], dtype=float)
        vals = [float(f(np.asarray(x, float))) for x in np.asarray(X[:take], float)]
        self.nfe += take
        return np.asarray(vals, dtype=float)


class _FitnessProxy:
    def __init__(self, f: Callable[[np.ndarray], float], budget: _Budget, bounds: Tuple[float, float]):
        self._f = f
        self._budget = budget
        self._bounds = (float(bounds[0]), float(bounds[1]))
    def __call__(self, x: np.ndarray) -> float:
        x = np.clip(np.asarray(x, float), self._bounds[0], self._bounds[1])
        return self._budget.eval(self._f, x)
    def batch(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, float), self._bounds[0], self._bounds[1])
        return self._budget.eval_batch(self._f, X)


def optimize_with_modules(
    objective: Callable[[np.ndarray], float],
    dim: int,
    bounds: Tuple[float, float],
    budget_evals: int,
    seed: int = 123,
    modules: Optional[Sequence[str]] = None,
    problem_name: Optional[str] = None,
    warmup_frac: float = 0.2,
    aea_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Morphantic AEA on a user objective with optional SDK modules.

    - objective: callable x->[float], with optional .batch(X)->np.ndarray
    - dim: search dimension
    - bounds: (low, high) broadcast to all dims
    - budget_evals: total objective evaluations permitted
    - modules: list of module names ["safezone","drift_guard","turbo","diversity"]
    - aea_kwargs: overrides for AEA (e.g., {"n_islands":3, "pop_size":24, "max_generations":20})

    Returns a result dict with keys: best_f, best_x (maybe None), evaluations, elapsed_sec, config
    """
    modules = list(modules or [])
    aea_kwargs = dict(aea_kwargs or {})

    # Configure AEA defaults from budget if not provided
    n_islands = int(aea_kwargs.pop("n_islands", 3))
    pop_size = int(aea_kwargs.pop("pop_size", 24))
    max_generations = int(aea_kwargs.pop("max_generations", max(6, budget_evals // max(1, n_islands * pop_size))))

    # Set cfg safely (affects global engine defaults only during this call)
    _cfg.n_islands = n_islands
    _cfg.pop_size = pop_size
    _cfg.max_generations = max_generations

    budget = _Budget(int(budget_evals))
    fit = _FitnessProxy(objective, budget, bounds)
    wrapped = attach_modules(fit, dim=dim, bounds=bounds, budget_max=budget.max_evals,
                             seed=seed, modules=modules, problem_name=problem_name,
                             warmup_frac=warmup_frac)

    aea = AdvancedArchipelagoEvolution(n_islands=n_islands, pop_size=pop_size,
                                       max_generations=max_generations, bounds=bounds,
                                       dimension=int(dim), **aea_kwargs)
    t0 = time.time()
    try:
        result, best_cell = aea.optimize(wrapped)
        best_x = best_cell.get_solution() if best_cell is not None else None
        best_f = getattr(result, "final_fitness", None)
    except Exception:
        # Graceful fallback
        best_x = None
        best_f = None
    elapsed = time.time() - t0

    return {
        "best_f": (float(best_f) if best_f is not None else None),
        "best_x": (best_x.tolist() if isinstance(best_x, np.ndarray) else best_x),
        "evaluations": int(budget.nfe),
        "elapsed_sec": float(elapsed),
        "config": {
            "n_islands": n_islands,
            "pop_size": pop_size,
            "max_generations": max_generations,
            "modules": modules,
            "dim": int(dim),
            "bounds": (float(bounds[0]), float(bounds[1])),
        },
    }

