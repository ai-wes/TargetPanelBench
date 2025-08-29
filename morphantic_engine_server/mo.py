from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from .aea import optimize_with_aea  # your existing scalar AEA wrapper

# Your engine (for native MO if available)
try:
    from scripts.core.complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution
    _HAS_NATIVE_AEA = True
except Exception:
    _HAS_NATIVE_AEA = False


# ---------------------------------------------------------------------
# ObjectiveSpec — your public MO API contract
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectiveSpec:
    """
    One objective for MO optimization.

    name:      metric key returned by metrics_fn(x)
    weight:    relative importance (auto-normalized across objectives)
    baseline:  typical/starting performance mapped to 0.0 goodness
    target:    desired performance mapped to 1.0 goodness
    direction: 'min' or 'max'
    """
    name: str
    weight: float
    baseline: float
    target: float
    direction: str  # 'min' | 'max'

    def scale(self, x: float) -> float:
        eps = 1e-12
        if self.direction == "max":
            num = (x - self.baseline)
            den = (self.target - self.baseline) + eps
            g = num / den
        else:  # 'min'
            num = (self.baseline - x)
            den = (self.baseline - self.target) + eps
            g = num / den
        return float(np.clip(g, 0.0, 1.0))


# ---------------------------------------------------------------------
# Scalarization fallback (safe for any engine)
# ---------------------------------------------------------------------
def make_mo_loss(
    metrics_fn: Callable[[np.ndarray], Dict[str, float]],
    objectives: List[ObjectiveSpec],
):
    """
    Returns (loss_fn, inspector) where loss_fn(x)->float and 'inspector'
    stores the last metrics/goodness for telemetry.
    """
    # Precompute normalized weights
    w_sum = sum(max(0.0, o.weight) for o in objectives) or 1.0
    weights = np.asarray([max(0.0, o.weight) / w_sum for o in objectives], float)

    inspector: Dict[str, Any] = {
        "last_metrics": None,
        "last_goodness": None,
        "last_composite": None,
    }

    def loss_fn(x: np.ndarray) -> float:
        x = np.asarray(x, float)
        m = metrics_fn(x)
        m = {k: float(v) for k, v in m.items()}  # defensive cast
        # goodness per objective (0..1)
        g = np.asarray([obj.scale(m.get(obj.name, obj.baseline)) for obj in objectives], float)
        composite = float(np.dot(weights, g))
        # Store for inspection
        inspector["last_metrics"] = m
        inspector["last_goodness"] = {obj.name: float(gi) for obj, gi in zip(objectives, g)}
        inspector["last_composite"] = composite
        return float(1.0 - composite)

    return loss_fn, inspector


# ---------------------------------------------------------------------
# Main MO entrypoint: try native; fallback to scalarization
# ---------------------------------------------------------------------
def optimize_mo_with_aea(
    objectives: List[ObjectiveSpec],
    metrics_fn: Callable[[np.ndarray], Dict[str, float]],
    dim: int,
    bounds: Tuple[float, float],
    budget_max: int,
    seed: int,
    *,
    # optional AEA knobs
    problem_name: str = "mo_run",
    telemetry_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    aea_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Multi-objective optimization with AEA.

    If native MO is available in your engine (metrics_fn + objectives),
    we use it directly. Otherwise we scalarize with make_mo_loss and call
    optimize_with_aea.

    Returns:
      {
        "x_best": [...],
        "f_best": float,
        "metrics": {...},         # last metrics at x_best
        "goodness": {...},        # per-objective goodness at x_best (0..1)
        "composite_goodness": float
      }
    """
    lb, ub = float(bounds[0]), float(bounds[1])

    # Try native MO if your engine supports it
    if _HAS_NATIVE_AEA:
        try:
            # Build engine configured for MO
            kw = dict(
                bounds=(lb, ub),
                dimension=int(dim),
                n_islands=4,
                pop_size=30,
                max_generations=max(20, int(budget_max // 30)),
                early_stop_patience=12,
                seed=int(seed),
            )
            if aea_kwargs:
                kw.update(aea_kwargs)

            aea = AdvancedArchipelagoEvolution(
                metrics_fn=metrics_fn,
                objectives=list(objectives),
                **kw
            )
            # engine expects a fitness; the MO loss is installed internally
            res, champ = aea.optimize(fitness_func=lambda x: 0.0)
            x_best = np.clip(np.asarray(champ.get_solution(), float), lb, ub)

            # Make sure we have final metrics at x_best
            m = metrics_fn(x_best)
            m = {k: float(v) for k, v in m.items()}

            # Derive goodness from the specs
            w_sum = sum(max(0.0, o.weight) for o in objectives) or 1.0
            weights = np.asarray([max(0.0, o.weight) / w_sum for o in objectives], float)
            g = np.asarray([obj.scale(m.get(obj.name, obj.baseline)) for obj in objectives], float)
            composite = float(np.dot(weights, g))

            return {
                "x_best": x_best.tolist(),
                "f_best": float(1.0 - composite),  # aligned with scalar path
                "metrics": m,
                "goodness": {obj.name: float(gi) for obj, gi in zip(objectives, g)},
                "composite_goodness": composite,
            }
        except TypeError:
            # signature mismatch – fall back below
            pass
        except Exception:
            # any runtime issues – fall back below
            pass

    # Fallback: scalarize and use existing scalar optimizer
    loss_fn, inspector = make_mo_loss(metrics_fn, objectives)
    out = optimize_with_aea(
        inner_fitness=loss_fn,
        dim=dim, bounds=bounds,
        budget_max=budget_max, seed=seed,
        modules=["turbo", "diversity", "safezone", "drift_guard"],
        problem_name=problem_name,
        telemetry_dir=telemetry_dir, run_id=run_id,
        aea_kwargs=aea_kwargs,
    )
    # Pull the MO info recorded by the loss_fn
    m = inspector["last_metrics"] or {}
    gdict = inspector["last_goodness"] or {}
    composite = inspector["last_composite"] if inspector["last_composite"] is not None else (1.0 - out["f_best"])

    return {
        "x_best": out["x_best"],
        "f_best": out["f_best"],
        "metrics": m,
        "goodness": gdict,
        "composite_goodness": float(composite),
    }
