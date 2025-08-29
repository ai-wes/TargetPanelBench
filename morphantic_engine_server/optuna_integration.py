# samplers/morphantic_optuna.py
from __future__ import annotations
import math, json, time, numpy as np
from typing import Dict, List, Optional, Any

import optuna
from optuna.samplers import BaseSampler

# Your engine
from scripts.core.complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution
from morphantic_modules import attach_modules

# ---- small RF surrogate with variance (works w/o heavy deps)
try:
    from sklearn.ensemble import RandomForestRegressor
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def _fit_rf(X: np.ndarray, y: np.ndarray):
    if not _HAS_SK: return None, None
    if len(X) < max(12, X.shape[1] + 3): return None, None
    rf = RandomForestRegressor(
        n_estimators=80,
        max_depth=None,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf, rf

def _predict_mean_var(model, X: np.ndarray):
    if model is None:  # fallback: mean-only
        m = np.mean(X, axis=1, dtype=float) * 0.0
        v = np.ones_like(m)
        return m, v
    # variance from per-tree predictions
    trees = model.estimators_
    preds = np.stack([t.predict(X) for t in trees], axis=0)  # [T, N]
    mean = preds.mean(axis=0)
    var = preds.var(axis=0) + 1e-9
    return mean, var

class MorphanticSampler(BaseSampler):
    """
    Optuna sampler that uses YOUR AEA to optimize a surrogate-based acquisition
    (LCB) over the *current* search space, then maps AEA's [0,1]^d vector to
    concrete parameter values for the next suggestion(s).

    Effect: Optuna orchestrates trials; **we** decide what to try by running
    AEA inside the sampler. This is your “Optuna uses my optimizer” lane.
    """
    def __init__(
        self,
        dim_hint: Optional[int] = None,
        bounds: tuple[float, float] = (0.0, 1.0),
        eval_budget_per_suggest: int = 150,
        seed: int = 123,
        modules: List[str] = ("turbo", "diversity", "safezone", "drift_guard"),
        warmup_random: int = 20,
        beta: float = 2.0,
        telemetry_dir: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        self.dim_hint = dim_hint
        self.lb, self.ub = float(bounds[0]), float(bounds[1])
        self.budget = int(eval_budget_per_suggest)
        self.seed = int(seed)
        self.modules = list(modules or [])
        self.warmup_random = int(warmup_random)
        self.beta = float(beta)
        self.telemetry_dir = telemetry_dir
        self.run_id = run_id or f"morph_{int(time.time())}"
        self._rng = np.random.default_rng(self.seed)

    # --- Optuna API
    def infer_relative_search_space(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_relative(self, study, trial, search_space: Dict[str, optuna.distributions.BaseDistribution]):
        # If small evidence, do pure random to populate the surrogate
        completed = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < self.warmup_random or not search_space:
            return {name: self._random_from_dist(dist) for name, dist in search_space.items()}

        # Build X/y from completed trials
        names = sorted(search_space.keys())
        to_vec = lambda p: np.array([self._param_to_unit(p[n], search_space[n]) for n in names], float)
        X = np.stack([to_vec(t.params) for t in completed if set(names).issubset(t.params)], axis=0)
        y = np.array([t.values[0] if t.values is not None else t.value for t in completed if set(names).issubset(t.params)], float)

        # Fit surrogate; target is MINIMIZE (loss)
        model, _ = _fit_rf(X, y)

        # Acquisition: LCB = mu - beta*sqrt(var)
        def acq(xu: np.ndarray) -> float:
            xu = np.atleast_2d(np.asarray(xu, float))
            mu, var = _predict_mean_var(model, xu)
            return float(mu[0] - self.beta * math.sqrt(float(var[0])))

        # Optimize acquisition with YOUR AEA (+ modules)
        dim = self.dim_hint or len(names)
        inner = lambda x: acq(np.clip(x, 0.0, 1.0))
        wrapped = attach_modules(
            inner_fitness=inner,
            dim=dim,
            bounds=(0.0, 1.0),
            budget_max=self.budget,
            seed=self.seed,
            modules=self.modules,
            problem_name="optuna_acq",
            warmup_frac=0.2,
            telemetry_dir=self.telemetry_dir,
            run_id=self.run_id + "_acq"
        )

        engine = AdvancedArchipelagoEvolution(
            n_islands=4, pop_size=24, max_generations=max(12, self.budget // 24),
            bounds=(0.0, 1.0), dimension=dim, early_stop_patience=8
        )
        res, champ = engine.optimize(wrapped)
        try:
            xu_best = np.clip(np.asarray(champ.get_solution(), float), 0.0, 1.0)
        except Exception:
            xu_best = self._rng.random(dim)

        # Map unit vector -> concrete params per dist
        suggestion = {name: self._unit_to_param(u, search_space[name]) for name, u in zip(names, xu_best)}
        return suggestion

    def sample_independent(self, study, trial, param_name, param_distribution):
        # If Optuna falls back to independent sampling, keep it random but seeded.
        return self._random_from_dist(param_distribution)

    # --- helpers: distributions <-> unit hypercube
    def _random_from_dist(self, dist):
        r = self._rng.random()
        return self._unit_to_param(r, dist)

    def _param_to_unit(self, val, dist):
        # Map to [0,1]; we handle Float/Int/LogUniform/Categorical
        if isinstance(dist, optuna.distributions.FloatDistribution):
            low, high = dist.low, dist.high
            if dist.log:
                return (math.log(val) - math.log(low)) / (math.log(high) - math.log(low))
            return (val - low) / (high - low)
        if isinstance(dist, optuna.distributions.IntDistribution):
            low, high = dist.low, dist.high
            return (val - low) / (high - low)
        if isinstance(dist, optuna.distributions.CategoricalDistribution):
            idx = dist.choices.index(val)
            return idx / max(1, len(dist.choices)-1)
        return float(val)

    def _unit_to_param(self, u, dist):
        u = float(np.clip(u, 0.0, 1.0))
        if isinstance(dist, optuna.distributions.FloatDistribution):
            low, high = dist.low, dist.high
            if dist.log:
                v = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
            else:
                v = low + u * (high - low)
            return float(v)
        if isinstance(dist, optuna.distributions.IntDistribution):
            low, high = dist.low, dist.high
            return int(round(low + u * (high - low)))
        if isinstance(dist, optuna.distributions.CategoricalDistribution):
            idx = int(np.floor(u * len(dist.choices)))
            idx = min(idx, len(dist.choices)-1)
            return dist.choices[idx]
        return u
