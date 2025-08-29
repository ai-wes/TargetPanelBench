# morphantic_modules.py  (DROP-IN REPLACEMENT)
# Adds a real runner that CALLS your AEA + optional telemetry logging.
from __future__ import annotations
import math, time, json, os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Sequence

# === import YOUR optimizer ===
from complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution


# -----------------------------
# Utilities
# -----------------------------
def _norm(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.dot(x, x)))

def _clip_bounds(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    return np.clip(np.asarray(x, float), lb, ub)

def _safe_len(x):
    try: return len(x)
    except Exception: return 0


# -----------------------------
# Telemetry (optional JSONL)
# -----------------------------
class _Telemetry:
    def __init__(self, dirpath: Optional[str], run_id: Optional[str]):
        self.fp = None
        if dirpath and run_id:
            os.makedirs(dirpath, exist_ok=True)
            self.path = os.path.join(dirpath, f"{run_id}_eval.jsonl")
            try:
                self.fp = open(self.path, "a", encoding="utf-8")
            except Exception:
                self.fp = None
        else:
            self.path = None

    def log_eval(self, ctx, x, f):
        if not self.fp: return
        rec = {
            "nfe": int(ctx.nfe),
            "stage": ctx.stage,
            "f": float(f),
            "best_f": float(ctx.best_f),
            "timestamp": time.time(),
        }
        # Avoid dumping giant vectors by default; add if needed
        try:
            self.fp.write(json.dumps(rec) + "\n")
            self.fp.flush()
        except Exception:
            pass

    def close(self):
        try:
            if self.fp: self.fp.close()
        except Exception:
            pass


# -----------------------------
# Context carried across modules
# -----------------------------
@dataclass
class ModuleContext:
    dim: int
    bounds: Tuple[float, float]
    budget_max: int
    seed: int
    problem_name: Optional[str] = None
    # live fields (updated by wrapper)
    nfe: int = 0
    best_x: Optional[np.ndarray] = None
    best_f: float = float("inf")
    stage: str = "early"   # "early" | "mid" | "late"
    events: Dict[str, Any] = field(default_factory=dict)
    # Optional callback to your core (hook is no-op unless you pass it)
    on_drift: Optional[callable] = None

    def update_stage(self):
        frac = 0.0 if self.budget_max <= 0 else (self.nfe / float(self.budget_max))
        if frac < 0.33: self.stage = "early"
        elif frac < 0.75: self.stage = "mid"
        else: self.stage = "late"


# -----------------------------
# Base module & manager
# -----------------------------
class Module:
    name = "base"
    def on_init(self, ctx: ModuleContext): pass
    def on_pre_eval(self, x: np.ndarray, ctx: ModuleContext) -> np.ndarray: return x
    def on_post_eval(self, x: np.ndarray, f: float, ctx: ModuleContext): pass

    # Batch hooks (optional)
    def on_pre_batch(self, X: np.ndarray, ctx: ModuleContext) -> np.ndarray:
        return np.arange(len(X), dtype=int)
    def on_post_batch(self, X: np.ndarray, vals: np.ndarray, ctx: ModuleContext): pass

    def on_event(self, event: str, payload: Dict[str, Any], ctx: ModuleContext): pass
    def finalize(self, ctx: ModuleContext): pass


class ModuleManager:
    def __init__(self, modules: Sequence[Module], ctx: ModuleContext, warmup_frac: float = 0.2, telemetry: Optional[_Telemetry] = None):
        self.modules = list(modules)
        self.ctx = ctx
        self.warmup_frac = float(max(0.0, min(0.8, warmup_frac)))
        self.telemetry = telemetry
        for m in self.modules:
            try: m.on_init(self.ctx)
            except Exception: pass

    def _in_warmup(self) -> bool:
        return (self.ctx.budget_max > 0) and (self.ctx.nfe < self.warmup_frac * self.ctx.budget_max)

    def pre_eval(self, x: np.ndarray) -> np.ndarray:
        self.ctx.update_stage()
        if self._in_warmup():
            return x
        for m in self.modules:
            try:
                x = m.on_pre_eval(x, self.ctx)
            except Exception:
                pass
        return x

    def post_eval(self, x: np.ndarray, f: float):
        # Update incumbent
        if f < self.ctx.best_f:
            self.ctx.best_f = float(f)
            self.ctx.best_x = np.asarray(x, float).copy()
        # Telemetry
        if self.telemetry:
            self.telemetry.log_eval(self.ctx, x, f)
        for m in self.modules:
            try:
                m.on_post_eval(x, f, self.ctx)
            except Exception:
                pass

    def pre_batch(self, X: np.ndarray) -> np.ndarray:
        self.ctx.update_stage()
        if self._in_warmup():
            return np.arange(len(X), dtype=int)
        idx = np.arange(len(X), dtype=int)
        for m in self.modules:
            try:
                p = np.asarray(m.on_pre_batch(X[idx], self.ctx), dtype=int)
                if p.ndim != 1 or len(p) != len(idx): continue
                idx = idx[p]
            except Exception:
                pass
        return idx

    def post_batch(self, X: np.ndarray, vals: np.ndarray):
        for m in self.modules:
            try:
                m.on_post_batch(X, vals, self.ctx)
            except Exception:
                pass
        try:
            ev = self.ctx.events
            if isinstance(ev, dict) and 'drift' in ev:
                self.notify('drift_detected', ev.get('drift', {}))
                try: del ev['drift']
                except Exception: pass
        except Exception:
            pass

    def notify(self, event: str, payload: Dict[str, Any]):
        for m in self.modules:
            try:
                m.on_event(event, payload, self.ctx)
            except Exception:
                pass

    def finalize(self):
        for m in self.modules:
            try: m.finalize(self.ctx)
            except Exception: pass
        if self.telemetry:
            self.telemetry.close()


# -----------------------------
# Fitness wrapper: drop-in
# -----------------------------
class FitnessProxyWithModules:
    """
    Wraps your existing FitnessProxy or raw callable.
    Guarantees: if you don't enable modules, behavior is identical.
    """
    def __init__(self, inner, module_manager: ModuleManager):
        self.inner = inner
        self.mm = module_manager
        self._bounds = getattr(inner, "_bounds", None)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, float)
        try:
            x_mod = self.mm.pre_eval(x)
        except Exception:
            x_mod = x
        try:
            v_raw = self.inner(x_mod)
        except StopIteration:
            v_raw = float('inf')
        try:
            v = float(v_raw)
        except Exception:
            v = float('inf')
        if not np.isfinite(v): v = float('inf')
        # nfe update: best effort
        self.mm.ctx.nfe += 1
        self.mm.post_eval(x_mod, v)
        try:
            ev = self.mm.ctx.events
            if isinstance(ev, dict) and 'drift' in ev:
                self.mm.notify('drift_detected', ev.get('drift', {}))
                try: del ev['drift']
                except Exception: pass
        except Exception:
            pass
        return v

    def batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        n = len(X)
        try:
            order = self.mm.pre_batch(X)
            order = np.asarray(order, int)
            if order.ndim != 1 or len(order) != n:
                order = np.arange(n, dtype=int)
        except Exception:
            order = np.arange(n, dtype=int)

        X_ord = X[order]
        try:
            vals_ord = np.asarray(self.inner.batch(X_ord), float)
        except StopIteration:
            vals_ord = np.array([], dtype=float)
        out = np.full(n, np.inf, dtype=float)
        m = min(len(vals_ord), n)
        out[:m] = vals_ord[:m]
        inv = np.empty_like(order)
        inv[order] = np.arange(n, dtype=int)
        out = out[inv]

        # nfe best-effort bump for any finite returns
        self.mm.ctx.nfe += int(np.isfinite(out).sum())
        finite_mask = np.isfinite(out)
        if np.any(finite_mask):
            best_val = float(np.min(out[finite_mask]))
            best_idx = int(np.argmin(out[finite_mask]))
            x_best = X[np.where(finite_mask)[0][best_idx]]
            if best_val < self.mm.ctx.best_f:
                self.mm.ctx.best_f = best_val
                self.mm.ctx.best_x = np.asarray(x_best, float).copy()

        self.mm.post_batch(X, out)
        return out


# -----------------------------
# Module 1: SafeZone (trust-region cap)
# -----------------------------
class SafeZoneModule(Module):
    name = "safezone"
    def __init__(self, radius_frac: float = 0.08, apply_after_stage: str = "mid", min_radius_frac: float = 2e-3, activation_frac: float = 0.25):
        self.radius_frac = float(radius_frac)
        self.min_radius_frac = float(min_radius_frac)
        assert apply_after_stage in ("mid","late")
        self.apply_after_stage = apply_after_stage
        self.activation_frac = float(max(0.0, min(1.0, activation_frac)))
        self._relax_until = -1

    def on_pre_eval(self, x: np.ndarray, ctx: ModuleContext) -> np.ndarray:
        if ctx.best_x is None: return x
        if self.apply_after_stage == "late" and ctx.stage != "late": return x
        if self.apply_after_stage == "mid" and ctx.stage not in ("mid", "late"): return x
        if self._relax_until >= 0 and ctx.nfe < self._relax_until: return x
        lb, ub = ctx.bounds
        span = float(ub - lb)
        r = max(self.min_radius_frac * span, self.radius_frac * span)
        dx = np.asarray(x, float) - ctx.best_x
        d = _norm(dx)
        if d > (self.activation_frac * span): return x
        if d <= r or d == 0.0: return x
        x_cap = ctx.best_x + dx * (r / d)
        return _clip_bounds(x_cap, lb, ub)

    def on_event(self, event: str, payload: Dict[str, Any], ctx: ModuleContext):
        if event == "drift_detected":
            self.radius_frac = min(0.15, self.radius_frac * 1.5)
            try: horizon = int(0.1 * ctx.budget_max)
            except Exception: horizon = 50
            self._relax_until = max(self._relax_until, ctx.nfe + max(10, horizon))


# -----------------------------
# Module 2: DriftGuard (Page-Hinkley)
# -----------------------------
class DriftGuardModule(Module):
    name = "drift_guard"
    def __init__(self, delta: float = 0.02, lam: float = 40.0):
        self.delta = float(delta); self.lam = float(lam)
        self._mean = 0.0; self._mT = 0.0; self._PH = 0.0; self._t = 0

    def on_post_eval(self, x: np.ndarray, f: float, ctx: ModuleContext):
        self._t += 1
        if self._t == 1:
            self._mean = f; self._mT = 0.0; self._PH = 0.0; return
        self._mean = self._mean + (f - self._mean) / self._t
        self._mT = self._mT + (f - self._mean - self.delta)
        self._PH = min(self._PH, self._mT)
        if (self._mT - self._PH) > self.lam:
            ctx.events["drift"] = {"nfe": ctx.nfe, "f": float(f)}
            if callable(ctx.on_drift):
                try: ctx.on_drift(ctx)
                except Exception: pass
            self._mean = f; self._mT = 0.0; self._PH = 0.0


# -----------------------------
# Module 3: TurboSurrogate (pre-screen order)
# -----------------------------
class TurboSurrogateModule(Module):
    name = "turbo_surrogate"
    def __init__(self, train_after: int = 40, retrain_every: int = 25, min_batch: int = 8):
        self.train_after = int(train_after)
        self.retrain_every = int(retrain_every)
        self.min_batch = int(min_batch)
        self._X = []; self._y = []; self._last_train_nfe = -999; self._model = None

    def _fit(self):
        X = np.asarray(self._X, float); y = np.asarray(self._y, float)
        if len(X) < max(10, X.shape[1]+2):
            self._model = None; return
        try:
            from sklearn.ensemble import RandomForestRegressor
            self._model = RandomForestRegressor(n_estimators=60, random_state=0)
            self._model.fit(X, y)
            return
        except Exception:
            pass
        try:
            self._model = _RbfRidge().fit(X, y)
        except Exception:
            self._model = None

    def on_post_eval(self, x: np.ndarray, f: float, ctx: ModuleContext):
        self._X.append(np.asarray(x, float).copy()); self._y.append(float(f))

    def on_pre_batch(self, X: np.ndarray, ctx: ModuleContext) -> np.ndarray:
        n = len(X)
        if len(self._X) < self.train_after or n < self.min_batch:
            return np.arange(n, dtype=int)
        if (ctx.nfe - self._last_train_nfe) >= self.retrain_every or (self._model is None):
            self._fit(); self._last_train_nfe = ctx.nfe
        if self._model is None:
            return np.arange(n, dtype=int)
        try:
            preds = np.asarray(self._model.predict(np.asarray(X, float)), float)
        except Exception:
            return np.arange(n, dtype=int)
        order = np.argsort(preds)  # ascending (lower loss first)
        return np.asarray(order, dtype=int)


# -----------------------------
# Module 4: BatchDiversity (distance penalty)
# -----------------------------
class BatchDiversityModule(Module):
    name = "batch_diversity"
    def __init__(self, penalty: float = 0.2, k_neigh: int = 5):
        self.penalty = float(penalty); self.k_neigh = int(k_neigh)
        self._archive = []

    def on_post_eval(self, x: np.ndarray, f: float, ctx: ModuleContext):
        self._archive.append(np.asarray(x, float).copy())

    def on_pre_batch(self, X: np.ndarray, ctx: ModuleContext) -> np.ndarray:
        n = len(X)
        if n <= 2: return np.arange(n, dtype=int)
        lb, ub = ctx.bounds; span = float(ub - lb)
        A = np.asarray(self._archive, float) if self._archive else None
        remaining = list(range(n)); perm = []
        if ctx.best_x is not None:
            dists = [np.linalg.norm((X[i]-ctx.best_x)/span) for i in remaining]
            start = int(np.argmax(dists))
        else:
            start = 0
        perm.append(remaining.pop(start))
        while remaining:
            scores = []
            for i in remaining:
                d_sum = 0.0
                for j in perm:
                    d_sum += np.linalg.norm((X[i]-X[j])/span)
                if A is not None and len(A) > 0:
                    idx = np.random.choice(len(A), size=min(self.k_neigh, len(A)), replace=False)
                    d_sum += self.penalty * float(np.mean(np.linalg.norm((X[i]-A[idx])/span, axis=1)))
                scores.append(d_sum)
            pick = int(np.argmax(scores))
            perm.append(remaining.pop(pick))
        return np.asarray(perm, dtype=int)


# -----------------------------
# Optional fallback surrogate
# -----------------------------
class _RbfRidge:
    def __init__(self, gamma: float = None, alpha: float = 1e-6):
        self.gamma = gamma; self.alpha = float(alpha)
        self.X = None; self.w = None

    def _rbf(self, A, B):
        A = np.asarray(A, float); B = np.asarray(B, float)
        if self.gamma is None:
            D = np.linalg.norm(A[:,None,:]-A[None,:,:], axis=2)
            med = np.median(D[D>0]) if np.any(D>0) else 1.0
            self.gamma = 1.0 / (2*med*med)
        G = np.linalg.norm(A[:,None,:]-B[None,:,:], axis=2)
        return np.exp(-self.gamma * (G**2))

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        K = self._rbf(X, X) + self.alpha * np.eye(len(X))
        self.w = np.linalg.solve(K, y); self.X = X
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        K = self._rbf(X, self.X)
        return K @ self.w


# -----------------------------
# Public: builders & RUNNER
# -----------------------------
def _make_modules(modules: Sequence[str] | Sequence[Module]) -> List[Module]:
    name2cls = {
        "safezone": SafeZoneModule,
        "drift_guard": DriftGuardModule,
        "turbo":    TurboSurrogateModule,
        "diversity":BatchDiversityModule,
    }
    mod_objs: List[Module] = []
    for m in modules:
        if isinstance(m, Module):
            mod_objs.append(m)
        elif isinstance(m, str):
            cls = name2cls.get(m.lower().strip())
            if cls is not None: mod_objs.append(cls())
    return mod_objs

@dataclass
class WrappedFitness:
    f: FitnessProxyWithModules
    ctx: ModuleContext
    mm: ModuleManager

def build_wrapped_fitness(inner_fitness,
                          dim: int,
                          bounds: Tuple[float,float],
                          budget_max: int,
                          seed: int,
                          modules: Sequence[str] | Sequence[Module],
                          problem_name: Optional[str] = None,
                          warmup_frac: float = 0.2,
                          telemetry_dir: Optional[str] = None,
                          run_id: Optional[str] = None) -> WrappedFitness:
    ctx = ModuleContext(dim=dim, bounds=bounds, budget_max=budget_max, seed=seed, problem_name=problem_name)
    telemetry = _Telemetry(telemetry_dir, run_id)
    mm = ModuleManager(_make_modules(modules), ctx, warmup_frac=warmup_frac, telemetry=telemetry)
    wrapped = FitnessProxyWithModules(inner_fitness, mm)
    return WrappedFitness(f=wrapped, ctx=ctx, mm=mm)

def attach_modules(inner_fitness,
                   dim: int,
                   bounds: Tuple[float,float],
                   budget_max: int,
                   seed: int,
                   modules: Sequence[str] | Sequence[Module],
                   problem_name: Optional[str] = None,
                   warmup_frac: float = 0.2,
                   on_drift_callback: Optional[callable] = None,
                   telemetry_dir: Optional[str] = None,
                   run_id: Optional[str] = None):
    """
    Back-compat helper: returns only the wrapped callable (like before).
    """
    bundle = build_wrapped_fitness(
        inner_fitness, dim, bounds, budget_max, seed, modules,
        problem_name, warmup_frac, telemetry_dir, run_id
    )
    bundle.ctx.on_drift = on_drift_callback
    return bundle.f

def optimize_with_aea(inner_fitness,
                      dim: int,
                      bounds: Tuple[float,float] = (0.0,1.0),
                      budget_max: int = 1200,
                      seed: int = 123,
                      modules: Sequence[str] | Sequence[Module] = ("turbo","diversity","safezone","drift_guard"),
                      problem_name: Optional[str] = None,
                      warmup_frac: float = 0.2,
                      telemetry_dir: Optional[str] = None,
                      run_id: Optional[str] = None,
                      # engine params (exposed so you can tune)
                      n_islands: int = 4,
                      pop_size: int = 30,
                      max_generations: Optional[int] = None,
                      early_stop_patience: int = 12) -> Dict[str, Any]:
    """
    >>> # THIS is where YOUR algorithm is invoked:
    >>> # engine = AdvancedArchipelagoEvolution(...); engine.optimize(wrapped.f)
    """
    # Build wrapper + modules + telemetry
    bundle = build_wrapped_fitness(
        inner_fitness=inner_fitness,
        dim=dim, bounds=bounds, budget_max=budget_max, seed=seed,
        modules=modules, problem_name=problem_name,
        warmup_frac=warmup_frac, telemetry_dir=telemetry_dir, run_id=run_id
    )

    # Derive default max_generations from budget if not given
    if max_generations is None:
        # rough budget partition across archipelago
        max_generations = max(20, int(budget_max / max(pop_size, 1)))

    # === YOUR OPTIMIZER CALLED HERE ===
    engine = AdvancedArchipelagoEvolution(
        n_islands=n_islands, pop_size=pop_size, max_generations=max_generations,
        bounds=bounds, dimension=dim, early_stop_patience=early_stop_patience
    )
    res, champ = engine.optimize(bundle.f)

    # Extract solution; fall back to ctx best if needed
    try:
        x_best = np.clip(np.asarray(champ.get_solution(), float), bounds[0], bounds[1])
        f_best = float(bundle.f.mm.ctx.best_f)  # incumbent after final call
    except Exception:
        x_best = np.asarray(bundle.ctx.best_x if bundle.ctx.best_x is not None else np.zeros(dim), float)
        f_best = float(bundle.ctx.best_f)

    out = {
        "x_best": x_best.tolist(),
        "f_best": float(f_best),
        "nfe": int(bundle.ctx.nfe),
        "run_id": run_id,
        "telemetry_path": bundle.mm.telemetry.path if bundle.mm.telemetry else None,
        "engine_cfg": {
            "n_islands": n_islands, "pop_size": pop_size,
            "max_generations": max_generations, "early_stop_patience": early_stop_patience
        },
        "context": {
            "dim": dim, "bounds": list(bounds), "seed": seed,
            "budget_max": budget_max, "problem": problem_name
        }
    }
    # finalize modules & close telemetry
    try: bundle.mm.finalize()
    except Exception: pass
    return out
