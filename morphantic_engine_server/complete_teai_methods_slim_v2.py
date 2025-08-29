


#export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

import os

# Respect existing environment; set conservative defaults only when not already set.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Do not override CUDA devices that may be configured by the harness; only hide CUDA if explicitly requested
if os.getenv("AEA_FORCE_CUDA", "0") != "1" and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
import math
from typing import Tuple, List, Dict, Any, Optional
import time
import random
from collections import deque, defaultdict
import uuid
from torch import amp
from scipy.optimize import minimize
# --- PyTorch and Torch diffeq for ODEs ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from typing import Callable
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List
from itertools import combinations_with_replacement

from data_types import BenchmarkResult
# Import the real CausalTapestry from the full implementation
from causal_tapestry_v2 import CausalTapestry

# Global tapestry instance reused across AEA objects within a process
_GLOBAL_TAPESTRY: Optional[CausalTapestry] = None
# Telemetry (single-call generation state write)
try:
    from utils.telemetry import write_archipelago_visualization_state, init_async_telemetry
except Exception:
    write_archipelago_visualization_state = None
    init_async_telemetry = None
import os

# --- Configuration Object (Simplified for PoC) ---
import logging

# Color logging setup
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels and special events"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[37m',       # White
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        # Special event colors
        'LIGHTNING': '\033[93m',  # Bright Yellow
        'EXPLOIT': '\033[92m',    # Bright Green
        'EXPLORE': '\033[94m',    # Bright Blue
        'IMPROVEMENT': '\033[96m', # Bright Cyan
        'NEW_BEST': '\033[92m',   # Bright Green for NEW BEST SO FAR
        'PNN_LOCKED': '\033[95m', # Bright Magenta
        'PNN_OPEN': '\033[93m',   # Bright Yellow
        'PNN_CLOSING': '\033[33m', # Yellow
        'STRESS': '\033[91m',     # Bright Red
        'DREAM': '\033[90m',      # Dark Gray
        'MIGRATION': '\033[97m',  # Bright White
    }
    
    def format(self, record):
        # Check for special event types in the message
        message = record.getMessage()
        
        # Apply special colors based on message content
        if 'NEW BEST SO FAR' in message:
            color = self.COLORS['NEW_BEST']
        elif '[LIGHTNING]' in message:
            color = self.COLORS['LIGHTNING']
        elif 'PNN EXPLOIT' in message:
            color = self.COLORS['EXPLOIT']
        elif 'PNN EXPLORE' in message:
            color = self.COLORS['EXPLORE']
        elif 'improvement:' in message:
            color = self.COLORS['IMPROVEMENT']
        elif 'LOCKED' in message and 'PNN' in message:
            color = self.COLORS['PNN_LOCKED']
        elif 'OPEN' in message and 'PNN' in message:
            color = self.COLORS['PNN_OPEN']
        elif 'CLOSING' in message and 'PNN' in message:
            color = self.COLORS['PNN_CLOSING']
        elif 'STRESS' in message:
            color = self.COLORS['STRESS']
        elif 'DREAM' in message:
            color = self.COLORS['DREAM']
        elif 'MIGRATION' in message:
            color = self.COLORS['MIGRATION']
        else:
            # Default color based on log level
            color = self.COLORS.get(record.levelname, self.COLORS['INFO'])
        
        # Apply color to the message
        record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


# Configure logging for normal AEA output with file output
import os
from datetime import datetime

from utils.detailed_logger import get_logger 

logger = get_logger()



# Configure module logger conservatively; avoid global side effects during import.
# File logging can be enabled by setting TEAI_ENABLE_FILE_LOG=1.

      
# You may need to add this to your imports at the top of the file
from scipy.optimize import minimize

class ExploitationToolkit:
    def __init__(self, bounds, safe_eval_func, safe_eval_batch_func=None):
        self.bounds = bounds
        self.safe_eval = safe_eval_func
        # Optional batch evaluator for vectorized evaluation
        self.safe_eval_batch = safe_eval_batch_func
        # Wrap for scipy minimize (expects scalar function)
        self.objective_func = lambda x: self.safe_eval(x)

    def nelder_mead_step(self, solution: np.ndarray) -> np.ndarray:
        """
        Performs a single iteration (or a few steps) of the Nelder-Mead algorithm.
        """
        # The `maxiter` option controls how many steps the optimizer takes.
        # We keep it very low (e.g., 1-3) to manage our NFE budget per generation.
        options = {'maxiter': 2, 'adaptive': True}

        # Convert bounds to scipy format: list of (lower, upper) tuples for each dimension
        low, high = self.bounds
        scipy_bounds = [(low, high)] * len(solution)

        # Wrap objective to clamp inputs within bounds to prevent overflow/NaNs
        def clamped_obj(x):
            x = np.asarray(x, dtype=float)
            x_clamped = np.clip(x, low, high)
            return float(self.objective_func(x_clamped))

        # The `minimize` function will run the local search.
        res = minimize(
            clamped_obj,
            solution,
            method='Nelder-Mead',
            bounds=scipy_bounds,
            options=options
        )

        # Return the improved solution found by the optimizer (clamped)
        x_opt = np.asarray(res.x, dtype=float)
        return np.clip(x_opt, low, high)

    def champion_linesearch_step(self, solution: np.ndarray, population_mean: np.ndarray, dimension: int) -> np.ndarray:
        """
        Performs a single step of the Champion Linesearch.
        (This is the logic you already have, now encapsulated in a method).
        """
        direction = solution - population_mean
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return solution # No direction to search

        direction /= norm
        
        low, high = self.bounds
        step_size = 0.1 * (high - low) / np.sqrt(dimension)
        
        cand1 = np.clip(solution - step_size * direction, low, high)
        cand2 = np.clip(solution + step_size * direction, low, high)

        f_current = self.objective_func(solution)
        f1 = self.objective_func(cand1)
        f2 = self.objective_func(cand2)

        if f1 < f_current or f2 < f_current:
            return cand1 if f1 <= f2 else cand2
        else:
            return solution # No improvement

    def trust_region_nelder_mead(self, solution: np.ndarray, radius: float) -> np.ndarray:
        low, high = self.bounds
        local_lb = np.maximum(low, solution - radius)
        local_ub = np.minimum(high, solution + radius)
        def local_obj(x):
            x = np.asarray(x, dtype=float)
            x_clamped = np.clip(x, local_lb, local_ub)
            return float(self.objective_func(x_clamped))
        try:
            res = minimize(
                local_obj,
                solution,
                method='Nelder-Mead',
                options={'maxiter': 2, 'adaptive': True}
            )
            x_opt = np.asarray(res.x, dtype=float)
            return np.clip(x_opt, low, high)
        except Exception:
            return solution

    def powell_step(self, solution: np.ndarray) -> np.ndarray:
        """Single Powell step with a tiny budget and trust‑region clamp."""
        low, high = self.bounds
        try:
            # Clamp inputs inside objective to avoid numeric overflows outside bounds
            def clamped_obj(x):
                x = np.asarray(x, dtype=float)
                x_clamped = np.clip(x, low, high)
                return float(self.objective_func(x_clamped))

            res = minimize(clamped_obj, solution, method='Powell', options={'maxiter': 2})
            x_opt = np.asarray(res.x, dtype=float)
            # Clamp to trust region sized like Nelder‑Mead radius
            radius = 0.5 * (high - low) / max(1.0, np.sqrt(len(solution)))
            step = x_opt - solution
            nrm = np.linalg.norm(step)
            if nrm > radius:
                x_opt = solution + step * (radius / nrm)
            return np.clip(x_opt, low, high)
        except Exception:
            return solution

    # --- MODIFIED: cmaes_step with Strict NFE Budgeting ---
    def cmaes_step(self, solution: np.ndarray, radius: float, budget: int,
                   dir_hint: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Separable CMA-ES burst with a strict NFE budget.
        Terminates exactly when the budget is spent.
        """
        try:
            x0 = np.asarray(solution, dtype=float).copy()
            d = x0.size
            low, high = self.bounds
            lb = np.maximum(low, x0 - radius)
            ub = np.minimum(high, x0 + radius)

            # Initial evaluation counts as 1 NFE
            f_best = float(self.objective_func(x0))
            evals_used = 1
            x_best = x0.copy()
            
            # Track heavy operations (CMA-ES evaluations)
            if hasattr(self, '_heavy_ops_count'):
                self._heavy_ops_count += 1
            
            if budget <= evals_used:
                return x_best, evals_used

            lam = int(min(max(4, 4 + np.floor(3*np.log(d))), getattr(cfg, 'cmaes_lambda_cap', 16)))
            if lam < 4: return x_best, evals_used
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = (w / w.sum()).astype(float)
            
            sigma = 0.3 * radius
            var = np.full(d, (sigma**2) / max(1.0, d))
            m = x0.copy()
            
            mu_eff = 1.0 / (w**2).sum()
            c_cov = min(0.5, (2.0 + mu_eff) / (d + 2.0 + 2.0 * mu_eff))

            while evals_used < budget:
                # Check if we have enough budget for a full generation
                if evals_used + lam > budget:
                    break # Not enough budget for another full generation

                Z = np.random.randn(lam, d)
                if dir_hint is not None:
                    n_hint = np.linalg.norm(dir_hint)
                    if n_hint > 1e-9:
                        Z[0] = 0.5 * Z[0] + 0.5 * (dir_hint / n_hint)

                X = m + Z * np.sqrt(var)
                X = np.clip(X, lb, ub)

                # Evaluate generation and count NFE
                fit = np.array([float(self.objective_func(x)) for x in X])
                evals_used += lam
                
                # Track heavy operations (CMA-ES generation evaluations)
                if hasattr(self, '_heavy_ops_count'):
                    self._heavy_ops_count += lam

                i_best_in_gen = int(fit.argmin())
                if fit[i_best_in_gen] < f_best:
                    f_best = float(fit[i_best_in_gen])
                    x_best = X[i_best_in_gen].copy()

                idx = np.argsort(fit)[:mu]
                Xsel = X[idx]
                m = np.sum((w[:, None] * Xsel), axis=0)
                
                Yrel = (Xsel - m)
                denom = np.sqrt(np.maximum(var, 1e-20))
                var = (1.0 - c_cov) * var + c_cov * np.sum((w[:, None] * ((Yrel / denom)**2)), axis=0) * var

            return x_best, int(evals_used)
        except Exception:
            return np.asarray(solution, dtype=float), 1 # Return 1 for the initial eval

    # --- NEW: Surrogate-Assisted CMA-ES Step ---
    def cmaes_surrogate_step(self, solution: np.ndarray, radius: float,
                             dir_hint: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Runs CMA-ES on a cheap local surrogate model to save NFE.
        """
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge

            x0 = np.asarray(solution, dtype=float).copy()
            d = x0.size
            low, high = self.bounds
            
            # 1. Sample data to build the surrogate
            n_samples = int(getattr(cfg, 'surrogate_sampling_nfe', 2 * d))
            sample_points = [x0]
            noise = np.random.randn(n_samples - 1, d) * (0.5 * radius)
            sample_points.extend([np.clip(x0 + n, low, high) for n in noise])
            
            fitness_values = np.array([float(self.objective_func(p)) for p in sample_points])
            evals_used = n_samples
            
            # Track heavy operations (surrogate sampling evaluations)
            if hasattr(self, '_heavy_ops_count'):
                self._heavy_ops_count += n_samples

            best_idx_initial = np.argmin(fitness_values)
            f_best_initial = fitness_values[best_idx_initial]
            x_best_initial = sample_points[best_idx_initial]

            # 2. Build the surrogate model (Quadratic)
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_features = poly.fit_transform(sample_points)
            model = Ridge(alpha=1e-3)
            model.fit(X_features, fitness_values)

            def surrogate_objective(x):
                x_features = poly.transform(x.reshape(1, -1))
                return model.predict(x_features)[0]

            # 3. Run CMA-ES on the cheap surrogate
            # Give it a large virtual budget as it's nearly free
            x_surrogate_best, _ = self.cmaes_step(x_best_initial, radius, budget=500, dir_hint=dir_hint)

            # 4. Final true evaluation
            f_final = float(self.objective_func(x_surrogate_best))
            evals_used += 1
            
            # Track heavy operations (surrogate final evaluation)
            if hasattr(self, '_heavy_ops_count'):
                self._heavy_ops_count += 1

            # 5. Return the best solution found
            if f_final < f_best_initial:
                return x_surrogate_best, evals_used
            else:
                return x_best_initial, evals_used
        except Exception:
            return np.asarray(solution, dtype=float), 1
        
        
        
    
class ContextualUCBSelector:
    """Contextual discounted UCB over discrete context keys (tuples).
    Maintains a separate UCB state per context bucket, with a discount factor to adapt under non‑stationarity.
    """
    def __init__(self, strategies: List[str], exploration_factor: float = 0.5, discount: float = 0.97):
        self.strategies = list(strategies)
        self.C = float(exploration_factor)
        self.gamma = float(discount)
        self.state: Dict[tuple, Dict[str, Dict[str, float]]] = {}
        self.total_counts: Dict[tuple, float] = {}

    def _ensure_bucket(self, ctx: tuple):
        if ctx not in self.state:
            self.state[ctx] = {s: {"count": 0.0, "value": 0.0} for s in self.strategies}
            self.total_counts[ctx] = 0.0

    def select_strategy(self, context_key: tuple) -> str:
        self._ensure_bucket(context_key)
        # Bootstrap: try unpulled arms first
        for s, rec in self.state[context_key].items():
            if rec["count"] < 1.0:
                return s
        total = max(1.0, self.total_counts[context_key])
        best_s, best_score = None, -1e18
        for s, rec in self.state[context_key].items():
            mean = rec["value"] / max(1e-12, rec["count"])
            bonus = self.C * math.sqrt(math.log(total + 1.0) / max(1e-12, rec["count"]))
            score = mean + bonus
            if score > best_score:
                best_score, best_s = score, s
        return best_s or self.strategies[0]

    def update(self, context_key: tuple, strategy: str, reward: float):
        self._ensure_bucket(context_key)
        # Discount all arms in this context
        for s in self.strategies:
            self.state[context_key][s]["count"] *= self.gamma
            self.state[context_key][s]["value"] *= self.gamma
        # Update chosen arm
        self.state[context_key][strategy]["count"] += 1.0
        self.state[context_key][strategy]["value"] += float(reward)
        self.total_counts[context_key] = self.total_counts.get(context_key, 0.0) * self.gamma + 1.0



    

# =============================================================================
# LIGHTWEIGHT BIOLOGICAL MECHANISMS
# =============================================================================

class PNN_STATE(Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    LOCKED = "LOCKED"

# In complete_teai_methods_slim.py
# REPLACE the existing PerineuronalNet class with this one.

class PerineuronalNet:
    """
    Enhanced PNN with a finite Exploitation Budget to prevent stagnation.
    The PNN is now responsible for tracking its own performance while LOCKED
    and draining its budget. The decision to unlock is handled at the
    archipelago level.
    """
    def __init__(self, cell_id: str, exploit_budget: float = 100.0):
        self.cell_id = cell_id
        self.state = PNN_STATE.OPEN
        self.generations_in_phase = 0
        self.plasticity_multiplier = 1.0
        self.stability_history = deque(maxlen=10)
        self.refractory_until = None  # Generation until which cell must stay OPEN

        # --- NEW: Exploitation Budget attributes ---
        self.exploit_budget = exploit_budget
        self.initial_exploit_budget = exploit_budget
        self.fitness_at_lock: Optional[float] = None
        self.best_fitness_in_lock_phase: Optional[float] = None
        # Recent exploitation telemetry (used to modulate decay)
        self.recent_exploit_evals: float = 0.0
        self.recent_exploit_improvement: float = 0.0

        # --- NEW: Track fitness history during LOCKED phase (for downstream logic)
        self.locked_phase_fitness_history: List[float] = []
        self.locked_phase_start_gen: Optional[int] = None
        self.locked_generation = None
        self.best_fitness_when_locked = None

    def note_exploit_outcome(self, evals_used: float, improvement: float) -> None:
        """Record outcome of a recent exploit step to inform decay policy."""
        try:
            self.recent_exploit_evals = float(max(0.0, evals_used))
            self.recent_exploit_improvement = float(improvement)
        except Exception:
            pass

    def update(self, current_fitness: float, generation: int, island_median_fitness: float = None, locking_fitness_threshold: float = None) -> None:
        """Updates the PNN state and, if locked, drains the exploitation budget.

        Args:
            current_fitness: Latest fitness value for this cell
            generation: Current generation index
            island_median_fitness: Backwards-compatible gating metric (legacy)
            locking_fitness_threshold: If provided, stricter lock gate; only allow CLOSING when
                current_fitness <= threshold (e.g., top 25% best on island)
        """
        # Always accumulate stability history so OPEN/CLOSING decisions have signal
        self.stability_history.append(current_fitness)

        # Check refractory period first
        if self.refractory_until is not None and generation < self.refractory_until:
            # Remain OPEN; allow history accumulation but block phase advancement
            self.generations_in_phase = 0
            return
        elif self.refractory_until is not None and generation >= self.refractory_until:
            self.refractory_until = None # Refractory period over

        self.generations_in_phase += 1

        if self.state == PNN_STATE.OPEN:
            if len(self.stability_history) >= 5:
                # Check for fitness stability to transition to CLOSING
                recent_mean = np.mean(list(self.stability_history)[-5:])
                relative_change = abs(current_fitness - recent_mean) / max(1e-8, abs(recent_mean))
                if relative_change < 0.05: # If fitness is stable within 5%
                    # Prefer stricter threshold if provided; otherwise fallback to median-based gating
                    if locking_fitness_threshold is not None:
                        if current_fitness <= locking_fitness_threshold:
                            self.state = PNN_STATE.CLOSING
                            self.generations_in_phase = 0
                            logger.info(f"    PNN TRANSITION: Cell {self.cell_id} fitness {current_fitness:.6f} <= top-25% threshold {locking_fitness_threshold:.6f}. Allowing CLOSING transition.")
                        else:
                            # Stable but not in top tier → keep OPEN
                            self.generations_in_phase = 0
                            logger.info(f"    PNN LOCKOUT: Cell {self.cell_id} fitness {current_fitness:.6f} > top-25% threshold {locking_fitness_threshold:.6f}. Forcing OPEN state.")
                    elif island_median_fitness is not None:
                        if current_fitness <= island_median_fitness:
                            self.state = PNN_STATE.CLOSING
                            self.generations_in_phase = 0
                            logger.info(f"    PNN LOCKOUT: Cell {self.cell_id} fitness {current_fitness:.6f} <= median {island_median_fitness:.6f}. Allowing CLOSING transition.")
                        else:
                            self.generations_in_phase = 0
                            logger.info(f"    PNN LOCKOUT: Cell {self.cell_id} fitness {current_fitness:.6f} > median {island_median_fitness:.6f}. Forcing OPEN state.")
                    else:
                        # No thresholds provided → allow CLOSING based on stability alone
                        self.state = PNN_STATE.CLOSING
                        self.generations_in_phase = 0

        elif self.state == PNN_STATE.CLOSING:
            # Gradually decrease plasticity
            progress = min(1.0, self.generations_in_phase / 5.0)
            self.plasticity_multiplier = max(0.2, 1.0 - 0.8 * progress)
            if progress >= 1.0:
                # Transition to LOCKED
                self.state = PNN_STATE.LOCKED
                self.plasticity_multiplier = 0.2
                self.generations_in_phase = 0
                self.fitness_at_lock = current_fitness
                self.best_fitness_in_lock_phase = current_fitness
                self.exploit_budget = self.initial_exploit_budget # Reset budget upon locking
                # Initialize locked-phase tracking
                self.locked_phase_fitness_history = [float(current_fitness)]
                self.locked_phase_start_gen = int(generation)

        elif self.state == PNN_STATE.LOCKED:
            self.plasticity_multiplier = 0.2 # Maintain low plasticity
            
            # --- Enhanced Budget Decay Logic ---
            # Maintain fitness history while locked
            try:
                self.locked_phase_fitness_history.append(float(current_fitness))
                # Keep history bounded to avoid growth
                if len(self.locked_phase_fitness_history) > 100:
                    self.locked_phase_fitness_history = self.locked_phase_fitness_history[-100:]
            except Exception:
                pass
            # Update the best fitness achieved during this locked phase
            if self.best_fitness_in_lock_phase is None:
                # Initialize if not set
                self.best_fitness_in_lock_phase = current_fitness
                stagnation_generations = 0
            elif current_fitness < self.best_fitness_in_lock_phase:
                self.best_fitness_in_lock_phase = current_fitness
                # REWARD: Stagnation is reset for making progress
                stagnation_generations = 0
            else:
                # PENALTY: No improvement, increment stagnation counter
                stagnation_generations = self.generations_in_phase

            # 1) Time-based baseline decay each generation while LOCKED
            time_decay = float(getattr(cfg, 'pnn_time_decay_per_gen', 0.2))

            # 2) Stagnation penalty scales with how long we've been unproductive
            if stagnation_generations >= getattr(cfg, 'pnn_stagnation_penalty_threshold', 6):
                stagnation_decay = float(getattr(cfg, 'pnn_stagnation_penalty_multiplier', 2.0)) * (1.0 + (stagnation_generations / 6.0))
            else:
                stagnation_decay = 0.2 * (stagnation_generations / 6.0)

            # 3) Usage/performance modulation from most recent exploit burst
            #    Fewer evals and larger improvement => smaller drain; failures => larger drain
            used = float(getattr(self, 'recent_exploit_evals', 0.0))
            imp = float(getattr(self, 'recent_exploit_improvement', 0.0))
            success = imp > float(getattr(cfg, 'pnn_improvement_eps', 1e-9))
            usage_scale_success = float(getattr(cfg, 'pnn_drain_success_factor', 0.5))
            usage_scale_failure = float(getattr(cfg, 'pnn_drain_failure_factor', 1.5))
            usage_decay = (usage_scale_success if success else usage_scale_failure) * used

            total_decay = time_decay + stagnation_decay + usage_decay
            if total_decay > 0:
                self.exploit_budget = max(0.0, self.exploit_budget - total_decay)

            # Softly forget usage telemetry so it fades over time
            self.recent_exploit_evals *= 0.5
            self.recent_exploit_improvement *= 0.5

            # NOTE: The decision to unlock is now made by the _strategic_pnn_unlocking method.
            
    def force_unlock(self, generation: int, refractory_period: int = 5):
        """Forcibly unlocks the PNN and sets a refractory period to prevent immediate re-locking."""
        self.state = PNN_STATE.OPEN
        self.plasticity_multiplier = 1.0
        self.generations_in_phase = 0
        self.stability_history.clear()
        self.fitness_at_lock = None
        self.best_fitness_in_lock_phase = None
        self.locked_phase_fitness_history = []
        self.locked_phase_start_gen = None
        # Dynamic refractory based on ambient stress: high stress => longer refractory (harder to re-lock quickly)
        base_ref = int(refractory_period)
        try:
            # Estimate ambient stress from last known stability window
            recent = list(self.stability_history)[-5:]
            stress_proxy = 0.0
            if recent:
                avg = float(np.mean(recent))
                std = float(np.std(recent))
                if avg > 1e-8:
                    stress_proxy = min(1.0, std / avg)
            stress_bias = 1.0 + 1.5 * stress_proxy
            base_ref = int(max(refractory_period, round(base_ref * stress_bias)))
        except Exception:
            pass
        self.refractory_until = generation + base_ref
        
        
        
        
        
        
        
        
class CircadianController:
    """Lightweight circadian rhythm controller"""
    def __init__(self, island_id: int, num_islands: int):
        self.period = cfg.circadian_period  # generations
        base_offset = (island_id / max(num_islands, 1)) * self.period
        jitter = (0.5 - (hash((island_id, num_islands)) % 1000) / 1000.0) * cfg.circadian_phase_jitter * self.period
        self.phase_offset = base_offset + jitter
        self.is_day = True
        self.plasticity_multiplier = 1.0
        
    def update(self, generation: int) -> None:
        time_in_cycle = (generation + self.phase_offset) % self.period
        # Wider day fraction for high-D or dynamic modes if available via cfg
        day_frac = getattr(cfg, 'circadian_day_fraction_static', 0.25)
        if getattr(cfg, 'dimension', None) and int(getattr(cfg, 'dimension')) >= getattr(cfg, 'high_dim_threshold', 30):
            day_frac = getattr(cfg, 'circadian_day_fraction_highd', 0.45)
        self.is_day = time_in_cycle < self.period * day_frac
        self.plasticity_multiplier = 1.0 if self.is_day else 0.5

class StressField:
    """Lightweight stress field implementation"""
    def __init__(self, device: torch.device):
        self.device = device
        self.stress_values = {}

    def update(self, 
               population_by_island: Dict[int, List['Cell']], 
               global_best_fitness: float) -> None:
        """
        Updates stress values for all cells based on a combination of their performance
        relative to the global best and how long they have been stuck in a LOCKED state.

        Args:
            population_by_island: A dictionary mapping island IDs to lists of cells.
            global_best_fitness: The best fitness value found so far across all islands.
        """
        # --- START REFACTOR ---
        
        # Ensure global_best_fitness is a valid float to prevent errors
        try:
            gb_fit = float(global_best_fitness)
            if not np.isfinite(gb_fit):
                # If global best is inf, we can't compute relative stress. Fallback to a default.
                for island_cells in population_by_island.values():
                    for cell in island_cells:
                        cell.local_stress = cfg.default_stress
                return
        except (ValueError, TypeError):
            # Fallback if conversion fails
            for island_cells in population_by_island.values():
                for cell in island_cells:
                    cell.local_stress = cfg.default_stress
            return

        for island_cells in population_by_island.values():
            for cell in island_cells:
                if not hasattr(cell, 'fitness_history') or not cell.fitness_history:
                    cell.local_stress = cfg.default_stress
                    continue

                current_fitness = cell.fitness_history[-1]

                # 1. Calculate Performance-Based Stress (Relative Stagnation)
                # How far is this cell from the global best?
                # A cell with fitness much worse than the global best has high stress.
                # We add a small epsilon for numerical stability if global_best_fitness is near zero.
                denominator = abs(gb_fit) + 1e-9
                performance_stress = (current_fitness - gb_fit) / denominator
                # Clamp at 0; a cell better than the global best has zero performance stress.
                performance_stress = max(0.0, performance_stress)

                # 2. Calculate Time-Based Stress (Generations Stuck)
                # This only applies to LOCKED cells to penalize long-term stagnation.
                time_stress = 0.0
                if cell.pnn.state == PNN_STATE.LOCKED:
                    # Stress increases linearly with the number of generations in the LOCKED phase,
                    # capping out at 1.0 after a configured period.
                    stagnation_period = getattr(cfg, 'stress_v2_time_stagnation_period', 20.0)
                    time_stress = min(1.0, cell.pnn.generations_in_phase / stagnation_period)

                # 3. Combine Metrics into a Final Stress Score
                # The final stress is a weighted average of the two components.
                w_perf = getattr(cfg, 'stress_v2_relative_stagnation_weight', 0.7)
                w_time = getattr(cfg, 'stress_v2_time_stagnation_weight', 0.3)
                
                final_stress = (w_perf * performance_stress) + (w_time * time_stress)
                
                # Ensure the final stress value is capped at 1.0
                final_stress = min(1.0, final_stress)

                # Assign the new, meaningful stress value to the cell
                cell.local_stress = final_stress
                self.stress_values[f"{cell.id}"] = final_stress
        # --- END REFACTOR ---
        
        
        
        
        
class StressImmuneLayer:
    """Lightweight immune cache implementation"""
    def __init__(self, transposition_engine=None, sigma_threshold: float = 1.0, max_cache: int = 128):
        # FIX: Use OrderedDict for proper LRU
        from collections import OrderedDict
        self.memory_cells = OrderedDict()
        self.max_cache = max_cache
        
    def add_memory(self, signature: str):
        # FIX: Proper LRU implementation
        if signature in self.memory_cells:
            # Move to end (most recently used)
            self.memory_cells.move_to_end(signature)
        else:
            self.memory_cells[signature] = True
            # Remove oldest if over limit
            if len(self.memory_cells) > self.max_cache:
                self.memory_cells.popitem(last=False)  # Remove oldest (FIFO)

# Lightweight threat signature computation
def compute_threat_signature(fitness_history: List[float], generation: int):
    if not fitness_history:
        return "unknown"
    recent = fitness_history[-5:] if len(fitness_history) >= 5 else fitness_history
    avg = np.mean(recent)
    trend = "improving" if len(recent) > 1 and recent[-1] > recent[0] else "declining"
    return f"{trend}_{avg:.3f}_{generation}"

# Normalized tapestry weighting function (expects normalized fitness in [0,1])
def tapestry_weight(norm_fitness: float, low_threshold: float = 0.2, high_threshold: float = 0.8):
    """
    Calculate tapestry influence weight as PERCENTAGE (0-100%).
    Input and thresholds are on the normalized scale where:
      - 0.0 corresponds to the problem optimum
      - 1.0 corresponds to the problem success_threshold
    HIGH normalized fitness (worse) => HIGH tapestry influence.

    Returns: 0.0% at/below low_threshold, 100.0% at/above high_threshold, linear in between.
    """
    try:
        nf = float(norm_fitness)
        lo = float(low_threshold)
        hi = float(high_threshold)
    except Exception:
        return 0.0
    if nf <= lo:
        return 0.0
    if nf >= hi:
        return 100.0
    return float((nf - lo) / max(1e-12, (hi - lo)) * 100.0)

def percentile_weight_from_history(current_value: float, history: List[float], 
                                  low_pct: float = 10.0, high_pct: float = 90.0, 
                                  default_weight: float = 0.5) -> Tuple[float, float, float, int]:
    """
    Calculate percentile-based weight from fitness history.
    
    Args:
        current_value: Current best fitness value
        history: List of historical best fitness values
        low_pct: Low percentile threshold (default 10%)
        high_pct: High percentile threshold (default 90%)
        default_weight: Default weight when insufficient history
        
    Returns:
        Tuple of (weight, low_threshold, high_threshold, n_history)
    """
    if len(history) < 5:  # Need minimum history for meaningful percentiles
        return default_weight, 0.0, 0.0, len(history)
    
    # Calculate percentiles from history
    lo_thr = float(np.percentile(history, low_pct))
    hi_thr = float(np.percentile(history, high_pct))
    
    # Calculate weight based on current value's position in the distribution
    if current_value <= lo_thr:
        weight = 0.0  # Doing well, no tapestry influence
    elif current_value >= hi_thr:
        weight = 1.0  # Struggling, full tapestry influence
    else:
        # Linear interpolation between low and high thresholds
        weight = (current_value - lo_thr) / max(1e-8, hi_thr - lo_thr)
    
    return weight, lo_thr, hi_thr, len(history)


# === Problem bounds helper (SO & MO) ===
BOUNDS_MAP: Dict[str, Any] = {
    # Single-objective
    "sphere": (-5.12, 5.12),
    "rastrigin": (-5.12, 5.12),
    "rastrigin-dynamic": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    "ackley-dynamic": (-32.768, 32.768),
    "griewank": (-600.0, 600.0),
    "griewank-dynamic": (-600.0, 600.0),
    "schwefel": (-500.0, 500.0),

    # Additions from fairness_benchmark_eval_v2.py context
    "happycat": (-5.0, 5.0),
    "dixonprice": (-10.0, 10.0),
    "zakharov": (-5.0, 10.0),
    "composition1": (-5.0, 5.0),

    # ZDT/DTLZ (decision space)
    "zdt1": (0.0, 1.0),
    "zdt2": (0.0, 1.0),
    "zdt3": (0.0, 1.0),
    "zdt4": None,  # x1 in [0,1], x2..n in [-5,5]
    "zdt6": (0.0, 1.0),
    "dtlz1": (0.0, 1.0),
    "dtlz2": (0.0, 1.0),
}

def problem_bounds(name: str, dim: int):
    name = (name or "").lower()
    if name == "zdt4":
        lb = np.array([0.0] + [-5.0] * (dim - 1), dtype=float)
        ub = np.array([1.0] + [5.0] * (dim - 1), dtype=float)
        return lb, ub
    lbub = BOUNDS_MAP.get(name, (-5.12, 5.12))
    if isinstance(lbub, tuple) and isinstance(lbub[0], (int, float)):
        return float(lbub[0]), float(lbub[1])
    return lbub


# === MO utilities: HV ref points and island weights (presets) ===
MO_HV_REF: Dict[str, np.ndarray] = {
    "zdt1": np.array([1.1, 1.1], dtype=float),
    "zdt2": np.array([1.1, 1.1], dtype=float),
    "zdt3": np.array([1.1, 1.1], dtype=float),
    "zdt4": np.array([1.1, 1.1], dtype=float),
    "zdt6": np.array([1.1, 1.1], dtype=float),
    # Future 3D HV reference examples
    "dtlz1": np.array([0.6, 0.6, 0.6], dtype=float),
    "dtlz2": np.array([1.1, 1.1, 1.1], dtype=float),
}

MO_ISLAND_WEIGHTS: Dict[str, List[np.ndarray]] = {
    # 5 islands, ZDT (M=2)
    k: [
        np.array([1.00, 0.00], dtype=float),
        np.array([0.75, 0.25], dtype=float),
        np.array([0.50, 0.50], dtype=float),
        np.array([0.25, 0.75], dtype=float),
        np.array([0.00, 1.00], dtype=float),
    ]
    for k in ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
}

MO_ISLAND_WEIGHTS.update({
    # 5 islands, DTLZ (M=3), H=3 grid (subset)
    "dtlz1": [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.5, 0.5, 0.0], dtype=float),
        np.array([1/3, 1/3, 1/3], dtype=float),
    ],
    "dtlz2": [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.5, 0.5, 0.0], dtype=float),
        np.array([1/3, 1/3, 1/3], dtype=float),
    ],
})


# Import the centralized configuration loader
from config import cfg


# === Optional: load objective overrides from config/ ===
def _load_objective_overrides() -> Optional[Dict[str, Any]]:
    try:
        candidates = [
            r'C:\Users\wes\new_aea_repository_extraction-1\benchmark_overrides.json',
            r'C:\Users\wes\new_aea_repository_extraction-1\objectives.json',
        ]
        for p in candidates:
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
    except Exception:
        return None
    return None

_OBJ_OVERRIDES = _load_objective_overrides()
if isinstance(_OBJ_OVERRIDES, dict):
    # Override HV ref points if provided
    try:
        hv_map = _OBJ_OVERRIDES.get('MO_HV_REF')
        if isinstance(hv_map, dict):
            for k, v in hv_map.items():
                try:
                    MO_HV_REF[str(k).lower()] = np.asarray(v, dtype=float)
                except Exception:
                    pass
    except Exception:
        pass
    # Override per-problem island weights if provided
    try:
        w_map = _OBJ_OVERRIDES.get('MO_ISLAND_WEIGHTS')
        if isinstance(w_map, dict):
            for k, arrs in w_map.items():
                try:
                    MO_ISLAND_WEIGHTS[str(k).lower()] = [np.asarray(a, dtype=float) for a in arrs]
                except Exception:
                    pass
    except Exception:
        pass
    # Optionally override universal island roles / n_islands via config
    try:
        roles = _OBJ_OVERRIDES.get('island_roles')
        if isinstance(roles, list) and roles:
            cfg.island_roles = [str(r) for r in roles]
    except Exception:
        pass
    try:
        n_is = _OBJ_OVERRIDES.get('n_islands')
        if isinstance(n_is, (int, float)) and int(n_is) > 0:
            cfg.n_islands = int(n_is)
    except Exception:
        pass
# CAN: Here is a Config class that includes ALL variables from benchmark_overrides.json (including AEA_CFG, DIMENSIONS, ENABLE_ALGOS, EVAL_BACKEND, FIXED_NFE_BUDGET, NUM_RUNS_PER_CASE, ONLY_DIMS, ONLY_PROBLEMS, OUTPUT_DIR), as well as all the algorithmic parameters. All variables are present and named exactly as in the JSON for maximum compatibility.





# --- Integrated ODE Gene Module ---
class NeuralODEFunc(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        try:
            # Ensure hidden_dim is a valid integer
            hidden_dim = int(hidden_dim)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        except Exception as e:
            logger.error(f"Error creating NeuralODEFunc with hidden_dim={hidden_dim}: {e}")
            raise

    def forward(self, t, h):
        return self.net(h)

class ContinuousDepthGeneModule(nn.Module):
    def __init__(self, gene_type: str, variant_id: int):
        super().__init__()
        self.gene_type = gene_type
        self.variant_id = variant_id
        self.gene_id = f"{gene_type}{variant_id}-{uuid.uuid4().hex[:4]}"
        
        # Ensure dimensions are valid integers
        feature_dim = int(cfg.feature_dim) if hasattr(cfg, 'feature_dim') and cfg.feature_dim else 30
        hidden_dim = int(cfg.hidden_dim) if hasattr(cfg, 'hidden_dim') and cfg.hidden_dim else 32
        
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        self.ode_func = NeuralODEFunc(hidden_dim)
        self.log_depth = nn.Parameter(torch.tensor(0.0))
        self.is_active = True
        self.fitness_contribution = 0.0
        # Adaptive features
        self.depth_ema = 1.0  # Start with baseline depth
        self.fitness_gradient = 0.0
        # Ensure the skip flag exists and defaults to False
        self._skip_ode = False

    def compute_depth(self) -> torch.Tensor:
        return torch.exp(self.log_depth).clamp(cfg.min_depth, cfg.max_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIX: Performance optimization - skip ODE when not needed
        # Check if we should skip ODE computation for performance
        if hasattr(self, '_skip_ode') and self._skip_ode:
            # Use simple linear projection instead of ODE
            h = self.input_projection(x)
            h_final = self.output_projection(h)
            return h_final
        
        # Dynamically adjust integration points based on fitness progress
        if abs(self.fitness_gradient) < 0.001:  # Stable fitness
            integration_points = 2  # Minimal integration
        elif abs(self.fitness_gradient) > 0.1:  # Rapid change
            integration_points = 5  # Full integration
        else:
            integration_points = 3  # Medium integration
            
        # Use learned depth scaling
        effective_depth = self.compute_depth() * self.depth_ema
        t = torch.linspace(0, effective_depth.item(), integration_points).to(x.device)
        
        # Option: Use simpler solver for fewer steps
        if integration_points <= 2:
            # Direct linear interpolation instead of ODE
            h = self.input_projection(x)
            h_final = h + self.ode_func(0, h) * effective_depth
        else:
            # Full ODE integration only when needed
            h = self.input_projection(x)
            h_trajectory = odeint(self.ode_func, h, t, method='euler')  # Faster solver
            h_final = h_trajectory[-1]
            
        return self.output_projection(h_final)
    


# AdaptiveODE functionality now integrated into ContinuousDepthGeneModule

    
# --- Integrated Quantum Gene Module ---
class QuantumGeneModule(ContinuousDepthGeneModule):
    def __init__(self, gene_type: str, variant_id: int):
        super().__init__(gene_type, variant_id)
        self.alpha = nn.Parameter(torch.tensor(1.0 / np.sqrt(2.0)))
        self.beta = nn.Parameter(torch.tensor(1.0 / np.sqrt(2.0)))
        # Use the same safeguarded dimension
        hidden_dim = int(cfg.hidden_dim) if hasattr(cfg, 'hidden_dim') and cfg.hidden_dim else 32
        self.alt_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(self.alpha**2 + self.beta**2)
        alpha_norm, beta_norm = self.alpha / norm, self.beta / norm
        h_ode = super().forward(x)
        h_alt = self.alt_path(self.input_projection(x))
        h_alt = self.output_projection(h_alt)
        return alpha_norm * h_ode + beta_norm * h_alt

# --- Integrated Self-Modifying Architecture ---
class SelfModifyingArchitecture:
    def __init__(self, cell):
        self.cell = cell
        self.modification_history = []

    def decide_and_apply_modification(self, fitness_trend: float):
        # NEW: ensure we're not mutating a shared genome instance
        if getattr(cfg, 'torch_genome_enabled', False):
            try:
                self.cell._ensure_private_genome()
            except Exception:
                pass
        if fitness_trend > -0.01:
            mod_type = random.choice(['add_gene', 'remove_gene'])
            
            if mod_type == 'add_gene' and len(self.cell.genome) < cfg.max_genes_per_clone:
                new_gene = ContinuousDepthGeneModule(random.choice(['V','D','J']), random.randint(1,100))
                self.cell.genome.append(new_gene)
                self.modification_history.append("add_gene")
            elif mod_type == 'remove_gene' and len(self.cell.genome) > 1:
                # Convert ModuleList to list to find index, then remove by index
                genome_list = list(self.cell.genome)
                gene_to_remove = random.choice(genome_list)
                gene_index = genome_list.index(gene_to_remove)
                del self.cell.genome[gene_index]
                self.modification_history.append("remove_gene")


class QuantumDreamer:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.denoise_net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, feature_dim)
        )
        self.noise_schedule = torch.linspace(1e-4, 0.02, 100)

    def generate_dream(self, num_dreams: int, device: torch.device) -> List[np.ndarray]:
        # Determine the device from the network's parameters
        model_device = next(self.denoise_net.parameters()).device
        x = torch.randn(num_dreams, self.feature_dim, device=model_device)
        for t in reversed(range(len(self.noise_schedule))):
            noise_pred = self.denoise_net(x)
            alpha = 1.0 - self.noise_schedule[t]
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            if t > 0:
                z = torch.randn_like(x)  # torch.randn_like(x) correctly uses the same device as x
                sigma = torch.sqrt(
                    (1.0 - self.noise_schedule[t - 1]) / (1.0 - self.noise_schedule[t]) * self.noise_schedule[t]
                )
                x += sigma * z
        return [d.detach().cpu().numpy() for d in x]


# Advanced biological mechanisms are now imported from scripts.core.advanced.biological_mechanisms
# The basic implementations below are replaced with the sophisticated versions above

# Causal Tapestry is now imported from scripts.core.causal_tapestry (the networkx version)


    
      
class Cell(nn.Module):
    def __init__(self, cell_id: int, dimension: int, bounds: Tuple[float, float], is_stem: bool = False):
        super().__init__()
        self.id = cell_id
        self.dimension = dimension
        self.bounds = bounds
        
        # CRITICAL: Each cell now holds its own solution vector
        self.solution = np.random.uniform(bounds[0], bounds[1], dimension)
        
        # Initialize advanced PNN with immune layer
        self.pnn = PerineuronalNet(cell_id=str(cell_id))
        self.epigenetic_shadow_mark = torch.randn(dimension) * 0.1
        # Store genes in a plain Python list to avoid ModuleList native mutations
        self.genome: list = []
        # NEW: CoW flags/metadata for genome
        self._genome_shared: bool = False
        self._genome_shared_from: Optional[int] = None
        
        # FIX: Initialize stress-related attributes
        self.local_stress = cfg.stress_threshold # Default stress value
        self.fitness_history = []  # Track fitness history for stress calculation
        
        self.is_stem = is_stem
        if getattr(cfg, 'torch_genome_enabled', False) and not is_stem:
            for gene_type in ['V', 'D', 'J']:
                try:
                    self.genome.append(ContinuousDepthGeneModule(gene_type, random.randint(1, 100)))
                except Exception:
                    # Fallback: skip gene if module construction fails
                    pass
        
        self.architecture_modifier = SelfModifyingArchitecture(self)

    def get_solution(self) -> np.ndarray:
        """Returns the actual solution vector for fitness evaluation."""
        return self.solution

    
    def update_solution(self, delta: np.ndarray):
        """Update solution based on evolved changes"""
        self.solution += delta
        # Ensure bounds
        self.solution = np.clip(self.solution, self.bounds[0], self.bounds[1])
    
    def differentiate(self, target_type: str):
        """Differentiate stem cell into specific gene type"""
        if not self.is_stem:
            return
        
        self.is_stem = False
        if getattr(cfg, 'torch_genome_enabled', False):
            try:
                self.genome.append(ContinuousDepthGeneModule(target_type, random.randint(1, 100)))
            except Exception:
                pass
    
    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """Generate phenotype modifications based on genome"""
        if x is None:
            # Create a dummy input tensor on the cell's tensor device if available
            try:
                dev = self.epigenetic_shadow_mark.device
            except Exception:
                dev = None
            x = torch.zeros(self.dimension, device=dev)
        
        if not self.genome:
            bounds = (-5.12, 5.12)
            random_phenotype = torch.rand_like(x) * (bounds[1] - bounds[0]) + bounds[0]
            return random_phenotype
        outputs = [gene(x) for gene in self.genome]
        phenotype = torch.stack(outputs).mean(dim=0)
        phenotype += self.epigenetic_shadow_mark.to(phenotype.device) * 0.1
        return phenotype

    # Ensure moving the cell also moves unregistered gene modules and tensors
    def to(self, device):
        super().to(device)
        try:
            self.epigenetic_shadow_mark = self.epigenetic_shadow_mark.to(device)
        except Exception:
            pass
        try:
            for gene in self.genome:
                if hasattr(gene, 'to'):
                    gene.to(device)
        except Exception:
            pass
        return self

    # === Copy-on-Write helpers ===
    def _share_genome_from(self, parent: 'Cell'):
        """Make this cell share the parent's genome (zero copy) until a write occurs."""
        self.genome = parent.genome
        self._genome_shared = True
        self._genome_shared_from = parent.id

    def _ensure_private_genome(self):
        """If genome is shared (CoW), materialize a private deep copy before any mutation."""
        if not getattr(cfg, 'torch_genome_enabled', False):
            return
        if getattr(self, '_genome_shared', False):
            new_list = []
            for gene in self.genome:
                if isinstance(gene, QuantumGeneModule):
                    g = QuantumGeneModule(gene.gene_type, gene.variant_id)
                else:
                    g = ContinuousDepthGeneModule(gene.gene_type, gene.variant_id)
                with torch.no_grad():
                    g.load_state_dict(gene.state_dict())
                new_list.append(g)
            self.genome = new_list
            self._genome_shared = False
            self._genome_shared_from = None

    def copy(self, new_id: int = None, device: torch.device = None) -> 'Cell':
        """Create a copy of the cell with CoW genome semantics."""
        if new_id is None:
            new_id = random.randint(50000, 60000)

        new_cell = Cell(new_id, self.dimension, self.bounds, self.is_stem)

        # Solution / tensors
        new_cell.solution = self.solution.copy()
        new_cell.epigenetic_shadow_mark = self.epigenetic_shadow_mark.clone()

        # PNN state (lightweight)
        new_cell.pnn = PerineuronalNet(cell_id=str(new_id))
        new_cell.pnn.state = self.pnn.state
        new_cell.pnn.generations_in_phase = self.pnn.generations_in_phase
        new_cell.pnn.stability_history = self.pnn.stability_history.copy()
        new_cell.pnn.refractory_until = self.pnn.refractory_until
        new_cell.pnn.exploit_budget = getattr(self.pnn, 'exploit_budget', new_cell.pnn.exploit_budget)
        new_cell.pnn.initial_exploit_budget = getattr(self.pnn, 'initial_exploit_budget', new_cell.pnn.initial_exploit_budget)
        new_cell.pnn.fitness_at_lock = getattr(self.pnn, 'fitness_at_lock', None)
        new_cell.pnn.best_fitness_in_lock_phase = getattr(self.pnn, 'best_fitness_in_lock_phase', None)

        # Stress/fitness histories
        new_cell.local_stress = self.local_stress
        new_cell.fitness_history = self.fitness_history.copy()

        # GENOME: Copy-on-Write when enabled; else keep prior lightweight behavior
        new_cell.genome = []
        if getattr(cfg, 'torch_genome_enabled', False) and self.genome:
            new_cell._share_genome_from(self)
        else:
            new_cell.genome = list(self.genome) if self.genome else []

        new_cell.architecture_modifier = SelfModifyingArchitecture(new_cell)

        if device is not None:
            new_cell.to(device)
        return new_cell


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    name: metric key returned by metrics_fn (e.g., 'cost', 'latency_ms', 'efficiency', 'stability').
    weight: non-negative; weights are auto-normalized to sum to 1.
    baseline: expected 'meh' performance (e.g., current prod).
    target: desired performance. For minimize objectives, target < baseline.
    direction: 'min' or 'max'.
    """
    name: str
    weight: float
    baseline: float
    target: float
    direction: str  # 'min' | 'max'

    def scale(self, x: float) -> float:
        # Map to [0,1] where 1 is good (closer to target).
        eps = 1e-12
        if self.direction == 'max':
            num = (x - self.baseline)
            den = (self.target - self.baseline) + eps
            return float(np.clip(num / den, 0.0, 1.0))
        else:  # 'min'
            num = (self.baseline - x)
            den = (self.baseline - self.target) + eps
            return float(np.clip(num / den, 0.0, 1.0))


class BaseEvolutionaryAlgorithm:
    """Base class for all evolutionary algorithms"""
    
    def __init__(self, pop_size: int = cfg.pop_size, max_generations: int = cfg.max_generations, 
                 bounds: Tuple[float, float] = (-10, 10), dimension: int = 30):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.bounds = bounds
        self.dimension = dimension
        self.population = None
        self.fitness_values = None
        self.best_fitness_history = []
        self.diversity_history = []
        self.function_evaluations = 0
        
    def initialize_population(self):
        """Initialize random population within bounds"""
        low, high = self.bounds
        self.population = np.random.uniform(low, high, (self.pop_size, self.dimension))
        
    def evaluate_population(self, fitness_func):
        """Evaluate fitness for entire population"""
        self.fitness_values = np.array([fitness_func(ind) for ind in self.population])
        self.function_evaluations += len(self.population)
        
    def calculate_diversity(self) -> float:
        """Calculate population diversity using average pairwise distance"""
        if self.population is None:
            return 0.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.linalg.norm(self.population[i] - self.population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self, fitness_func) -> Tuple[BenchmarkResult, Optional[Cell]]:
        """Main optimization loop - to be implemented by subclasses"""
        raise NotImplementedError





# --- Main Algorithm Class with All Integrations ---
class AdvancedArchipelagoEvolution(BaseEvolutionaryAlgorithm):
    def __init__(self, n_islands: int = cfg.n_islands, migration_interval: int = cfg.migration_interval, 
                 enable_epigenetic: bool = True, enable_immune: bool = True,
                 enable_dream: bool = True, enable_pnn: bool = True, 
                 enable_quantum: bool = True, enable_self_modify: bool = True,
                 enable_batched_eval: bool = True, early_stop_patience: int = cfg.early_stop_patience, 
                 rel_tol: float = cfg.rel_tol, abs_tol: float = cfg.abs_tol, 
                 solution_mutation_strength: float = cfg.solution_mutation_strength,
                 neural_mutation_probability: float = cfg.neural_mutation_probability,
                 neural_mutation_strength: float = cfg.neural_mutation_strength,
                 causal_tapestry: Optional[CausalTapestry] = None,
                 seed: Optional[int] = None,
                 metrics_fn: Callable[[np.ndarray], Dict[str, float]] = None,
                 objectives: List[ObjectiveSpec] = None,
                 **kwargs):
        # Performance tracking
        self.batched_eval_times = []
        self.individual_eval_times = []
        # Extract only the parameters that BaseEvolutionaryAlgorithm expects
        base_kwargs = {}
        for key in ['pop_size', 'max_generations', 'bounds', 'dimension']:
            if key in kwargs:
                base_kwargs[key] = kwargs[key]
        super().__init__(**base_kwargs)
        cfg.feature_dim = self.dimension
        
        # Auto-tune population/islands for low-budget regime (e.g., 5k NFE)
        if getattr(cfg, 'auto_low_budget_tuning', True):
            try:
                if self.dimension <= 5:
                    self.pop_size = 32
                    n_islands = min(max(2, n_islands), 3)
                elif self.dimension <= 10:
                    self.pop_size = max(32, min(48, self.pop_size))
                    n_islands = min(max(2, n_islands), 2)
                elif self.dimension <= 30:
                    self.pop_size = max(48, min(64, self.pop_size))
                    n_islands = min(max(1, n_islands), 2)
            except Exception:
                pass
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.island_pop_size = max(1, self.pop_size // n_islands)
        self.islands: List[List[Cell]] = []
        # Deterministic RNGs (harness passes seed via wrapper)
        self.seed = int(seed) if seed is not None else None
        try:
            import numpy as _np
            self._np_rng = _np.random.default_rng(self.seed) if self.seed is not None else _np.random.default_rng()
        except Exception:
            self._np_rng = None
        try:
            import random as _random
            self._py_rng = _random.Random(self.seed) if self.seed is not None else _random.Random()
        except Exception:
            self._py_rng = None
        
        # FIX: Ensure island roles are properly initialized
        if not hasattr(self, 'island_roles') or len(self.island_roles) < self.n_islands:
            base_roles = cfg.island_roles
            if len(base_roles) < self.n_islands:
                # Extend with 'raw' roles if we don't have enough
                self.island_roles = base_roles + ['raw'] * (self.n_islands - len(base_roles))
            else:
                self.island_roles = base_roles[:self.n_islands]
        
        # --- TARGETED CHANGE 1: Keep per-island histories ---
        from collections import deque
            
        self._metrics_fn = metrics_fn
        self._objectives = list(objectives) if objectives else None

        # Lightweight cache to avoid re-evaluating identical solutions
        self._mo_cache: Dict[tuple, Dict[str, float]] = {}

        if self._metrics_fn and self._objectives:
            # Replace scalar objective with MO composite (minimization compatible)
            self.current_fitness_func = self._mo_loss  # <-- plugs into your evaluator calls
            
        # Rolling windows for per-island health
        self.island_best_hist = [deque(maxlen=getattr(cfg, "percentile_window", 50))
                                 for _ in range(self.n_islands)]
        self.island_median_hist = [deque(maxlen=getattr(cfg, "percentile_window", 50))
                                   for _ in range(self.n_islands)]
        
        self.enable_epigenetic = enable_epigenetic
        self.enable_immune = enable_immune
        self.enable_dream = enable_dream
        self.enable_pnn = enable_pnn
        self.enable_quantum = enable_quantum
        self.enable_self_modify = enable_self_modify
        self.enable_batched_eval = enable_batched_eval
        
        # FIX: Initialize causal tapestry - reuse provided instance or create new one
        global _GLOBAL_TAPESTRY
        if causal_tapestry is not None:
            self.causal_tapestry = causal_tapestry
            # Ensure the provided instance supports persistence APIs; upgrade to v2 if not
            if not (hasattr(self.causal_tapestry, 'load_from_json') and hasattr(self.causal_tapestry, 'load_vectors')):
                logger.info("Upgrading provided CausalTapestry instance to v2 for persistence support")
                self.causal_tapestry = CausalTapestry()
            _GLOBAL_TAPESTRY = self.causal_tapestry
            logger.info(f"Reusing existing CausalTapestry: {self.causal_tapestry.run_id}")
        else:
            # Reuse global tapestry if available; else create and cache
            if _GLOBAL_TAPESTRY is not None:
                self.causal_tapestry = _GLOBAL_TAPESTRY
                logger.info(f"Reusing global CausalTapestry: {self.causal_tapestry.run_id}")
            else:
                self.causal_tapestry = CausalTapestry()
                _GLOBAL_TAPESTRY = self.causal_tapestry
            logger.info(f"Created new CausalTapestry: {self.causal_tapestry.run_id}")
        
        # FIX: Disable tapestry saving by default for benchmark runs (improves performance)
        self.save_tapestry_enabled = True
        self.dormant_bay: Dict[int, Cell] = {}
        self.breeding_summit_interval = cfg.breeding_summit_interval
        self.microchimerism_interval = cfg.microchimerism_interval
        
        self.immune_patches = {}
        self.dream_buffer = []
        self.quantum_dreamer = QuantumDreamer(self.dimension)
        self.dream_interval = cfg.dream_interval
        
        self.phase_transition_threshold = 0.01
        self.epigenetic_memory = [np.random.uniform(-0.1, 0.1, self.dimension) for _ in range(self.pop_size)]
        self.max_epigenetic_memory = self.pop_size * 2
        self.early_stop_patience = early_stop_patience
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        
        # Mutation parameters
        self.solution_mutation_strength = solution_mutation_strength
        self.neural_mutation_probability = neural_mutation_probability
        self.neural_mutation_strength = neural_mutation_strength
        
        # --- NEW: Adaptive 1/5th Rule Threshold ---
        self.one_fifth_threshold = 0.2  # Default threshold
        
        # Initialize mixed precision training
        self.scaler = self.enable_mixed_precision()
        
        # Set device for batched evaluation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize advanced biological systems
        self.circadian_controller = CircadianController(island_id=0, num_islands=n_islands)
        self.stress_field = StressField(device=self.device)

        # Initialize telemetry async writer if enabled
        if getattr(cfg, 'enable_telemetry', True) and callable(init_async_telemetry):
            try:
                init_async_telemetry()
            except Exception:
                pass

        # Per-generation diagnostics counters (updated inside optimize)
        self._dreams_created_last_gen: int = 0
        self._exploit_actions_last_gen: int = 0
        self._strategy_usage_last_gen: Dict[str, int] = {}
        
        # Initialize stress-immune layer for each island
        self.stress_immune_layers = []
        for i in range(n_islands):
            # Create a mock transposition engine for now
            mock_engine = type('MockEngine', (), {
                'sync_graph': lambda: None,
                'eval_loss_sample': lambda: 0.0,
                'atomic_context': lambda: type('MockContext', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})()
            })()
            immune_layer = StressImmuneLayer(
                transposition_engine=mock_engine,
                sigma_threshold=cfg.stress_immune_sigma_threshold,
                max_cache=cfg.stress_immune_max_cache
            )
            self.stress_immune_layers.append(immune_layer)
        
        self._strategy_budgets: Dict[tuple, Dict[str, float]] = {}
        
    def _tap_event(self, event_type: str, details: Dict[str, Any]) -> str:
        # FIX: Add safety wrapper to prevent SystemError from crashing optimization
        try:
            eid = f"{event_type}_{self.generation}_{uuid.uuid4().hex[:4]}"
            self.causal_tapestry.add_event_node(eid, event_type, self.generation, details)
            logger.debug(f"Tapestry event: {eid}")
            return eid
        except (SystemError, Exception) as e:
            # FIX: Graceful fallback if Causal Tapestry event logging fails
            logger.warning(f"CAUSAL TAPESTRY EVENT LOGGING FAILED: {e}")
            return f"fallback_{event_type}_{self.generation}_{uuid.uuid4().hex[:4]}"

    def _tap_cell(self, cell, island_name: str, fit_cache: Dict[int, float]):
        # Do NOT force an evaluation; only log if we already have it.
        if cell.id in fit_cache:
            try:
                f = float(fit_cache[cell.id])
                genes = [g.gene_id for g in (cell.genome or [])]
                self.causal_tapestry.add_cell_node(str(cell.id), self.generation, island_name, f, genes)
                logger.debug(f"Tapestry cell: {str(cell.id)}")
            except (SystemError, Exception) as e:
                # FIX: Graceful fallback if Causal Tapestry cell logging fails
                logger.warning(f"CAUSAL TAPESTRY CELL LOGGING FAILED: {e}")

    def _tap_lineage(self, parent, child):
        try:
            self.causal_tapestry.log_lineage(str(parent.id), str(child.id))
        except (SystemError, Exception) as e:
            # FIX: Graceful fallback if Causal Tapestry lineage logging fails
            logger.warning(f"CAUSAL TAPESTRY LINEAGE LOGGING FAILED: {e}")
        
    def initialize_population(self):
        self.islands = []
        cell_id_counter = 0
        for isl in range(self.n_islands):
            # Last island gets the remainder
            size = self.island_pop_size if isl < self.n_islands - 1 else (self.pop_size - (self.n_islands - 1)*self.island_pop_size)
            island_pop = []
            for _ in range(size):
                try:
                    is_stem = (self._py_rng.random() < 0.1) if self._py_rng is not None else (random.random() < 0.1)
                except Exception:
                    is_stem = (random.random() < 0.1)
                cell = Cell(cell_id_counter, self.dimension, self.bounds, is_stem)
                # Move cell to correct device
                cell.to(self.device)
                island_pop.append(cell); cell_id_counter += 1
            self.islands.append(island_pop)
            logger.debug(f"Initialized island {isl} with {len(island_pop)} cells")
            
        # Log initial population (no fitness yet)
        # Show first few initial vectors to confirm within bounds
        try:
            flat = [c.solution for island in self.islands for c in island][:5]
            logger.info(f"Initial sample (first 5) within bounds? {[bool(np.all((v>=self.bounds[0]) & (v<=self.bounds[1]))) for v in flat]}")
        except Exception:
            pass
        for isl_idx, island in enumerate(self.islands):
            for cell in island:
                # no fitness yet; record with default -1
                genes = [g.gene_id for g in (cell.genome or [])]
                self.causal_tapestry.add_cell_node(str(cell.id), 0, f"island_{isl_idx}", -1.0, genes)
                # Log gene nodes and composition for richer queries
                for g in (cell.genome or []):
                    try:
                        self.causal_tapestry.add_gene_node(g.gene_id, g.gene_type, g.variant_id)
                        self.causal_tapestry.log_gene_composition(str(cell.id), g.gene_id)
                    except Exception:
                        pass

    def _get_phenotype(self, cell: Cell) -> np.ndarray:
        # This method should now simply return the cell's stored solution
        return cell.get_solution()

    def _evaluate_cell(self, cell: Cell) -> float:
        # Evaluate the fitness of the cell's solution, not the cell object itself
        return self.current_fitness_func(cell.get_solution())

    def _evaluate_island_once(self, island: List[Cell], cache: Dict[int, float]) -> List[float]:
        vals: List[float] = []
        for c in island:
            if c.id in cache:
                vals.append(cache[c.id])
            else:
                v = self.current_fitness_func(c.get_solution())  # counted by the harness proxy
                cache[c.id] = float(v)
                self.function_evaluations += 1  # local counter only
                vals.append(float(v))
        return vals

    # --- FIX: New SOC Stress Calculation ---
    # REMOVED: _calculate_stress method - now using StressField.local_stress values


    def immune_response(self, phenotype, stress):
        if not self.enable_immune or stress < cfg.stress_threshold:
            logger.debug(f"Immune response: no immune response needed")
            return phenotype
        signature = int(np.sum(phenotype * 1000) % 1000)
        if signature in self.immune_patches:
            return np.clip(phenotype + self.immune_patches[signature] * 0.1, self.bounds[0], self.bounds[1])
        else:
            patch = np.random.uniform(-0.1, 0.1, len(phenotype))
            self.immune_patches[signature] = patch
            return phenotype


    def epigenetic_inheritance(self, parent: Cell, child: Cell):
        if not self.enable_epigenetic: return
        child.epigenetic_shadow_mark = 0.8 * parent.epigenetic_shadow_mark + 0.2 * torch.randn(self.dimension) * 0.05
        logger.debug(f"Epigenetic inheritance: child.epigenetic_shadow_mark={child.epigenetic_shadow_mark}")

            

    @lru_cache(maxsize=4096)
    def _mo_key(self, x_tuple: tuple) -> tuple:
        return x_tuple  # split out for readability / future hashing changes

    def _get_metrics(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        key = self._mo_key(tuple(np.round(x, 8)))
        hit = self._mo_cache.get(key)
        if hit is not None:
            return hit
        m = self._metrics_fn(x)
        # Defensive: ensure floats
        m = {k: float(v) for k, v in m.items()}
        self._mo_cache[key] = m
        return m

    def _mo_loss(self, x: np.ndarray) -> float:
        """
        Returns a single scalar loss (lower is better) for use by your optimizer.
        Internally:
          1) Pulls raw enterprise metrics from metrics_fn(x)
          2) Normalizes each to [0,1] 'goodness' via ObjectiveSpec.scale
          3) Weighted average to a composite goodness in [0,1]
          4) Convert to loss: loss = 1 - goodness
        """
        assert self._metrics_fn is not None and self._objectives is not None, \
            "MO path requires metrics_fn and objectives"
        logger.debug("Starting _mo_loss computation")
        logger.debug(f"Input x: {x}")

        metrics = self._get_metrics(x)
        logger.debug(f"Metrics obtained: {metrics}")

        # Normalize weights
        w_sum = sum(max(0.0, obj.weight) for obj in self._objectives) or 1.0
        logger.debug(f"Sum of positive objective weights: {w_sum}")
        w = [max(0.0, obj.weight) / w_sum for obj in self._objectives]
        logger.debug(f"Normalized weights: {w}")

        # Compute goodness per objective
        goods = []
        for obj in self._objectives:
            if obj.name not in metrics:
                logger.debug(f"Metric '{obj.name}' not found in metrics. Appending 0.0 goodness.")
                goods.append(0.0)
                continue
            scaled = obj.scale(metrics[obj.name])
            logger.debug(f"Objective '{obj.name}': metric={metrics[obj.name]}, scaled_goodness={scaled}")
            goods.append(scaled)

        composite_goodness = float(np.dot(w, np.asarray(goods, dtype=float)))
        logger.debug(f"Composite goodness (weighted sum): {composite_goodness}")
        loss = 1.0 - composite_goodness  # lower is better for your optimizer
        logger.debug(f"Final loss (1 - composite_goodness): {loss}")

        # Optional: attach last metrics for logging/telemetry
        try:
            self.last_metrics = metrics
            self.last_objective_goodness = dict(zip([o.name for o in self._objectives], goods))
            self.last_composite_goodness = composite_goodness
            logger.debug(f"Updated last_metrics, last_objective_goodness, last_composite_goodness")
        except Exception as e:
            logger.debug(f"Exception updating last metrics: {e}")

        return loss


    def _strategic_pnn_unlocking(self, island: List[Cell], island_idx: int, generation: int, island_champion_fitness: float = None):
        """
        Identifies locked cells, prioritizes them by 'need-to-unlock', and
        unlocks the top K most urgent candidates to maintain exploration.
        """
        import logging
        logger = logging.getLogger("teai.strategic_pnn_unlocking")
        unlock_candidates = []
        unlocks = 0
        logger.debug(f"Starting _strategic_pnn_unlocking for island {island_idx}, generation {generation}")
        # Island-wide best fitness (last known) for stagnation checks
        try:
            island_fitnesses = [c.fitness_history[-1] for c in island if getattr(c, 'fitness_history', None)]
            island_best_fitness = min(island_fitnesses) if island_fitnesses else None
            logger.debug(f"Island best fitness: {island_best_fitness}")
        except Exception as e:
            logger.debug(f"Exception getting island best fitness: {e}")
            island_best_fitness = None
        
        # 1. Identify all locked cells that are eligible for unlocking
        for cell in island:
            logger.debug(f"Checking cell {cell.id} for unlocking (PNN state: {cell.pnn.state})")
            if cell.pnn.state == PNN_STATE.LOCKED:
                budget_exhausted = cell.pnn.exploit_budget <= 0
                critical_stress = cell.local_stress > cfg.pnn_stress_unlock_threshold
                current_fitness = cell.fitness_history[-1] if cell.fitness_history else float('inf')
                logger.debug(f"  Cell {cell.id}: exploit_budget={cell.pnn.exploit_budget}, local_stress={cell.local_stress}, current_fitness={current_fitness}")

                # Relative stagnation vs island champion (if provided)
                is_stagnant_relative_to_champ = False
                try:
                    if island_champion_fitness is not None and cell.pnn.generations_in_phase > 5:
                        if current_fitness > (1.1 * float(island_champion_fitness)):
                            is_stagnant_relative_to_champ = True
                    logger.debug(f"  Cell {cell.id}: is_stagnant_relative_to_champ={is_stagnant_relative_to_champ}")
                except Exception as e:
                    logger.debug(f"  Exception in relative stagnation check for cell {cell.id}: {e}")
                    is_stagnant_relative_to_champ = False

                # NEW: Config-driven stagnation vs island best
                is_stagnated_cfg = False
                try:
                    if island_best_fitness is not None and cell.pnn.generations_in_phase > getattr(cfg, 'pnn_stagnation_generations', 8):
                        thresh = float(island_best_fitness) + abs(float(island_best_fitness) * float(getattr(cfg, 'pnn_improvement_threshold', 0.01)))
                        if current_fitness > thresh:
                            is_stagnated_cfg = True
                    logger.debug(f"  Cell {cell.id}: is_stagnated_cfg={is_stagnated_cfg}")
                except Exception as e:
                    logger.debug(f"  Exception in config stagnation check for cell {cell.id}: {e}")
                    is_stagnated_cfg = False

                if budget_exhausted or critical_stress or is_stagnant_relative_to_champ or is_stagnated_cfg:
                    # 2. Calculate unlock priority score
                    generations_stuck = cell.pnn.generations_in_phase
                    logger.debug(f"  Cell {cell.id}: generations_stuck={generations_stuck}")
                    
                    # Priority = How bad the solution is * how long it has been unproductive
                    priority = current_fitness * (1 + 0.1 * generations_stuck)
                    logger.debug(f"  Cell {cell.id}: priority={priority}")

                    if is_stagnated_cfg:
                        reason = "Stagnated"
                    elif is_stagnant_relative_to_champ:
                        reason = f"Relative Stagnation (self={current_fitness:.4f} > champ={float(island_champion_fitness):.4f}*1.1)"
                    elif budget_exhausted:
                        reason = "Budget Exhausted"
                    else:
                        reason = f"Critical Stress ({cell.local_stress:.2f})"
                    logger.debug(f"  Cell {cell.id}: unlock reason: {reason}")
                    unlock_candidates.append((priority, cell, reason))

        logger.debug(f"Total unlock candidates: {len(unlock_candidates)}")
        if not unlock_candidates:
            logger.debug("No unlock candidates found, returning 0 unlocks.")
            return unlocks

        # 3. Sort candidates by priority (highest priority first)
        unlock_candidates.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"Sorted unlock candidates by priority: {[ (c[1].id, c[0]) for c in unlock_candidates ]}")
        
        # 4. Enforce the unlock limit (Top-K), dynamically modulated by island stress
        island_stresses = [getattr(c, 'local_stress', 0.0) for c in island]
        stress_level = float(np.mean(island_stresses)) if island_stresses else 0.0
        logger.debug(f"Island stress level: {stress_level}")
        frac = float(getattr(cfg, 'unlock_fraction_static', 0.05))
        try:
            dim = int(getattr(self, 'dimension', getattr(cfg, 'dimension', 0)))
            logger.debug(f"Problem dimension: {dim}")
        except Exception as e:
            logger.debug(f"Exception getting dimension: {e}")
            dim = 0
        if dim >= getattr(cfg, 'high_dim_threshold', 30):
            frac = float(getattr(cfg, 'unlock_fraction_high_dim', 0.08))
            logger.debug(f"High dimension detected, using unlock_fraction_high_dim: {frac}")
        base_limit = max(1, int(round(len(unlock_candidates) * frac)))
        logger.debug(f"Base unlock limit: {base_limit}")
        stress_low = float(getattr(cfg, 'unlock_stress_low', 0.2))
        stress_high = float(getattr(cfg, 'unlock_stress_high', 0.8))
        if stress_level <= stress_low:
            scale = 2.0
        elif stress_level >= stress_high:
            scale = 0.5
        else:
            t = (stress_high - stress_level) / max(1e-8, (stress_high - stress_low))
            scale = 0.5 + 1.5 * t
        logger.debug(f"Unlock scale factor: {scale}")
        limit = max(1, int(round(base_limit * scale)))
        logger.debug(f"Final unlock limit: {limit}")
        cells_to_unlock = unlock_candidates[:limit]
        
        logger.info(f"    STRATEGIC UNLOCK (Island {island_idx}): {len(unlock_candidates)} candidates, unlocking top {len(cells_to_unlock)}.")

        for priority, cell, reason in cells_to_unlock:
            logger.info(f"        -> Unlocking cell {cell.id} (Priority: {priority:.2f}, Reason: {reason})")
            logger.debug(f"Unlocking cell {cell.id} at generation {generation} with reason: {reason}")
            cell.pnn.force_unlock(generation) # Use the new safe unlocking method
            unlocks += 1
        logger.debug(f"Total unlocks performed: {unlocks}")
        return unlocks

    def _choose_exploitation_strategy(self, cell: Cell, island_idx: int) -> str:
        """
        Queries the Causal Tapestry to decide between local search strategies
        for a LOCKED cell based on historical performance.
        
        Returns:
            str: The name of the chosen strategy ('nelder-mead' or 'champion-linesearch').
        """
        logger.debug(f"Choosing exploitation strategy for cell {cell.id} on island {island_idx}")
        # Forced override
        forced = getattr(cfg, 'force_exploit_strategy', None)
        if forced in ('nelder-mead','champion-linesearch'):
            logger.debug(f"Forced strategy override: {forced}")
            return forced
        # Safety bypass: when causal queries are disabled, avoid any tapestry access
        if getattr(cfg, 'disable_causal_queries', True):
            logger.debug("Causal queries disabled, choosing randomly")
            return random.choice(['nelder-mead', 'champion-linesearch'])
        # BOOTSTRAP PHASE: Use random strategy selection for first N generations to build unbiased dataset
        if self.generation < cfg.tapestry_bootstrap_generations:
            choice = random.choice(['nelder-mead', 'champion-linesearch'])
            logger.info(f"      BOOTSTRAP EXPLOIT CHOICE for cell {cell.id}: Random '{choice}' (generation {self.generation}/{cfg.tapestry_bootstrap_generations})")
            logger.debug(f"Bootstrap phase, random choice: {choice}")
            return choice

        # Define context for the query based on the cell's current state
        context = {
            'island': f"island_{island_idx}",
            'stress_bin': int(cell.local_stress * 5), # Bin stress for better matching
            'pnn_state': PNN_STATE.LOCKED.value
        }
        logger.debug(f"Context for causal query: {context}")
        
        # Use per-generation snapshot if available; otherwise safe fallbacks
        s_nm = {'effect': 0.0, 'std': 0.0, 'count': 0}
        s_cl = {'effect': 0.0, 'std': 0.0, 'count': 0}
        try:
            if not getattr(cfg, 'disable_causal_queries', True) and hasattr(self, '_causal_stats_snapshot'):
                items_base = [
                    ('island', f"island_{island_idx}"),
                    ('pnn_state', PNN_STATE.LOCKED.value),
                    ('stress_bin', int(cell.local_stress * 5)),
                ]
                key_nm = ('exploit', tuple(sorted(items_base + [('strategy_used','nelder-mead')])))
                key_cl = ('exploit', tuple(sorted(items_base + [('strategy_used','champion-linesearch')])))
                s_nm = self._causal_stats_snapshot.get(key_nm, s_nm)
                s_cl = self._causal_stats_snapshot.get(key_cl, s_cl)
                logger.debug(f"Snapshot stats for NM: {s_nm}, CL: {s_cl}")
        except Exception as e:
            logger.debug(f"Exception in causal stats snapshot: {e}")

        def lcb(s):
            cnt = max(1, int(s.get('count', 0)))
            std = float(s.get('std', 0.0))
            eff = float(s.get('effect', 0.0))
            lcb_val = eff - std / (cnt ** 0.5)
            logger.debug(f"LCB calculation: effect={eff}, std={std}, count={cnt}, lcb={lcb_val}")
            return lcb_val

        score_nm = lcb(s_nm)
        score_cl = lcb(s_cl)
        logger.debug(f"LCB scores: NM={score_nm}, CL={score_cl}")
        if score_nm > score_cl:
            choice = 'nelder-mead'
        elif score_cl > score_nm:
            choice = 'champion-linesearch'
        else:
            choice = random.choice(['nelder-mead', 'champion-linesearch'])
        logger.info(f"      CAUSAL EXPLOIT CHOICE for cell {cell.id}: '{choice}' (LCB: NM={score_nm:.3f}, CL={score_cl:.3f})")
        logger.debug(f"Final strategy choice: {choice}")
        return choice

    def _exploit_locked_cells(self, island: List[Cell], island_idx: int, cache: Dict[str, float], generation: int, safe_eval: Callable[[np.ndarray], float]):
        
        if generation < 20 or generation % 5 != 0:
            return
        
        import logging
        logger = logging.getLogger("teai.exploit_locked_cells")
        logger.debug(f"Starting _exploit_locked_cells for island {island_idx}, generation {generation}")
        # 1. GATING: Check if exploitation is enabled for the current phase/day cycle
        if not getattr(cfg, 'enable_exploit_queue', False):
            logger.debug("Exploit queue not enabled, returning.")
            return
        allow_mid = getattr(cfg, 'exploit_queue_allow_mid_phase', False)
        if not allow_mid and generation < cfg.late_phase_start * cfg.max_generations:
            logger.debug("Not in late phase, and mid-phase not allowed. Returning.")
            return
        if getattr(cfg, 'exploit_queue_circadian_day_only', False) and not self.circadian_controller.is_day:
            logger.debug("Circadian controller: not day, skipping exploitation.")
            return
            
        locked = [c for c in island if c.pnn.state == PNN_STATE.LOCKED]
        logger.debug(f"Locked cells found: {[c.id for c in locked]}")
        if not locked:
            logger.debug("No locked cells to exploit.")
            return
            
        # 2. SELECTION: Prioritize the worst-performing locked cells to fix them
        scored = [(cache.get(c.id, float('inf')), c) for c in locked]
        logger.debug(f"Locked cell scores: {[(c.id, score) for score, c in scored]}")
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [c for _, c in scored[:cfg.exploit_queue_top_k]]
        logger.debug(f"Top-k locked cells selected for exploitation: {[c.id for c in top_k]}")

        # 3. SETUP: Initialize tools for this exploitation burst
        lb, ub = self.bounds
        base_radius = cfg.trust_radius_init_fraction * (ub - lb)
        logger.debug(f"Base trust region radius: {base_radius}")
        toolkit = ExploitationToolkit(self.bounds, safe_eval, getattr(safe_eval, 'batch', None))

        if not hasattr(self, "_ucb_selectors"):
            logger.debug("Initializing UCB selectors for islands.")
            self._ucb_selectors = [ContextualUCBSelector(
                ['nelder-mead', 'champion-linesearch', 'powell', 'cmaes'],
                exploration_factor=getattr(cfg, 'ucb_exploration_factor', 0.5)
            ) for _ in range(self.n_islands)]
        ucb_selector = self._ucb_selectors[island_idx]

        # 4. EXECUTION LOOP: Iterate through the selected cells
        for cell in top_k:
            logger.debug(f"Processing cell {cell.id} (exploit_budget={getattr(cell.pnn, 'exploit_budget', 1.0)})")
            if getattr(cell.pnn, 'exploit_budget', 1.0) <= 0:
                logger.debug(f"Cell {cell.id} has no exploit budget left, skipping.")
                continue

            # 4a. CONTEXT & STRATEGY SELECTION
            context_key = (f"island_{island_idx}", PNN_STATE.LOCKED.value, int(cell.local_stress * 5))
            logger.debug(f"Context key for UCB selector: {context_key}")
            strategy = ucb_selector.select_strategy(context_key)
            logger.debug(f"Selected strategy: {strategy}")

            # 4b. ADAPTIVE BUDGETING: Determine NFE budget for this specific action
            context_budgets = self._strategy_budgets.setdefault(context_key, {})
            adaptive_budget = context_budgets.get(strategy, getattr(cfg, 'exploit_budget_initial', 16))
            eval_budget = int(max(0, min(adaptive_budget, getattr(cell.pnn, 'exploit_budget', 0.0))))
            logger.debug(f"Adaptive budget: {adaptive_budget}, eval_budget: {eval_budget}")

            if eval_budget < getattr(cfg, 'exploit_budget_min', 8):
                logger.debug(f"Eval budget {eval_budget} below minimum, skipping cell {cell.id}.")
                continue

            # 4c. PREPARATION: Get current state and optional causal hint
            best_before = cache.get(cell.id, float('inf'))
            logger.debug(f"Cell {cell.id} best_before: {best_before}")
            current = np.array(cell.get_solution(), dtype=float)
            logger.debug(f"Cell {cell.id} current solution: {current}")
            radius = float(base_radius * (0.5 + 0.5 * min(2.0, max(0.0, cell.local_stress))))
            logger.debug(f"Cell {cell.id} trust region radius: {radius}")
            
            dir_hint = None
            if not getattr(cfg, 'disable_causal_queries', True):
                try:
                    dir_hint = self.causal_tapestry.query_causal_direction('exploit', {
                        'island': f'island_{island_idx}',
                        'pnn_state': PNN_STATE.LOCKED.value,
                        'stress_bin': int(cell.local_stress * 5)
                    })
                    logger.debug(f"Cell {cell.id} causal direction hint: {dir_hint}")
                except Exception as e:
                    logger.debug(f"Exception getting causal direction for cell {cell.id}: {e}")
                    dir_hint = None

            improvement = 0.0
            used_evals = 0
            
            # 4d. STRATEGY EXECUTION: Run the chosen local search method
            logger.debug(f"Cell {cell.id} executing strategy: {strategy}")
            if strategy == 'cmaes' and getattr(cfg, 'cmaes_enabled', True):
                if getattr(cfg, 'cmaes_use_surrogate', False):
                    logger.debug(f"Cell {cell.id} using cmaes_surrogate_step")
                    cand, used = toolkit.cmaes_surrogate_step(current, radius, dir_hint=dir_hint)
                else:
                    logger.debug(f"Cell {cell.id} using cmaes_step")
                    cand, used = toolkit.cmaes_step(current, radius, eval_budget, dir_hint=dir_hint)
                
                f_candidate = safe_eval(cand)
                logger.debug(f"Cell {cell.id} cmaes candidate fitness: {f_candidate}, evals used: {used}")
                used_evals = used
                improvement = max(0.0, best_before - f_candidate)
                logger.debug(f"Cell {cell.id} improvement: {improvement}")
                if improvement > 0:
                    current = cand
                    cache[cell.id] = float(f_candidate)
                    logger.debug(f"Cell {cell.id} updated current and cache with improved candidate.")
            else:
                # Logic for all other, non-CMA-ES strategies
                steps = int(max(1, cfg.exploit_queue_steps))
                switched = False
                initial_sol_fitness = best_before
                logger.debug(f"Cell {cell.id} non-CMA-ES strategy, steps: {steps}, initial_sol_fitness: {initial_sol_fitness}")
                
                for s_idx in range(steps):
                    logger.debug(f"Cell {cell.id} step {s_idx+1}/{steps}, used_evals={used_evals}")
                    if (used_evals + 1) > eval_budget: # Respect the NFE budget
                        logger.debug(f"Cell {cell.id} would exceed eval budget, breaking loop.")
                        break
                    
                    if strategy == 'nelder-mead' and hasattr(toolkit, 'trust_region_nelder_mead'):
                        logger.debug(f"Cell {cell.id} using trust_region_nelder_mead")
                        candidate = toolkit.trust_region_nelder_mead(current, radius)
                    elif strategy == 'powell' and hasattr(toolkit, 'powell_step'):
                        logger.debug(f"Cell {cell.id} using powell_step")
                        candidate = toolkit.powell_step(current)
                    else: # Fallback to champion-linesearch logic
                        logger.debug(f"Cell {cell.id} using champion_linesearch_step")
                        pop_mean = np.mean([c.solution for c in island], axis=0)
                        candidate = toolkit.champion_linesearch_step(current, pop_mean, self.dimension)
                    
                    f_candidate = safe_eval(candidate)
                    logger.debug(f"Cell {cell.id} candidate fitness: {f_candidate}")
                    used_evals += 1
                    
                    if f_candidate < best_before - 1e-12:
                        logger.debug(f"Cell {cell.id} found improvement: {f_candidate} < {best_before}")
                        current = candidate
                        best_before = f_candidate
                        cache[cell.id] = float(f_candidate)
                        radius *= cfg.trust_radius_expand
                        logger.debug(f"Cell {cell.id} expanded radius to {radius}")
                    else:
                        radius *= cfg.trust_radius_shrink
                        logger.debug(f"Cell {cell.id} shrunk radius to {radius}")
                    
                    if not switched and (s_idx + 1) >= max(1, int(0.25 * steps)) and best_before >= initial_sol_fitness - 1e-12:
                        switched = True
                        old_strategy = strategy
                        strategy = 'powell' if strategy in ('nelder-mead', 'champion-linesearch') else 'nelder-mead'
                        logger.debug(f"Cell {cell.id} switched strategy from {old_strategy} to {strategy}")
                
                improvement = max(0.0, initial_sol_fitness - best_before)
                logger.debug(f"Cell {cell.id} total improvement after steps: {improvement}")

            # 5. POST-EXPLOIT UPDATES: Update all relevant system states
            
            # 5a. Update Adaptive Budget based on performance
            new_budget = adaptive_budget
            if improvement > 1e-9:
                new_budget *= getattr(cfg, 'exploit_budget_reward_factor', 1.2)
                logger.debug(f"Cell {cell.id} improved, increasing budget to {new_budget}")
            else:
                new_budget *= getattr(cfg, 'exploit_budget_penalty_factor', 0.8)
                logger.debug(f"Cell {cell.id} did not improve, decreasing budget to {new_budget}")
            
            context_budgets[strategy] = np.clip(new_budget,
                                                getattr(cfg, 'exploit_budget_min', 8),
                                                getattr(cfg, 'exploit_budget_max', 64))
            logger.debug(f"Cell {cell.id} context_budgets[{strategy}] set to {context_budgets[strategy]}")
            
            # 5b. Update UCB selector with the reward (improvement)
            ucb_selector.update(context_key, strategy, improvement)
            logger.debug(f"Cell {cell.id} UCB selector updated with improvement {improvement}")

            # 5c. Drain the cell's PNN budget by the actual NFE used
            old_budget = getattr(cell.pnn, 'exploit_budget', 0.0)
            cell.pnn.exploit_budget = max(0.0, old_budget - used_evals)
            logger.debug(f"Cell {cell.id} exploit_budget reduced from {old_budget} to {cell.pnn.exploit_budget}")
            cell.pnn.note_exploit_outcome(evals_used=used_evals, improvement=improvement)
            logger.debug(f"Cell {cell.id} noted exploit outcome: evals_used={used_evals}, improvement={improvement}")

            # 5d. Update the cell's solution in the main population
            cell.solution = np.asarray(current, dtype=float)
            logger.debug(f"Cell {cell.id} solution updated in main population.")

            # 5e. Log the event to the Causal Tapestry
            tapestry_event = self._tap_event("EXPLOIT", {
                "action": "exploit",
                "cell_id": str(cell.id),
                "strategy_used": strategy,
                "effect": improvement,
                "island": f"island_{island_idx}",
                "pnn_state": PNN_STATE.LOCKED.value,
                "stress_bin": int(getattr(cell, 'local_stress', 0.0) * 5),
                "evals": int(used_evals),
                "budget_before": adaptive_budget,
                "budget_after": context_budgets[strategy]
                })
            logger.debug(f"Cell {cell.id} tapestry event: {tapestry_event}")


    def diversity_injection(self):
        logger.info(f"    Injecting new random cells for diversity")
        for island_idx, island in enumerate(self.islands):
            idx = random.randrange(len(island))
            # Corrected: Inject a Cell object, not a numpy array
            old_cell_id = island[idx].id
            new_cell = Cell(random.randint(80000, 90000), self.dimension, self.bounds)
            # FIX: Move diversity cell to device
            new_cell.to(self.device)
            island[idx] = new_cell
            logger.info(f"      Island {island_idx}: Replaced cell {old_cell_id} with new cell {new_cell.id}")
            
            # Log diversity injection event
            eid = self._tap_event("DIVERSITY_INJECTION", {
                "replaced": str(old_cell_id),
                "replacement": str(island[idx].id),
                "island": f"island_{island_idx}"
            })
            self.causal_tapestry.log_event_output(eid, str(island[idx].id), "injected_cell")
            logger.debug(f"Cell {island[idx].id} tapestry event: {eid}")




            
                
                    
    def dream_consolidation(self, generation: int, cache: Dict[int, float], _safe_eval):
        if not self.enable_dream or generation % self.dream_interval != 0:
            return

        # FIX G: Remove duplicate logging - this is now logged once per generation in the main loop
        lb, ub = self.bounds
        range_scale = (ub - lb) / np.sqrt(self.dimension)

        # FIX: Adaptive dream generation - more dreams when stuck
        base_num_dreams = max(1, min(self.island_pop_size // 10, 2))
        if hasattr(self, 'no_improve') and self.no_improve >= cfg.dream_stagnation_threshold:  # Use config
            # Generate more dreams when stuck
            num_dreams = min(base_num_dreams * cfg.dream_stagnation_multiplier,
                             int(self.island_pop_size * cfg.dream_max_fraction))  # Use config
            logger.info(f"    STAGNATION DETECTED: Generating {num_dreams} dreams (vs {base_num_dreams} normally)")
        else:
            num_dreams = base_num_dreams

        # FIX: Increase dream burst on stagnation - generate 10x dreams when no_improve >= 3
        if cfg.enable_aggressive_dreams and hasattr(self, 'no_improve') and self.no_improve >= 3:
            num_dreams = min(num_dreams * 10, int(self.island_pop_size * 0.2))  # Replace up to 20% of population
            logger.info(f"    PLATEAU DREAM BURST: Generating {num_dreams} dreams due to plateau (no_improve={self.no_improve})")

        dreams_created_total = 0
        for island_idx, island in enumerate(self.islands):
            if not island:
                continue

            # Use cached fitness - no extra evaluations
            fit = [cache.get(c.id, 0.5) for c in island]  # Default to 0.5 if not cached

            # Identify champion and anti-champion
            champ_idx = int(np.argmin(fit))
            anti_champ_idx = int(np.argmax(fit))
            champ = island[champ_idx]
            anti_champ = island[anti_champ_idx]
            logger.debug(f"Cell {champ.id} is champion, Cell {anti_champ.id} is anti-champion")
            sigma = cfg.dream_sigma_scale * range_scale  # Use config

            current_fitness = fit[:]  # local copy
            # Half-and-half: split dreams between champion and anti-champion
            n_anti = num_dreams // 2
            n_champ = num_dreams - n_anti
            dreams_champ = [np.clip(champ.solution + np.random.normal(0, sigma, self.dimension), lb, ub) for _ in range(n_champ)]
            dreams_anti = [np.clip(anti_champ.solution + np.random.normal(0, sigma, self.dimension), lb, ub) for _ in range(n_anti)]
            dreams = dreams_champ + dreams_anti
            parents = ([champ.id] * n_champ) + ([anti_champ.id] * n_anti)
            logger.debug(f"Cell {champ.id} is parent {n_champ} times, Cell {anti_champ.id} is parent {n_anti} times")
            # Evaluate in batch when available
            f_dreams = []
            try:
                if hasattr(_safe_eval, 'batch') and callable(getattr(_safe_eval, 'batch')) and len(dreams) > 0:
                    f_dreams = _safe_eval.batch(np.asarray(dreams, dtype=float)).tolist()
                else:
                    f_dreams = [float(_safe_eval(d)) for d in dreams]
            except Exception:
                f_dreams = [float(_safe_eval(d)) for d in dreams]
            logger.debug(f"Cell {champ.id} dreams fitness: {f_dreams[:n_champ]}, Cell {anti_champ.id} dreams fitness: {f_dreams[n_champ:]}")
            for d, f_d, parent_id in zip(dreams, f_dreams, parents):
                worst_idx = int(np.argmax(current_fitness))
                worst_fit = current_fitness[worst_idx]
                if worst_fit > cfg.dream_fitness_threshold and f_d < worst_fit:
                    old = island[worst_idx]
                    new_cell = Cell(cell_id=random.randint(10000, 20000),
                                    dimension=self.dimension, bounds=self.bounds)
                    new_cell.genome = nn.ModuleList([ContinuousDepthGeneModule('DREAM', 1)])
                    new_cell.solution = d.astype(float)
                    new_cell.to(self.device)
                    island[worst_idx] = new_cell
                    cache.pop(old.id, None)
                    cache[new_cell.id] = float(f_d)
                    current_fitness[worst_idx] = float(f_d)
                    dreams_created_total += 1
                    eid = self._tap_event("DREAM_SPAWN", {
                        "parent": str(parent_id),
                        "spawn": str(new_cell.id),
                        "island": f"island_{island_idx}",
                        "child_fitness": float(f_d)
                    })
                    logger.debug(f"Cell {new_cell.id} tapestry event: {eid}")
                    role = "champion" if parent_id == champ.id else "anti_champion"
                    self.causal_tapestry.log_event_participation(str(parent_id), eid, role)
                    self.causal_tapestry.log_event_output(eid, str(new_cell.id), "spawn")
        # expose per-generation metric
        try:
            self._dreams_created_last_gen = int(dreams_created_total)
        except Exception:
            pass

        
    def adaptive_mutation_rate(self, generation: int, diversity: float) -> float:
        """Adapt mutation rate based on population state"""
        base_rate = 0.1
        
        # Increase mutation if diversity is low
        if diversity < 0.01:
            return min(base_rate * 2.0, 0.3)
        
        # Decrease if population is still diverse
        elif diversity > 0.1:
            return max(base_rate * 0.5, 0.01)
        
        return base_rate
    
    

    def detect_phase_transition(self) -> bool:
        # Need enough history
        if len(self.best_fitness_history) < 30:
            return False
        recent = np.asarray(self.best_fitness_history[-20:], dtype=float)

        # If variance is ~0, we're stagnant but don't compute corrcoef
        var = float(np.var(recent))
        logger.debug(f"Phase transition detection: var={var}")
        if var < 1e-12:
            return True  # clearly flat, treat as transition

        # Safe autocorrelation (lag-1) only if std > 0
        x = recent - recent.mean()
        denom = float(np.dot(x[:-1], x[:-1])**0.5 * np.dot(x[1:], x[1:])**0.5)
        ac1 = float('nan')
        if denom > 0:
            try:
                ac1 = float(np.dot(x[:-1], x[1:]) / denom)
            except Exception:
                ac1 = float('nan')
            if np.isfinite(ac1) and ac1 > 0.95:
                return True
        try:
            logger.debug(f"Phase transition detection: ac1={ac1}")
        except Exception:
            pass
        return var < self.phase_transition_threshold

    def epigenetic_inheritance(self, parent: Cell, child: Cell):
        if not self.enable_epigenetic: return
        child.epigenetic_shadow_mark = 0.8 * parent.epigenetic_shadow_mark + 0.2 * torch.randn(self.dimension) * 0.05
        logger.debug(f"Epigenetic inheritance: child.epigenetic_shadow_mark={child.epigenetic_shadow_mark}")
        logger.debug(f"In epigenetic_inheritance: child.epigenetic_shadow_mark={child.epigenetic_shadow_mark}")
        
        
    # --- START: ADD THIS NEW METHOD ---
    def _select_gene_to_mutate(self, cell: Cell) -> Optional[int]:
        """
        Selects a random, active gene from a cell's genome for mutation.

        This helper function ensures that we only attempt to mutate genes
        that are currently contributing to the cell's phenotype.

        Args:
            cell: The cell whose genome will be mutated.

        Returns:
            The index of the gene to mutate, or None if no active genes are found.
        """
        if not cell.genome:
            return None
        logger.debug(f"In _select_gene_to_mutate:   Cell {cell.id} genome: {cell.genome}")
        # Find the indices of all active genes
        active_gene_indices = [i for i, gene in enumerate(cell.genome) if gene.is_active]
        logger.debug(f"Cell {cell.id} active gene indices: {active_gene_indices}")
        if not active_gene_indices:
            return None
        logger.debug(f"In _select_gene_to_mutate: Cell {cell.id} returning random index from active genes: {random.choice(active_gene_indices)}")
        # Return a random index from the list of active genes
        return random.choice(active_gene_indices)
    # --- END: ADD THIS NEW METHOD ---

    
    def champion_linesearch(self, island, cache, _safe_eval):
        if not island: return
        # Use cached fitness - no extra evaluations
        fit = [cache.get(c.id, 0.5) for c in island]  # Default to 0.5 if not cached

        best_idx = int(np.argmin(fit))
        champ = island[best_idx]
        mu = np.mean([c.solution for c in island], axis=0)
        logger.debug(f"In Champion Linesearch: Cell {champ.id} is champion, Cell {mu} is mean")
        d = champ.solution - mu
        n = np.linalg.norm(d)
        if n == 0: return
        d = d / n

        # Use adaptive step size that shrinks on non-improvement
        step = self.champion_step_scale * (self.bounds[1] - self.bounds[0]) / np.sqrt(self.dimension)
        cand1 = np.clip(champ.solution - step * d, *self.bounds)
        cand2 = np.clip(champ.solution + step * d, *self.bounds)

        # Only do line search if champion fitness is poor enough to warrant it
        thr = cfg.champion_linesearch_fitness_threshold
        f_cmp = fit[best_idx]
        try:
            nf = self._normalize_func(fit[best_idx]) if hasattr(self, '_normalize_func') else None
            if nf is not None:
                f_cmp = float(nf)
        except Exception:
            pass
        if f_cmp > thr:
            f1 = _safe_eval(cand1); f2 = _safe_eval(cand2)
            # Track heavy operations (linesearch evaluations)
            self._heavy_ops_count += 2
            improved = False
            if f1 < fit[best_idx] or f2 < fit[best_idx]:
                if f1 <= f2:
                    champ.solution = cand1
                    cache[champ.id] = float(f1)
                else:
                    champ.solution = cand2
                    cache[champ.id] = float(f2)
                improved = True
                self.champion_no_improve = 0
            
            # Adaptive step reduction on non-improvement
            if not improved:
                self.champion_no_improve += 1
                if self.champion_no_improve >= 2:  # Halve step after 2 non-improvements
                    self.champion_step_scale *= 0.5
                    self.champion_no_improve = 0
                    logger.info(f"      Champion linesearch: Halving step to {self.champion_step_scale:.6f}")
    
    def optimize(self, fitness_func) -> Tuple[BenchmarkResult, Optional[Cell]]:
        # FIX: Remove profiler to prevent RuntimeError on subsequent calls
        # import cProfile
        # profiler = cProfile.Profile()
        # profiler.enable()
        start_time = time.time()
        self.current_fitness_func = fitness_func
        # Default persistence: try to load latest if available; else create new run_id
        from pathlib import Path
        latest_dir = Path("runs") / "latest_run"
        try:
            if self.causal_tapestry.run_id is None:
                if latest_dir.exists():
                    json_path = latest_dir / "tapestry.json"
                    vec_path = latest_dir / "priors_v1.npz"
                    if json_path.exists() and not getattr(self.causal_tapestry, '_did_load_once', False):
                        try:
                            self.causal_tapestry.load_from_json(str(json_path))
                            logger.info(f"Loaded existing CausalTapestry from {json_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load tapestry.json: {e}")
                    if vec_path.exists() and not getattr(self.causal_tapestry, '_did_load_once', False):
                        try:
                            self.causal_tapestry.load_vectors(str(vec_path))
                            logger.info(f"Loaded tapestry vectors from {vec_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load vectors: {e}")
                    # keep/assign run_id after load to a fresh timestamped one
                    self.causal_tapestry.run_id = f"run_{int(time.time())}"
                else:
                    self.causal_tapestry.run_id = f"run_{int(time.time())}"
                logger.info(f"Initialized CausalTapestry with run_id: {self.causal_tapestry.run_id}")
            else:
                logger.info(f"Continuing with existing CausalTapestry: {self.causal_tapestry.run_id}")
        except Exception:
            # Fallback to fresh run id
            self.causal_tapestry.run_id = self.causal_tapestry.run_id or f"run_{int(time.time())}"
        
        # Ensure per-run file logging for module logger and detailed logger
        try:
            # Prefer repository logs directory
            from pathlib import Path as _Path
            _log_dir = _Path(project_root) / "logs"
            _log_dir.mkdir(parents=True, exist_ok=True)
            _run_tag = getattr(self.causal_tapestry, 'run_id', f"run_{int(time.time())}")
            _mod_log_path = _log_dir / f"teai_methods_{_run_tag}.log"
            # Add a DEBUG file handler to module logger if not present
            _has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            if not _has_file:
                _fh = logging.FileHandler(str(_mod_log_path), encoding='utf-8')
                _fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
                _fh.setLevel(logging.DEBUG)
                logger.addHandler(_fh)
                logger.setLevel(logging.DEBUG)
        except Exception:
            pass

        # Detailed telemetry logger (force DEBUG to capture all .debug calls)
        try:
            os.environ.setdefault('TEAI_LOG_LEVEL', 'DEBUG')
            os.environ.setdefault('TEAI_CONSOLE_LOG_LEVEL', 'INFO')
        except Exception:
            pass
        # Use the single module-level DetailedLogger instance; do not re-initialize per run

        logger.debug(f"""\n\n##################################################\n\n NEW OPTIMIZE: current_fitness_func={self.current_fitness_func}\n\n###############################################\n\n""")

        logger.info(f"=== AEA Optimization Started ===")
        logger.info(f"Problem: {getattr(fitness_func, '__name__', 'Unknown')}")
        logger.info(f"Population: {self.pop_size}, Islands: {self.n_islands}, Dimensions: {self.dimension}")
        logger.info(f"Bounds: {self.bounds}, Max Generations: {self.max_generations}")
        
        self.initialize_population()
        logger.info(f"Population initialized across {self.n_islands} islands")
        # --- SAFE INIT: prevents UnboundLocalError if nothing ever updates ---
        best_cell: Optional[Cell] = None
        best_fitness: float = float("inf")
        # --- END: SAFE INIT ---
        # Build or reuse per-island objective wrappers around base function
        base_f = fitness_func

        # Ensure island roles are defined
        if not hasattr(self, 'island_roles') or len(self.island_roles) < self.n_islands:
            base_roles = cfg.island_roles
            if len(base_roles) < self.n_islands:
                self.island_roles = base_roles + ['raw'] * (self.n_islands - len(base_roles))
            else:
                self.island_roles = base_roles[:self.n_islands]

        # Respect externally provided island objectives if caller set them
        use_external = False
        if hasattr(self, 'island_objectives') and isinstance(self.island_objectives, (list, tuple)):
            try:
                use_external = (len(self.island_objectives) == self.n_islands) and all(callable(f) for f in self.island_objectives)
            except Exception:
                use_external = False

        if use_external:
            logger.info(f"Using externally provided island objectives (n={len(self.island_objectives)}); skipping rebuild")
        else:
            self._build_island_objectives(base_f)
            logger.info(f"Built {len(self.island_objectives)} island objectives for {self.n_islands} islands")
            logger.info(f"Island roles: {self.island_roles}")

            # Fallback if still not initialized correctly
            if not hasattr(self, 'island_objectives') or len(self.island_objectives) < self.n_islands:
                logger.warning(f"island_objectives not properly initialized. Expected {self.n_islands}, got {len(getattr(self, 'island_objectives', []))}")
                logger.info(f"Island roles: {self.island_roles}")
                self.island_objectives = [base_f] * self.n_islands
        
        # Per-island fitness caches so migration re-evaluates under the new island
        fitness_caches = [dict() for _ in range(self.n_islands)]
        
        self.generation = 0
        self.best_fitness_history.clear()
        self.diversity_history.clear()
        
        best_so_far = float("inf")
        no_improve = 0
        def _safe_eval(x: np.ndarray) -> float:
            # Let StopIteration bubble up so the harness can end the run exactly at the NFE cap.
            try:
                logger.debug("In _safe_eval")
            except Exception:
                pass
            return fitness_func(x)

        # (Removed unused local helper `_eval_cell` to satisfy linter)

        # Initialize per-island sigma (independent of 1/5th rule)
        if not hasattr(self, "_island_sigma"):
            lb, ub = self.bounds
            init_scale = getattr(cfg, 'sigma_init_scale', 0.10)
            self._island_sigma = [init_scale * (ub - lb) / np.sqrt(self.dimension)
                                  for _ in self.islands]
        # Success/trial tracking for optional 1/5th rule
        self._succ = [0]*len(self.islands); self._trials = [0]*len(self.islands)
        
        # --- NEW: Improvement tracking for 1/5th rule ---
        self._improvement_magnitudes = [[] for _ in self.islands]  # Track improvement magnitudes per island
        
        # FIX H: Add plateau unlock cooldown to make it idempotent
        self._plateau_cooldown_until = -1
        
        # Track champion line search step size for adaptive reduction
        self.champion_step_scale = cfg.champion_linesearch_step_scale
        self.champion_no_improve = 0
        
        # === Phase Crisis Tracking ===
        self.phase_crisis_consecutive_count = 0  # Track consecutive generations in crisis
        self.phase_crisis_cooldown_until = -1  # Cooldown to prevent repeated crisis interventions
        
        # Determine normalization mapping if the provided fitness has hints
        norm_optimum = getattr(fitness_func, "_norm_optimum", None)
        norm_threshold = getattr(fitness_func, "_norm_threshold", None)
        logger.debug(f"In _normalize: norm_optimum={norm_optimum}, norm_threshold={norm_threshold}")
        
        
        def _normalize(v: float) -> Optional[float]:
            try:
                if norm_optimum is None or norm_threshold is None:
                    return None
                denom = max(1e-18, abs(float(norm_threshold)))
                return (float(v) - float(norm_optimum)) / denom
            except Exception:
                return None
        # Expose normalizer for helper methods
        self._normalize_func = _normalize
        # Dynamic objective hint (clears caches proactively on shifting landscapes)
        is_dynamic = bool(getattr(fitness_func, "_is_dynamic", False))
        logger.debug(f"In optimize: is_dynamic={is_dynamic}")
        # Track NFE at last improvement to modulate step size under tight budgets
        self._last_improve_nfe = getattr(self, '_last_improve_nfe', 0)
        # === Mechanism usage totals (for run-level telemetry) ===
        self._total_exploit_actions = 0
        self._total_unlocks = 0
        self._total_migration_rounds = 0
        self._total_migrants = 0
        self._total_dream_rounds = 0
        self._total_linesearch_rounds = 0
        # === Additional metrics for telemetry ===
        self._vetoed_breeds = 0
        self._accepted_breeds = 0
        self._heavy_ops_count = 0
        # === Lock fraction tracking for telemetry ===
        self._lock_fraction_history = []
        for generation in range(self.max_generations):
            self.generation = generation
            # Per-generation counters and feature flags
            locked_this_gen = 0
            unlocked_this_gen = 0
            features_active = {
                'migration': False,
                'dream': False,
                'linesearch': False,
                'exploit_queue': False,
                'phase_crisis': False,
            }
            logger.debug(f"In optimize: features_active={features_active}")
            # Reset rolling counters for this generation
            try:
                self._dreams_created_last_gen = 0
                self._exploit_actions_last_gen = 0
                self._strategy_usage_last_gen = {}
            except Exception:
                pass
            # Build causal stats snapshot once per generation (safe read-only)
            if not getattr(cfg, 'disable_causal_queries', True):
                try:
                    self._causal_stats_snapshot = self.causal_tapestry.build_stats_snapshot(
                        generation_window=getattr(cfg, 'causal_tapestry_generation_window', 10),
                        decay_rate=getattr(cfg, 'causal_tapestry_decay_rate', 0.1)
                    )
                except Exception:
                    self._causal_stats_snapshot = {}
            else:
                self._causal_stats_snapshot = {}
            # Note: avoid duplicate snapshot build per generation
            logger.debug(f"In optimize: _causal_stats_snapshot={self._causal_stats_snapshot}")
            # === LATE PHASE OPTIMIZATION SWITCH ===
            late_phase_active = generation >= cfg.late_phase_start * cfg.max_generations
            if late_phase_active:
                # Turn off exploration features
                self.enable_dream = False
                enable_dream = self.enable_dream
                logger.debug(f"In optimize:  late_phase_active={late_phase_active}, enable_dream={enable_dream}")
                # Effectively disable migration and microchimerism
                migration_interval = cfg.migration_interval
                microchimerism_interval = cfg.microchimerism_interval
                logger.debug(f"In optimize:  migration_interval={migration_interval}, microchimerism_interval={microchimerism_interval}")
                # Freeze island roles to mostly 'raw' with one 'rescale' for mild diversity
                self.island_roles = ['raw'] * (cfg.n_islands - 1) + ['rescale']
                logger.debug(f"In optimize:  island_roles={self.island_roles}")
                # Raise stress threshold to prevent stress-driven kicks
                cfg.stress_threshold = cfg.late_phase_stress_threshold
                logger.debug(f"In optimize:  cfg.stress_threshold={cfg.stress_threshold}")
                # Zero out causal tapestry influence when doing well
                cfg.min_weight_when_history_bad = cfg.late_phase_min_weight
                logger.debug(f"In optimize:  cfg.min_weight_when_history_bad={cfg.min_weight_when_history_bad}")
                # Always-on champion linesearch in late phase (every generation)
                champion_linesearch_regular_interval = cfg.champion_linesearch_regular_interval
                logger.debug(f"In optimize:  champion_linesearch_regular_interval={champion_linesearch_regular_interval}")
                # Apply late-phase sigma cap
                lb, ub = self.bounds
                sigma_max_late = cfg.late_phase_sigma_factor * (ub - lb) / np.sqrt(self.dimension)
                for i in range(len(self._island_sigma)):
                    self._island_sigma[i] = min(self._island_sigma[i], sigma_max_late)
                
                # --- NEW: Aggressive Sigma Damping ---
                # Force a decay in mutation strength to switch to fine-tuning mode.
                for i in range(len(self._island_sigma)):
                    self._island_sigma[i] *= 0.98  # Apply a 2% decay each generation
                
                logger.info(f"    LATE PHASE SIGMA DAMPING: Decaying mutation sigmas.")
            
            self.circadian_controller.update(generation)
            # Clear per-island caches on dynamic problems to avoid stale decisions
            if is_dynamic:
                for c in fitness_caches:
                    c.clear()
            
            # FIX E: Update stress field per island instead of globally
            pop_by_island = {}
            for island_idx, island in enumerate(self.islands):
                for cell in island:
                    # Add fitness history for stress field
                    if not hasattr(cell, 'fitness_history'):
                        cell.fitness_history = []
                    # Use cached fitness if available, otherwise default
                    cached_fitness = fitness_caches[island_idx].get(cell.id, 0.5) if island_idx < len(fitness_caches) else 0.5
                    cell.fitness_history.append(cached_fitness)
                
                # Group cells by island
                if island_idx not in pop_by_island:
                    pop_by_island[island_idx] = []
                pop_by_island[island_idx].extend(island)
            
            self.stress_field.update(pop_by_island, best_so_far)
            
            # Structured generation header
            logger.info(generation, {
                'population': self.pop_size,
                'islands': self.n_islands,
                'dimension': self.dimension,
                'late_phase': bool(generation >= cfg.late_phase_start * cfg.max_generations),
            })
            logger.info(f"\n--- Generation {generation} ---")
            logger.info(f"Circadian: {'DAY' if self.circadian_controller.is_day else 'NIGHT'} (plasticity: {self.circadian_controller.plasticity_multiplier:.2f})")
            
            fitness_cache: Dict[int, float] = {}
            all_fitness = []
            all_phenotypes = []
            island_champions = []
            parallel_island_fitness: Optional[List[List[float]]] = None
            
            
            
            if getattr(cfg, 'enable_parallel_islands', False):
                try:
                    parallel_island_fitness = self._evaluate_islands_parallel(fitness_caches)
                except Exception as e:
                    logger.warning(f"Parallel evaluation failed; proceeding sequentially: {e}")
                    parallel_island_fitness = None

            for island_idx, island in enumerate(self.islands):
                # FIX: Add bounds checking to prevent IndexError
                if island_idx >= len(self.island_objectives):
                    logger.error(f"IndexError: island_idx {island_idx} >= len(island_objectives) {len(self.island_objectives)}")
                    # Use the first objective as fallback
                    island_obj = self.island_objectives[0] if self.island_objectives else fitness_func
                else:
                    island_obj = self.island_objectives[island_idx]
                cache = fitness_caches[island_idx]
                
                logger.info(f"  Island {island_idx}: Processing {len(island)} cells")
                
                # Stem Cell Differentiation
                gene_counts = defaultdict(int)
                stem_cells = 0
                for cell in island:
                    if cell.is_stem:
                        stem_cells += 1
                    else:
                        for gene in cell.genome: gene_counts[gene.gene_type] += 1
                
                logger.info(f"    Stem cells: {stem_cells}, Gene counts: {dict(gene_counts)}")
                
                needed = next((gt for gt in ['V','D','J'] if gene_counts[gt] == 0), None)
                if needed:
                    for cell in island:
                        if cell.is_stem:
                            cell.differentiate(needed)
                            # Log differentiation event using the helper method
                            eid = self._tap_event('DIFFERENTIATION', {'cell_id': cell.id, 'to_type': needed})
                            self.causal_tapestry.log_event_participation(str(cell.id), eid, 'differentiating_cell')
                            logger.info(f"    Cell {cell.id} differentiated to {needed}")
                            break

                # Define a local safe evaluator for this island (attach batch if available)
                def make_safe_eval(obj_fn):
                    def _safe_eval_x(x: np.ndarray) -> float:
                        return float(obj_fn(x))
                    try:
                        if hasattr(obj_fn, 'batch') and callable(getattr(obj_fn, 'batch')):
                            setattr(_safe_eval_x, 'batch', lambda X: obj_fn.batch(np.asarray(X, dtype=float)))
                    except Exception:
                        pass
                    return _safe_eval_x

                _safe_eval = make_safe_eval(island_obj)

                # === 1. Evaluate the entire island population for the CURRENT generation first ===
                if parallel_island_fitness is not None and island_idx < len(parallel_island_fitness):
                    island_fitness = parallel_island_fitness[island_idx]
                    eval_time = 0.0
                else:
                    eval_start = time.time()
                    island_fitness = self._evaluate_island_once_with_obj(island, island_obj, cache)
                    eval_time = time.time() - eval_start

                # === 2. Update the fitness history on each cell object with the NEW fitness ===
                for i, cell in enumerate(island):
                    if not hasattr(cell, 'fitness_history'):
                        cell.fitness_history = []
                    cell.fitness_history.append(island_fitness[i])

                # ========================================================================
                # 3. NOW, with fresh data, run the PNN Lock Quota logic.
                # ========================================================================
                allowed_to_lock_quota = len(island)  # Default to all cells allowed

                if cfg.enable_pnn_lock_quota:
                    # 1. Calculate the dynamic lock quota for this island
                    best_fitness_on_island = float(np.min(island_fitness))

                    # Normalize the fitness to a [0, 1] scale for the quota calculation.
                    # We'll use a simple exponential decay. As fitness -> inf, scale -> 0. As fitness -> 0, scale -> 1.
                    # The `pnn_quota_fitness_scale` controls how fast this decay happens.
                    performance_scale = math.exp(-best_fitness_on_island / cfg.pnn_quota_fitness_scale)

                    # Calculate the allowed lock fraction based on performance
                    max_lock_fraction = cfg.pnn_quota_min_fraction + (cfg.pnn_quota_base_fraction - cfg.pnn_quota_min_fraction) * performance_scale

                    # Convert fraction to a hard number of cells
                    allowed_to_lock_quota = int(max_lock_fraction * len(island))

                    logger.info(f"    PNN QUOTA: Best fitness={best_fitness_on_island:.4f}, Perf_scale={performance_scale:.2f}, Allowed_lock_fraction={max_lock_fraction:.2%}, Quota={allowed_to_lock_quota}/{len(island)}")

                # 2. Identify candidates for locking
                # A cell is a candidate if it's in the OPEN state but is stable enough to transition to CLOSING.
                locking_candidates = []
                for i, cell in enumerate(island):
                    if cell.pnn.state == PNN_STATE.OPEN:
                        # Re-use the stability check logic from the PNN update method
                        if hasattr(cell.pnn, "stability_history") and len(cell.pnn.stability_history) >= 5:
                            recent_mean = np.mean(list(cell.pnn.stability_history)[-5:])
                            relative_change = abs(island_fitness[i] - recent_mean) / max(1e-8, abs(recent_mean))
                            if relative_change < 0.05:  # If fitness is stable
                                # This cell is a candidate to start locking. Prioritize by fitness (lower is better).
                                locking_candidates.append((island_fitness[i], cell))

                # 3. Enforce the quota
                if len(locking_candidates) > allowed_to_lock_quota:
                    # Sort candidates: best fitness first
                    locking_candidates.sort(key=lambda x: x[0])

                    # Allow the best K candidates to lock, where K is the quota
                    cells_allowed_to_lock = {c.id for _, c in locking_candidates[:allowed_to_lock_quota]}

                    # For all other candidates, force them to remain OPEN, even if stable.
                    for fitness, cell in locking_candidates[allowed_to_lock_quota:]:
                        # This is a "soft" force. We just prevent the transition for this generation.
                        # We can add a special attribute to the cell that the pnn.update() method will check.
                        cell.pnn.force_open_due_to_quota = True
                        logger.info(f"    PNN QUOTA VETO: Cell {cell.id} (fitness {fitness:.4f}) is stable but vetoed from locking due to quota.")

                # ========================================================================
                # END: NEW DYNAMIC PNN LOCK QUOTA IMPLEMENTATION
                # ========================================================================

                logger.info(f"    Evaluation: {len(island)} cells in {eval_time:.3f}s")
                logger.info(f"GEN {generation} | island {island_idx} evaluated {len(island)} cells in {eval_time:.3f}s; fitness [{min(island_fitness):.6f}, {max(island_fitness):.6f}]")
                logger.info(f"    Fitness range: [{min(island_fitness):.6f}, {max(island_fitness):.6f}]")
                # Feasibility ratio (pre-eval bounds check) — target >95%
                lb, ub = self.bounds
                feas = [float(np.all((c.solution >= lb) & (c.solution <= ub))) for c in island]
                feasible_ratio = float(np.mean(feas)) if feas else 1.0
                logger.info(f"    Feasible ratio: {feasible_ratio:.2%}")

                # Log evaluated cells to tapestry
                for cell in island:
                    self._tap_cell(cell, island_name=f"island_{island_idx}", fit_cache=cache)

                # Calculate island statistics first
                best_idx = int(np.argmin(island_fitness))
                best_fitness = float(island_fitness[best_idx])
                median_fitness = float(np.median(island_fitness))
                # New: stricter locking threshold (top quartile, lower is better)
                try:
                    locking_threshold = float(np.percentile(np.asarray(island_fitness, dtype=float), 25))
                except Exception:
                    locking_threshold = None

                # === 4. FINALLY, run the PNN update loop. ===
                for i, cell in enumerate(island):
                    parent_stress = cell.local_stress  # This is set by the StressField.update() call
                    if self.enable_pnn:
                        # Before updating, check if this cell was vetoed by the quota system
                        if getattr(cell.pnn, 'force_open_due_to_quota', False):
                            # If vetoed, ensure it stays OPEN and reset the flag
                            cell.pnn.state = PNN_STATE.OPEN
                            cell.pnn.generations_in_phase = 0
                            cell.pnn.force_open_due_to_quota = False  # Reset for next generation
                            # Also add its current fitness to history to maintain the record
                            if hasattr(cell.pnn, "stability_history"):
                                cell.pnn.stability_history.append(island_fitness[i])
                        else:
                            # If not vetoed, update as normal
                            prev_state = cell.pnn.state
                            cell.pnn.update(
                                current_fitness=island_fitness[i],
                                generation=generation,
                                island_median_fitness=median_fitness,
                                locking_fitness_threshold=locking_threshold
                            )
                            # Count LOCK transitions for generation summary
                            try:
                                if prev_state != PNN_STATE.LOCKED and cell.pnn.state == PNN_STATE.LOCKED:
                                    locked_this_gen += 1
                            except Exception:
                                pass
                            if i == 0:  # Log first cell's PNN state as example
                                logger.info(f"    Cell {cell.id} PNN: {cell.pnn.state} (stress: {parent_stress:.3f})")

                new_island_cells = []
                # --- TARGETED CHANGE 2: Track per-island history (both best and median; we'll use median by default) ---
                self.island_best_hist[island_idx].append(best_fitness)
                self.island_median_hist[island_idx].append(median_fitness)
                
                new_island_cells.append(island[best_idx].copy(device=self.device))
                island_champions.append((island[best_idx].copy(device=self.device), island_fitness[best_idx]))
                
                logger.info(f"    Best fitness: {best_fitness:.6f} (cell {island[best_idx].id})")
                logger.info(f"GEN {generation} | island {island_idx} best={best_fitness:.6f} median={median_fitness:.6f} stem_cells={stem_cells}")

                # Perform exploitation bursts for current LOCKED cells (before creating new children)
                try:
                    # Delay exploit if feasibility is poor to avoid wasting NFE
                    if feasible_ratio >= 0.95:
                        self._exploit_locked_cells(island, island_idx, cache, generation, _safe_eval)
                        if any(c.pnn.state == PNN_STATE.LOCKED for c in island):
                            features_active['exploit_queue'] = True
                except Exception as e:
                    msg = str(e)
                    # Downgrade expected budget exhaustion to info to avoid alarming stderr on Windows
                    if isinstance(e, StopIteration) or 'NFE budget exhausted' in msg:
                        logger.info(f"    Exploit queue ended due to NFE cap on island {island_idx}")
                    else:
                        logger.warning(f"    Exploit queue failure on island {island_idx}: {e}")

                # === VECTORIZED REPRODUCTION (SoA + batched evaluate) ===
                self._reproduce_island_vectorized(island, island_fitness, island_idx, generation, island_obj, cache)
                final_island_cells = island  # downstream code expects this symbol
                # Legacy per-child reproduction loop; disabled by default. Vectorized path is authoritative.
                while getattr(cfg, 'enable_legacy_per_child_loop', False):
                    p1_idx, p2_idx = np.random.choice(len(island), 2, replace=False)
                    if getattr(cfg, 'pure_random_parents', False):
                        parent1 = island[p1_idx]
                        parent2 = island[p2_idx]
                    else:
                        parent1 = island[p1_idx] if island_fitness[p1_idx] < island_fitness[p2_idx] else island[p2_idx]
                        parent2 = island[p2_idx] if island_fitness[p1_idx] < island_fitness[p2_idx] else island[p1_idx]

                    # Causal veto on parent pairing based on tapestry history
                    if not getattr(cfg, 'disable_causal_queries', True):
                        try:
                            p1_type = parent1.genome[0].gene_type if parent1.genome else "NONE"
                            p2_type = parent2.genome[0].gene_type if parent2.genome else "NONE"
                            ctx = {
                                'island': f"island_{island_idx}",
                                'parent_types': tuple(sorted([p1_type, p2_type]))
                            }
                            stats = self.causal_tapestry.query_action_effect_with_stats('recombine', ctx, generation_window=20, decay_rate=0.1)
                            attempts = 0
                            while stats.get('count', 0) >= 5 and stats.get('effect', 0.0) > -1e-6 and attempts < 5:
                                # re-pick a different second parent if history predicts non-improvement
                                candidates = [j for j in range(len(island)) if j != p1_idx]
                                p2_idx = np.random.choice(candidates)
                                parent2 = island[p2_idx]
                                p2_type = parent2.genome[0].gene_type if parent2.genome else "NONE"
                                ctx['parent_types'] = tuple(sorted([p1_type, p2_type]))
                                stats = self.causal_tapestry.query_action_effect_with_stats('recombine', ctx, generation_window=20, decay_rate=0.1)
                                attempts += 1
                        except Exception:
                            pass

                    child_cell = parent1.copy(device=self.device)
                    children_created += 1

                    if parent1.pnn.state == PNN_STATE.LOCKED:
                        # --- UCB-driven exploit phase (patched) ---
                        child_cell.solution = parent1.solution.copy()
                        context_key = (f"island_{island_idx}", PNN_STATE.LOCKED.value, int(parent1.local_stress * 5))
                        if not hasattr(self, "_ucb_selectors"):
                            self._ucb_selectors = [ContextualUCBSelector(
                                ['nelder-mead','champion-linesearch','powell','cmaes'],
                                exploration_factor=getattr(cfg, 'ucb_exploration_factor', 0.5)
                            ) for _ in range(self.n_islands)]
                        ucb_selector = self._ucb_selectors[island_idx]
                        strategy = ucb_selector.select_strategy(context_key)
                        logger.debug(f"In exploit: strategy={strategy}")
                        # Track strategy usage and exploit actions
                        try:
                            self._exploit_actions_last_gen += 1
                            if not isinstance(getattr(self, '_strategy_usage_last_gen', None), dict):
                                self._strategy_usage_last_gen = {}
                            self._strategy_usage_last_gen[strategy] = self._strategy_usage_last_gen.get(strategy, 0) + 1
                        except Exception:
                            pass
                        fitness_before_exploit = cache.get(parent1.id, float('inf'))
                        population_mean = np.mean([c.solution for c in island], axis=0)
                        if strategy == 'nelder-mead' and hasattr(toolkit, 'trust_region_nelder_mead'):
                            child_cell.solution = toolkit.trust_region_nelder_mead(child_cell.solution, radius=0.5 * (self.bounds[1]-self.bounds[0]) / max(1.0, np.sqrt(self.dimension)))
                        elif strategy == 'powell' and hasattr(toolkit, 'powell_step'):
                            child_cell.solution = toolkit.powell_step(child_cell.solution)
                        else:
                            child_cell.solution = toolkit.champion_linesearch_step(child_cell.solution, population_mean, self.dimension)

                        fitness_after_exploit = _safe_eval_for_toolkit(child_cell.solution)
                        cache[child_cell.id] = fitness_after_exploit
                        improvement = max(0.0, float(fitness_before_exploit - fitness_after_exploit))
                        try:
                            ucb_selector.update(context_key, strategy, improvement)
                        except Exception:
                            pass

                        self._tap_event("EXPLOIT", {
                            "action": "exploit",
                            "cell_id": str(parent1.id),
                            "strategy_used": strategy,
                            "effect": improvement,
                            "island": f"island_{island_idx}",
                            "pnn_state": PNN_STATE.LOCKED.value
                        })
                        logger.debug(f"In optimize: exploit: child_cell.solution={child_cell.solution}")
                    else:
                        # --- EXPLORE PHASE (Only for non-locked parents) ---
                        # 1. Standard Crossover
                        best_idx = int(np.argmin(island_fitness))
                        best_sol = island[best_idx].solution
                        alpha = random.random()
                        # Anneal beta from strong exploitation early to more exploration later
                        anneal_beta = max(0.01, 0.05 * (1.0 - generation / self.max_generations))
                        child_cell.solution = (
                            alpha * parent1.solution + (1 - alpha) * parent2.solution
                            + anneal_beta * (best_sol - parent1.solution)
                        )
                        # SILENCED: PNN EXPLORE crossover details (too noisy)
                        
                        # 2. Standard Mutation (Causal Tapestry, Stress, etc.)
                        lb, ub = self.bounds
                        range_scale = (ub - lb) / np.sqrt(self.dimension)
                        anneal = max(0.1, 1.0 - generation / max(1, self.max_generations))  # cools over time
                        # FIX A: Re-enable biological plasticity in mutation
                        plasticity = parent1.pnn.plasticity_multiplier * self.circadian_controller.plasticity_multiplier
                        logger.debug(f"In optimize: plasticity={plasticity}")
                        # FIX B: TACTICAL STRESS RESPONSE - Allow gentle boost for CLOSING state
                        parent_stress = parent1.local_stress  # Use the stress from StressField
                        stress_boost = 1.0
                        if parent_stress > cfg.stress_threshold:
                            # --- CALM DOWN: Only apply stress boost during DAY cycle ---
                            if self.circadian_controller.is_day:
                                if parent1.pnn.state == PNN_STATE.OPEN:
                                    stress_boost = 1.5  # Full boost for unstable parents
                                    logger.info(f"      TACTICAL STRESS BOOST (DAY): High stress ({parent_stress:.3f} > {cfg.stress_threshold}) + OPEN PNN state -> boosting mutation")
                                elif parent1.pnn.state == PNN_STATE.CLOSING:
                                    stress_boost = 1.15  # Gentle nudge for CLOSING state
                                    logger.info(f"      TACTICAL STRESS MICRO-BOOST (DAY): High stress ({parent_stress:.3f} > {cfg.stress_threshold}) + CLOSING PNN state -> small boost")
                                else:
                                    logger.info(f"      STRESS IGNORED: High stress ({parent_stress:.3f} > {cfg.stress_threshold}) but PNN state is {parent1.pnn.state.value} -> no boost (focusing on refinement)")
                            else:
                                logger.info(f"      STRESS BOOST SKIPPED: High stress detected ({parent_stress:.3f} > {cfg.stress_threshold}), but it is NIGHT cycle.")
                        
                        # Use adaptive island-specific sigma
                        mutation_strength = self._island_sigma[island_idx] * plasticity * anneal * stress_boost
                        # SILENCED: PNN EXPLORE mutation strength details (too noisy)

                    # --- TARGETED FIX: MOVE CAUSAL INTERVENTION BEFORE MUTATION ---
                    # --- Percentile-weighted causal intervention ---
                    if generation > 10 and random.random() < 0.3:  # Increased chance to query and act
                        # FIX: Add safety wrapper to prevent SystemError from crashing optimization
                        mutation_history = 0.0
                        if not getattr(cfg, 'disable_causal_queries', True):
                            try:
                                mutation_history = self.causal_tapestry.query_action_effect(
                                    action='recombine',
                                    context_filters={
                                        'island': f'island_{island_idx}',
                                        'stress_bin': int(parent_stress * 5),     # same binning as you log
                                        'pnn_state': parent1.pnn.state.value      # OPEN / CLOSING / LOCKED
                                    },
                                    generation_window=cfg.causal_tapestry_generation_window,
                                    decay_rate=cfg.causal_tapestry_decay_rate
                                )
                            except Exception as e:
                                logger.warning(f"      CAUSAL TAPESTRY QUERY FAILED: {e} - using fallback")

                        if mutation_history != 0.0:
                            logger.info(f"      CAUSAL FEEDBACK: Historical recombination effect (island {island_idx}): {mutation_history:.3f}")

                            # --- TARGETED CHANGE 3: Percentile weighting (LOCAL instead of global) ---
                            pct_window    = cfg.percentile_window
                            low_pct       = cfg.low_percentile
                            high_pct      = cfg.high_percentile
                            default_w     = cfg.default_percentile_weight
                            
                            # Global (small) context, to keep some signal when ALL islands are great
                            glob_hist = self.best_fitness_history[-pct_window:] if pct_window > 0 else self.best_fitness_history
                            w_global, glo, ghi, gn = percentile_weight_from_history(
                                current_value=best_so_far,
                                history=glob_hist,
                                low_pct=low_pct, high_pct=high_pct,
                                default_weight=default_w
                            )
                            
                            # Local island health: use median to reflect overall island state (robust vs. a single good champ)
                            island_hist = list(self.island_median_hist[island_idx])
                            island_stat_now = float(np.median(island_fitness))
                            w_island, ilo, ihi, in_hist = percentile_weight_from_history(
                                current_value=island_stat_now,
                                history=island_hist,
                                low_pct=low_pct, high_pct=high_pct,
                                default_weight=default_w
                            )
                            
                            # Stress-driven weight: engage more when parent is under stress
                            stress_relax = getattr(cfg, "stress_relax", 0.45)   # new knob, see cfg below
                            w_stress = float(np.clip(
                                (parent_stress - stress_relax) / max(1e-8, cfg.stress_threshold - stress_relax),
                                0.0, 1.0
                            ))
                            logger.debug(f"In optimize: Stress-driven weight:w_stress={w_stress}")
                            # Final influence: prioritize LOCAL signal; keep some GLOBAL/STRESS input
                            w = max(w_island, 0.5 * w_global, w_stress)

                            # Add normalized-fitness gating (0..1 -> 0..100% via tapestry_weight)
                            try:
                                nf = self._normalize_func(best_so_far) if hasattr(self, '_normalize_func') else None
                            except Exception:
                                nf = None
                            if nf is not None and np.isfinite(nf):
                                try:
                                    w_fit_pct = tapestry_weight(
                                        float(nf),
                                        low_threshold=float(getattr(cfg, 'low_fitness_threshold_norm', 0.2)),
                                        high_threshold=float(getattr(cfg, 'high_fitness_threshold_norm', 0.8))
                                    )
                                    w_fit = float(w_fit_pct) / 100.0
                                    w = max(w, w_fit)
                                except Exception:
                                    pass
                            
                            # Ensure a minimum influence when history says recombination is bad
                            min_w_bad = getattr(cfg, "min_weight_when_history_bad", 0.15)  # new knob
                            if mutation_history > 0.1:
                                w = max(w, min_w_bad)
                                logger.debug(f"In optimize: Bad Mutation Historyw={w}")
                            # Clamp causal weighting to zero when local is good and improving
                            # 'impr' is improvement in this context; compute conservatively if not present
                            improvement_proxy = 0.0
                            try:
                                # Use recent improvement tracker if available in scope
                                improvement_proxy = float(locals().get('impr', 0.0))
                            except Exception:
                                improvement_proxy = 0.0
                            if late_phase_active and w_island < 0.1 and improvement_proxy > 0:
                                w = 0.0
                                logger.info(f"      LATE PHASE: Clamping causal weight to 0 (local doing well, improving)")
                            
                            logger.info(
                                f"      PERCENTILE WEIGHTING (local): w_island={w_island:.2%} "
                                f"| low@{low_pct:.1f}->{ilo:.6g} | high@{high_pct:.1f}->{ihi:.6g} | n={in_hist} "
                                f"|| w_global={w_global:.2%}, w_stress={w_stress:.2%} -> w_final={w:.2%}"
                            )
                            
                            # Modulate aggressiveness by recent island success (as before)
                            recent_success_rate = self._succ[island_idx] / max(1, self._trials[island_idx])
                            if   recent_success_rate > 0.30: mult, mode = 0.10, "conservative"
                            elif recent_success_rate > 0.15: mult, mode = 0.30, "moderate"
                            else:                            mult, mode = 0.80, "aggressive"
                            
                            if mutation_history < -0.1:
                                base_reduction = 0.50
                                eff = base_reduction * w * mult
                                mutation_strength *= (1.0 - eff)
                                logger.info(f"      CAUSAL INTERVENTION ({mode}): -{eff:.2%} (history good, lighten touch)")
                            elif mutation_history > 0.1:
                                base_increase = 0.20
                                eff = base_increase * w * mult
                                mutation_strength *= (1.0 + eff)
                                logger.info(f"      CAUSAL INTERVENTION ({mode}): +{eff:.2%} (history bad, stronger push)")

                    # --- APPLY PHENOTYPE FROM GENOME ---
                    # FIX: Apply phenotype delta BEFORE Gaussian noise to prevent overshooting
                    # Convert current solution to a tensor for the NN
                    solution_tensor = torch.tensor(child_cell.solution, dtype=torch.float32).to(self.device)
                    
                    # Ensure the cell's genome modules are on the same device
                    child_cell.to(self.device)
                    
                    # FIX: Skip ODE when mutation strength is very small or PNN is LOCKED (performance optimization)
                    phenotype_delta = np.zeros(self.dimension)
                    # Ensure scalar mutation strength for boolean logic
                    try:
                        ms_arr = np.asarray(mutation_strength)
                        ms_scalar = float(np.max(np.abs(ms_arr))) if ms_arr.size > 1 else float(ms_arr)
                    except Exception:
                        ms_scalar = float(mutation_strength)
                    # Guard config flags to be plain bool/float
                    min_ms = float(getattr(cfg, 'phenotype_min_mutation', 1e-6))
                    ode_skip_enabled = bool(getattr(cfg, 'enable_ode_skip', True))

                    should_compute_phenotype = (
                        (ms_scalar >= min_ms)
                        and (parent1.pnn.state != PNN_STATE.LOCKED)
                        and (not ode_skip_enabled or parent1.pnn.state != PNN_STATE.LOCKED)
                    )
                    
                    # DEBUG: Log phenotype computation decision
                    logger.debug(
                        f"      PHENOTYPE: should_compute_phenotype={should_compute_phenotype} "
                        f"(ms_scalar={ms_scalar:.3g}, min_ms={min_ms:.3g}, "
                        f"pnn_locked={parent1.pnn.state == PNN_STATE.LOCKED}, "
                        f"ode_skip_enabled={ode_skip_enabled})"
                    )
                    
                    if should_compute_phenotype:
                        # FIX: Set skip_ode flag for performance optimization
                        # Skip ODE when mutation strength is very small or PNN is locked
                        if ode_skip_enabled:
                            for gene in child_cell.genome:
                                gene._skip_ode = (ms_scalar < 1e-3 or parent1.pnn.state == PNN_STATE.LOCKED)
                        else:
                            # Reset skip_ode flag if optimization is disabled
                            for gene in child_cell.genome:
                                gene._skip_ode = False
                        
                        # Get the phenotype delta from the cell's forward pass
                        # The phenotype is an update vector, not a new position
                        phenotype_delta = child_cell.forward(solution_tensor).detach().cpu().numpy()
                        
                        # DEBUG: Log phenotype delta stats
                        logger.debug(
                            f"      PHENOTYPE: delta norm={np.linalg.norm(phenotype_delta):.4g}, "
                            f"min={phenotype_delta.min():.4g}, max={phenotype_delta.max():.4g}"
                        )
                        
                        # FIX: Scale phenotype delta by mutation strength and range to prevent huge overshoots
                        range_scale = (ub - lb) / np.sqrt(self.dimension)
                        lr = cfg.phenotype_scale_factor * (mutation_strength / range_scale)
                        logger.debug(
                            f"      PHENOTYPE: range_scale={range_scale:.4g}, "
                            f"phenotype_scale_factor={cfg.phenotype_scale_factor}, "
                            f"mutation_strength={mutation_strength:.4g}, lr={lr:.4g}"
                        )
                        child_cell.solution += phenotype_delta * lr
                        child_cell.solution = np.clip(child_cell.solution, lb, ub)  # Re-clip after phenotype update
                        logger.debug(
                            f"      PHENOTYPE: solution post-phenotype min={child_cell.solution.min():.4g}, "
                            f"max={child_cell.solution.max():.4g}"
                        )
                    
                    # --- END OF PHENOTYPE APPLICATION ---
                    
                    # Now apply Gaussian noise around the phenotype-adjusted position
                    # For LOCKED cells, this is the only noise applied (no phenotype)
                    if parent1.pnn.state != PNN_STATE.LOCKED:
                        # Mixture: Gaussian + occasional Cauchy heavy tail (annealed probability)
                        try:
                            p_ht = float(getattr(cfg, 'heavy_tail_mutation_prob', 0.0)) * float(anneal)
                            p_ht = max(0.0, min(1.0, p_ht))
                        except Exception:
                            p_ht = float(getattr(cfg, 'heavy_tail_mutation_prob', 0.0))
                        logger.debug(f"      NOISE: heavy_tail_prob={p_ht:.3g}, anneal={anneal:.3g}")
                        if random.random() < p_ht:
                            noise = np.random.standard_cauchy(size=self.dimension) * (mutation_strength * getattr(cfg, 'heavy_tail_scale', 3.0))
                            logger.debug(f"      NOISE: Using Cauchy noise, scale={mutation_strength * getattr(cfg, 'heavy_tail_scale', 3.0):.4g}")
                        else:
                            noise = np.random.normal(0, mutation_strength, self.dimension)
                            logger.debug(f"      NOISE: Using Gaussian noise, std={mutation_strength:.4g}")
                        # Query directional oracle from tapestry
                        direction = None
                        if not getattr(cfg, 'disable_causal_queries', True) and hasattr(self, '_causal_stats_snapshot'):
                            try:
                                dir_ctx = (
                                    'recombine',
                                    tuple(sorted([
                                        ('island', f"island_{island_idx}"),
                                        ('pnn_state', parent1.pnn.state.value),
                                        ('stress_bin', int(parent1.local_stress * 5)),
                                        ('parent_types', tuple(sorted([p1_type, p2_type])) if 'p1_type' in locals() and 'p2_type' in locals() else None)
                                    ]))
                                )
                                # Prefer snapshot-derived direction if available; fallback to live query
                                ds = self._causal_stats_snapshot.get(dir_ctx) if isinstance(self._causal_stats_snapshot, dict) else None
                                if ds and isinstance(ds.get('effect'), (int, float)) and ds.get('effect', 0.0) < 0.0:
                                    # No direct vector from snapshot; trust effect sign only → gently nudge along parent diff
                                    base_dir = (parent1.solution - parent2.solution)
                                    nrm = np.linalg.norm(base_dir)
                                    if nrm > 1e-12:
                                        direction = base_dir / nrm
                                if direction is None:
                                    # Live query fallback to directional memory
                                    direction = self.causal_tapestry.query_causal_direction('recombine', {
                                        'island': f"island_{island_idx}",
                                        'pnn_state': parent1.pnn.state.value,
                                        'stress_bin': int(parent1.local_stress * 5),
                                        'parent_types': tuple(sorted([p1_type, p2_type])) if 'p1_type' in locals() and 'p2_type' in locals() else None
                                    })
                            except Exception:
                                direction = None
                        if direction is not None:
                            logger.debug(f"      NOISE: Adding causal direction, norm={np.linalg.norm(direction):.4g}, influence={cfg.causal_mutagen_influence:.4g}")
                            noise = noise + direction * (mutation_strength * cfg.causal_mutagen_influence)
                        child_cell.solution += noise
                        logger.debug(
                            f"      NOISE: solution post-noise min={child_cell.solution.min():.4g}, "
                            f"max={child_cell.solution.max():.4g}, noise norm={np.linalg.norm(noise):.4g}"
                        )
                    else:
                        logger.info(f"      PNN EXPLOIT: Skipped additional Gaussian noise (LOCKED state - linesearch will handle refinement)")
                    
                    # FIX F: Clamp sigma aggressively when many steps clip at bounds
                    pre_clip = child_cell.solution.copy()
                    child_cell.solution = np.clip(child_cell.solution, lb, ub)
                    hit_fraction = np.mean((pre_clip != child_cell.solution).astype(np.float32))
                    if hit_fraction > 0.25:  # >25% coords clipped
                        self._island_sigma[island_idx] *= 0.8  # shrink sigma
                        try:
                            cur_sigma = float(np.mean(self._island_sigma[island_idx]))
                        except Exception:
                            cur_sigma = self._island_sigma[island_idx]
                        logger.info(f"      SIGMA CLAMP: {hit_fraction:.1%} bounds hit, shrinking sigma to {cur_sigma}")
                    logger.debug(
                        f"      CLIP: hit_fraction={hit_fraction:.3g}, solution min={child_cell.solution.min():.4g}, max={child_cell.solution.max():.4g}"
                    )
                    
                    # Cap the island sigma to prevent excessive values
                    # Use a scalar cap derived from average search range to avoid ambiguous array comparisons
                    sigma_max_scalar = float(0.2 * np.mean(self.bounds[1] - self.bounds[0]) / np.sqrt(self.dimension))
                    cur = self._island_sigma[island_idx]
                    if isinstance(cur, np.ndarray):
                        if np.any(cur > sigma_max_scalar):
                            self._island_sigma[island_idx] = np.minimum(cur, sigma_max_scalar)
                            logger.info(f"      SIGMA CAP: Capped vector sigma to <= {sigma_max_scalar:.6f}")
                    else:
                        if cur > sigma_max_scalar:
                            self._island_sigma[island_idx] = sigma_max_scalar
                            logger.info(f"      SIGMA CAP: Capped sigma to {sigma_max_scalar:.6f}")
                    
                    # Log mutation details for first few children
                    if children_created <= 2:
                        try:
                            ms = float(np.asarray(mutation_strength).mean())
                        except Exception:
                            ms = float(mutation_strength)
                        try:
                            pl = float(np.asarray(plasticity).mean())
                        except Exception:
                            pl = float(plasticity)
                        try:
                            an = float(np.asarray(anneal).mean())
                        except Exception:
                            an = float(anneal)
                        logger.info(f"      Child {children_created}: mutation_strength={ms:.6f}, plasticity={pl:.3f}, anneal={an:.3f}")
                        logger.debug(
                            f"      Child {children_created}: solution min={child_cell.solution.min():.4g}, max={child_cell.solution.max():.4g}"
                        )
                    
                    # ========================================================================
                    # END: PNN-DRIVEN EXPLOITATION PHASE IMPLEMENTATION
                    # ========================================================================
                    
                    # Evaluate child once for greedy selection
                    f_parent = island_fitness[p1_idx]
                    f_child = _safe_eval(child_cell.get_solution())
                    fitness_caches[island_idx][child_cell.id] = f_child
                    logger.debug(
                        f"      FITNESS: f_parent={f_parent:.6f}, f_child={f_child:.6f}, delta={f_child - f_parent:.6g}"
                    )
                    
                    # Log lineage and recombination event
                    self._tap_lineage(parent1, child_cell)
                    
                    # Log recombination effect (using cached fitness)
                    parent_fit = float(island_fitness[p1_idx])
                    other_fit = float(island_fitness[p2_idx])
                    avg_parent = 0.5 * (parent_fit + other_fit)
                    effect = float(f_child - avg_parent)
                    logger.debug(
                        f"      RECOMBINE: parent_fit={parent_fit:.6f}, other_fit={other_fit:.6f}, avg_parent={avg_parent:.6f}, effect={effect:.6g}"
                    )
                    
                    # Get parent gene types for context
                    p1_gene_type = parent1.genome[0].gene_type if parent1.genome else "NONE"
                    p2_gene_type = parent2.genome[0].gene_type if parent2.genome else "NONE"
                    
                    # Check if child has quantum genes
                    child_has_quantum = any(isinstance(gene, QuantumGeneModule) for gene in child_cell.genome) if child_cell.genome and self.enable_quantum else False
                    
                    eid = self._tap_event("MUTATION", {
                        "action": "recombine",
                        "parent_types": tuple(sorted([p1_gene_type, p2_gene_type])),
                        "parent_ids": [str(parent1.id), str(parent2.id)],
                        "child_id": str(child_cell.id),
                        "parent_fitness_avg": avg_parent,
                        "child_fitness": float(f_child),
                        # keep effect sign as defined elsewhere (minimization: improvement < 0)
                        "effect": effect,
                        # record the applied mutation vector when improvement occurs
                        "mutation_vector": (child_cell.solution - (alpha * parent1.solution + (1 - alpha) * parent2.solution)).tolist() if effect < 0 else None,
                        "island": f"island_{island_idx}",
                        "stress_bin": int(parent_stress*5),
                        "child_has_quantum": child_has_quantum
                    })
                    self.causal_tapestry.log_event_participation(str(parent1.id), eid, "parent")
                    self.causal_tapestry.log_event_participation(str(parent2.id), eid, "parent")
                    self.causal_tapestry.log_event_output(eid, str(child_cell.id), "child")
                    
                    # Track success for 1/5th rule
                    self._trials[island_idx] += 1
                    if f_child < f_parent:
                        self._succ[island_idx] += 1
                        # --- NEW: Track improvement magnitude ---
                        improvement_magnitude = f_parent - f_child  # Positive improvement
                        self._improvement_magnitudes[island_idx].append(improvement_magnitude)
                        if children_created <= 2:
                            logger.info(f"      Child {children_created}: SUCCESS! {f_child:.6f} < {f_parent:.6f} (improvement: {improvement_magnitude:.6f})")
                            logger.debug(
                                f"      Child {children_created}: improvement_magnitude={improvement_magnitude:.6g}"
                            )
                    else:
                        logger.debug(
                            f"      Child {children_created}: NO IMPROVEMENT ({f_child:.6f} >= {f_parent:.6f})"
                        )
                    
                    # FIX: immune step may change the solution -> must re-evaluate and update cache
                    # Use child_cell's local_stress instead of _calculate_stress
                    cell_stress = child_cell.local_stress
                    modified = self.immune_response(child_cell.get_solution(), cell_stress)
                    if not np.array_equal(modified, child_cell.solution):
                        logger.debug("      IMMUNE: Solution modified by immune_response, re-evaluating fitness.")
                        child_cell.solution = modified
                        # solution changed -> force recompute
                        cache.pop(child_cell.id, None)
                        f_child = _safe_eval(child_cell.solution)
                        cache[child_cell.id] = f_child
                    
                    if self.enable_self_modify:
                        # Self-modification based on fitness trend
                        fitness_trend = 0.0
                        if len(parent1.fitness_history) > 5:
                            recent_fitness = parent1.fitness_history[-5:]
                            fitness_trend = (recent_fitness[-1] - recent_fitness[0]) / len(recent_fitness)
                        logger.debug(f"      SELF-MODIFY: fitness_trend={fitness_trend:.6g}")
                        child_cell.architecture_modifier.decide_and_apply_modification(fitness_trend)
                        self.causal_tapestry.log_event_participation(str(child_cell.id), eid, "host")
                        
                        # Update cache with new fitness if architecture changed
                        cache[child_cell.id] = f_child
                    else:
                        gene_to_mutate_idx = self._select_gene_to_mutate(child_cell)
                        
                        # Check if a valid gene was found before attempting mutation
                        if gene_to_mutate_idx is not None and getattr(cfg, 'torch_genome_enabled', False):
                            # NEW: CoW materialization before any param writes
                            try:
                                child_cell._ensure_private_genome()
                            except Exception:
                                pass
                            with torch.no_grad():
                                for param in child_cell.genome[gene_to_mutate_idx].parameters():
                                    noise = torch.randn(param.size(), device=param.device) * self.neural_mutation_strength
                                    param.add_(noise)
                            logger.debug(f"      TORCH-GENOME: Mutated gene {gene_to_mutate_idx} with neural_mutation_strength={self.neural_mutation_strength:.4g}")
                    # Apply stability-based architecture modification
                    if hasattr(parent1, 'pnn') and hasattr(parent1.pnn, 'stability_history'):
                        stability_list = list(parent1.pnn.stability_history)
                        if len(stability_list) >= 10:
                            fitness_trend = np.mean(stability_list[-5:]) - np.mean(stability_list[:5])
                            logger.debug(f"      STABILITY-MOD: fitness_trend={fitness_trend:.6g}")
                            child_cell.architecture_modifier.decide_and_apply_modification(fitness_trend)
                    
                    new_island_cells.append(child_cell)

                # Legacy post-loop replaced by vectorized path above

                # ========================================================================
                # EXPLOITATION TOOLKIT INTEGRATION
                # ========================================================================
                
                # We need the island's objective function for the toolkit
                island_obj = self.island_objectives[island_idx]
                cache = fitness_caches[island_idx]
                
                def make_safe_eval(obj_fn):
                    def _safe_eval_x(x: np.ndarray) -> float:
                        return float(obj_fn(x))
                    # Attach batch if available on the objective wrapper
                    try:
                        if hasattr(obj_fn, 'batch') and callable(getattr(obj_fn, 'batch')):
                            setattr(_safe_eval_x, 'batch', lambda X, obj_fn=obj_fn: np.asarray(obj_fn.batch(np.asarray(X, dtype=float)), dtype=float))
                    except Exception:
                        pass
                    return _safe_eval_x
                
                _safe_eval_for_toolkit = make_safe_eval(island_obj)
                
                # Instantiate the toolkit for this generation
                toolkit = ExploitationToolkit(self.bounds, _safe_eval_for_toolkit, getattr(_safe_eval_for_toolkit, 'batch', None))

                self.islands[island_idx] = final_island_cells
                
                # Champion-aware strategic unlocking
                unlocked_this_gen += int(self._strategic_pnn_unlocking(final_island_cells, island_idx, generation, island_champion_fitness=best_fitness) or 0)

                # ========================================================================
                # END: STRATEGIC PNN UNLOCKING IMPLEMENTATION
                # ========================================================================

                all_phenotypes.extend([self._get_phenotype(c) for c in final_island_cells])

                # ========================================================================
                # START: NEW HIGH-LEVEL ISLAND SUMMARY LOGGING
                # ========================================================================

                # Log PNN phase statistics for this island
                pnn_states = [cell.pnn.state for cell in final_island_cells]
                open_count = pnn_states.count(PNN_STATE.OPEN)
                closing_count = pnn_states.count(PNN_STATE.CLOSING)
                locked_count = pnn_states.count(PNN_STATE.LOCKED)
                total_cells = len(final_island_cells)

                # Determine the primary mode of the island for this generation
                if locked_count / total_cells > 0.5:
                    island_mode = "EXPLOITING"
                else:
                    island_mode = "EXPLORING"

                logger.info(f"  ISLAND {island_idx} SUMMARY (Mode: {island_mode}): "
                            f"Open={open_count}, Closing={closing_count}, Locked={locked_count}")

                # Log average mutation sigma for this island
                try:
                    avg_sigma_val = float(np.asarray(self._island_sigma[island_idx]).mean())
                except Exception:
                    avg_sigma_val = float(self._island_sigma[island_idx])
                logger.info(f"      Avg. Mutation Sigma: {avg_sigma_val:.4f}")
                logger.debug(
                    f"      ISLAND {island_idx}: sigma vector min={np.min(self._island_sigma[island_idx]):.4g}, "
                    f"max={np.max(self._island_sigma[island_idx]):.4g}"
                )

                # ========================================================================
                # END: NEW HIGH-LEVEL ISLAND SUMMARY LOGGING
                # ========================================================================

            # compute base_f on each island champion once per generation
            # Per-generation feature/PNN summary line
            try:
                logger.info(
                    f"GEN {generation} | features: migration={features_active['migration']} "
                    f"dream={features_active['dream']} linesearch={features_active['linesearch']} "
                    f"exploit_queue={features_active['exploit_queue']} phase_crisis={features_active['phase_crisis']} "
                    f"| PNN: locked+={locked_this_gen} unlocked+={unlocked_this_gen} "
                    f"| dreams_created={getattr(self, '_dreams_created_last_gen', 0)} "
                    f"| exploit_actions={getattr(self, '_exploit_actions_last_gen', 0)} "
                    f"| strategy_usage={getattr(self, '_strategy_usage_last_gen', {})}"
                )
                logger.debug(
                    f"GEN {generation} | best_fitness_history={self.best_fitness_history[-5:] if len(self.best_fitness_history) >= 5 else self.best_fitness_history}"
                )
                # Accumulate generation totals
                try:
                    self._total_exploit_actions += int(getattr(self, '_exploit_actions_last_gen', 0))
                    self._total_unlocks += int(unlocked_this_gen)
                    if features_active['migration']:
                        self._total_migration_rounds += 1
                    if features_active['dream']:
                        self._total_dream_rounds += 1
                    if features_active['linesearch']:
                        self._total_linesearch_rounds += 1
                    
                    # Calculate and track lock fraction for telemetry
                    total_cells = sum(len(island) for island in self.islands)
                    if total_cells > 0:
                        # Fraction of cells currently LOCKED (not just new locks this gen)
                        num_locked_now = sum(1 for island in self.islands for c in island if c.pnn.state == PNN_STATE.LOCKED)
                        lock_fraction = num_locked_now / total_cells
                        self._lock_fraction_history.append(lock_fraction)
                        logger.debug(f"GEN {generation} | lock_fraction={lock_fraction:.4g}")
                except Exception as e:
                    logger.debug(f"GEN {generation} | Exception in generation totals accumulation: {e}")
            except Exception as e:
                logger.debug(f"GEN {generation} | Exception in per-generation summary: {e}")
            base_champs = []
            for i, island in enumerate(self.islands):
                if not island:
                    continue
                island_obj = self.island_objectives[i]
                cache = fitness_caches[i]
                # best under island objective
                fits = [self._eval_cell_with_obj(c, island_obj, cache) for c in island]
                best_idx = int(np.argmin(fits))
                champ = island[best_idx]
                # evaluate "true" base objective ONCE (reduce MO vectors to scalar via sum)
                _f = base_f(champ.get_solution())
                try:
                    if isinstance(_f, (list, tuple, np.ndarray)):
                        _f = float(np.sum(np.asarray(_f, dtype=float)))
                    else:
                        _f = float(_f)
                except Exception:
                    _f = float(np.nan)
                f_true = _f
                base_champs.append(f_true)
                logger.debug(f"GEN {generation} | ISLAND {i} champion fitness: {f_true:.6f}")

            best_true = min(base_champs) if base_champs else float('inf')
            self.best_fitness_history.append(best_true)
            logger.debug(f"GEN {generation} | best_true={best_true:.6f}")

            # Use the cache to compute best/diversity without extra evals
            all_cells = [c for island in self.islands for c in island]
            all_fitness = [
                self._eval_cell_with_obj(c, self.island_objectives[i], fitness_caches[i])
                for i, island in enumerate(self.islands) for c in island
            ]
            logger.debug(f"GEN {generation} | all_fitness: min={np.min(all_fitness):.6f}, max={np.max(all_fitness):.6f}")

            # migration: island_champions holds (cell, fitness)
            if generation > 0 and generation % self.migration_interval == 0:
                features_active['migration'] = True
                logger.info(f"  MIGRATION EVENT: Moving champions between islands")
                for i in range(self.n_islands):
                    migrant_cell, migrant_fit = island_champions[i]
                    target = (i + 1) % self.n_islands
                    target_obj = self.island_objectives[target]
                    tcache = fitness_caches[target]
                    # find worst in target island using cache
                    target_fits = [self._eval_cell_with_obj(c, target_obj, tcache) for c in self.islands[target]]
                    worst_idx = int(np.argmax(target_fits))
                    worst_fit = target_fits[worst_idx]
                    self.islands[target][worst_idx] = migrant_cell.copy(device=self.device)
                    # refresh cache entry under the TARGET objective
                    tcache[self.islands[target][worst_idx].id] = float(target_obj(self.islands[target][worst_idx].get_solution()))
                    logger.info(f"    Island {i} -> {target}: {migrant_fit:.6f} replaces {worst_fit:.6f}")
                    logger.info(f"GEN {generation} | migration i{i}->i{target} migrant_fit={migrant_fit:.6f} replaced={worst_fit:.6f}")

                    # DEBUG: Log migration details
                    logger.debug(
                        f"    MIGRATION: i{i} champion id={migrant_cell.id} fit={migrant_fit:.6f} "
                        f"-> i{target} replaces idx={worst_idx} fit={worst_fit:.6f}"
                    )

                    # Log migration event
                    eid = self._tap_event("MUTATION", {
                        "action": "migrate",
                        "from": f"island_{i}",
                        "to": f"island_{target}",
                        "migrant_id": str(migrant_cell.id)
                    })
                    self.causal_tapestry.log_event_participation(str(migrant_cell.id), eid, "migrant")

                    # Update node metadata with target-island fitness (already computed above)
                    self._tap_cell(self.islands[target][worst_idx], island_name=f"island_{target}", fit_cache=tcache)

            # Dream consolidation uses the last island's safe evaluator (will be updated per island)
            if self.enable_dream and generation % self.dream_interval == 0:
                features_active['dream'] = True
                logger.info(f"  DREAM CONSOLIDATION: Generating new solutions")
                logger.info(f"GEN {generation} | dream_consolidation across islands")
                for island_idx, island in enumerate(self.islands):
                    island_obj = self.island_objectives[island_idx]
                    cache = fitness_caches[island_idx]

                    def make_safe_eval(obj_fn):
                        def _safe_eval_x(x: np.ndarray) -> float:
                            return float(obj_fn(x))
                        return _safe_eval_x

                    _safe_eval = make_safe_eval(island_obj)
                    self.dream_consolidation(generation, cache, _safe_eval)

            # Champion line search with uncertainty-aware strategy selection via tapestry
            # FIX: Adaptive Champion Linesearch - run more frequently when progress stalls
            should_run_linesearch = (
                generation % cfg.champion_linesearch_regular_interval == 0 and cfg.enable_champion_linesearch or  # Regular schedule
                (no_improve >= cfg.champion_linesearch_stagnation_threshold and generation > 10)  # Adaptive: run when stuck
            )

            if should_run_linesearch:
                features_active['linesearch'] = True
                trigger_reason = "regular schedule" if generation % cfg.champion_linesearch_regular_interval == 0 else f"stagnation ({no_improve} generations)"
                logger.info(f"  CHAMPION LINE SEARCH: Performing local optimization (trigger: {trigger_reason})")
                logger.info(f"GEN {generation} | champion_linesearch trigger={trigger_reason}")
                for island_idx, island in enumerate(self.islands):
                    island_obj = self.island_objectives[island_idx]
                    cache = fitness_caches[island_idx]

                    def make_safe_eval(obj_fn):
                        def _safe_eval_x(x: np.ndarray) -> float:
                            return float(obj_fn(x))
                        return _safe_eval_x

                    _safe_eval = make_safe_eval(island_obj)
                    best_before = min([cache.get(c.id, float('inf')) for c in island])
                    self.champion_linesearch(island, cache, _safe_eval)
                    best_after = min([cache.get(c.id, float('inf')) for c in island])
                    if best_after < best_before:
                        logger.info(f"    Island {island_idx}: {best_before:.6f} -> {best_after:.6f} (improvement: {best_before - best_after:.6f})")
                        logger.info(f"GEN {generation} | linesearch_improve island={island_idx} dF={(best_before - best_after):.6f}")

            # Update sigma using 1/5th rule every 10 generations (optional)
            if getattr(cfg, 'enable_one_fifth_rule', False) and (generation + 1) % 10 == 0:
                logger.info(f"  1/5TH RULE UPDATE: Adapting mutation strengths")
                for i in range(len(self.islands)):
                    rate = (self._succ[i] / max(1, self._trials[i]))
                    old_sigma = self._island_sigma[i]

                    # --- NEW: Improvement-aware 1/5th rule ---
                    if len(self._improvement_magnitudes[i]) > 0:
                        avg_improvement = np.mean(self._improvement_magnitudes[i])
                        # Get current median fitness for this island to calculate relative improvement
                        current_median_fitness = np.median([cache.get(c.id, 0.5) for c in self.islands[i]]) if len(self.islands[i]) > 0 else 0.5
                        relative_improvement = avg_improvement / max(1e-8, abs(current_median_fitness))

                        # NEW LOGIC: Check for high success but tiny improvement
                        if rate > 0.2 and relative_improvement < 0.01:
                            # This is a trap! We are stuck. DECREASE sigma to enable fine-tuning.
                            factor = cfg.conservative_factor_decrease
                            logger.info(f"    1/5TH RULE TRAP DETECTED: High success ({rate:.3f}) but low impact ({relative_improvement:.2%}). Decreasing sigma.")
                        elif rate > self.one_fifth_threshold:
                            factor = cfg.conservative_factor_increase  # Gentle 2% increase
                        else:
                            factor = cfg.conservative_factor_decrease  # Gentle 2% decrease
                    else:
                        # No improvement data available, use original logic
                        if rate > self.one_fifth_threshold:
                            factor = cfg.conservative_factor_increase  # Gentle 2% increase
                        else:
                            factor = cfg.conservative_factor_decrease  # Gentle 2% decrease

                    self._island_sigma[i] *= factor
                    logger.info(f"GEN {generation} | sigma_update island={i} rate={rate:.3f} factor={factor:.3f} new_sigma={self._island_sigma[i]:.6f}")

                    # --- NEW: Add a hard cap to sigma ---
                    # Use scalar cap to avoid ambiguous array comparisons when sigma is scalar
                    sigma_max_scalar = float(0.2 * np.mean(self.bounds[1] - self.bounds[0]))
                    if self._island_sigma[i] > sigma_max_scalar:
                        try:
                            cur_val = float(np.asarray(self._island_sigma[i]).mean())
                        except Exception:
                            cur_val = float(self._island_sigma[i])
                        logger.info(f"      SIGMA CAP: Capping sigma from {cur_val:.4f} to {sigma_max_scalar:.4f}")
                        self._island_sigma[i] = sigma_max_scalar

                    # Apply late-phase sigma cap if in late phase
                    if late_phase_active:
                        lb, ub = self.bounds
                        sigma_max_late = float(cfg.late_phase_sigma_factor * np.mean(ub - lb) / np.sqrt(self.dimension))
                        if self._island_sigma[i] > sigma_max_late:
                            self._island_sigma[i] = sigma_max_late

                    try:
                        old_val = float(np.asarray(old_sigma).mean())
                    except Exception:
                        old_val = float(old_sigma)
                    try:
                        new_val = float(np.asarray(self._island_sigma[i]).mean())
                    except Exception:
                        new_val = float(self._island_sigma[i])
                    logger.info(f"    Island {i}: success_rate={rate:.3f}, sigma {old_val:.6f} -> {new_val:.6f} (factor: {factor})")
                    if getattr(cfg, 'enable_one_fifth_rule', False):
                        self._succ[i] = 0
                        self._trials[i] = 0
                        # --- NEW: Clear improvement magnitudes after 1/5th rule update ---
                        self._improvement_magnitudes[i].clear()

            # --- NEW: Perform Strategic PNN Unlocking ---
            for island_idx, island in enumerate(self.islands):
                self._strategic_pnn_unlocking(island, island_idx, generation)

            # Per-generation best (normalized only)
            gen_best_norm = _normalize(best_true)
            if gen_best_norm is not None and np.isfinite(gen_best_norm):
                logger.info(f"Gen {generation} best: {gen_best_norm:.6g}")
            # --- Early stopping logic ---
            if not np.isfinite(best_so_far):
                best_so_far = best_true
                # Do not emit per-improvement logs; only final end-of-gen line will be printed
                no_improve = 0
            else:
                old_best = best_so_far
                impr_raw = old_best - best_true
                updated = False
                # Always track the strict running minimum for reporting
                if best_true < old_best - 1e-12:
                    best_so_far = best_true
                    updated = True
                    # Suppress mid-generation best-change logs per user preference
                # Apply tolerance gating only for early-stop counter
                threshold = max(self.abs_tol, self.rel_tol * max(1.0, abs(best_so_far)))
                if impr_raw > threshold:
                    no_improve = 0
                    # Reset NFE improvement marker
                    try:
                        self._last_improve_nfe = int(self.function_evaluations)
                    except Exception:
                        self._last_improve_nfe = 0
                else:
                    no_improve += 1
                    if not updated and no_improve % 5 == 0:  # Log every 5 generations without material improvement
                        # Keep this progress hint, but without raw fitness values
                        logger.info(f"NoImprove: {no_improve} gens")
                # --- NEW: Adaptive 1/5th Rule Threshold (optional) ---
                if getattr(cfg, 'enable_one_fifth_rule', False):
                    relative_improvement = (self.best_fitness_history[-5] - best_so_far) / max(1e-8, abs(self.best_fitness_history[-5])) if len(self.best_fitness_history) > 5 else 1.0
                    if relative_improvement < 0.001 and not late_phase_active:
                        self.one_fifth_threshold = 0.1  # Become stricter
                        logger.info(f"    ADAPTIVE 1/5th RULE: Low improvement ({relative_improvement:.2%}). Threshold set to {self.one_fifth_threshold}")
                    else:
                        self.one_fifth_threshold = 0.2  # Default threshold

            # End-of-generation final best (normalized only)
            final_best_norm = _normalize(best_so_far if np.isfinite(best_so_far) else best_true)
            if final_best_norm is not None and np.isfinite(final_best_norm):
                logger.info(f"Gen {generation} final best: {final_best_norm:.6g}")

            # FIX H: Plateau unlock mechanism - force PNN OPEN state when stuck (with cooldown)
            if cfg.enable_plateau_unlock and no_improve >= 3 and generation >= self._plateau_cooldown_until:
                logger.info(f"  PLATEAU UNLOCK: Forcing PNN OPEN state for all cells (no_improve={no_improve})")
                for island in self.islands:
                    for cell in island:
                        if cell.pnn.state == PNN_STATE.LOCKED:
                            cell.pnn.state = PNN_STATE.OPEN
                            cell.pnn.plasticity_multiplier = 1.0
                            cell.pnn.generations_in_phase = 0
                            cell.pnn.stability_history.clear()
                            cell.pnn.locked_generation = None
                            cell.pnn.best_fitness_when_locked = None
                            # FIX C: Set refractory period to prevent instant re-closing
                            cell.pnn.refractory_until = generation + 5
                
                # Set cooldown to prevent repeated unlocks
                self._plateau_cooldown_until = generation + 6

            # Stagnation adaptive restart: refresh a fraction of population per island
            if no_improve >= getattr(cfg, 'stagnation_restart_generations', 10):
                logger.info(f"  STAGNATION RESTART: Re-seeding a fraction of each island")
                for island_idx, island in enumerate(self.islands):
                    if not island:
                        continue
                    k = max(1, int(len(island) * getattr(cfg, 'stagnation_restart_fraction', 0.2)))
                    for _ in range(k):
                        idx = random.randrange(len(island))
                        old = island[idx].id
                        new_cell = Cell(random.randint(90000, 100000), self.dimension, self.bounds)
                        new_cell.to(self.device)
                        island[idx] = new_cell
                        logger.info(f"    Island {island_idx}: Replaced cell {old} with new cell {new_cell.id} due to stagnation")
                no_improve = 0
                self._trials = [0]*len(self.islands)
                self._succ = [0]*len(self.islands)

            # NFE-driven trust-region shrink when stalled under tight budget
            try:
                nfe_since = int(self.function_evaluations) - int(self._last_improve_nfe)
                if nfe_since >= 200:
                    for i in range(len(self._island_sigma)):
                        self._island_sigma[i] *= 0.85
                    self._last_improve_nfe = int(self.function_evaluations)
                    logger.info("  STEP SCHEDULE: No improvement in 200 NFE — shrinking trust region by 15%")
            except Exception:
                pass

            if no_improve >= self.early_stop_patience:
                logger.info(f"  EARLY STOPPING: No improvement for {no_improve} generations")
                print(f"  EARLY STOPPING: No improvement for {no_improve} generations")
                break
            # --- End early stopping logic ---

            # === Telemetry: Single consolidated generation write ===
            try:
                if getattr(cfg, 'enable_telemetry', True) and callable(write_archipelago_visualization_state):
                    # Optionally, attach last-gen KPIs for the telemetry file
                    try:
                        self.generation_kpis = {
                            'features_active': features_active,
                            'pnn_locked_plus': int(locked_this_gen),
                            'pnn_unlocked_plus': int(unlocked_this_gen),
                            'dreams_created': int(getattr(self, '_dreams_created_last_gen', 0)),
                            'exploit_actions': int(getattr(self, '_exploit_actions_last_gen', 0)),
                            'strategy_usage': dict(getattr(self, '_strategy_usage_last_gen', {})),
                        }
                    except Exception:
                        pass
                    write_archipelago_visualization_state(self, generation)
            except Exception:
                pass
            
            # Log global PNN phase statistics for this generation
            all_cells = [c for island in self.islands for c in island]
            global_pnn_states = [cell.pnn.state.value for cell in all_cells]
            global_open_count = global_pnn_states.count('OPEN')
            global_closing_count = global_pnn_states.count('CLOSING')
            global_locked_count = global_pnn_states.count('LOCKED')
            global_total_cells = len(all_cells)
            
            logger.info(f"  GLOBAL PNN PHASE STATS: OPEN={global_open_count}/{global_total_cells} ({global_open_count/global_total_cells*100:.1f}%), "
                       f"CLOSING={global_closing_count}/{global_total_cells} ({global_closing_count/global_total_cells*100:.1f}%), "
                       f"LOCKED={global_locked_count}/{global_total_cells} ({global_locked_count/global_total_cells*100:.1f}%)")
            
            # Log average generations in phase for LOCKED cells globally
            if global_locked_count > 0:
                global_locked_generations = [cell.pnn.generations_in_phase for cell in all_cells if cell.pnn.state == PNN_STATE.LOCKED]
                global_avg_locked_generations = np.mean(global_locked_generations)
                logger.info(f"  GLOBAL LOCKED CELLS: avg_generations_in_phase={global_avg_locked_generations:.1f}")
            
            # === PHASE CRISIS DETECTION AND INTERVENTION ===
            locked_ratio = global_locked_count / global_total_cells
            if locked_ratio > cfg.phase_crisis_locked_threshold:
                # Crisis condition detected
                self.phase_crisis_consecutive_count += 1
                logger.info(f"  PHASE CRISIS DETECTED: {locked_ratio:.1%} cells locked (consecutive: {self.phase_crisis_consecutive_count}/{cfg.phase_crisis_consecutive_generations})")
                
                # Check if we should trigger crisis intervention
                if (self.phase_crisis_consecutive_count >= cfg.phase_crisis_consecutive_generations and 
                    generation >= self.phase_crisis_cooldown_until):
                    
                    logger.info(f"  PHASE CRISIS INTERVENTION: Forcing unlock of {cfg.phase_crisis_unlock_fraction:.1%} of locked cells")
                    
                    # Collect all locked cells
                    locked_cells = [cell for cell in all_cells if cell.pnn.state == PNN_STATE.LOCKED]
                    num_to_unlock = max(1, int(len(locked_cells) * cfg.phase_crisis_unlock_fraction))
                    
                    # Randomly select cells to unlock
                    cells_to_unlock = random.sample(locked_cells, min(num_to_unlock, len(locked_cells)))
                    
                    for cell in cells_to_unlock:
                        cell.pnn.force_unlock(generation, refractory_period=3)
                        logger.info(f"    Crisis unlock: Cell {cell.id} forced to OPEN state")
                        unlocked_this_gen += 1
                    
                    # Set cooldown to prevent repeated interventions
                    self.phase_crisis_cooldown_until = generation + 10
                    self.phase_crisis_consecutive_count = 0  # Reset counter
                    
                    # Log crisis intervention event
                    self._tap_event("PHASE_CRISIS", {
                        "action": "crisis_intervention",
                        "locked_ratio": locked_ratio,
                        "cells_unlocked": len(cells_to_unlock),
                        "generation": generation
                    })
            else:
                # Reset crisis counter if not in crisis
                if self.phase_crisis_consecutive_count > 0:
                    logger.info(f"  PHASE CRISIS RESOLVED: Locked ratio dropped to {locked_ratio:.1%}")
                self.phase_crisis_consecutive_count = 0
            
            if self.detect_phase_transition():
                # --- CALM DOWN: Only inject diversity if we are truly stuck for a long time ---
                if no_improve > 15: 
                    logger.info(f"  PHASE TRANSITION DETECTED & STAGNATED ({no_improve} gens): Injecting diversity")
                    self.diversity_injection()
                else:
                    logger.info(f"  Phase transition detected, but holding off on diversity injection (stagnation: {no_improve} < 15).")

            if all_phenotypes:
                distances = [np.linalg.norm(p1 - p2) for i, p1 in enumerate(all_phenotypes) for p2 in all_phenotypes[i+1:]]
                self.diversity_history.append(np.mean(distances) if distances else 0)

        # Find the best cell across all islands using true base fitness
        for i, island in enumerate(self.islands):
            if not island: continue
            island_obj = self.island_objectives[i]
            cache = fitness_caches[i]
            # best under island objective
            fits = [self._eval_cell_with_obj(c, island_obj, cache) for c in island]
            best_idx = int(np.argmin(fits))
            champ = island[best_idx]
            # evaluate "true" base objective (reduce MO vectors to scalar via sum)
            _f = base_f(champ.get_solution())
            try:
                if isinstance(_f, (list, tuple, np.ndarray)):
                    _f = float(np.sum(np.asarray(_f, dtype=float)))
                else:
                    _f = float(_f)
            except Exception:
                _f = float(np.nan)
            f_true = _f
            if f_true < best_fitness:
                best_fitness = f_true
                best_cell = champ
        if best_cell is None:
        # Prefer any existing cell without new evaluations
            for island in self.islands:
                if island:
                    best_cell = island[0]
                    break
            if best_cell is None:
                # Absolute last resort: materialize a bounded cell (no eval)
                best_cell = Cell(random.randint(90000, 100000), self.dimension, self.bounds)
                best_cell.to(self.device)

        # If best_fitness never improved, try to use the last computed best_true from the loop.
        if (not np.isfinite(best_fitness)) or best_fitness == float("inf"):
            if 'best_true' in locals() and np.isfinite(best_true):
                best_fitness = float(best_true)
            else:
                best_fitness = float("inf")  # keep as-is; logs will show "inf"
        total_time = time.time() - start_time
        logger.info(f"\n=== AEA Optimization Completed ===")
        logger.info(f"Final best fitness: {best_fitness:.6f}")
        logger.info(f"Total generations: {self.generation + 1}")
        logger.info(f"Total function evaluations: {self.function_evaluations}")
        logger.info(f"Wall time: {total_time:.2f}s")
        logger.info(f"Success: {best_fitness < 1e-6}")
        
        # Default: always save tapestry (JSON + vectors) and refresh latest_run link
        if hasattr(self, 'save_tapestry_enabled') and self.save_tapestry_enabled:
            try:
                from pathlib import Path
                outdir = Path("runs") / (self.causal_tapestry.run_id or "latest_run")
                outdir.mkdir(parents=True, exist_ok=True)
                self.causal_tapestry.save_tapestry(str(outdir / "tapestry.graphml"))
                self.causal_tapestry.export_to_json(str(outdir / "tapestry.json"), generation_window=50)
                # Save vectors alongside JSON
                try:
                    self.causal_tapestry.save_vectors(str(outdir / "priors_v1.npz"))
                except Exception as e:
                    logger.warning(f"Vector save failed: {e}")
                # Update latest_run snapshot
                latest_dir = Path("runs") / "latest_run"
                latest_dir.mkdir(parents=True, exist_ok=True)
                # Write small manifest with pointer
                try:
                    (latest_dir / "_pointer.txt").write_text(str(outdir))
                    # Also copy JSON and VEC for convenience
                    import shutil
                    shutil.copy2(str(outdir / "tapestry.json"), str(latest_dir / "tapestry.json"))
                    vec_src = outdir / "priors_v1.npz"
                    if vec_src.exists():
                        shutil.copy2(str(vec_src), str(latest_dir / "priors_v1.npz"))
                except Exception as e:
                    logger.warning(f"Failed to update latest_run: {e}")
                # Optional: visualize tapestry (can be commented out to save time)
                self.causal_tapestry.visualize_tapestry(str(outdir / "tapestry.png"), generation_window=50)
                logger.info(f"Causal tapestry saved to {outdir}")
            except Exception as e:
                logger.warning(f"Tapestry export failed: {e}")
        else:
            logger.info("Causal tapestry saving disabled for benchmark runs")
        
        # Analyze quantum gene performance
        self._analyze_quantum_gene_performance()
        
        # FIX: Properly disable profiler to prevent RuntimeError on subsequent calls
        # profiler.disable()
        
        benchmark_result = BenchmarkResult(
            algorithm_name="AEA_Full_With_ODE_Quantum_SelfMod",
            problem_name=getattr(self.current_fitness_func, '__name__', ''),
            run_id=0,
            final_fitness=best_fitness,
            convergence_generation=self.generation,
            function_evaluations=self.function_evaluations,
            wall_time=time.time() - start_time,
            success=best_fitness < 1e-6,
            fitness_history=self.best_fitness_history.copy(),
            diversity_history=self.diversity_history.copy(),
            extra_metrics={
                "total_exploit_actions": float(getattr(self, '_total_exploit_actions', 0)),
                "total_unlocks": float(getattr(self, '_total_unlocks', 0)),
                "total_migration_rounds": float(getattr(self, '_total_migration_rounds', 0)),
                "total_migrants": float(getattr(self, '_total_migrants', 0)),
                "total_dream_rounds": float(getattr(self, '_total_dream_rounds', 0)),
                "total_linesearch_rounds": float(getattr(self, '_total_linesearch_rounds', 0)),
                "heavy_ops_count": float(getattr(self, '_heavy_ops_count', 0)),
                "vetoed_breeds": float(getattr(self, '_vetoed_breeds', 0)),
                "accepted_breeds": float(getattr(self, '_accepted_breeds', 0)),
                "lock_fraction_p95": float(np.nanpercentile(self._lock_fraction_history, 95)) if self._lock_fraction_history else 0.0,
            }
        )
        logger.info(f"\n=== AEA Optimization Completed ===")
        logger.info(f"Final best fitness: {best_fitness}")   # prints 'inf' safely if needed
        
        # --- START OF FIX ---
        # Return both the results object AND the best cell object
        return benchmark_result, best_cell
        # --- END OF FIX ---
        
        
        
        

        
    
    def _analyze_quantum_gene_performance(self):
        """Analyze the performance impact of quantum gene transpositions with proper statistical analysis"""
        try:
            # Get current problem context for scoping
            current_problem = getattr(self, 'current_problem', 'unknown')
            current_dim = getattr(self, 'dimension', 'unknown')
            generation_window = getattr(self, 'generation', 0)
            
            logger.info(f"QUANTUM GENE ANALYSIS (scope: {current_problem} d={current_dim}, gen_window={generation_window}):")
            
            # === 1. STRATIFIED ANALYSIS BY STRESS/ISLAND/PNN STATE ===
            # Query for recombination events with quantum vs non-quantum children
            # Use stratified approach to avoid regime bias
            quantum_recombinations_stats = self.causal_tapestry.query_action_effect_with_stats(
                action='recombine',
                context_filters={'child_has_quantum': True},
                generation_window=generation_window,
                decay_rate=0.0
            )
            
            non_quantum_recombinations_stats = self.causal_tapestry.query_action_effect_with_stats(
                action='recombine',
                context_filters={'child_has_quantum': False},
                generation_window=generation_window,
                decay_rate=0.0
            )
            
            quantum_recombinations = quantum_recombinations_stats['effect']
            non_quantum_recombinations = non_quantum_recombinations_stats['effect']
            
            # === 2. FIXED SIGN LOGIC: Convert effects to improvements ===
            # Effects are (child_fitness - avg_parent_fitness), so negative = improvement
            # Convert to improvements where positive = better
            q_imp = -quantum_recombinations  # Convert effect to improvement
            r_imp = -non_quantum_recombinations  # Convert effect to improvement
            
            delta_imp = q_imp - r_imp  # >0 means quantum improves MORE than regular
            
            # === 3. ENHANCED FITNESS ANALYSIS WITH N AND DISPERSION ===
            quantum_count = 0
            total_genes = 0
            quantum_cell_fitness = []
            regular_cell_fitness = []
            
            # Collect fitness data with proper stratification
            for island_idx, island in enumerate(self.islands):
                for cell in island:
                    if cell.genome:
                        cell_has_quantum = False
                        for gene in cell.genome:
                            total_genes += 1
                            if isinstance(gene, QuantumGeneModule):
                                quantum_count += 1
                                cell_has_quantum = True
                        
                        # Get cell fitness (approximate)
                        if hasattr(cell, 'fitness_history') and cell.fitness_history:
                            fitness = cell.fitness_history[-1]
                            if cell_has_quantum:
                                quantum_cell_fitness.append(fitness)
                            else:
                                regular_cell_fitness.append(fitness)
            
            # Calculate statistics with N and dispersion
            quantum_percentage = (quantum_count / max(1, total_genes)) * 100
            nq = len(quantum_cell_fitness)
            nr = len(regular_cell_fitness)
            
            logger.info(f"  Quantum genes: {quantum_count}/{total_genes} ({quantum_percentage:.1f}%)")
            
            if nq > 0 and nr > 0:
                avg_q = float(np.mean(quantum_cell_fitness))
                avg_r = float(np.mean(regular_cell_fitness))
                sd_q = float(np.std(quantum_cell_fitness))
                sd_r = float(np.std(regular_cell_fitness))
                median_q = float(np.median(quantum_cell_fitness))
                median_r = float(np.median(regular_cell_fitness))
                advantage = avg_r - avg_q  # Positive if quantum is better
                
                logger.info(f"  [CHART] Avg regular cell fitness: {avg_r:.6f} ± {sd_r:.6f} (n={nr}, median={median_r:.6f})")
                logger.info(f"  [CHART] Avg quantum  cell fitness: {avg_q:.6f} ± {sd_q:.6f} (n={nq}, median={median_q:.6f})")
                logger.info(f"  [LIGHTNING] Quantum advantage: {advantage:.6f} ({'BETTER' if advantage > 0 else 'WORSE'})")
            else:
                logger.info(f"  [WARNING] Insufficient data: quantum_cells={nq}, regular_cells={nr}")
            
            # === 4. FIXED RECOMBINATION COMPARISON WITH ENHANCED STATS ===
            if quantum_recombinations != 0.0 and non_quantum_recombinations != 0.0:
                logger.info(f"  Quantum vs Regular recombination (improvement): {delta_imp:.6f} ({'QUANTUM BETTER' if delta_imp > 0 else 'REGULAR BETTER'})")
                logger.info(f"    - Quantum improvement: {q_imp:.6f} (n={quantum_recombinations_stats['count']}, std={quantum_recombinations_stats['std']:.6f})")
                logger.info(f"    - Regular improvement: {r_imp:.6f} (n={non_quantum_recombinations_stats['count']}, std={non_quantum_recombinations_stats['std']:.6f})")
                logger.info(f"    - Raw effects (child - avg_parent): quantum={quantum_recombinations:.6f}, regular={non_quantum_recombinations:.6f}")
            elif quantum_recombinations_stats['count'] > 0 or non_quantum_recombinations_stats['count'] > 0:
                logger.info(f"  [WARNING] Insufficient recombination data: quantum_n={quantum_recombinations_stats['count']}, regular_n={non_quantum_recombinations_stats['count']}")
            
            # === 5. TRANSPOSITION ANALYSIS WITH ENHANCED STATS ===
            transposition_stats = self.causal_tapestry.query_action_effect_with_stats(
                action='transpose',
                context_filters={},
                generation_window=generation_window,
                decay_rate=0.0
            )
            
            if transposition_stats['effect'] != 0.0:
                logger.info(f"  Quantum vs Regular transposition (improvement): {transposition_stats['effect']:.6f} ({'QUANTUM BETTER' if transposition_stats['effect'] > 0 else 'REGULAR BETTER'})")
            
            # === 6. STRATIFIED ANALYSIS BY STRESS BINS ===
            # Perform stratified analysis to avoid regime bias
            if nq > 0 and nr > 0:
                logger.info(f"  [STRATIFIED] Stress-bin analysis:")
                
                # Analyze by stress bins (0-4, where 0=low stress, 4=high stress)
                for stress_bin in range(5):
                    q_stats = self.causal_tapestry.query_action_effect_with_stats(
                        action='recombine',
                        context_filters={'child_has_quantum': True, 'stress_bin': stress_bin},
                        generation_window=generation_window,
                        decay_rate=0.0
                    )
                    
                    r_stats = self.causal_tapestry.query_action_effect_with_stats(
                        action='recombine',
                        context_filters={'child_has_quantum': False, 'stress_bin': stress_bin},
                        generation_window=generation_window,
                        decay_rate=0.0
                    )
                    
                    if q_stats['count'] > 0 or r_stats['count'] > 0:
                        q_imp_strat = -q_stats['effect']
                        r_imp_strat = -r_stats['effect']
                        delta_strat = q_imp_strat - r_imp_strat
                        
                        logger.info(f"    Stress bin {stress_bin}: quantum_imp={q_imp_strat:.6f}(n={q_stats['count']}), regular_imp={r_imp_strat:.6f}(n={r_stats['count']}), delta={delta_strat:.6f}")
                
                # Analyze by PNN states if available
                pnn_states = ['OPEN', 'CLOSING', 'LOCKED']
                logger.info(f"  [STRATIFIED] PNN state analysis:")
                
                for pnn_state in pnn_states:
                    q_stats = self.causal_tapestry.query_action_effect_with_stats(
                        action='recombine',
                        context_filters={'child_has_quantum': True, 'pnn_state': pnn_state},
                        generation_window=generation_window,
                        decay_rate=0.0
                    )
                    
                    r_stats = self.causal_tapestry.query_action_effect_with_stats(
                        action='recombine',
                        context_filters={'child_has_quantum': False, 'pnn_state': pnn_state},
                        generation_window=generation_window,
                        decay_rate=0.0
                    )
                    
                    if q_stats['count'] > 0 or r_stats['count'] > 0:
                        q_imp_strat = -q_stats['effect']
                        r_imp_strat = -r_stats['effect']
                        delta_strat = q_imp_strat - r_imp_strat
                        
                        logger.info(f"    PNN {pnn_state}: quantum_imp={q_imp_strat:.6f}(n={q_stats['count']}), regular_imp={r_imp_strat:.6f}(n={r_stats['count']}), delta={delta_strat:.6f}")
                
        except Exception as e:
            logger.warning(f"Quantum gene analysis failed: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            
            
            
    def run_breeding_summit(self, champions):
        if len(champions) < 2: return
        
        logger.info(f"  BREEDING SUMMIT: Cross-island champion breeding")
        
        for i in range(self.n_islands):
            parent1 = champions[i]
            parent2 = champions[(i + 1) % self.n_islands]
            
            # Query Tapestry for historical breeding effect prediction
            # Use gene types from the champions' genomes for context
            parent1_genes = [g.gene_type for g in parent1.genome] if parent1.genome else ['NONE']
            parent2_genes = [g.gene_type for g in parent2.genome] if parent2.genome else ['NONE']
            
            # Query using the actual query_action_effect method with proper context
            context_filters = {
                'parent_types': tuple(sorted([parent1_genes[0] if parent1_genes else 'NONE', 
                                             parent2_genes[0] if parent2_genes else 'NONE']))
            }
            predicted_effect = self.causal_tapestry.query_action_effect(
                action='recombine',
                context_filters=context_filters,
                generation_window=20,
                decay_rate=0.1
            )
            
            # Skip breeding if historical data suggests negative effects
            if predicted_effect < -0.1:
                logger.info(f"    Island {i}: Skipping breeding due to negative history ({predicted_effect:.3f})")
                self._vetoed_breeds += 1
                continue

            # Simple genome crossover
            child = Cell(random.randint(70000, 80000), self.dimension, self.bounds)
            if parent1.genome and parent2.genome:
                # Mix genomes from both parents
                p1_size = len(parent1.genome)
                p2_size = len(parent2.genome)
                split1 = p1_size // 2
                split2 = p2_size // 2
                child.genome = nn.ModuleList(list(parent1.genome[:split1]) + list(parent2.genome[split2:]))
            
            # Solution crossover
            alpha = random.random()
            child.solution = alpha * parent1.solution + (1 - alpha) * parent2.solution
            
            # Evaluate child fitness
            fitness_before = (self._evaluate_cell(parent1) + self._evaluate_cell(parent2)) / 2
            fitness_after = self._evaluate_cell(child)
            effect = fitness_after - fitness_before
            
            # Log breeding outcome to tapestry
            eid = self._tap_event('BREEDING', {
                'parent_types': tuple(sorted([parent1_genes[0] if parent1_genes else 'NONE',
                                             parent2_genes[0] if parent2_genes else 'NONE'])),
                'effect': effect,
                'parent1_id': str(parent1.id),
                'parent2_id': str(parent2.id),
                'child_fitness': fitness_after,
                'predicted_effect': predicted_effect
            })
            self.causal_tapestry.log_event_participation(str(parent1.id), eid, "parent")
            self.causal_tapestry.log_event_participation(str(parent2.id), eid, "parent")
            self.causal_tapestry.log_event_output(eid, str(child.id), "offspring")
            
            logger.info(f"    Island {i}: Bred child {child.id} (effect: {effect:.3f}, predicted: {predicted_effect:.3f})")

            # Inject into next island
            target_island = self.islands[(i + 2) % self.n_islands]
            worst_idx = np.argmax([self._evaluate_cell(c) for c in target_island])
            target_island[worst_idx] = child
            self._accepted_breeds += 1

    def run_microchimerism(self, champions, generation):
        # Seeding high-performing cells into dormant bay
        for champ in champions:
            if random.random() < 0.3:  # 30% chance to seed
                self.dormant_bay[champ.id] = champ.copy(device=self.device)
                logger.info(f"    Seeded champion {champ.id} to dormant bay")
                
                # Log seeding event
                eid = self._tap_event('MICROCHIMERISM', {
                    'action': 'seed',
                    'cell_id': str(champ.id),
                    'generation': generation
                })
                self.causal_tapestry.log_event_participation(str(champ.id), eid, "seeded_cell")

        # Revival when diversity is low
        for island_idx, island in enumerate(self.islands):
            # Calculate diversity using fitness variance
            fitness_values = [self._evaluate_cell(c) for c in island]
            diversity = np.std(fitness_values)
            
            # Revive dormant cells if diversity is too low
            if diversity < 0.05 and self.dormant_bay:
                # Query tapestry for successful revival history
                revival_success = self.causal_tapestry.query_action_effect(
                    action='revive',
                    context_filters={'island': f'island_{island_idx}'},
                    generation_window=30,
                    decay_rate=0.05
                )
                
                # Only revive if history suggests it's beneficial or we have no history
                if revival_success >= 0 or revival_success == 0.0:
                    revival_id = random.choice(list(self.dormant_bay.keys()))
                    revived_cell = self.dormant_bay.pop(revival_id)
                    
                    # Find worst cell to replace
                    worst_idx = np.argmax(fitness_values)
                    old_fitness = fitness_values[worst_idx]
                    island[worst_idx] = revived_cell
                    
                    # Evaluate revived cell
                    new_fitness = self._evaluate_cell(revived_cell)
                    effect = old_fitness - new_fitness  # Positive if improvement
                    
                    logger.info(f"    Island {island_idx}: Revived cell {revival_id} (diversity: {diversity:.3f}, effect: {effect:.3f})")
                    
                    # Log revival event
                    eid = self._tap_event('MICROCHIMERISM', {
                        'action': 'revive',
                        'cell_id': str(revival_id),
                        'island': f'island_{island_idx}',
                        'generation': generation,
                        'diversity': diversity,
                        'effect': effect
                    })
                    self.causal_tapestry.log_event_output(eid, str(revival_id), "revived_cell")


    def _build_island_objectives(self, base_f):
        """Create per-island objective wrappers.
        - SO: scalar objective per island with role-specific shaping
        - Note: Do NOT probe the provided objective, because in the benchmark harness the
          provided callable is instrumented (counts NFE and logs). Probing would consume
          budget and contaminate anytime logs. We assume a scalar objective here; multi-
          objective cases should be scalarized by the caller (benchmark harness already does this).
        """
        # Assume scalar objective; caller must scalarize MO before passing in
        is_mo = False
        M = 1
        # Ensure we have enough roles for all islands
        if not hasattr(self, 'island_roles') or len(self.island_roles) < self.n_islands:
            # Use config roles and extend if needed
            base_roles = cfg.island_roles
            if len(base_roles) < self.n_islands:
                # Extend with 'raw' roles if we don't have enough
                self.island_roles = base_roles + ['raw'] * (self.n_islands - len(base_roles))
            else:
                self.island_roles = base_roles[:self.n_islands]
        
        # Initialize variables outside the if block
        try:
            if self.seed is not None:
                ss = np.random.SeedSequence(self.seed)
                rngs = [np.random.default_rng(s) for s in ss.spawn(self.n_islands)]
            else:
                rngs = [np.random.default_rng(1000 + i) for i in range(self.n_islands)]
        except Exception:
            rngs = [np.random.default_rng(1000 + i) for i in range(self.n_islands)]
        lb, ub = self.bounds
        dim = self.dimension
        objs = []

        for i, role in enumerate(self.island_roles):
            if role == 'raw':
                def f_i(x, base_f=base_f):
                    bx = base_f(x)
                    if isinstance(bx, (list, tuple, np.ndarray)):
                        return float(np.sum(np.asarray(bx, dtype=float)))
                    return float(bx)
                # Attach batch if available
                if hasattr(base_f, 'batch') and callable(getattr(base_f, 'batch')):
                    def _batch_raw(X, base_f=base_f):
                        V = base_f.batch(np.asarray(X, dtype=float))
                        V = np.asarray(V)
                        if V.ndim == 2:
                            return np.sum(V.astype(float), axis=1)
                        return V.astype(float)
                    setattr(f_i, 'batch', _batch_raw)

            elif role == 'gaussian_smooth':
                # NOTE: increases NFE by k per eval. Keep k small.
                k = 2  # was 3; reduce to limit NFE
                def f_i(x, base_f=base_f, rng=rngs[i]):
                    r = max(1e-12, np.linalg.norm(x)) / np.sqrt(dim)
                    sigma = 0.10 * r  # a bit smaller
                    vals = []
                    for _ in range(k):
                        z = np.clip(x + rng.normal(0, sigma, size=dim), lb, ub)
                        bz = base_f(z)
                        if isinstance(bz, (list, tuple, np.ndarray)):
                            bz = float(np.sum(np.asarray(bz, dtype=float)))
                        else:
                            bz = float(bz)
                        vals.append(bz)
                    return float(np.mean(vals))

            elif role == 'trust_region':
                m_i = rngs[i].uniform(lb, ub, size=dim)
                lam = 0.02  # milder pull
                def f_i(x, base_f=base_f, m=m_i, lam=lam):
                    bx = base_f(x)
                    if isinstance(bx, (list, tuple, np.ndarray)):
                        bx = float(np.sum(np.asarray(bx, dtype=float)))
                    else:
                        bx = float(bx)
                    return float(bx + lam * np.sum((x - m)**2))
                if hasattr(base_f, 'batch') and callable(getattr(base_f, 'batch')):
                    def batch_TR(X, base_f=base_f, m=m_i, lam=lam):
                        X = np.asarray(X, dtype=float)
                        V = np.asarray(base_f.batch(X))
                        if V.ndim == 2:
                            V = np.sum(V.astype(float), axis=1)
                        else:
                            V = V.astype(float)
                        pen = lam * np.sum((X - m)**2, axis=1)
                        return V + pen
                    setattr(f_i, 'batch', batch_TR)

            elif role == 'explore':
                # Heavy-tailed explorer with mild rescale
                scale = 0.05
                def f_i(x, base_f=base_f, rng=rngs[i], scale=scale):
                    # proposal only; evaluate base_f on clipped x (selection uses base_f)
                    bx = base_f(np.clip(x, lb, ub))
                    if isinstance(bx, (list, tuple, np.ndarray)):
                        return float(np.sum(np.asarray(bx, dtype=float)))
                    return float(bx)

            elif role == 'annealed':
                def f_i(x, base_f=base_f, self_ref=self):
                    t = max(1e-6, (self_ref.max_generations - self_ref.generation) / max(1, self_ref.max_generations))
                    bx = base_f(x)
                    if isinstance(bx, (list, tuple, np.ndarray)):
                        bx = float(np.sum(np.asarray(bx, dtype=float)))
                    else:
                        bx = float(bx)
                    return float(bx + 0.01 * t * np.linalg.norm(x))
                if hasattr(base_f, 'batch') and callable(getattr(base_f, 'batch')):
                    def batch_ann(X, base_f=base_f, self_ref=self):
                        X = np.asarray(X, dtype=float)
                        t = max(1e-6, (self_ref.max_generations - self_ref.generation) / max(1, self_ref.max_generations))
                        V = np.asarray(base_f.batch(X))
                        if V.ndim == 2:
                            V = np.sum(V.astype(float), axis=1)
                        else:
                            V = V.astype(float)
                        pen = 0.01 * t * np.linalg.norm(X, axis=1)
                        return V + pen
                    setattr(f_i, 'batch', batch_ann)

            elif role == 'two_point':
                # NOTE: +2x NFE per eval. Keep small or restrict to champions only.
                def f_i(x, base_f=base_f, rng=rngs[i]):
                    r = max(1e-12, np.linalg.norm(x)) / np.sqrt(dim)
                    delta = 0.08 * r
                    d = rng.normal(0, 1, size=dim); d /= (np.linalg.norm(d) + 1e-12)
                    x1 = np.clip(x + delta * d, lb, ub)
                    x2 = np.clip(x - delta * d, lb, ub)
                    b1 = base_f(x1); b2 = base_f(x2)
                    if isinstance(b1, (list, tuple, np.ndarray)):
                        b1 = float(np.sum(np.asarray(b1, dtype=float)))
                    else:
                        b1 = float(b1)
                    if isinstance(b2, (list, tuple, np.ndarray)):
                        b2 = float(np.sum(np.asarray(b2, dtype=float)))
                    else:
                        b2 = float(b2)
                    return 0.5 * (b1 + b2)
                if hasattr(base_f, 'batch') and callable(getattr(base_f, 'batch')):
                    def batch_two(X, base_f=base_f, rng=rngs[i]):
                        X = np.asarray(X, dtype=float)
                        n = X.shape[0]
                        # Use a fixed random direction per vector for reproducibility
                        D = rng.normal(0, 1, size=X.shape); 
                        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
                        r = np.maximum(1e-12, np.linalg.norm(X, axis=1)) / np.sqrt(dim)
                        delta = (0.08 * r)[:, None]
                        X1 = np.clip(X + delta * D, lb, ub)
                        X2 = np.clip(X - delta * D, lb, ub)
                        V1 = np.asarray(base_f.batch(X1))
                        V2 = np.asarray(base_f.batch(X2))
                        V = 0.5 * (V1 + V2)
                        if V.ndim == 2:
                            return np.sum(V.astype(float), axis=1)
                        return V.astype(float)
                    setattr(f_i, 'batch', batch_two)

            elif role == 'rescale':
                A = np.diag(rngs[i].uniform(0.7, 1.3, size=dim))
                def f_i(x, base_f=base_f, A=A):
                    bx = base_f(np.clip(A @ x, lb, ub))
                    if isinstance(bx, (list, tuple, np.ndarray)):
                        return float(np.sum(np.asarray(bx, dtype=float)))
                    return float(bx)
                if hasattr(base_f, 'batch') and callable(getattr(base_f, 'batch')):
                    def _batch_rescale(X, base_f=base_f, A=A):
                        V = base_f.batch(np.clip(np.asarray(X, dtype=float) @ A.T, lb, ub))
                        V = np.asarray(V)
                        if V.ndim == 2:
                            return np.sum(V.astype(float), axis=1)
                        return V.astype(float)
                    setattr(f_i, 'batch', _batch_rescale)

            elif role == 'decomp_or_indicator' and is_mo:
                # Build MO scalarizer for this island
                # weight set per island
                def weights_2d(K=9):
                    return [np.array([i/(K-1), 1.0 - i/(K-1)], dtype=float) for i in range(K)]
                def weights_simplex(Mm=3, H=6):
                    base = []
                    for bars in combinations_with_replacement(range(H+Mm-1), Mm-1):
                        prev = -1; parts = []
                        for b in bars + (H+Mm-1-1,):
                            parts.append(b - prev - 1); prev = b
                        w = np.array(parts, dtype=float) / H
                        base.append(w)
                    return base
                def tchebycheff(f, w, z):
                    return float(np.max(w * np.abs(f - z)))
                def pbi(f, w, z, theta=5.0):
                    wv = w / (np.linalg.norm(w) + 1e-12)
                    d1 = np.dot(f - z, wv)
                    proj = z + d1 * wv
                    d2 = np.linalg.norm(f - proj)
                    return float(d1 + theta * d2)

                # init shared MO state
                if not hasattr(self, '_pareto_archive'):
                    self._pareto_archive = []  # list of (x, f)
                if not hasattr(self, '_ideal') or self._ideal is None:
                    self._ideal = np.full(M, np.inf, dtype=float)

                # choose weight for island i
                # Prefer problem-specific island weights if provided
                prob_name = getattr(self, 'problem_name', getattr(self, 'current_problem', None))
                prob_name = (prob_name or '').lower()
                W = MO_ISLAND_WEIGHTS.get(prob_name)
                if W is None:
                    if M == 2:
                        W = weights_2d(K=max(5, self.n_islands))
                    else:
                        W = weights_simplex(Mm=M, H=3)
                # ensure length
                W = (W + W)[:self.n_islands]
                w = np.asarray(W[i % len(W)], dtype=float)
                use_pbi = (i % 5 == 0)

                def f_i(x, base_f=base_f, w=w, use_pbi=use_pbi, self_ref=self):
                    f = np.asarray(base_f(x), dtype=float)
                    self_ref._ideal = np.minimum(self_ref._ideal, f)
                    # maintain simple non-dominated set (cheap, approximate)
                    try:
                        dominated = []
                        for idx,(xx,ff) in enumerate(getattr(self_ref, '_pareto_archive', [])):
                            if np.all(ff <= f) and np.any(ff < f):
                                return pbi(f, w, self_ref._ideal) if use_pbi else tchebycheff(f, w, self_ref._ideal)
                            if np.all(f <= ff) and np.any(f < ff):
                                dominated.append(idx)
                        for idx in reversed(dominated):
                            self_ref._pareto_archive.pop(idx)
                        self_ref._pareto_archive.append((np.asarray(x, dtype=float), f))
                    except Exception:
                        pass
                    return pbi(f, w, self_ref._ideal) if use_pbi else tchebycheff(f, w, self_ref._ideal)

            else:
                def f_i(x, base_f=base_f):
                    return base_f(x)

            objs.append(f_i)

        self.island_objectives = objs

    def _evaluate_island_once_with_obj(self, island, obj_fn, cache):
        vals = []
        lb, ub = self.bounds
        # Fast path: batch evaluate all uncached if obj_fn exposes a batch method
        uncached_idx = []
        X = []
        for idx, c in enumerate(island):
            v = cache.get(c.id)
            if v is not None:
                vals.append(float(v))
            else:
                vals.append(None)  # placeholder
                uncached_idx.append(idx)
                X.append(c.get_solution())
        if uncached_idx and hasattr(obj_fn, 'batch') and callable(getattr(obj_fn, 'batch')):
            X = np.asarray(X, dtype=float)
            try:
                V = obj_fn.batch(X)
            except Exception:
                V = None
            if isinstance(V, np.ndarray) and V.size == len(uncached_idx):
                self.function_evaluations += int(V.size)
                for k, idx in enumerate(uncached_idx):
                    v = float(V[k])
                    cache[island[idx].id] = v
                    vals[idx] = v
        # Fallback for any remaining None entries
        for i, v in enumerate(vals):
            if v is None:
                vv = float(obj_fn(island[i].get_solution()))
                cache[island[i].id] = vv
                vals[i] = vv
                self.function_evaluations += 1
        return [float(v) for v in vals]

    def _evaluate_islands_parallel(self, fitness_caches: List[Dict[int, float]]) -> List[List[float]]:
        """Evaluate all islands concurrently using threads (Windows-safe).

        Returns a list of per-island fitness arrays aligned by island index.
        """
        try:
            num_islands = len(self.islands)
            if num_islands == 0:
                return []
            # Determine worker count
            try:
                workers_hint = int(getattr(cfg, 'island_workers', 0) or 0)
            except Exception:
                workers_hint = 0
            max_workers = workers_hint if workers_hint > 0 else max(1, min((os.cpu_count() or 1), num_islands))

            results: List[Optional[List[float]]] = [None] * num_islands

            def eval_one(island_idx: int) -> tuple:
                # Robustly pick objective function
                if island_idx < len(self.island_objectives):
                    obj_fn = self.island_objectives[island_idx]
                else:
                    obj_fn = self.island_objectives[0] if self.island_objectives else self.current_fitness_func
                cache = fitness_caches[island_idx]
                vals = self._evaluate_island_once_with_obj(self.islands[island_idx], obj_fn, cache)
                return island_idx, vals

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(eval_one, i) for i in range(num_islands)]
                for fut in as_completed(futures):
                    idx, vals = fut.result()
                    results[idx] = vals

            # Fill any None (shouldn't happen) with empty lists
            return [r if r is not None else [] for r in results]
        except Exception:
            # Fallback to sequential evaluation if anything goes wrong
            out = []
            for i in range(len(self.islands)):
                obj_fn = self.island_objectives[i] if i < len(self.island_objectives) else (self.island_objectives[0] if self.island_objectives else self.current_fitness_func)
                out.append(self._evaluate_island_once_with_obj(self.islands[i], obj_fn, fitness_caches[i]))
            return out

    def _eval_cell_with_obj(self, cell, obj_fn, cache):
        fid = cache.get(cell.id)
        if fid is None:
            fid = float(obj_fn(cell.get_solution()))
            cache[cell.id] = fid
            self.function_evaluations += 1
            
            # FIX: Track fitness history for stress calculation
            cell.fitness_history.append(fid)
            # Keep only recent history to avoid memory bloat
            if len(cell.fitness_history) > 20:
                cell.fitness_history = cell.fitness_history[-20:]
        return fid

    def enable_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """Enable automatic mixed precision training"""
        if torch.cuda.is_available():
            scaler = amp.GradScaler('cuda') if torch.cuda.is_available() else None
            logger.info("Enabled automatic mixed precision (AMP)")
            return scaler
        return None

    def _reproduce_island_vectorized(
        self,
        island: List[Cell],
        island_fitness: List[float],
        island_idx: int,
        generation: int,
        island_obj: Callable[[np.ndarray], float],
        cache: Dict[int, float]
    ) -> None:
        """
        Vectorized exploration-phase reproduction for a single island.
        - Builds SoA arrays for solutions/fitness
        - Vectorized parent selection, crossover, mutation
        - Batched evaluation (if available)
        - In-place writeback to existing Cell objects (no object-churn)
        """
        N = len(island)
        if N == 0:
            return
        D = int(self.dimension)
        lb, ub = self.bounds
        lbv = float(lb); ubv = float(ub)

        # SoA snapshots (arrays for heavy math)
        S = np.empty((N, D), dtype=np.float64)
        F = np.empty((N,), dtype=np.float64)
        plast = np.empty((N,), dtype=np.float64)
        stress = np.empty((N,), dtype=np.float64)
        pnn_state_codes = np.empty((N,), dtype=np.int8)  # 0=OPEN,1=CLOSING,2=LOCKED

        for i, c in enumerate(island):
            S[i, :] = c.solution
            F[i] = float(island_fitness[i])
            plast[i] = float(getattr(c.pnn, 'plasticity_multiplier', 1.0))
            stress[i] = float(getattr(c, 'local_stress', 0.0))
            st = c.pnn.state
            pnn_state_codes[i] = 0 if st == PNN_STATE.OPEN else (1 if st == PNN_STATE.CLOSING else 2)

        # Champion retention
        best_idx = int(np.argmin(F))
        best_sol = S[best_idx].copy()
        best_fit = float(F[best_idx])

        # Random source
        rng = np.random.default_rng(self.seed if hasattr(self, 'seed') and self.seed is not None else None)

        # Masks for PNN states
        locked_mask = (pnn_state_codes == 2)
        explore_mask = ~locked_mask  # OPEN or CLOSING

        # Initialize children as copy
        children = S.copy()

        # Retain existing exploitation for LOCKED individuals (previous crossover toward best, no mutation)
        if np.any(locked_mask):
            # Draw parents and perform existing crossover only for locked subset
            idx_a = rng.integers(0, N, size=N)
            idx_b = rng.integers(0, N, size=N)
            collide = (idx_a == idx_b)
            if collide.any():
                max_tries = 4
                tries = 0
                while collide.any() and tries < max_tries:
                    idx_b[collide] = rng.integers(0, N, size=collide.sum())
                    collide = (idx_a == idx_b)
                    tries += 1
                if collide.any():
                    idx_b[collide] = (idx_b[collide] + 1) % N

            fA = F[idx_a]; fB = F[idx_b]
            p1_idx = np.where(fA <= fB, idx_a, idx_b)
            p2_idx = np.where(p1_idx == idx_a, idx_b, idx_a)

            alpha = rng.random((N, 1))
            anneal_beta = max(0.01, 0.05 * (1.0 - generation / max(1, self.max_generations)))

            locked_idx = np.where(locked_mask)[0]
            if locked_idx.size > 0:
                children_locked = (
                    alpha[locked_idx] * S[p1_idx[locked_idx]]
                    + (1.0 - alpha[locked_idx]) * S[p2_idx[locked_idx]]
                    + anneal_beta * (best_sol - S[p1_idx[locked_idx]])
                )
                children[locked_idx] = np.clip(children_locked, lbv, ubv)

        # Adaptive Differential Evolution (DE/current-to-best/1) for exploration (OPEN or CLOSING)
        e_idx = np.where(explore_mask)[0]
        if e_idx.size > 0:
            # Adaptive parameters
            base_F = float(getattr(cfg, 'de_F', 0.6))
            base_CR = float(getattr(cfg, 'de_CR', 0.9))

            stress_thr = float(getattr(cfg, 'stress_threshold', 0.0))
            anneal = max(0.1, 1.0 - generation / max(1, self.max_generations))

            plast_e = plast[e_idx]
            stress_e = stress[e_idx]
            state_e = pnn_state_codes[e_idx]

            # Scale F by plasticity and stress; ensure reasonable bounds
            F_i = base_F * plast_e * (1.0 + 0.25 * (stress_e > stress_thr)) * (0.5 + 0.5 * anneal)
            F_i = np.clip(F_i, 0.3, 1.0)

            # CR slightly lower when CLOSING to bias exploitation
            CR_i = np.where(state_e == 1, base_CR * 0.75, base_CR)
            CR_i = np.clip(CR_i, 0.1, 1.0)

            # Random distinct r1, r2 for each target in e_idx
            M = e_idx.size
            r1 = rng.integers(0, N, size=M)
            r2 = rng.integers(0, N, size=M)
            collide = (r1 == r2) | (r1 == e_idx) | (r2 == e_idx)
            if collide.any():
                max_tries = 6
                tries = 0
                while collide.any() and tries < max_tries:
                    r1[collide] = rng.integers(0, N, size=collide.sum())
                    r2[collide] = rng.integers(0, N, size=collide.sum())
                    collide = (r1 == r2) | (r1 == e_idx) | (r2 == e_idx)
                    tries += 1
                if collide.any():
                    # deterministic fallback
                    r1[collide] = (e_idx[collide] + 1) % N
                    r2[collide] = (e_idx[collide] + 2) % N

            # Mutation: v = x + F*(x_best - x) + F*(x_r1 - x_r2)
            x = S[e_idx]
            v = x + F_i[:, None] * (best_sol - x) + F_i[:, None] * (S[r1] - S[r2])

            # Binomial crossover to form trial u
            cross_mask = rng.random((M, D)) < CR_i[:, None]
            jrand = rng.integers(0, D, size=M)
            cross_mask[np.arange(M), jrand] = True
            u = np.where(cross_mask, v, x)

            # Bounds
            u = np.clip(u, lbv, ubv)
            children[e_idx] = u

        # Champion retention (don't overwrite)
        children[best_idx, :] = best_sol

        # Batched evaluation with strict-budget safety: allow partial completion
        child_fit = np.full(N, np.inf, dtype=np.float64)
        did_eval_mask = np.zeros(N, dtype=bool)
        try:
            if hasattr(island_obj, 'batch') and callable(getattr(island_obj, 'batch')):
                V = island_obj.batch(children.astype(np.float64))
                V = np.asarray(V, dtype=np.float64)
                take = min(N, int(V.size))
                if take > 0:
                    child_fit[:take] = V[:take]
                    did_eval_mask[:take] = True
            else:
                # Scalar fallback; stop on budget exhaustion
                for i in range(N):
                    try:
                        child_fit[i] = float(island_obj(children[i]))
                        did_eval_mask[i] = True
                    except StopIteration:
                        break
        except Exception:
            # Last-resort scalar loop with StopIteration guard
            child_fit.fill(np.inf)
            did_eval_mask[:] = False
            for i in range(N):
                try:
                    child_fit[i] = float(island_obj(children[i]))
                    did_eval_mask[i] = True
                except StopIteration:
                    break

        # Keep the champion fitness (always finite)
        child_fit[best_idx] = best_fit
        did_eval_mask[best_idx] = True

        # In-place writeback only for evaluated kids (avoid mutating unevaluated state)
        for i, c in enumerate(island):
            if did_eval_mask[i]:
                c.solution = children[i].astype(np.float64, copy=True)
                cache[c.id] = float(child_fit[i])

    def parallel_batch_processing(
        self,
        func: Callable,
        batch: torch.Tensor,
        chunk_size: int = 16,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Process a large batch in chunks to avoid OOM

        Args:
            func: Function to apply to each chunk
            batch: Input batch
            chunk_size: Size of each chunk
            device: Device to use

        Returns:
            Concatenated results
        """
        if device is None:
            device = batch.device

        results = []

        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size].to(device)
            with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                result = func(chunk)
            results.append(result.cpu())  # Move to CPU to save GPU memory

            # Clear cache between chunks
            if i + chunk_size < len(batch):
                torch.cuda.empty_cache()

        # Concatenate results
        return torch.cat(results, dim=0).to(device)
    
    @property
    def lock_fraction(self) -> float:
        """Current lock fraction across all islands"""
        try:
            if not self._lock_fraction_history:
                return 0.0
            return float(self._lock_fraction_history[-1])
        except Exception:
            return 0.0
    
    @property
    def lock_fraction_current(self) -> float:
        """Alias for lock_fraction property"""
        return self.lock_fraction
