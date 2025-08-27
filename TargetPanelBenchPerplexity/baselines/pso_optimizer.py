"""
Particle Swarm Optimization (PSO) optimizer for TargetPanelBench.

This implements PSO for learning optimal evidence combination weights
for target prioritization. PSO is inspired by social behavior of bird flocking.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
from base_optimizer import BaseTargetOptimizer, EvidenceProcessor


class Particle:
    """
    Individual particle in the PSO swarm.

    Each particle represents a potential solution (weight vector)
    and maintains position, velocity, and best known position.
    """

    def __init__(self, dimensions: int, bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize particle with random position and velocity.

        Args:
            dimensions: Number of parameters (evidence types)
            bounds: (min, max) bounds for parameter values
        """
        self.dimensions = dimensions
        self.bounds = bounds

        # Initialize position randomly within bounds
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        # Normalize to sum to 1 (for weights)
        self.position = self.position / np.sum(self.position)

        # Initialize velocity
        self.velocity = np.random.uniform(-0.1, 0.1, dimensions)

        # Personal best
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf

        # Current fitness
        self.fitness = -np.inf

    def update_velocity(self, 
                       global_best_position: np.ndarray,
                       w: float = 0.7,
                       c1: float = 1.4,
                       c2: float = 1.4):
        """
        Update particle velocity based on PSO equations.

        Args:
            global_best_position: Best position found by any particle
            w: Inertia weight
            c1: Cognitive acceleration coefficient
            c2: Social acceleration coefficient
        """
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

        # Velocity clamping
        max_velocity = (self.bounds[1] - self.bounds[0]) * 0.1
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def update_position(self):
        """Update particle position based on velocity."""
        self.position = self.position + self.velocity

        # Enforce bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

        # Renormalize weights to sum to 1
        if np.sum(self.position) > 0:
            self.position = self.position / np.sum(self.position)
        else:
            self.position = np.ones(self.dimensions) / self.dimensions

    def update_best(self):
        """Update personal best if current position is better."""
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()


class PSOOptimizer(BaseTargetOptimizer):
    """
    Particle Swarm Optimization for target prioritization.

    Uses swarm intelligence to optimize evidence combination weights
    by maintaining a population of particles that explore the solution space.
    """

    def __init__(self,
                 swarm_size: int = 30,
                 max_iterations: int = 100,
                 w: float = 0.7,
                 c1: float = 1.4,
                 c2: float = 1.4,
                 convergence_threshold: float = 1e-6,
                 fitness_function: str = 'precision_at_k',
                 k: int = 20,
                 bounds: Tuple[float, float] = (0.01, 1.0),
                 **kwargs):
        """
        Initialize PSO optimizer.

        Args:
            swarm_size: Number of particles in swarm
            max_iterations: Maximum number of iterations
            w: Inertia weight (controls exploration vs exploitation)
            c1: Cognitive acceleration coefficient (particle memory)
            c2: Social acceleration coefficient (swarm influence)
            convergence_threshold: Convergence criterion
            fitness_function: Fitness metric ('precision_at_k', 'ndcg', 'mrr')
            k: Top-k for precision calculation
            bounds: (min, max) bounds for weight values
            **kwargs: Additional parameters
        """
        super().__init__("PSO", **kwargs)
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.convergence_threshold = convergence_threshold
        self.fitness_function = fitness_function
        self.k = k
        self.bounds = bounds

        # PSO state
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.fitness_history = []

        # Training data
        self.evidence_columns = None
        self.training_targets = None
        self.training_evidence = None
        self.ground_truth = None

        # Dynamic parameters
        self.adaptive_parameters = kwargs.get('adaptive_parameters', False)
        self.w_min = 0.4
        self.w_max = 0.9

    def fit(self, 
            targets: List[str], 
            evidence_matrix: pd.DataFrame,
            ground_truth: Optional[List[str]] = None) -> 'PSOOptimizer':
        """
        Fit PSO optimizer to training data.

        Args:
            targets: List of target identifiers
            evidence_matrix: Evidence matrix with targets as rows
            ground_truth: Known positive targets for fitness evaluation

        Returns:
            self
        """
        if ground_truth is None or len(ground_truth) == 0:
            raise ValueError("PSO requires ground truth targets for fitness evaluation")

        # Store training data
        self.training_targets = targets
        self.training_evidence = evidence_matrix.copy()
        self.ground_truth = ground_truth
        self.evidence_columns = evidence_matrix.select_dtypes(include=[np.number]).columns.tolist()

        # Normalize and clean evidence
        self.training_evidence = EvidenceProcessor.normalize_evidence(
            self.training_evidence, method='min_max'
        )
        self.training_evidence = EvidenceProcessor.handle_missing_values(
            self.training_evidence, strategy='median'
        )

        # Initialize swarm
        self._initialize_swarm()

        # Run PSO
        self._optimize()

        self.is_fitted = True
        self.logger.info(f"PSO fitted with best fitness: {self.global_best_fitness:.4f}")
        return self

    def _initialize_swarm(self):
        """Initialize particle swarm."""
        n_dimensions = len(self.evidence_columns)

        self.swarm = []
        for _ in range(self.swarm_size):
            particle = Particle(n_dimensions, self.bounds)
            self.swarm.append(particle)

        # Initialize global best
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.fitness_history = []

    def _optimize(self):
        """Run PSO optimization."""
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all particles
            for particle in self.swarm:
                particle.fitness = self._evaluate_fitness(particle.position)
                particle.update_best()

                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()

            self.fitness_history.append(self.global_best_fitness)

            # Update parameters if adaptive
            if self.adaptive_parameters:
                self._update_adaptive_parameters(iteration)

            # Update particle velocities and positions
            for particle in self.swarm:
                particle.update_velocity(
                    self.global_best_position, self.w, self.c1, self.c2
                )
                particle.update_position()

            # Check convergence
            if iteration > 10:
                recent_improvement = (
                    max(self.fitness_history[-5:]) - 
                    max(self.fitness_history[-10:-5])
                )
                if abs(recent_improvement) < self.convergence_threshold:
                    self.logger.info(f"PSO converged at iteration {iteration}")
                    break

            # Log progress
            if iteration % 20 == 0:
                self.logger.debug(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f}")

    def _update_adaptive_parameters(self, iteration: int):
        """Update PSO parameters dynamically."""
        # Linearly decrease inertia weight
        self.w = self.w_max - ((self.w_max - self.w_min) * iteration / self.max_iterations)

        # Optionally adjust acceleration coefficients
        # c1 decreases, c2 increases over time (exploration -> exploitation)
        progress = iteration / self.max_iterations
        self.c1 = 2.5 - 1.5 * progress
        self.c2 = 0.5 + 1.5 * progress

    def _evaluate_fitness(self, weights: np.ndarray) -> float:
        """
        Evaluate fitness of weight vector.

        Args:
            weights: Weight vector for evidence types

        Returns:
            Fitness score (higher is better)
        """
        # Score all training targets
        scores = []
        for target in self.training_targets:
            if target in self.training_evidence.index:
                target_evidence = self.training_evidence.loc[target, self.evidence_columns]
                score = np.sum(weights * target_evidence.values)
                scores.append(score)
            else:
                scores.append(0.0)

        # Create ranking
        target_scores_df = pd.DataFrame({
            'target': self.training_targets,
            'score': scores
        }).sort_values('score', ascending=False)

        # Calculate fitness based on ground truth
        if self.fitness_function == 'precision_at_k':
            top_k = target_scores_df.head(self.k)['target'].tolist()
            precision = len(set(top_k) & set(self.ground_truth)) / len(top_k)
            return precision

        elif self.fitness_function == 'ndcg':
            return self._calculate_ndcg(target_scores_df, self.ground_truth, self.k)

        elif self.fitness_function == 'mrr':
            return self._calculate_mrr(target_scores_df, self.ground_truth)

        else:
            raise ValueError(f"Unknown fitness function: {self.fitness_function}")

    def _calculate_ndcg(self, ranked_df: pd.DataFrame, ground_truth: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # DCG calculation
        dcg = 0.0
        for i, target in enumerate(ranked_df.head(k)['target']):
            if target in ground_truth:
                dcg += 1.0 / np.log2(i + 2)

        # IDCG calculation
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, ranked_df: pd.DataFrame, ground_truth: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, target in enumerate(ranked_df['target']):
            if target in ground_truth:
                return 1.0 / (i + 1)
        return 0.0

    def rank_targets(self, 
                    targets: List[str], 
                    evidence_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Rank targets using optimized weights.

        Args:
            targets: Target identifiers to rank
            evidence_matrix: Evidence matrix

        Returns:
            DataFrame with rankings
        """
        if not self.is_fitted or self.global_best_position is None:
            raise ValueError("Optimizer must be fitted before ranking")

        # Normalize evidence
        evidence_norm = EvidenceProcessor.normalize_evidence(evidence_matrix, method='min_max')
        evidence_norm = EvidenceProcessor.handle_missing_values(evidence_norm, strategy='median')

        # Calculate scores using optimized weights
        scores = []
        for target in targets:
            if target in evidence_norm.index:
                target_evidence = evidence_norm.loc[target, self.evidence_columns]
                score = np.sum(self.global_best_position * target_evidence.values)
                scores.append(score)
            else:
                scores.append(0.0)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'target': targets,
            'score': scores
        })

        # Sort and rank
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)

        return results_df

    def select_panel(self, 
                    ranked_targets: pd.DataFrame,
                    panel_size: int = 15,
                    diversity_weight: float = 0.3) -> List[str]:
        """
        Select diverse panel from ranked targets.

        Args:
            ranked_targets: Rankings from rank_targets()
            panel_size: Number of targets to select
            diversity_weight: Weight for diversity vs score

        Returns:
            List of selected targets
        """
        # Simple diversity selection for PSO
        return self._probabilistic_selection(ranked_targets, panel_size, diversity_weight)

    def _probabilistic_selection(self, 
                                ranked_targets: pd.DataFrame, 
                                panel_size: int, 
                                diversity_weight: float) -> List[str]:
        """
        Probabilistic selection balancing score and diversity.

        Uses scores as selection probabilities with some randomization for diversity.
        """
        # Get top candidates
        n_candidates = min(panel_size * 3, len(ranked_targets))
        top_candidates = ranked_targets.head(n_candidates)

        if len(top_candidates) <= panel_size:
            return top_candidates['target'].tolist()

        # Create selection probabilities
        scores = top_candidates['score'].values
        if np.max(scores) > np.min(scores):
            # Normalize scores to [0, 1]
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            normalized_scores = np.ones(len(scores)) / len(scores)

        # Apply diversity weight (higher diversity = more uniform probabilities)
        if diversity_weight > 0:
            uniform_probs = np.ones(len(scores)) / len(scores)
            probabilities = (
                (1 - diversity_weight) * normalized_scores + 
                diversity_weight * uniform_probs
            )
        else:
            probabilities = normalized_scores

        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)

        # Select panel
        selected_indices = np.random.choice(
            len(top_candidates), 
            size=panel_size, 
            replace=False, 
            p=probabilities
        )

        return top_candidates.iloc[selected_indices]['target'].tolist()

    def get_swarm_diversity(self) -> float:
        """Calculate diversity of current swarm."""
        if not self.swarm:
            return 0.0

        positions = np.array([particle.position for particle in self.swarm])

        # Calculate pairwise distances
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def get_convergence_history(self) -> List[float]:
        """Get fitness convergence history."""
        return self.fitness_history.copy()

    def get_optimized_weights(self) -> Dict[str, float]:
        """Get optimized evidence weights."""
        if self.global_best_position is None:
            return {}

        return dict(zip(self.evidence_columns, self.global_best_position))

    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current swarm."""
        if not self.swarm:
            return {}

        fitnesses = [p.fitness for p in self.swarm]
        return {
            'swarm_size': len(self.swarm),
            'best_fitness': self.global_best_fitness,
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'diversity': self.get_swarm_diversity(),
            'convergence_iterations': len(self.fitness_history)
        }


class PSOFactory:
    """Factory for creating PSO optimizers with different configurations."""

    @staticmethod
    def create_standard(**kwargs) -> PSOOptimizer:
        """Create standard PSO optimizer."""
        return PSOOptimizer(
            swarm_size=30,
            w=0.7,
            c1=1.4,
            c2=1.4,
            **kwargs
        )

    @staticmethod
    def create_adaptive(**kwargs) -> PSOOptimizer:
        """Create PSO with adaptive parameters."""
        return PSOOptimizer(
            swarm_size=30,
            adaptive_parameters=True,
            **kwargs
        )

    @staticmethod
    def create_exploration_focused(**kwargs) -> PSOOptimizer:
        """Create PSO focused on exploration."""
        return PSOOptimizer(
            swarm_size=50,
            w=0.9,
            c1=2.0,
            c2=1.0,
            **kwargs
        )

    @staticmethod
    def create_exploitation_focused(**kwargs) -> PSOOptimizer:
        """Create PSO focused on exploitation."""
        return PSOOptimizer(
            swarm_size=20,
            w=0.4,
            c1=1.0,
            c2=2.0,
            **kwargs
        )
