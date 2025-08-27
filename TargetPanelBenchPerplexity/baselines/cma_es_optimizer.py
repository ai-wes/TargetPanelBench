"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer for TargetPanelBench.

This implements a sophisticated evolutionary algorithm that learns the covariance
structure of the optimization landscape to efficiently search for optimal 
target prioritization weights.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Any
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import logging
from base_optimizer import BaseTargetOptimizer, EvidenceProcessor


class CMAESOptimizer(BaseTargetOptimizer):
    """
    CMA-ES optimizer for target prioritization.

    Uses evolutionary strategy to learn optimal weights for evidence combination
    and target selection. Particularly effective for high-dimensional, 
    multi-modal optimization problems.
    """

    def __init__(self,
                 population_size: Optional[int] = None,
                 sigma: float = 0.3,
                 max_generations: int = 100,
                 tolerance: float = 1e-8,
                 diversity_network: Optional[np.ndarray] = None,
                 fitness_function: str = 'precision_at_k',
                 k: int = 20,
                 **kwargs):
        """
        Initialize CMA-ES optimizer.

        Args:
            population_size: Number of individuals per generation (auto if None)
            sigma: Initial step size (mutation strength)
            max_generations: Maximum number of generations
            tolerance: Convergence tolerance
            diversity_network: Adjacency matrix for diversity calculation
            fitness_function: Fitness metric ('precision_at_k', 'ndcg', 'mrr')
            k: Top-k for precision calculation
            **kwargs: Additional parameters
        """
        super().__init__("CMA-ES", **kwargs)
        self.population_size = population_size
        self.sigma = sigma
        self.max_generations = max_generations
        self.tolerance = tolerance
        self.diversity_network = diversity_network
        self.fitness_function = fitness_function
        self.k = k

        # CMA-ES state variables (initialized during fit)
        self.n_params = None
        self.mean = None
        self.cov_matrix = None
        self.evolution_path = None
        self.conjugate_evolution_path = None
        self.best_weights = None
        self.fitness_history = []

        # Evidence processing
        self.evidence_columns = None
        self.training_targets = None
        self.training_evidence = None
        self.ground_truth = None

    def fit(self, 
            targets: List[str], 
            evidence_matrix: pd.DataFrame,
            ground_truth: Optional[List[str]] = None) -> 'CMAESOptimizer':
        """
        Fit CMA-ES optimizer to training data.

        Args:
            targets: List of target identifiers
            evidence_matrix: Evidence matrix with targets as rows
            ground_truth: Known positive targets for fitness evaluation

        Returns:
            self
        """
        if ground_truth is None or len(ground_truth) == 0:
            raise ValueError("CMA-ES requires ground truth targets for fitness evaluation")

        # Store training data
        self.training_targets = targets
        self.training_evidence = evidence_matrix.copy()
        self.ground_truth = ground_truth
        self.evidence_columns = evidence_matrix.select_dtypes(include=[np.number]).columns.tolist()

        # Normalize and clean evidence matrix
        self.training_evidence = EvidenceProcessor.normalize_evidence(
            self.training_evidence, method='min_max'
        )
        self.training_evidence = EvidenceProcessor.handle_missing_values(
            self.training_evidence, strategy='median'
        )

        # Initialize CMA-ES parameters
        self.n_params = len(self.evidence_columns)

        if self.population_size is None:
            self.population_size = 4 + int(3 * np.log(self.n_params))

        self._initialize_cma_es()

        # Run evolution
        self._evolve()

        self.is_fitted = True
        self.logger.info(f"CMA-ES fitted with best fitness: {max(self.fitness_history):.4f}")
        return self

    def _initialize_cma_es(self):
        """Initialize CMA-ES state variables."""
        # Initialize mean as equal weights
        self.mean = np.ones(self.n_params) / self.n_params

        # Initialize covariance matrix as identity
        self.cov_matrix = np.eye(self.n_params)

        # Evolution paths
        self.evolution_path = np.zeros(self.n_params)
        self.conjugate_evolution_path = np.zeros(self.n_params)

        # CMA-ES hyperparameters
        self.weights = self._compute_recombination_weights()
        self.mueff = 1 / np.sum(self.weights ** 2)

        # Learning rates
        self.cc = (4 + self.mueff / self.n_params) / (self.n_params + 4 + 2 * self.mueff / self.n_params)
        self.cs = (self.mueff + 2) / (self.n_params + self.mueff + 5)
        self.c1 = 2 / ((self.n_params + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n_params + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n_params + 1)) - 1) + self.cs

    def _compute_recombination_weights(self) -> np.ndarray:
        """Compute recombination weights for CMA-ES."""
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        return weights

    def _evolve(self):
        """Run CMA-ES evolution."""
        for generation in range(self.max_generations):
            # Generate population
            population = self._generate_population()

            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual) for individual in population]

            # Track best fitness
            current_best = max(fitness_scores)
            self.fitness_history.append(current_best)

            # Update best weights
            best_idx = np.argmax(fitness_scores)
            if self.best_weights is None or current_best > max(self.fitness_history[:-1]):
                self.best_weights = population[best_idx].copy()

            # Selection and recombination
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_population = [population[i] for i in sorted_indices[:len(self.weights)]]

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.sum([w * ind for w, ind in zip(self.weights, selected_population)], axis=0)

            # Update evolution paths
            self._update_evolution_paths(old_mean)

            # Update covariance matrix
            self._update_covariance_matrix(selected_population)

            # Update step size
            self._update_step_size()

            # Check convergence
            if generation > 10:
                recent_improvement = max(self.fitness_history[-5:]) - max(self.fitness_history[-10:-5])
                if abs(recent_improvement) < self.tolerance:
                    self.logger.info(f"Converged at generation {generation}")
                    break

    def _generate_population(self) -> List[np.ndarray]:
        """Generate population of candidate solutions."""
        # Eigendecomposition for sampling
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-14)  # Ensure positive definiteness

        population = []
        for _ in range(self.population_size):
            # Sample from multivariate normal
            z = np.random.randn(self.n_params)
            y = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ z
            x = self.mean + self.sigma * y

            # Ensure weights are positive and sum to reasonable range
            x = np.maximum(x, 0.01)
            x = x / (np.sum(x) + 1e-8)  # Normalize to sum to 1

            population.append(x)

        return population

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
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # IDCG calculation (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, ranked_df: pd.DataFrame, ground_truth: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, target in enumerate(ranked_df['target']):
            if target in ground_truth:
                return 1.0 / (i + 1)
        return 0.0

    def _update_evolution_paths(self, old_mean: np.ndarray):
        """Update evolution paths for CMA-ES."""
        # Update conjugate evolution path
        mean_diff = (self.mean - old_mean) / self.sigma

        # Simplified update (full CMA-ES would use matrix square root)
        self.conjugate_evolution_path = (
            (1 - self.cs) * self.conjugate_evolution_path + 
            np.sqrt(self.cs * (2 - self.cs) * self.mueff) * mean_diff
        )

        # Update evolution path
        hsig = np.linalg.norm(self.conjugate_evolution_path) / np.sqrt(
            1 - (1 - self.cs) ** (2 * len(self.fitness_history))
        ) < 1.4 + 2 / (self.n_params + 1)

        self.evolution_path = (
            (1 - self.cc) * self.evolution_path + 
            hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * mean_diff
        )

    def _update_covariance_matrix(self, selected_population: List[np.ndarray]):
        """Update covariance matrix."""
        # Rank-1 update
        rank1_update = np.outer(self.evolution_path, self.evolution_path)

        # Rank-mu update
        rankmu_update = np.zeros((self.n_params, self.n_params))
        for i, (weight, individual) in enumerate(zip(self.weights, selected_population)):
            diff = (individual - self.mean) / self.sigma
            rankmu_update += weight * np.outer(diff, diff)

        # Update covariance matrix
        self.cov_matrix = (
            (1 - self.c1 - self.cmu) * self.cov_matrix + 
            self.c1 * rank1_update + 
            self.cmu * rankmu_update
        )

        # Ensure positive definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-14)
        self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _update_step_size(self):
        """Update step size (sigma)."""
        # Simplified step size update
        cs_norm = np.linalg.norm(self.conjugate_evolution_path)
        expected_norm = np.sqrt(self.n_params) * (
            1 - 1/(4*self.n_params) + 1/(21*self.n_params**2)
        )

        self.sigma *= np.exp(
            (self.cs / self.damps) * (cs_norm / expected_norm - 1)
        )

    def rank_targets(self, 
                    targets: List[str], 
                    evidence_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Rank targets using learned weights.

        Args:
            targets: Target identifiers to rank
            evidence_matrix: Evidence matrix

        Returns:
            DataFrame with rankings
        """
        if not self.is_fitted or self.best_weights is None:
            raise ValueError("Optimizer must be fitted before ranking")

        # Normalize evidence
        evidence_norm = EvidenceProcessor.normalize_evidence(evidence_matrix, method='min_max')
        evidence_norm = EvidenceProcessor.handle_missing_values(evidence_norm, strategy='median')

        # Calculate scores using learned weights
        scores = []
        for target in targets:
            if target in evidence_norm.index:
                target_evidence = evidence_norm.loc[target, self.evidence_columns]
                score = np.sum(self.best_weights * target_evidence.values)
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
        Select diverse panel using learned preferences.

        Args:
            ranked_targets: Rankings from rank_targets()
            panel_size: Number of targets to select
            diversity_weight: Weight for diversity vs score

        Returns:
            List of selected targets
        """
        if self.diversity_network is None:
            # Fall back to greedy selection without network
            return self._greedy_selection_no_network(ranked_targets, panel_size, diversity_weight)
        else:
            return self._greedy_selection_with_network(ranked_targets, panel_size, diversity_weight)

    def _greedy_selection_no_network(self, 
                                   ranked_targets: pd.DataFrame, 
                                   panel_size: int, 
                                   diversity_weight: float) -> List[str]:
        """Greedy selection without network information."""
        # Simple implementation - just take top targets with some randomization
        top_candidates = ranked_targets.head(min(panel_size * 2, len(ranked_targets)))

        if diversity_weight > 0.5:
            # Higher diversity - more random sampling
            selected_indices = np.random.choice(
                len(top_candidates), 
                size=min(panel_size, len(top_candidates)), 
                replace=False,
                p=None  # Uniform probability
            )
        else:
            # Lower diversity - bias towards top scores
            scores = top_candidates['score'].values
            probs = scores / np.sum(scores) if np.sum(scores) > 0 else None
            selected_indices = np.random.choice(
                len(top_candidates), 
                size=min(panel_size, len(top_candidates)), 
                replace=False,
                p=probs
            )

        return top_candidates.iloc[selected_indices]['target'].tolist()

    def _greedy_selection_with_network(self, 
                                     ranked_targets: pd.DataFrame, 
                                     panel_size: int, 
                                     diversity_weight: float) -> List[str]:
        """Greedy selection with network-based diversity."""
        # Implementation would use self.diversity_network for PPI-based diversity
        # For now, fall back to no-network version
        return self._greedy_selection_no_network(ranked_targets, panel_size, diversity_weight)

    def get_convergence_history(self) -> List[float]:
        """Get fitness convergence history."""
        return self.fitness_history.copy()

    def get_learned_weights(self) -> Dict[str, float]:
        """Get learned evidence weights."""
        if self.best_weights is None:
            return {}

        return dict(zip(self.evidence_columns, self.best_weights))
