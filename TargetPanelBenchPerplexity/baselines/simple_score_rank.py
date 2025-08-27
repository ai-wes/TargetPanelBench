"""
Simple Score & Rank baseline method for TargetPanelBench.

This is a naive baseline that normalizes all evidence scores and adds them up,
then ranks targets by this simple sum. This serves as the "strawman" baseline
that more sophisticated methods should easily outperform.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from base_optimizer import BaseTargetOptimizer, EvidenceProcessor


class SimpleScoreRankOptimizer(BaseTargetOptimizer):
    """
    Simple baseline that sums normalized evidence scores.

    This method:
    1. Normalizes all evidence scores to [0,1]
    2. Computes weighted sum of evidence types
    3. Ranks targets by total score
    4. Selects panel using greedy diversity approach
    """

    def __init__(self, 
                 evidence_weights: Optional[dict] = None,
                 diversity_method: str = 'greedy',
                 **kwargs):
        """
        Initialize Simple Score & Rank optimizer.

        Args:
            evidence_weights: Dict mapping evidence type to weight (default: equal weights)
            diversity_method: Method for panel diversification ('greedy', 'random', 'none')
            **kwargs: Additional parameters
        """
        super().__init__("SimpleScoreRank", **kwargs)
        self.evidence_weights = evidence_weights or {}
        self.diversity_method = diversity_method
        self.evidence_columns = None
        self.ppi_network = None

    def fit(self, 
            targets: List[str], 
            evidence_matrix: pd.DataFrame,
            ground_truth: Optional[List[str]] = None) -> 'SimpleScoreRankOptimizer':
        """
        Fit the simple scorer (just stores evidence column names).

        Args:
            targets: List of target identifiers
            evidence_matrix: Evidence scores matrix
            ground_truth: Known positive targets (unused by this method)

        Returns:
            self
        """
        self.evidence_columns = evidence_matrix.select_dtypes(include=[np.number]).columns.tolist()

        # Set default equal weights if not provided
        if not self.evidence_weights:
            self.evidence_weights = {col: 1.0 for col in self.evidence_columns}

        self.is_fitted = True
        self.logger.info(f"Fitted SimpleScoreRank with {len(self.evidence_columns)} evidence types")
        return self

    def rank_targets(self, 
                    targets: List[str], 
                    evidence_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Rank targets by weighted sum of normalized evidence scores.

        Args:
            targets: Target identifiers to rank
            evidence_matrix: Evidence matrix

        Returns:
            DataFrame with target rankings
        """
        if not self.is_fitted:
            raise ValueError("Optimizer must be fitted before ranking targets")

        # Normalize evidence matrix
        evidence_norm = EvidenceProcessor.normalize_evidence(evidence_matrix, method='min_max')
        evidence_norm = EvidenceProcessor.handle_missing_values(evidence_norm, strategy='median')

        # Calculate weighted scores
        total_scores = []
        for target in targets:
            if target in evidence_norm.index:
                target_scores = evidence_norm.loc[target, self.evidence_columns]
                weighted_score = sum(
                    target_scores[col] * self.evidence_weights.get(col, 1.0) 
                    for col in self.evidence_columns
                )
                total_scores.append(weighted_score)
            else:
                total_scores.append(0.0)  # Unknown target gets zero score

        # Create ranking DataFrame
        results_df = pd.DataFrame({
            'target': targets,
            'score': total_scores
        })

        # Sort by score (descending) and assign ranks
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
            ranked_targets: DataFrame from rank_targets()
            panel_size: Number of targets to select
            diversity_weight: Weight for diversity vs score

        Returns:
            List of selected targets
        """
        if self.diversity_method == 'none':
            # Just take top-ranked targets
            return ranked_targets.head(panel_size)['target'].tolist()

        elif self.diversity_method == 'random':
            # Random sampling from top candidates
            top_candidates = ranked_targets.head(min(panel_size * 3, len(ranked_targets)))
            selected_indices = np.random.choice(
                len(top_candidates), 
                size=min(panel_size, len(top_candidates)), 
                replace=False
            )
            return top_candidates.iloc[selected_indices]['target'].tolist()

        elif self.diversity_method == 'greedy':
            # Greedy diversity selection (simplified - no network data)
            return self._greedy_diversity_selection(ranked_targets, panel_size, diversity_weight)

        else:
            raise ValueError(f"Unknown diversity method: {self.diversity_method}")

    def _greedy_diversity_selection(self, 
                                  ranked_targets: pd.DataFrame,
                                  panel_size: int,
                                  diversity_weight: float) -> List[str]:
        """
        Greedy selection balancing score and diversity.

        Since we don't have PPI network data in the simple method,
        we'll use a simplified diversity metric based on target names.
        """
        selected = []
        candidates = ranked_targets.copy()

        # Select highest scoring target first
        if len(candidates) > 0:
            selected.append(candidates.iloc[0]['target'])
            candidates = candidates.iloc[1:]

        # Greedily select remaining targets
        while len(selected) < panel_size and len(candidates) > 0:
            best_target = None
            best_score = -np.inf

            for idx, row in candidates.iterrows():
                # Score component
                score_component = row['score']

                # Simple diversity component (based on string similarity)
                diversity_component = self._calculate_simple_diversity(
                    row['target'], selected
                )

                # Combined score
                combined_score = (
                    (1 - diversity_weight) * score_component + 
                    diversity_weight * diversity_component
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_target = row['target']

            if best_target:
                selected.append(best_target)
                candidates = candidates[candidates['target'] != best_target]

        return selected

    def _calculate_simple_diversity(self, target: str, selected_targets: List[str]) -> float:
        """
        Calculate simple diversity score based on string differences.

        This is a placeholder diversity metric for the simple baseline.
        Real implementations would use protein-protein interaction networks.
        """
        if not selected_targets:
            return 1.0

        # Simple string-based diversity (character differences)
        diversities = []
        for selected in selected_targets:
            # Calculate normalized string distance
            min_len = min(len(target), len(selected))
            if min_len == 0:
                diversity = 1.0
            else:
                common_chars = sum(1 for a, b in zip(target, selected) if a == b)
                diversity = 1.0 - (common_chars / min_len)
            diversities.append(diversity)

        return np.mean(diversities)


class SimpleScoreRankFactory:
    """Factory class for creating SimpleScoreRank optimizers with different configurations."""

    @staticmethod
    def create_equal_weights(**kwargs) -> SimpleScoreRankOptimizer:
        """Create optimizer with equal weights for all evidence types."""
        return SimpleScoreRankOptimizer(
            evidence_weights=None,  # Will default to equal weights
            diversity_method='greedy',
            **kwargs
        )

    @staticmethod
    def create_genetics_weighted(**kwargs) -> SimpleScoreRankOptimizer:
        """Create optimizer with higher weight on genetic evidence."""
        return SimpleScoreRankOptimizer(
            evidence_weights={
                'genetic_association': 2.0,
                'expression_specificity': 1.5,
                'ppi_centrality': 1.0,
                'druggability_score': 1.0,
                'literature_score': 0.5
            },
            diversity_method='greedy',
            **kwargs
        )

    @staticmethod
    def create_no_diversity(**kwargs) -> SimpleScoreRankOptimizer:
        """Create optimizer that ignores diversity (pure score-based selection)."""
        return SimpleScoreRankOptimizer(
            evidence_weights=None,
            diversity_method='none',
            **kwargs
        )
