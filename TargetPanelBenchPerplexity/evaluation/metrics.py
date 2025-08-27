"""
Evaluation metrics for TargetPanelBench.

This module implements standard ranking and classification metrics used
in computational biology and machine learning for assessing target prioritization
and panel selection performance.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import networkx as nx
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import logging


class TargetPrioritizationMetrics:
    """
    Comprehensive evaluation metrics for target prioritization benchmarks.

    Implements metrics commonly used in drug discovery and computational biology
    for evaluating ranking quality and panel diversity.
    """

    def __init__(self):
        self.logger = logging.getLogger("TargetPanelBench.Metrics")

    def precision_at_k(self, 
                      ranked_targets: List[str], 
                      ground_truth: List[str], 
                      k: int) -> float:
        """
        Calculate Precision@K metric.

        Args:
            ranked_targets: List of targets ordered by score (best first)
            ground_truth: List of known positive targets
            k: Number of top targets to consider

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k <= 0 or len(ranked_targets) == 0:
            return 0.0

        top_k = ranked_targets[:min(k, len(ranked_targets))]
        correct = len(set(top_k) & set(ground_truth))

        return correct / len(top_k)

    def recall_at_k(self, 
                   ranked_targets: List[str], 
                   ground_truth: List[str], 
                   k: int) -> float:
        """
        Calculate Recall@K metric.

        Args:
            ranked_targets: List of targets ordered by score
            ground_truth: List of known positive targets
            k: Number of top targets to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if len(ground_truth) == 0 or k <= 0:
            return 0.0

        top_k = ranked_targets[:min(k, len(ranked_targets))]
        correct = len(set(top_k) & set(ground_truth))

        return correct / len(ground_truth)

    def mean_reciprocal_rank(self, 
                           ranked_targets: List[str], 
                           ground_truth: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            ranked_targets: List of targets ordered by score
            ground_truth: List of known positive targets

        Returns:
            MRR score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.0

        reciprocal_ranks = []
        for target in ground_truth:
            try:
                rank = ranked_targets.index(target) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)  # Target not in ranking

        return np.mean(reciprocal_ranks)

    def ndcg_at_k(self, 
                 ranked_targets: List[str], 
                 ground_truth: List[str], 
                 k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        Args:
            ranked_targets: List of targets ordered by score
            ground_truth: List of known positive targets
            k: Number of top targets to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if k <= 0 or len(ground_truth) == 0:
            return 0.0

        # Calculate DCG@K
        dcg = 0.0
        for i, target in enumerate(ranked_targets[:min(k, len(ranked_targets))]):
            if target in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # Calculate IDCG@K (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))

        return dcg / idcg if idcg > 0 else 0.0

    def average_precision(self, 
                         ranked_targets: List[str], 
                         ground_truth: List[str]) -> float:
        """
        Calculate Average Precision (AP).

        Args:
            ranked_targets: List of targets ordered by score
            ground_truth: List of known positive targets

        Returns:
            Average Precision score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.0

        precisions = []
        num_correct = 0

        for i, target in enumerate(ranked_targets):
            if target in ground_truth:
                num_correct += 1
                precision = num_correct / (i + 1)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def auc_roc(self, 
               ranked_df: pd.DataFrame, 
               ground_truth: List[str],
               score_column: str = 'score') -> float:
        """
        Calculate Area Under ROC Curve.

        Args:
            ranked_df: DataFrame with 'target' and score columns
            ground_truth: List of known positive targets
            score_column: Name of score column

        Returns:
            AUC-ROC score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.5

        # Create binary labels
        y_true = [1 if target in ground_truth else 0 for target in ranked_df['target']]
        y_scores = ranked_df[score_column].values

        if len(set(y_true)) < 2:  # Need both positive and negative examples
            return 0.5

        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return 0.5

    def auc_pr(self, 
              ranked_df: pd.DataFrame, 
              ground_truth: List[str],
              score_column: str = 'score') -> float:
        """
        Calculate Area Under Precision-Recall Curve.

        Args:
            ranked_df: DataFrame with 'target' and score columns
            ground_truth: List of known positive targets
            score_column: Name of score column

        Returns:
            AUC-PR score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.0

        # Create binary labels
        y_true = [1 if target in ground_truth else 0 for target in ranked_df['target']]
        y_scores = ranked_df[score_column].values

        if len(set(y_true)) < 2:
            return 0.0

        try:
            return average_precision_score(y_true, y_scores)
        except ValueError:
            return 0.0


class PanelDiversityMetrics:
    """
    Metrics for evaluating diversity of selected target panels.

    Implements network-based and feature-based diversity measures
    commonly used in drug discovery for assessing target panel quality.
    """

    def __init__(self, ppi_network: Optional[nx.Graph] = None):
        """
        Initialize diversity metrics.

        Args:
            ppi_network: NetworkX graph representing protein-protein interactions
        """
        self.ppi_network = ppi_network
        self.logger = logging.getLogger("TargetPanelBench.Diversity")

    def network_diversity_score(self, selected_targets: List[str]) -> float:
        """
        Calculate network-based diversity score.

        Computes average shortest path distance between all pairs of selected targets
        in the protein-protein interaction network.

        Args:
            selected_targets: List of selected target identifiers

        Returns:
            Average network distance (higher = more diverse)
        """
        if self.ppi_network is None:
            self.logger.warning("No PPI network provided, returning default diversity score")
            return 1.0

        if len(selected_targets) < 2:
            return 1.0

        # Filter targets that are in the network
        network_targets = [t for t in selected_targets if t in self.ppi_network.nodes]

        if len(network_targets) < 2:
            return 1.0

        distances = []
        for i, target1 in enumerate(network_targets):
            for target2 in network_targets[i+1:]:
                try:
                    distance = nx.shortest_path_length(self.ppi_network, target1, target2)
                    distances.append(distance)
                except nx.NetworkXNoPath:
                    # No path between targets - they are in different components
                    distances.append(float('inf'))

        # Handle infinite distances (different components)
        finite_distances = [d for d in distances if d != float('inf')]

        if not finite_distances:
            return 10.0  # High diversity for disconnected targets

        avg_distance = np.mean(finite_distances)

        # Add bonus for disconnected targets
        infinite_count = len(distances) - len(finite_distances)
        if infinite_count > 0:
            avg_distance += infinite_count / len(distances) * 5.0

        return avg_distance

    def feature_diversity_score(self, 
                              selected_targets: List[str],
                              feature_matrix: pd.DataFrame,
                              method: str = 'euclidean') -> float:
        """
        Calculate feature-based diversity score.

        Args:
            selected_targets: List of selected targets
            feature_matrix: DataFrame with targets as rows, features as columns
            method: Distance metric ('euclidean', 'cosine', 'manhattan')

        Returns:
            Average pairwise distance in feature space
        """
        if len(selected_targets) < 2:
            return 1.0

        # Get features for selected targets
        available_targets = [t for t in selected_targets if t in feature_matrix.index]

        if len(available_targets) < 2:
            return 1.0

        target_features = feature_matrix.loc[available_targets]

        # Calculate pairwise distances
        distances = []
        for i, target1 in enumerate(available_targets):
            for target2 in available_targets[i+1:]:
                features1 = target_features.loc[target1].values
                features2 = target_features.loc[target2].values

                if method == 'euclidean':
                    distance = np.linalg.norm(features1 - features2)
                elif method == 'cosine':
                    dot_product = np.dot(features1, features2)
                    norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
                    distance = 1 - (dot_product / norm_product) if norm_product > 0 else 1.0
                elif method == 'manhattan':
                    distance = np.sum(np.abs(features1 - features2))
                else:
                    raise ValueError(f"Unknown distance method: {method}")

                distances.append(distance)

        return np.mean(distances) if distances else 1.0

    def panel_coverage_score(self, 
                           selected_targets: List[str],
                           pathway_annotations: Dict[str, List[str]]) -> float:
        """
        Calculate pathway coverage diversity.

        Args:
            selected_targets: List of selected targets
            pathway_annotations: Dict mapping pathways to target lists

        Returns:
            Fraction of pathways covered by selected targets
        """
        if not pathway_annotations or not selected_targets:
            return 0.0

        covered_pathways = set()
        for pathway, pathway_targets in pathway_annotations.items():
            if any(target in selected_targets for target in pathway_targets):
                covered_pathways.add(pathway)

        return len(covered_pathways) / len(pathway_annotations)

    def redundancy_penalty(self, 
                          selected_targets: List[str]) -> float:
        """
        Calculate redundancy penalty based on target similarity.

        This is a simplified version - in practice would use sequence similarity,
        functional similarity, or network clustering.

        Args:
            selected_targets: List of selected targets

        Returns:
            Redundancy penalty (0 = no redundancy, 1 = high redundancy)
        """
        if len(selected_targets) < 2:
            return 0.0

        # Simple implementation based on string similarity
        # In practice, would use protein sequences, domains, or GO annotations
        similarities = []
        for i, target1 in enumerate(selected_targets):
            for target2 in selected_targets[i+1:]:
                # Jaccard similarity of character sets (very simplified)
                set1 = set(target1.upper())
                set2 = set(target2.upper())

                if len(set1) == 0 and len(set2) == 0:
                    similarity = 1.0
                elif len(set1) == 0 or len(set2) == 0:
                    similarity = 0.0
                else:
                    similarity = len(set1 & set2) / len(set1 | set2)

                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0


class BenchmarkEvaluator:
    """
    Comprehensive benchmark evaluator combining ranking and diversity metrics.

    Provides standardized evaluation of target prioritization methods including
    ranking quality, panel diversity, and statistical significance testing.
    """

    def __init__(self, 
                 ground_truth: List[str],
                 ppi_network: Optional[nx.Graph] = None,
                 feature_matrix: Optional[pd.DataFrame] = None):
        """
        Initialize benchmark evaluator.

        Args:
            ground_truth: List of known positive targets
            ppi_network: Protein-protein interaction network
            feature_matrix: Target feature matrix for diversity calculation
        """
        self.ground_truth = ground_truth
        self.ranking_metrics = TargetPrioritizationMetrics()
        self.diversity_metrics = PanelDiversityMetrics(ppi_network)
        self.feature_matrix = feature_matrix

        self.logger = logging.getLogger("TargetPanelBench.Evaluator")

    def evaluate_ranking(self, 
                        ranked_df: pd.DataFrame,
                        k_values: List[int] = [10, 20, 50]) -> Dict[str, float]:
        """
        Evaluate ranking performance.

        Args:
            ranked_df: DataFrame with 'target', 'score', 'rank' columns
            k_values: List of k values for Precision@K and NDCG@K

        Returns:
            Dict of metric names to scores
        """
        ranked_targets = ranked_df['target'].tolist()
        results = {}

        # Precision@K and Recall@K
        for k in k_values:
            results[f'precision_at_{k}'] = self.ranking_metrics.precision_at_k(
                ranked_targets, self.ground_truth, k
            )
            results[f'recall_at_{k}'] = self.ranking_metrics.recall_at_k(
                ranked_targets, self.ground_truth, k
            )
            results[f'ndcg_at_{k}'] = self.ranking_metrics.ndcg_at_k(
                ranked_targets, self.ground_truth, k
            )

        # Overall metrics
        results['mean_reciprocal_rank'] = self.ranking_metrics.mean_reciprocal_rank(
            ranked_targets, self.ground_truth
        )
        results['average_precision'] = self.ranking_metrics.average_precision(
            ranked_targets, self.ground_truth
        )
        results['auc_roc'] = self.ranking_metrics.auc_roc(ranked_df, self.ground_truth)
        results['auc_pr'] = self.ranking_metrics.auc_pr(ranked_df, self.ground_truth)

        return results

    def evaluate_panel(self, 
                      selected_panel: List[str]) -> Dict[str, float]:
        """
        Evaluate panel diversity and quality.

        Args:
            selected_panel: List of selected target identifiers

        Returns:
            Dict of diversity metrics
        """
        results = {}

        # Panel recall (how many ground truth targets are included)
        panel_recall = len(set(selected_panel) & set(self.ground_truth)) / len(self.ground_truth)
        results['panel_recall'] = panel_recall

        # Network diversity
        results['network_diversity'] = self.diversity_metrics.network_diversity_score(selected_panel)

        # Feature diversity
        if self.feature_matrix is not None:
            results['feature_diversity'] = self.diversity_metrics.feature_diversity_score(
                selected_panel, self.feature_matrix
            )

        # Redundancy penalty
        results['redundancy_penalty'] = self.diversity_metrics.redundancy_penalty(selected_panel)

        # Combined diversity score (higher is better)
        diversity_components = []
        if 'network_diversity' in results:
            diversity_components.append(results['network_diversity'])
        if 'feature_diversity' in results:
            diversity_components.append(results['feature_diversity'])

        if diversity_components:
            results['combined_diversity'] = np.mean(diversity_components) * (1 - results['redundancy_penalty'])

        return results

    def evaluate_method(self, 
                       ranked_df: pd.DataFrame,
                       selected_panel: List[str],
                       k_values: List[int] = [10, 20, 50]) -> Dict[str, Any]:
        """
        Complete evaluation of a target prioritization method.

        Args:
            ranked_df: Target rankings from the method
            selected_panel: Selected target panel
            k_values: K values for evaluation metrics

        Returns:
            Dict with all evaluation results
        """
        results = {
            'method_name': ranked_df.get('method_name', 'Unknown'),
            'ranking_metrics': self.evaluate_ranking(ranked_df, k_values),
            'panel_metrics': self.evaluate_panel(selected_panel),
            'panel_size': len(selected_panel)
        }

        # Overall quality score combining ranking and diversity
        ranking_score = results['ranking_metrics'].get('precision_at_20', 0.0)
        diversity_score = results['panel_metrics'].get('combined_diversity', 1.0)
        panel_recall_score = results['panel_metrics'].get('panel_recall', 0.0)

        results['overall_score'] = (
            0.5 * ranking_score + 
            0.3 * panel_recall_score + 
            0.2 * min(diversity_score, 5.0) / 5.0  # Normalize diversity to [0,1]
        )

        return results

    def compare_methods(self, 
                       method_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple methods and generate summary table.

        Args:
            method_results: List of results from evaluate_method()

        Returns:
            DataFrame with method comparison
        """
        comparison_data = []

        for result in method_results:
            row = {
                'method': result.get('method_name', 'Unknown'),
                'precision_at_10': result['ranking_metrics']['precision_at_10'],
                'precision_at_20': result['ranking_metrics']['precision_at_20'],
                'ndcg_at_20': result['ranking_metrics']['ndcg_at_20'],
                'mrr': result['ranking_metrics']['mean_reciprocal_rank'],
                'auc_pr': result['ranking_metrics']['auc_pr'],
                'panel_recall': result['panel_metrics']['panel_recall'],
                'network_diversity': result['panel_metrics']['network_diversity'],
                'overall_score': result['overall_score']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by overall score (descending)
        df = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)

        return df

    def statistical_significance_test(self, 
                                    method1_scores: List[float],
                                    method2_scores: List[float],
                                    test: str = 'wilcoxon') -> Tuple[float, float]:
        """
        Test statistical significance between two methods.

        Args:
            method1_scores: List of scores from method 1
            method2_scores: List of scores from method 2
            test: Statistical test ('wilcoxon', 'ttest')

        Returns:
            (statistic, p_value)
        """
        if len(method1_scores) != len(method2_scores):
            raise ValueError("Score lists must have equal length")

        if test == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(method1_scores, method2_scores)
        elif test == 'ttest':
            statistic, p_value = stats.ttest_rel(method1_scores, method2_scores)
        else:
            raise ValueError(f"Unknown test: {test}")

        return statistic, p_value
