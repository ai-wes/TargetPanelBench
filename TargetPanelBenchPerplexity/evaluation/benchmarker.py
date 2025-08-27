"""
Main benchmarker class for TargetPanelBench.

Orchestrates the evaluation of different target prioritization methods
and generates comprehensive benchmark results.
"""
import pandas as pd
import numpy as np
import json
import time
import logging
from typing import List, Dict, Optional, Any, Tuple, Type
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Import optimizers and evaluation modules
from base_optimizer import BaseTargetOptimizer
from simple_score_rank import SimpleScoreRankOptimizer, SimpleScoreRankFactory
from cma_es_optimizer import CMAESOptimizer
from pso_optimizer import PSOOptimizer, PSOFactory
from metrics import BenchmarkEvaluator, TargetPrioritizationMetrics, PanelDiversityMetrics


class TargetPanelBenchmarker:
    """
    Main benchmarking class that evaluates multiple target prioritization methods.

    Provides standardized evaluation across different algorithms using consistent
    datasets, metrics, and statistical tests.
    """

    def __init__(self, 
                 data_dir: str = "data/processed",
                 results_dir: str = "results",
                 random_seed: int = 42):
        """
        Initialize benchmarker.

        Args:
            data_dir: Directory containing processed benchmark data
            results_dir: Directory for saving results
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.logger = logging.getLogger("TargetPanelBench.Benchmarker")

        # Data storage
        self.evidence_matrix = None
        self.target_list = None
        self.ground_truth = None
        self.ppi_network = None

        # Results storage
        self.method_results = {}
        self.comparison_table = None

        # Evaluation configuration
        self.k_values = [10, 20, 50]
        self.panel_size = 15
        self.diversity_weight = 0.3

    def load_benchmark_data(self) -> 'TargetPanelBenchmarker':
        """
        Load benchmark datasets.

        Returns:
            self for method chaining
        """
        try:
            # Load evidence matrix
            evidence_path = self.data_dir / "evidence_matrix.csv"
            if evidence_path.exists():
                self.evidence_matrix = pd.read_csv(evidence_path, index_col=0)
                self.target_list = self.evidence_matrix.index.tolist()
                self.logger.info(f"Loaded evidence matrix: {self.evidence_matrix.shape}")
            else:
                raise FileNotFoundError(f"Evidence matrix not found at {evidence_path}")

            # Load ground truth
            ground_truth_path = self.data_dir / "ground_truth.json"
            if ground_truth_path.exists():
                with open(ground_truth_path, 'r') as f:
                    self.ground_truth = json.load(f)
                self.logger.info(f"Loaded ground truth: {len(self.ground_truth)} targets")
            else:
                raise FileNotFoundError(f"Ground truth not found at {ground_truth_path}")

            # Load PPI network (optional)
            ppi_path = self.data_dir / "ppi_network.csv"
            if ppi_path.exists():
                ppi_df = pd.read_csv(ppi_path)
                self.logger.info(f"Loaded PPI network: {len(ppi_df)} interactions")
                # Convert to NetworkX graph if available
                try:
                    import networkx as nx
                    self.ppi_network = nx.Graph()
                    for _, row in ppi_df.iterrows():
                        self.ppi_network.add_edge(
                            row['protein1'], row['protein2'], 
                            weight=row.get('score', 1.0)
                        )
                except ImportError:
                    self.logger.warning("NetworkX not available, PPI network features disabled")
                    self.ppi_network = None

            return self

        except Exception as e:
            self.logger.error(f"Failed to load benchmark data: {e}")
            raise

    def register_baseline_methods(self) -> 'TargetPanelBenchmarker':
        """
        Register standard baseline methods for comparison.

        Returns:
            self for method chaining
        """
        self.baseline_methods = {}

        # Simple Score & Rank baselines
        self.baseline_methods['SimpleScoreRank_Equal'] = SimpleScoreRankFactory.create_equal_weights()
        self.baseline_methods['SimpleScoreRank_Genetics'] = SimpleScoreRankFactory.create_genetics_weighted()
        self.baseline_methods['SimpleScoreRank_NoDiv'] = SimpleScoreRankFactory.create_no_diversity()

        # CMA-ES optimizers
        self.baseline_methods['CMA-ES_Standard'] = CMAESOptimizer(
            population_size=None,  # Auto-size
            max_generations=50,
            fitness_function='precision_at_k',
            k=20
        )

        self.baseline_methods['CMA-ES_NDCG'] = CMAESOptimizer(
            population_size=None,
            max_generations=50,
            fitness_function='ndcg',
            k=20
        )

        # PSO optimizers
        self.baseline_methods['PSO_Standard'] = PSOFactory.create_standard()
        self.baseline_methods['PSO_Adaptive'] = PSOFactory.create_adaptive()
        self.baseline_methods['PSO_Exploration'] = PSOFactory.create_exploration_focused()

        self.logger.info(f"Registered {len(self.baseline_methods)} baseline methods")
        return self

    def run_single_method(self, 
                         method_name: str, 
                         method: BaseTargetOptimizer,
                         test_targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run evaluation for a single method.

        Args:
            method_name: Name of the method
            method: Optimizer instance
            test_targets: Optional list of test targets (default: all targets)

        Returns:
            Dict with method results
        """
        if test_targets is None:
            test_targets = self.target_list

        start_time = time.time()

        try:
            # Fit the method
            self.logger.info(f"Training {method_name}...")
            method.fit(
                targets=test_targets,
                evidence_matrix=self.evidence_matrix,
                ground_truth=self.ground_truth
            )

            # Rank targets
            self.logger.info(f"Ranking targets with {method_name}...")
            ranked_df = method.rank_targets(test_targets, self.evidence_matrix)
            ranked_df['method'] = method_name

            # Select panel
            self.logger.info(f"Selecting panel with {method_name}...")
            selected_panel = method.select_panel(
                ranked_df, 
                panel_size=self.panel_size,
                diversity_weight=self.diversity_weight
            )

            # Evaluate results
            evaluator = BenchmarkEvaluator(
                ground_truth=self.ground_truth,
                ppi_network=self.ppi_network,
                feature_matrix=self.evidence_matrix
            )

            results = evaluator.evaluate_method(
                ranked_df=ranked_df,
                selected_panel=selected_panel,
                k_values=self.k_values
            )

            # Add method metadata
            results['method_name'] = method_name
            results['runtime_seconds'] = time.time() - start_time
            results['parameters'] = method.get_params()
            results['ranked_targets'] = ranked_df.copy()
            results['selected_panel'] = selected_panel.copy()

            # Add convergence history if available
            if hasattr(method, 'get_convergence_history'):
                try:
                    results['convergence_history'] = method.get_convergence_history()
                except:
                    pass

            # Add learned weights if available
            if hasattr(method, 'get_learned_weights') or hasattr(method, 'get_optimized_weights'):
                try:
                    if hasattr(method, 'get_learned_weights'):
                        results['learned_weights'] = method.get_learned_weights()
                    else:
                        results['learned_weights'] = method.get_optimized_weights()
                except:
                    pass

            self.logger.info(f"Completed {method_name} in {results['runtime_seconds']:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Failed to evaluate {method_name}: {e}")
            return {
                'method_name': method_name,
                'error': str(e),
                'runtime_seconds': time.time() - start_time,
                'ranking_metrics': {},
                'panel_metrics': {},
                'overall_score': 0.0
            }

    def run_benchmark(self, 
                     methods: Optional[Dict[str, BaseTargetOptimizer]] = None,
                     parallel: bool = False,
                     max_workers: int = 4) -> 'TargetPanelBenchmarker':
        """
        Run complete benchmark evaluation.

        Args:
            methods: Dict of method names to optimizers (default: use registered baselines)
            parallel: Whether to run methods in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            self for method chaining
        """
        if methods is None:
            if not hasattr(self, 'baseline_methods'):
                self.register_baseline_methods()
            methods = self.baseline_methods

        self.logger.info(f"Starting benchmark with {len(methods)} methods...")

        if parallel and len(methods) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_method = {
                    executor.submit(self.run_single_method, name, method): name
                    for name, method in methods.items()
                }

                for future in as_completed(future_to_method):
                    method_name = future_to_method[future]
                    try:
                        results = future.result()
                        self.method_results[method_name] = results
                    except Exception as e:
                        self.logger.error(f"Method {method_name} failed: {e}")
        else:
            # Sequential execution
            for method_name, method in methods.items():
                results = self.run_single_method(method_name, method)
                self.method_results[method_name] = results

        # Generate comparison table
        self._generate_comparison_table()

        self.logger.info("Benchmark completed!")
        return self

    def add_archipelago_results(self, results_file: str = "results/archipelago_aea_results.json") -> 'TargetPanelBenchmarker':
        """
        Add pre-computed results from Archipelago AEA method.

        Args:
            results_file: Path to Archipelago AEA results file

        Returns:
            self for method chaining
        """
        results_path = Path(results_file)

        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    aea_results = json.load(f)

                self.method_results['Archipelago_AEA'] = aea_results
                self.logger.info("Added Archipelago AEA results to benchmark")

                # Regenerate comparison table
                if self.method_results:
                    self._generate_comparison_table()

            except Exception as e:
                self.logger.error(f"Failed to load Archipelago AEA results: {e}")
        else:
            self.logger.warning(f"Archipelago AEA results not found at {results_path}")

        return self

    def _generate_comparison_table(self):
        """Generate comparison table from method results."""
        if not self.method_results:
            return

        comparison_data = []

        for method_name, results in self.method_results.items():
            if 'error' in results:
                # Method failed
                row = {
                    'method': method_name,
                    'status': 'FAILED',
                    'error': results['error'],
                    'runtime': results.get('runtime_seconds', 0.0)
                }
            else:
                # Method succeeded
                ranking_metrics = results.get('ranking_metrics', {})
                panel_metrics = results.get('panel_metrics', {})

                row = {
                    'method': method_name,
                    'status': 'SUCCESS',
                    'precision_at_10': ranking_metrics.get('precision_at_10', 0.0),
                    'precision_at_20': ranking_metrics.get('precision_at_20', 0.0),
                    'ndcg_at_20': ranking_metrics.get('ndcg_at_20', 0.0),
                    'mrr': ranking_metrics.get('mean_reciprocal_rank', 0.0),
                    'auc_pr': ranking_metrics.get('auc_pr', 0.0),
                    'panel_recall': panel_metrics.get('panel_recall', 0.0),
                    'network_diversity': panel_metrics.get('network_diversity', 0.0),
                    'overall_score': results.get('overall_score', 0.0),
                    'runtime': results.get('runtime_seconds', 0.0)
                }

            comparison_data.append(row)

        self.comparison_table = pd.DataFrame(comparison_data)

        # Sort by overall score (descending)
        if 'overall_score' in self.comparison_table.columns:
            self.comparison_table = self.comparison_table.sort_values(
                'overall_score', ascending=False
            ).reset_index(drop=True)
            self.comparison_table['rank'] = range(1, len(self.comparison_table) + 1)

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of benchmark results.

        Returns:
            Dict with benchmark summary
        """
        summary = {
            'benchmark_info': {
                'num_methods': len(self.method_results),
                'num_targets': len(self.target_list) if self.target_list else 0,
                'num_ground_truth': len(self.ground_truth) if self.ground_truth else 0,
                'evidence_types': list(self.evidence_matrix.columns) if self.evidence_matrix is not None else [],
                'panel_size': self.panel_size,
                'k_values': self.k_values
            },
            'top_methods': [],
            'method_comparison': self.comparison_table.to_dict('records') if self.comparison_table is not None else []
        }

        # Get top 3 methods
        if self.comparison_table is not None:
            successful_methods = self.comparison_table[self.comparison_table['status'] == 'SUCCESS']
            if len(successful_methods) > 0:
                top_3 = successful_methods.head(3)
                summary['top_methods'] = top_3.to_dict('records')

        return summary

    def save_results(self, 
                    output_file: str = "benchmark_results.json",
                    save_detailed: bool = True) -> 'TargetPanelBenchmarker':
        """
        Save benchmark results to file.

        Args:
            output_file: Output file path
            save_detailed: Whether to save detailed method results

        Returns:
            self for method chaining
        """
        output_path = self.results_dir / output_file

        # Prepare results for JSON serialization
        results_to_save = {
            'benchmark_summary': self.get_results_summary(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'random_seed': self.random_seed,
                'k_values': self.k_values,
                'panel_size': self.panel_size,
                'diversity_weight': self.diversity_weight
            }
        }

        if save_detailed:
            # Clean method results for JSON serialization
            detailed_results = {}
            for method_name, results in self.method_results.items():
                cleaned_results = results.copy()

                # Convert DataFrames to dicts
                if 'ranked_targets' in cleaned_results and isinstance(cleaned_results['ranked_targets'], pd.DataFrame):
                    cleaned_results['ranked_targets'] = cleaned_results['ranked_targets'].to_dict('records')

                # Convert numpy arrays to lists
                for key, value in cleaned_results.items():
                    if isinstance(value, np.ndarray):
                        cleaned_results[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_results[key] = float(value)

                detailed_results[method_name] = cleaned_results

            results_to_save['detailed_results'] = detailed_results

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        self.logger.info(f"Results saved to {output_path}")

        # Also save comparison table as CSV
        if self.comparison_table is not None:
            csv_path = self.results_dir / "comparison_table.csv"
            self.comparison_table.to_csv(csv_path, index=False)
            self.logger.info(f"Comparison table saved to {csv_path}")

        return self

    def generate_report(self, template_file: Optional[str] = None) -> str:
        """
        Generate benchmark report.

        Args:
            template_file: Optional template file for report generation

        Returns:
            Report content as string
        """
        if not self.method_results:
            return "No benchmark results available."

        summary = self.get_results_summary()

        report = f"""
# TargetPanelBench Results Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Benchmark Configuration
- Number of methods evaluated: {summary['benchmark_info']['num_methods']}
- Number of targets: {summary['benchmark_info']['num_targets']}
- Ground truth targets: {summary['benchmark_info']['num_ground_truth']}
- Panel size: {summary['benchmark_info']['panel_size']}
- Evidence types: {', '.join(summary['benchmark_info']['evidence_types'])}

## Top Performing Methods
"""

        if summary['top_methods']:
            for i, method in enumerate(summary['top_methods'], 1):
                report += f"""
### {i}. {method['method']}
- Overall Score: {method['overall_score']:.4f}
- Precision@20: {method['precision_at_20']:.4f}
- Panel Recall: {method['panel_recall']:.4f}
- Network Diversity: {method['network_diversity']:.4f}
- Runtime: {method['runtime']:.2f}s
"""

        report += f"""
## Method Comparison

| Rank | Method | Overall Score | Precision@20 | Panel Recall | Runtime |
|------|--------|---------------|--------------|--------------|---------|
"""

        if self.comparison_table is not None:
            for _, row in self.comparison_table.iterrows():
                if row['status'] == 'SUCCESS':
                    report += f"| {row.get('rank', 'N/A')} | {row['method']} | {row['overall_score']:.4f} | {row['precision_at_20']:.4f} | {row['panel_recall']:.4f} | {row['runtime']:.2f}s |\n"

        report += f"""
## Key Findings

{self._generate_key_findings()}

## Methodology

This benchmark evaluates target prioritization methods on their ability to:
1. **Rank targets accurately**: Measured by Precision@K, NDCG, and MRR
2. **Select diverse panels**: Measured by network diversity and panel recall
3. **Balance performance and efficiency**: Overall score combines ranking and diversity

All methods were evaluated on the same dataset using identical evaluation metrics
to ensure fair comparison.
"""

        return report

    def _generate_key_findings(self) -> str:
        """Generate key findings section for report."""
        if not self.comparison_table is not None:
            return "No results available for analysis."

        successful_methods = self.comparison_table[self.comparison_table['status'] == 'SUCCESS']

        if len(successful_methods) == 0:
            return "No methods completed successfully."

        findings = []

        # Best overall method
        best_method = successful_methods.iloc[0]
        findings.append(f"- **{best_method['method']}** achieved the highest overall score ({best_method['overall_score']:.4f})")

        # Best ranking method
        best_ranking = successful_methods.loc[successful_methods['precision_at_20'].idxmax()]
        if best_ranking['method'] != best_method['method']:
            findings.append(f"- **{best_ranking['method']}** achieved the best ranking performance (Precision@20: {best_ranking['precision_at_20']:.4f})")

        # Best diversity method
        best_diversity = successful_methods.loc[successful_methods['network_diversity'].idxmax()]
        findings.append(f"- **{best_diversity['method']}** achieved the best network diversity ({best_diversity['network_diversity']:.4f})")

        # Runtime analysis
        fastest_method = successful_methods.loc[successful_methods['runtime'].idxmin()]
        findings.append(f"- **{fastest_method['method']}** was the fastest method ({fastest_method['runtime']:.2f}s)")

        # Simple vs complex methods
        simple_methods = successful_methods[successful_methods['method'].str.contains('Simple')]
        if len(simple_methods) > 0:
            best_simple = simple_methods.loc[simple_methods['overall_score'].idxmax()]
            findings.append(f"- Best simple baseline: **{best_simple['method']}** (score: {best_simple['overall_score']:.4f})")

        return '\n'.join(findings)


def main():
    """Main function for running benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize benchmarker
    benchmarker = TargetPanelBenchmarker()

    # Load data and run benchmark
    benchmarker.load_benchmark_data()
    benchmarker.register_baseline_methods()
    benchmarker.add_archipelago_results()
    benchmarker.run_benchmark()

    # Save results
    benchmarker.save_results()

    # Print summary
    summary = benchmarker.get_results_summary()
    print("\nBenchmark Summary:")
    print(f"Evaluated {summary['benchmark_info']['num_methods']} methods")
    print("\nTop 3 Methods:")
    for i, method in enumerate(summary['top_methods'][:3], 1):
        print(f"{i}. {method['method']} (Score: {method['overall_score']:.4f})")


if __name__ == "__main__":
    main()
