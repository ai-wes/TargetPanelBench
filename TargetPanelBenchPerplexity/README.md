# TargetPanelBench: A Benchmark for Target Prioritization & Panel Design

**A fair, reproducible, and open-source framework for evaluating computational methods for drug target prioritization and panel design.**

## Overview

TargetPanelBench is a comprehensive benchmarking framework that standardizes how the industry measures the effectiveness of target narrowing algorithms. Instead of just claiming superiority, we prove it with auditable data and code.

### Core Principle: "Show, Don't Just Tell"

All methods are evaluated on the same public datasets using identical metrics, ensuring fair and transparent comparisons.

## Key Features

- **🔬 Standardized Evaluation**: Consistent metrics across all methods
- **📊 Comprehensive Datasets**: Public data from Open Targets, ChEMBL, STRING
- **🎯 Dual Assessment**: Both ranking quality and panel diversity
- **🔄 Reproducible**: All code and data are openly available
- **📈 Multiple Baselines**: From simple scoring to advanced evolutionary algorithms

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ArchipelagoAnalytics/TargetPanelBench.git
cd TargetPanelBench
pip install -r requirements.txt
```

### 2. Download Benchmark Data

```bash
python -m data.download_data
```

### 3. Run the Benchmark

```python
from evaluation.benchmarker import TargetPanelBenchmarker

# Initialize benchmarker
benchmarker = TargetPanelBenchmarker()

# Load data and run all baseline methods
benchmarker.load_benchmark_data()
benchmarker.register_baseline_methods()
benchmarker.add_archipelago_results()
benchmarker.run_benchmark()

# Save results and generate report
benchmarker.save_results()
print(benchmarker.generate_report())
```

### 4. Explore Results

Open `notebooks/Run_Benchmark.ipynb` to interactively explore the benchmark results and generate publication-ready plots.

## Benchmark Components

### 1. Datasets (Public & Reproducible)

- **Disease Focus**: Alzheimer's Disease (EFO_0000249)
- **Target Universe**: ~500 protein targets from Open Targets
- **Ground Truth**: 15-20 clinically validated targets
- **Evidence Sources**:
  - Genetic associations (Open Targets)
  - Expression specificity (GTEx-derived)
  - Protein interactions (STRING)
  - Druggability scores (ChEMBL)
  - Literature evidence (PubMed-derived)

### 2. Evaluation Tasks

**Task 1: Target Prioritization (Ranking)**

- Rank 500 candidates to surface ground truth targets
- Metrics: Precision@K, NDCG@K, Mean Reciprocal Rank

**Task 2: Panel Design (Selection & Diversification)**

- Select 15-target panel balancing quality and diversity
- Metrics: Panel Recall, Network Diversity Score

### 3. Baseline Methods

| Method                       | Description                                   | Key Features                            |
| ---------------------------- | --------------------------------------------- | --------------------------------------- |
| **Simple Score & Rank**      | Weighted sum of normalized evidence           | Equal/genetics-weighted variants        |
| **CMA-ES**                   | Evolution strategy with covariance adaptation | Learns optimal evidence weights         |
| **PSO**                      | Particle swarm optimization                   | Adaptive parameters, swarm intelligence |
| **Morphantic Core AdaptEvo** | _Proprietary method_                          | Superior diversity optimization         |

## Results

### Performance Comparison

| Method                       | Precision@20 | Panel Recall | Network Diversity | Overall Score |
| ---------------------------- | ------------ | ------------ | ----------------- | ------------- |
| **Morphantic Core AdaptEvo** | **55.0%**    | **85.0%**    | **7.23**          | **0.712**     |
| CMA-ES Standard              | 42.0%        | 75.0%        | 4.20              | 0.580         |
| PSO Adaptive                 | 38.0%        | 70.0%        | 3.80              | 0.520         |
| Simple Score Rank            | 25.0%        | 60.0%        | 2.10              | 0.350         |

### Key Findings

- **Morphantic Core AdaptEvo outperforms** all baseline methods across ranking and diversity metrics
- **Diversity matters**: Methods that explicitly optimize for network diversity produce more robust panels
- **Evidence weighting is critical**: Learned weights significantly outperform equal weighting
- **Evolutionary algorithms excel**: CMA-ES and PSO both outperform simple approaches

## Framework Architecture

```
TargetPanelBench/
├── data/                    # Data download and processing
│   ├── download_data.py     # Public API integrations
│   └── process_data.py      # Data cleaning and curation
├── baselines/               # Baseline method implementations
│   ├── simple_score_rank.py # Simple weighted scoring
│   ├── cma_es_optimizer.py  # CMA-ES evolution strategy
│   └── pso_optimizer.py     # Particle swarm optimization
├── evaluation/              # Evaluation framework
│   ├── metrics.py           # Ranking and diversity metrics
│   └── benchmarker.py       # Main benchmark orchestrator
├── notebooks/               # Interactive analysis
│   └── Run_Benchmark.ipynb  # Complete walkthrough
└── results/                 # Benchmark outputs
    └── archipelago_aea_results.json
```

## Extending the Benchmark

### Adding New Methods

```python
from baselines.base_optimizer import BaseTargetOptimizer

class YourOptimizer(BaseTargetOptimizer):
    def fit(self, targets, evidence_matrix, ground_truth):
        # Implement your training logic
        pass

    def rank_targets(self, targets, evidence_matrix):
        # Return ranked DataFrame
        pass

    def select_panel(self, ranked_targets, panel_size=15):
        # Return diverse panel selection
        pass

# Add to benchmark
benchmarker = TargetPanelBenchmarker()
benchmarker.run_single_method("YourMethod", YourOptimizer())
```

### Adding New Diseases

```python
# Download data for different disease
from data.download_data import DataDownloader

downloader = DataDownloader(
    disease_id="MONDO_0007803",  # Parkinson's disease
    max_targets=500
)
datasets = downloader.download_all_data()
```

## API Reference

### Core Classes

- **`BaseTargetOptimizer`**: Abstract base class for all optimization methods
- **`TargetPanelBenchmarker`**: Main benchmarking orchestrator
- **`BenchmarkEvaluator`**: Comprehensive method evaluation
- **`DataDownloader`**: Public data integration

### Key Methods

- **`benchmarker.run_benchmark()`**: Execute complete benchmark
- **`benchmarker.add_method()`**: Add custom optimization method
- **`evaluator.evaluate_method()`**: Assess single method performance

## Citation

If you use TargetPanelBench in your research, please cite:

```bibtex
@software{targetpanelbench2024,
  title={TargetPanelBench: A Benchmark for Target Prioritization and Panel Design},
  author={Archipelago Analytics},
  year={2024},
  url={https://github.com/ArchipelagoAnalytics/TargetPanelBench}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

- **Issues**: [GitHub Issues](https://github.com/ArchipelagoAnalytics/TargetPanelBench/issues)
- **Email**: benchmark@archipelagoanalytics.com
- **Website**: [www.archipelagoanalytics.com](https://www.archipelagoanalytics.com)

---

**Built with ❤️ by Archipelago Analytics**
