from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

# Part 1: Core Data Structures and Enums

class BenchmarkMetric(Enum):
    """Standard benchmarking metrics for evolutionary algorithms"""
    BEST_FITNESS = "best_fitness"
    CONVERGENCE_RATE = "convergence_rate"
    DIVERSITY_MEASURE = "diversity_measure"
    SUCCESS_RATE = "success_rate"
    FUNCTION_EVALUATIONS = "function_evaluations"
    WALL_TIME = "wall_time"
    HYPERVOLUME = "hypervolume"
    SPREAD_INDICATOR = "spread_indicator"

class AlgorithmType(Enum):
    """Types of evolutionary algorithms for comparison"""
    AEA_FULL = "AEA_Full"  # Full Archipelago with all bio-inspired features
    AEA_BASIC = "AEA_Basic"  # Basic multi-island without advanced features
    STANDARD_GA = "Standard_GA"  # Traditional genetic algorithm
    NSGA_II = "NSGA_II"  # Multi-objective NSGA-II
    DE = "Differential_Evolution"  # Differential Evolution
    PSO = "Particle_Swarm"  # Particle Swarm Optimization

@dataclass
class BenchmarkResult:
    """Results from a single algorithm run"""
    algorithm_name: str
    problem_name: str
    run_id: int
    final_fitness: float
    convergence_generation: int
    function_evaluations: int
    wall_time: float
    success: bool
    fitness_history: List[float]
    diversity_history: List[float]
    extra_metrics: Dict[str, float]

@dataclass  
class StatisticalTestResult:
    """Results from statistical significance testing"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    algorithm_a: str
    algorithm_b: str

print("[OK] Core data structures defined")
print("[OK] Benchmark metrics enumerated")
print("[OK] Algorithm types specified")
print("[OK] Result containers created")