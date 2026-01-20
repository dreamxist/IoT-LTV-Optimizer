"""
OmniEvo - optimizacion de atribucion omnicanal con GA
proyecto de soft computing
"""

from omnievo.data_generator import DataGenerator, CHANNEL_CONFIG
from omnievo.genetic import GeneticOptimizer, OptimizationResult
from omnievo.fitness import calculate_rmse, calculate_fitness
from omnievo.baselines import (
    uniform_model, last_touch_model, random_model, compare_baselines,
)
from omnievo.visualization import (
    plot_convergence, plot_weights, plot_comparison,
    plot_prediction_scatter, plot_evolution_dashboard,
)
from omnievo.experiments import (
    grid_search, cross_validate, sensitivity_analysis,
    analyze_convergence, run_full_experiment,
    GridSearchResult, ExperimentResult,
)
from omnievo.benchmark import (
    run_benchmark, run_statistical_tests, generate_business_insights,
    generate_final_report, BenchmarkReport,
)

__version__ = "0.1.0"

# lo que se exporta con "from omnievo import *"
__all__ = [
    # data
    "DataGenerator",
    "CHANNEL_CONFIG",
    # ga
    "GeneticOptimizer",
    "OptimizationResult",
    # fitness
    "calculate_rmse",
    "calculate_fitness",
    # baselines
    "uniform_model",
    "last_touch_model",
    "random_model",
    "compare_baselines",
    # viz
    "plot_convergence",
    "plot_weights",
    "plot_comparison",
    "plot_prediction_scatter",
    "plot_evolution_dashboard",
    # experiments
    "grid_search",
    "cross_validate",
    "sensitivity_analysis",
    "analyze_convergence",
    "run_full_experiment",
    "GridSearchResult",
    "ExperimentResult",
    # benchmark
    "run_benchmark",
    "run_statistical_tests",
    "generate_business_insights",
    "generate_final_report",
    "BenchmarkReport",
]
