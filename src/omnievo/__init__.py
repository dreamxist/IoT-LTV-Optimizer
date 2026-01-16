"""
OmniEvo: Optimización de Atribución Omnicanal con Algoritmos Genéticos

Este paquete implementa un sistema de optimización metaheurística para resolver
el problema de atribución en entornos omnicanal (digital + físico/IoT).
"""

from omnievo.data_generator import DataGenerator, CHANNEL_CONFIG
from omnievo.genetic import GeneticOptimizer
from omnievo.fitness import calculate_rmse, calculate_fitness
from omnievo.baselines import (
    uniform_model,
    last_touch_model,
    random_model,
    compare_baselines,
)
from omnievo.visualization import (
    plot_convergence,
    plot_weights,
    plot_comparison,
    plot_prediction_scatter,
    plot_evolution_dashboard,
)
from omnievo.experiments import (
    grid_search,
    cross_validate,
    sensitivity_analysis,
    analyze_convergence,
    run_full_experiment,
    GridSearchResult,
    ExperimentResult,
)

__version__ = "0.1.0"
__all__ = [
    # Data Generation
    "DataGenerator",
    "CHANNEL_CONFIG",
    # Genetic Algorithm
    "GeneticOptimizer",
    # Fitness
    "calculate_rmse",
    "calculate_fitness",
    # Baselines
    "uniform_model",
    "last_touch_model",
    "random_model",
    "compare_baselines",
    # Visualization
    "plot_convergence",
    "plot_weights",
    "plot_comparison",
    "plot_prediction_scatter",
    "plot_evolution_dashboard",
    # Experiments (Sprint 3)
    "grid_search",
    "cross_validate",
    "sensitivity_analysis",
    "analyze_convergence",
    "run_full_experiment",
    "GridSearchResult",
    "ExperimentResult",
]
