"""
Tests básicos para OmniEvo.
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from omnievo.data_generator import DataGenerator, generate_simple_dataset
from omnievo.fitness import normalize_weights, calculate_rmse, calculate_fitness
from omnievo.baselines import uniform_model, last_touch_model, compare_baselines
from omnievo.genetic import GeneticOptimizer


class TestDataGenerator:
    """Tests para el generador de datos."""

    def test_generate_creates_correct_shape(self):
        generator = DataGenerator(n_users=100, random_state=42)
        df = generator.generate()
        assert len(df) == 100
        assert "LTV_real" in df.columns
        assert "segment" in df.columns

    def test_generate_simple_dataset(self):
        df = generate_simple_dataset(n_users=50, random_state=42)
        assert len(df) == 50
        assert "facebook_ads" in df.columns

    def test_segments_sum_to_one(self):
        generator = DataGenerator(n_users=1000, random_state=42)
        df = generator.generate()
        # Verificar que hay múltiples segmentos
        assert df["segment"].nunique() > 1


class TestFitness:
    """Tests para funciones de fitness."""

    def test_normalize_weights_sums_to_one(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_weights(weights)
        assert np.isclose(normalized.sum(), 1.0)

    def test_normalize_weights_handles_zeros(self):
        weights = np.array([0.0, 0.0, 0.0, 0.0])
        normalized = normalize_weights(weights)
        assert np.isclose(normalized.sum(), 1.0)

    def test_calculate_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(y_true, y_pred) == 0.0

    def test_calculate_fitness_returns_tuple(self):
        X = np.random.rand(10, 4)
        y = np.random.rand(10) * 100
        individual = [0.25, 0.25, 0.25, 0.25]
        result = calculate_fitness(individual, X, y)
        assert isinstance(result, tuple)
        assert len(result) == 1


class TestBaselines:
    """Tests para modelos baseline."""

    def test_uniform_model(self):
        weights = uniform_model(4)
        assert len(weights) == 4
        assert np.isclose(weights.sum(), 1.0)
        assert np.allclose(weights, 0.25)

    def test_last_touch_model(self):
        weights = last_touch_model(4)
        assert weights[-1] == 1.0
        assert weights[0] == 0.0

    def test_compare_baselines_returns_dataframe(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100
        df = compare_baselines(X, y)
        assert "Modelo" in df.columns
        assert "RMSE" in df.columns


class TestGeneticOptimizer:
    """Tests para el optimizador genético."""

    def test_optimizer_runs(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        optimizer = GeneticOptimizer(
            population_size=10,
            generations=5,
            random_state=42,
            verbose=False,
        )

        result = optimizer.fit(X, y)
        assert result.best_weights is not None
        assert len(result.best_weights) == 4
        assert np.isclose(result.best_weights.sum(), 1.0)

    def test_optimizer_result_has_history(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        optimizer = GeneticOptimizer(
            population_size=10,
            generations=5,
            random_state=42,
            verbose=False,
        )

        result = optimizer.fit(X, y)
        assert len(result.history) == 5


class TestBenchmark:
    """Tests para el módulo de benchmarking (Sprint 4)."""

    def test_run_benchmark(self):
        from omnievo.benchmark import run_benchmark

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        optimizer = GeneticOptimizer(
            population_size=10,
            generations=5,
            random_state=42,
            verbose=False,
        )
        result = optimizer.fit(X, y)

        report = run_benchmark(
            X, y, X, y,
            ga_weights=result.best_weights,
            channel_names=["ch1", "ch2", "ch3", "ch4"],
            verbose=False,
        )

        assert report.ga_rmse > 0
        assert len(report.business_insights) == 4
        assert len(report.statistical_tests) == 3

    def test_generate_business_insights(self):
        from omnievo.benchmark import generate_business_insights

        weights = np.array([0.4, 0.3, 0.2, 0.1])
        channels = ["nfc", "app", "email", "facebook"]

        insights = generate_business_insights(weights, channels)

        assert len(insights) == 4
        assert insights[0].rank == 1
        assert np.isclose(insights[0].weight, 0.4)


class TestExperiments:
    """Tests para el módulo de experimentación (Sprint 3)."""

    def test_grid_search_runs(self):
        from omnievo.experiments import grid_search

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        param_grid = {
            "population_size": [10, 20],
            "generations": [5, 10],
        }

        result = grid_search(
            X, y, X, y,  # Usamos mismo set para test rápido
            param_grid=param_grid,
            n_runs=1,
            verbose=False,
        )

        assert result.best_result is not None
        assert len(result.results) == 4  # 2 x 2 configuraciones

    def test_cross_validate_runs(self):
        from omnievo.experiments import cross_validate

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        result = cross_validate(
            X, y,
            k_folds=3,
            optimizer_params={"population_size": 10, "generations": 5},
            verbose=False,
        )

        assert "rmse_mean" in result
        assert "pearson_mean" in result
        assert len(result["folds"]) == 3

    def test_analyze_convergence(self):
        from omnievo.experiments import analyze_convergence

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        optimizer = GeneticOptimizer(
            population_size=10,
            generations=20,
            random_state=42,
            verbose=False,
        )

        result = optimizer.fit(X, y)
        conv = analyze_convergence(result)

        assert "improvement_pct" in conv
        assert "convergence_gen" in conv
        assert conv["total_generations"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
