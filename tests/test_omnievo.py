# -*- coding: utf-8 -*-
"""
tests basicos para omnievo
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
    def test_genera_shape_correcto(self):
        gen = DataGenerator(n_users=100, random_state=42)
        df = gen.generate()
        assert len(df) == 100
        assert "LTV_real" in df.columns
        assert "segment" in df.columns

    def test_simple_dataset(self):
        df = generate_simple_dataset(n_users=50, random_state=42)
        assert len(df) == 50
        assert "facebook_ads" in df.columns

    def test_hay_varios_segmentos(self):
        gen = DataGenerator(n_users=1000, random_state=42)
        df = gen.generate()
        assert df["segment"].nunique() > 1


class TestFitness:
    def test_normalize_suma_uno(self):
        w = np.array([1.0, 2.0, 3.0, 4.0])
        norm = normalize_weights(w)
        assert np.isclose(norm.sum(), 1.0)

    def test_normalize_con_ceros(self):
        w = np.array([0.0, 0.0, 0.0, 0.0])
        norm = normalize_weights(w)
        assert np.isclose(norm.sum(), 1.0)

    def test_rmse_perfecto(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(y, y) == 0.0

    def test_fitness_retorna_tupla(self):
        X = np.random.rand(10, 4)
        y = np.random.rand(10) * 100
        ind = [0.25, 0.25, 0.25, 0.25]
        res = calculate_fitness(ind, X, y)
        assert isinstance(res, tuple)
        assert len(res) == 1


class TestBaselines:
    def test_uniforme(self):
        w = uniform_model(4)
        assert len(w) == 4
        assert np.isclose(w.sum(), 1.0)
        assert np.allclose(w, 0.25)

    def test_last_touch(self):
        w = last_touch_model(4)
        assert w[-1] == 1.0
        assert w[0] == 0.0

    def test_compare_retorna_df(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100
        df = compare_baselines(X, y)
        assert "Modelo" in df.columns
        assert "RMSE" in df.columns


class TestGA:
    def test_optimizer_corre(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        opt = GeneticOptimizer(
            population_size=10,
            generations=5,
            random_state=42,
            verbose=False,
        )

        res = opt.fit(X, y)
        assert res["best_weights"] is not None
        assert len(res["best_weights"]) == 4
        assert np.isclose(res["best_weights"].sum(), 1.0)

    def test_tiene_history(self):
        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        opt = GeneticOptimizer(
            population_size=10,
            generations=5,
            random_state=42,
            verbose=False,
        )

        res = opt.fit(X, y)
        assert len(res["history"]) == 5


class TestBenchmark:
    def test_run_benchmark(self):
        from omnievo.benchmark import run_benchmark

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        opt = GeneticOptimizer(
            population_size=10,
            generations=5,
            random_state=42,
            verbose=False,
        )
        res = opt.fit(X, y)

        report = run_benchmark(
            X, y, X, y,
            ga_weights=res["best_weights"],
            channel_names=["ch1", "ch2", "ch3", "ch4"],
            verbose=False,
        )

        assert report["ga_rmse"] > 0
        assert len(report["business_insights"]) == 4
        assert len(report["statistical_tests"]) == 3

    def test_insights(self):
        from omnievo.benchmark import generate_business_insights

        w = np.array([0.4, 0.3, 0.2, 0.1])
        chs = ["nfc", "app", "email", "facebook"]

        insights = generate_business_insights(w, chs)

        assert len(insights) == 4
        assert insights[0]["rank"] == 1
        assert np.isclose(insights[0]["weight"], 0.4)


class TestExperiments:
    def test_grid_search(self):
        from omnievo.experiments import grid_search

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        params = {
            "population_size": [10, 20],
            "generations": [5, 10],
        }

        res = grid_search(X, y, X, y, param_grid=params, n_runs=1, verbose=False)

        assert res["best_result"] is not None
        assert len(res["results"]) == 4  # 2x2

    def test_cross_validate(self):
        from omnievo.experiments import cross_validate

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        res = cross_validate(
            X, y,
            k_folds=3,
            optimizer_params={"population_size": 10, "generations": 5},
            verbose=False,
        )

        assert "rmse_mean" in res
        assert "pearson_mean" in res
        assert len(res["folds"]) == 3

    def test_convergence(self):
        from omnievo.experiments import analyze_convergence

        X = np.random.rand(50, 4)
        y = np.random.rand(50) * 100

        opt = GeneticOptimizer(
            population_size=10,
            generations=20,
            random_state=42,
            verbose=False,
        )

        res = opt.fit(X, y)
        conv = analyze_convergence(res)

        assert "improvement_pct" in conv
        assert "convergence_gen" in conv
        assert conv["total_generations"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
