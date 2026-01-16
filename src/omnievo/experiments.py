"""
Módulo de Experimentación para Sprint 3.

Este módulo proporciona herramientas para:
- Grid Search de hiperparámetros
- Análisis de convergencia
- Validación cruzada (k-fold)
- Comparación estadística de resultados
"""

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from omnievo.genetic import GeneticOptimizer, OptimizationResult
from omnievo.fitness import calculate_rmse, calculate_pearson, predict_ltv
from omnievo.baselines import compare_baselines


@dataclass
class ExperimentResult:
    """Resultado de un experimento individual."""

    params: dict
    rmse_train: float
    rmse_test: float
    pearson_test: float
    best_weights: np.ndarray
    convergence_gen: int  # Generación donde convergió
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convierte a diccionario para DataFrame."""
        return {
            **self.params,
            "rmse_train": self.rmse_train,
            "rmse_test": self.rmse_test,
            "pearson_test": self.pearson_test,
            "convergence_gen": self.convergence_gen,
        }


@dataclass
class GridSearchResult:
    """Resultado completo del Grid Search."""

    results: list[ExperimentResult]
    best_result: ExperimentResult
    best_params: dict
    results_df: pd.DataFrame = None

    def __post_init__(self):
        self.results_df = pd.DataFrame([r.to_dict() for r in self.results])
        self.results_df = self.results_df.sort_values("rmse_test").reset_index(drop=True)


def detect_convergence(
    history: list,
    window: int = 10,
    threshold: float = 0.001,
) -> int:
    """
    Detecta en qué generación el AG convergió.

    Args:
        history: Lista de EvolutionStats
        window: Ventana de generaciones para evaluar
        threshold: Umbral de variación para considerar convergencia

    Returns:
        Generación de convergencia (-1 si no convergió)
    """
    if len(history) < window:
        return -1

    fitness_values = [s.best_fitness for s in history]

    for i in range(window, len(fitness_values)):
        recent = fitness_values[i - window:i]
        mean_val = np.mean(recent)
        if mean_val == 0:
            continue
        variation = np.std(recent) / (np.abs(mean_val) + 1e-8)
        if variation < threshold:
            return i - window  # Primera generación de la ventana convergente

    return -1  # No convergió


def analyze_convergence(result: OptimizationResult) -> dict:
    """
    Analiza las características de convergencia de una ejecución.

    Args:
        result: Resultado de optimización

    Returns:
        Diccionario con métricas de convergencia
    """
    history = result.history
    fitness_values = [-s.best_fitness for s in history]  # RMSE positivo
    avg_fitness = [-s.avg_fitness for s in history]

    # Mejora total
    initial_rmse = fitness_values[0]
    final_rmse = fitness_values[-1]
    improvement = (initial_rmse - final_rmse) / initial_rmse * 100

    # Velocidad de convergencia (generaciones para alcanzar 90% de mejora)
    target_90 = initial_rmse - 0.9 * (initial_rmse - final_rmse)
    gen_90 = -1
    for i, f in enumerate(fitness_values):
        if f <= target_90:
            gen_90 = i
            break

    # Diversidad (gap entre mejor y promedio)
    diversity = np.mean([avg - best for avg, best in zip(avg_fitness, fitness_values)])

    # Estabilidad (variación en últimas 10 generaciones)
    last_10 = fitness_values[-10:] if len(fitness_values) >= 10 else fitness_values
    stability = np.std(last_10)

    # Generación de convergencia
    conv_gen = detect_convergence(history)

    return {
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "improvement_pct": improvement,
        "gen_90_improvement": gen_90,
        "convergence_gen": conv_gen,
        "avg_diversity": diversity,
        "final_stability": stability,
        "total_generations": len(history),
        "converged": conv_gen > 0,
    }


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_grid: dict,
    n_runs: int = 1,
    verbose: bool = True,
) -> GridSearchResult:
    """
    Ejecuta Grid Search sobre hiperparámetros del AG.

    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        param_grid: Diccionario con listas de valores por parámetro
            Ejemplo: {
                "population_size": [30, 50, 100],
                "generations": [50, 100],
                "cxpb": [0.6, 0.8],
                "mutpb": [0.1, 0.2],
            }
        n_runs: Número de ejecuciones por configuración (para promediar)
        verbose: Mostrar progreso

    Returns:
        GridSearchResult con todos los resultados
    """
    # Generar todas las combinaciones
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    total_experiments = len(combinations) * n_runs
    if verbose:
        print(f"Grid Search: {len(combinations)} configuraciones × {n_runs} runs = {total_experiments} experimentos")
        print("=" * 60)

    results = []
    best_rmse = float("inf")
    best_result = None

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        run_rmses = []
        run_results = []

        for run in range(n_runs):
            # Configurar semilla diferente para cada run
            seed = 42 + run

            optimizer = GeneticOptimizer(
                n_channels=X_train.shape[1],
                random_state=seed,
                verbose=False,
                **params,
            )

            opt_result = optimizer.fit(X_train, y_train)

            # Evaluar en test
            y_pred_test = predict_ltv(X_test, opt_result.best_weights, scale_factor=y_test.max())
            rmse_test = calculate_rmse(y_test, y_pred_test)
            pearson_test = calculate_pearson(y_test, y_pred_test)

            # Evaluar en train
            y_pred_train = predict_ltv(X_train, opt_result.best_weights, scale_factor=y_train.max())
            rmse_train = calculate_rmse(y_train, y_pred_train)

            # Convergencia
            conv_gen = detect_convergence(opt_result.history)

            exp_result = ExperimentResult(
                params=params,
                rmse_train=rmse_train,
                rmse_test=rmse_test,
                pearson_test=pearson_test,
                best_weights=opt_result.best_weights,
                convergence_gen=conv_gen,
                history=opt_result.history,
            )

            run_results.append(exp_result)
            run_rmses.append(rmse_test)

        # Promediar si hay múltiples runs
        if n_runs > 1:
            avg_rmse = np.mean(run_rmses)
            best_run = run_results[np.argmin(run_rmses)]
            best_run.rmse_test = avg_rmse  # Usar promedio
            results.append(best_run)
        else:
            results.append(run_results[0])

        # Actualizar mejor
        if results[-1].rmse_test < best_rmse:
            best_rmse = results[-1].rmse_test
            best_result = results[-1]

        if verbose:
            status = "★" if results[-1] == best_result else " "
            print(f"[{i+1:3d}/{len(combinations)}] {status} RMSE={results[-1].rmse_test:.4f} | {params}")

    if verbose:
        print("=" * 60)
        print(f"Mejor configuración: RMSE={best_rmse:.4f}")
        print(f"Parámetros: {best_result.params}")

    return GridSearchResult(
        results=results,
        best_result=best_result,
        best_params=best_result.params,
    )


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    optimizer_params: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta validación cruzada k-fold.

    Args:
        X: Features completas
        y: Target completo
        k_folds: Número de folds
        optimizer_params: Parámetros para el GeneticOptimizer
        verbose: Mostrar progreso

    Returns:
        Diccionario con métricas por fold y agregadas
    """
    if optimizer_params is None:
        optimizer_params = {
            "population_size": 50,
            "generations": 50,
        }

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    all_weights = []

    if verbose:
        print(f"Validación Cruzada: {k_folds} folds")
        print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        optimizer = GeneticOptimizer(
            n_channels=X.shape[1],
            random_state=42 + fold,
            verbose=False,
            **optimizer_params,
        )

        result = optimizer.fit(X_train, y_train)

        # Evaluar en test fold
        y_pred = predict_ltv(X_test, result.best_weights, scale_factor=y_test.max())
        rmse = calculate_rmse(y_test, y_pred)
        pearson = calculate_pearson(y_test, y_pred)

        fold_results.append({
            "fold": fold + 1,
            "rmse": rmse,
            "pearson": pearson,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })
        all_weights.append(result.best_weights)

        if verbose:
            print(f"Fold {fold+1}/{k_folds}: RMSE={rmse:.4f}, Pearson={pearson:.4f}")

    # Métricas agregadas
    rmse_values = [f["rmse"] for f in fold_results]
    pearson_values = [f["pearson"] for f in fold_results]

    summary = {
        "folds": fold_results,
        "rmse_mean": np.mean(rmse_values),
        "rmse_std": np.std(rmse_values),
        "pearson_mean": np.mean(pearson_values),
        "pearson_std": np.std(pearson_values),
        "weights_mean": np.mean(all_weights, axis=0),
        "weights_std": np.std(all_weights, axis=0),
    }

    if verbose:
        print("=" * 60)
        print(f"RMSE:    {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
        print(f"Pearson: {summary['pearson_mean']:.4f} ± {summary['pearson_std']:.4f}")

    return summary


def sensitivity_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_params: dict,
    param_name: str,
    param_values: list,
    n_runs: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad para un hiperparámetro específico.

    Args:
        X_train, y_train, X_test, y_test: Datos
        base_params: Parámetros base del optimizador
        param_name: Nombre del parámetro a variar
        param_values: Lista de valores a probar
        n_runs: Runs por valor (para calcular media y std)
        verbose: Mostrar progreso

    Returns:
        DataFrame con resultados por valor del parámetro
    """
    results = []

    if verbose:
        print(f"Análisis de Sensibilidad: {param_name}")
        print("=" * 60)

    for value in param_values:
        params = {**base_params, param_name: value}
        rmses = []
        pearsons = []

        for run in range(n_runs):
            optimizer = GeneticOptimizer(
                n_channels=X_train.shape[1],
                random_state=42 + run,
                verbose=False,
                **params,
            )

            result = optimizer.fit(X_train, y_train)
            y_pred = predict_ltv(X_test, result.best_weights, scale_factor=y_test.max())

            rmses.append(calculate_rmse(y_test, y_pred))
            pearsons.append(calculate_pearson(y_test, y_pred))

        results.append({
            param_name: value,
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "pearson_mean": np.mean(pearsons),
            "pearson_std": np.std(pearsons),
        })

        if verbose:
            print(f"  {param_name}={value}: RMSE={results[-1]['rmse_mean']:.4f} ± {results[-1]['rmse_std']:.4f}")

    return pd.DataFrame(results)


def run_full_experiment(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
    test_size: float = 0.3,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta un experimento completo del Sprint 3.

    Incluye:
    1. Grid Search de hiperparámetros
    2. Validación cruzada con mejores parámetros
    3. Análisis de convergencia
    4. Comparación con baselines

    Args:
        X: Features
        y: Target (LTV real)
        channel_names: Nombres de los canales
        test_size: Proporción de test
        random_state: Semilla
        verbose: Mostrar progreso

    Returns:
        Diccionario con todos los resultados
    """
    from sklearn.model_selection import train_test_split

    if verbose:
        print("=" * 70)
        print("EXPERIMENTO COMPLETO - SPRINT 3")
        print("=" * 70)

    # Split inicial
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print(f"\nDatos: {len(X)} usuarios ({len(X_train)} train / {len(X_test)} test)")

    # 1. Grid Search
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 1: Grid Search de Hiperparámetros")
        print("-" * 70)

    param_grid = {
        "population_size": [30, 50, 100],
        "generations": [50, 100],
        "cxpb": [0.6, 0.8],
        "mutpb": [0.1, 0.2],
    }

    gs_result = grid_search(
        X_train, y_train, X_test, y_test,
        param_grid=param_grid,
        n_runs=1,
        verbose=verbose,
    )

    # 2. Validación Cruzada con mejores parámetros
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 2: Validación Cruzada (5-fold)")
        print("-" * 70)

    cv_result = cross_validate(
        X, y,
        k_folds=5,
        optimizer_params=gs_result.best_params,
        verbose=verbose,
    )

    # 3. Análisis de convergencia del mejor modelo
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 3: Análisis de Convergencia")
        print("-" * 70)

    conv_analysis = analyze_convergence(
        OptimizationResult(
            best_weights=gs_result.best_result.best_weights,
            best_fitness=-gs_result.best_result.rmse_test,
            history=gs_result.best_result.history,
            generations_run=len(gs_result.best_result.history),
        )
    )

    if verbose:
        print(f"  Mejora total: {conv_analysis['improvement_pct']:.1f}%")
        print(f"  Generación 90% mejora: {conv_analysis['gen_90_improvement']}")
        print(f"  Convergencia en gen: {conv_analysis['convergence_gen']}")
        print(f"  Estabilidad final: {conv_analysis['final_stability']:.4f}")

    # 4. Comparación con baselines
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 4: Comparación con Baselines")
        print("-" * 70)

    comparison = compare_baselines(
        X_test, y_test,
        ga_weights=gs_result.best_result.best_weights,
        channel_names=channel_names,
    )

    if verbose:
        print(comparison.to_string(index=False))

    # 5. Pesos finales
    if verbose:
        print("\n" + "-" * 70)
        print("PESOS DE ATRIBUCIÓN OPTIMIZADOS")
        print("-" * 70)
        weights = gs_result.best_result.best_weights
        for i, (ch, w) in enumerate(sorted(zip(channel_names, weights), key=lambda x: -x[1])):
            bar = "█" * int(w * 40)
            print(f"  {ch:20s} {w:.4f} ({w*100:5.1f}%) {bar}")

    return {
        "grid_search": gs_result,
        "cross_validation": cv_result,
        "convergence": conv_analysis,
        "baseline_comparison": comparison,
        "best_params": gs_result.best_params,
        "best_weights": gs_result.best_result.best_weights,
        "final_rmse": gs_result.best_result.rmse_test,
        "final_pearson": gs_result.best_result.pearson_test,
    }
