"""
modulo de experimentacion - grid search, cv, etc
Sprint 3 del proyecto
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from omnievo.genetic import GeneticOptimizer, OptimizationResult
from omnievo.fitness import calculate_rmse, calculate_pearson, predict_ltv
from omnievo.baselines import compare_baselines


# resultado de un experimento - uso dict en vez de dataclass
def _make_experiment_result(params, rmse_train, rmse_test, pearson_test,
                           best_weights, conv_gen, history):
    return {
        "params": params,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "pearson_test": pearson_test,
        "best_weights": best_weights,
        "convergence_gen": conv_gen,
        "history": history,
    }


def detect_convergence(history, window=10, thresh=0.001):
    """detecta cuando convergio el GA"""
    if len(history) < window:
        return -1

    # extraer fitness (puede ser dict o objeto)
    if isinstance(history[0], dict):
        fits = [h["best"] for h in history]
    else:
        fits = [h.best_fitness for h in history]

    for i in range(window, len(fits)):
        recent = fits[i-window:i]
        mean_val = np.mean(recent)
        if mean_val == 0:
            continue
        var = np.std(recent) / (abs(mean_val) + 1e-8)
        if var < thresh:
            return i - window

    return -1  # no convergio


def analyze_convergence(result):
    """analiza convergencia de una corrida"""
    # manejar ambos formatos (dict o OptimizationResult)
    if isinstance(result, dict):
        history = result["history"]
        if isinstance(history[0], dict):
            fits = [-h["best"] for h in history]  # rmse positivo
            avg_fits = [-h["avg"] for h in history]
        else:
            fits = [-h.best_fitness for h in history]
            avg_fits = [-h.avg_fitness for h in history]
    else:
        history = result.history
        fits = [-h.best_fitness for h in history]
        avg_fits = [-h.avg_fitness for h in history]

    initial = fits[0]
    final = fits[-1]
    improvement = (initial - final) / initial * 100

    # generacion para 90% de mejora
    target_90 = initial - 0.9 * (initial - final)
    gen_90 = -1
    for i, f in enumerate(fits):
        if f <= target_90:
            gen_90 = i
            break

    # diversidad
    diversity = np.mean([a - b for a, b in zip(avg_fits, fits)])

    # estabilidad
    last_10 = fits[-10:] if len(fits) >= 10 else fits
    stability = np.std(last_10)

    conv_gen = detect_convergence(history)

    return {
        "initial_rmse": initial,
        "final_rmse": final,
        "improvement_pct": improvement,
        "gen_90_improvement": gen_90,
        "convergence_gen": conv_gen,
        "avg_diversity": diversity,
        "final_stability": stability,
        "total_generations": len(history),
        "converged": conv_gen > 0,
    }


def grid_search(X_train, y_train, X_test, y_test, param_grid,
               n_runs=1, verbose=True):
    """
    grid search sobre hiperparametros
    """
    param_names = list(param_grid.keys())
    param_vals = list(param_grid.values())
    combos = list(itertools.product(*param_vals))

    total = len(combos) * n_runs
    if verbose:
        print(f"Grid Search: {len(combos)} configs x {n_runs} runs = {total} experimentos")
        print("=" * 60)

    results = []
    best_rmse = float("inf")
    best_result = None

    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))

        run_rmses = []
        run_results = []

        for run in range(n_runs):
            seed = 42 + run

            opt = GeneticOptimizer(
                n_channels=X_train.shape[1],
                random_state=seed,
                verbose=False,
                **params,
            )

            res = opt.fit(X_train, y_train)
            weights = res["best_weights"]

            # eval en test
            y_pred_test = predict_ltv(X_test, weights, scale_factor=y_test.max())
            rmse_test = calculate_rmse(y_test, y_pred_test)
            pearson_test = calculate_pearson(y_test, y_pred_test)

            # eval en train
            y_pred_train = predict_ltv(X_train, weights, scale_factor=y_train.max())
            rmse_train = calculate_rmse(y_train, y_pred_train)

            conv_gen = detect_convergence(res["history"])

            exp_res = _make_experiment_result(
                params, rmse_train, rmse_test, pearson_test,
                weights, conv_gen, res["history"]
            )

            run_results.append(exp_res)
            run_rmses.append(rmse_test)

        # promediar si hay varios runs
        if n_runs > 1:
            avg_rmse = np.mean(run_rmses)
            best_run = run_results[np.argmin(run_rmses)]
            best_run["rmse_test"] = avg_rmse
            results.append(best_run)
        else:
            results.append(run_results[0])

        # actualizar mejor
        if results[-1]["rmse_test"] < best_rmse:
            best_rmse = results[-1]["rmse_test"]
            best_result = results[-1]

        if verbose:
            star = "*" if results[-1] == best_result else " "
            print(f"[{i+1:3d}/{len(combos)}] {star} RMSE={results[-1]['rmse_test']:.4f} | {params}")

    if verbose:
        print("=" * 60)
        print(f"Mejor config: RMSE={best_rmse:.4f}")
        print(f"Params: {best_result['params']}")

    # armar resultado
    results_df = pd.DataFrame([
        {**r["params"], "rmse_train": r["rmse_train"],
         "rmse_test": r["rmse_test"], "pearson_test": r["pearson_test"],
         "convergence_gen": r["convergence_gen"]}
        for r in results
    ])
    results_df = results_df.sort_values("rmse_test").reset_index(drop=True)

    return {
        "results": results,
        "best_result": best_result,
        "best_params": best_result["params"],
        "results_df": results_df,
    }


def cross_validate(X, y, k_folds=5, optimizer_params=None, verbose=True):
    """k-fold cross validation"""
    if optimizer_params is None:
        optimizer_params = {"population_size": 50, "generations": 50}

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    all_weights = []

    if verbose:
        print(f"Cross Validation: {k_folds} folds")
        print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        opt = GeneticOptimizer(
            n_channels=X.shape[1],
            random_state=42 + fold,
            verbose=False,
            **optimizer_params,
        )

        res = opt.fit(X_tr, y_tr)
        weights = res["best_weights"]

        y_pred = predict_ltv(X_te, weights, scale_factor=y_te.max())
        rmse = calculate_rmse(y_te, y_pred)
        pearson = calculate_pearson(y_te, y_pred)

        fold_results.append({
            "fold": fold + 1,
            "rmse": rmse,
            "pearson": pearson,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })
        all_weights.append(weights)

        if verbose:
            print(f"Fold {fold+1}/{k_folds}: RMSE={rmse:.4f}, Pearson={pearson:.4f}")

    # resumen
    rmses = [f["rmse"] for f in fold_results]
    pearsons = [f["pearson"] for f in fold_results]

    summary = {
        "folds": fold_results,
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "pearson_mean": np.mean(pearsons),
        "pearson_std": np.std(pearsons),
        "weights_mean": np.mean(all_weights, axis=0),
        "weights_std": np.std(all_weights, axis=0),
    }

    if verbose:
        print("=" * 60)
        print(f"RMSE:    {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f}")
        print(f"Pearson: {summary['pearson_mean']:.4f} +/- {summary['pearson_std']:.4f}")

    return summary


def sensitivity_analysis(X_train, y_train, X_test, y_test,
                        base_params, param_name, param_values,
                        n_runs=3, verbose=True):
    """analisis de sensibilidad para un parametro"""
    results = []

    if verbose:
        print(f"Sensitivity Analysis: {param_name}")
        print("=" * 60)

    for val in param_values:
        params = {**base_params, param_name: val}
        rmses = []
        pearsons = []

        for run in range(n_runs):
            opt = GeneticOptimizer(
                n_channels=X_train.shape[1],
                random_state=42 + run,
                verbose=False,
                **params,
            )

            res = opt.fit(X_train, y_train)
            y_pred = predict_ltv(X_test, res["best_weights"],
                               scale_factor=y_test.max())

            rmses.append(calculate_rmse(y_test, y_pred))
            pearsons.append(calculate_pearson(y_test, y_pred))

        results.append({
            param_name: val,
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "pearson_mean": np.mean(pearsons),
            "pearson_std": np.std(pearsons),
        })

        if verbose:
            print(f"  {param_name}={val}: RMSE={results[-1]['rmse_mean']:.4f} +/- {results[-1]['rmse_std']:.4f}")

    return pd.DataFrame(results)


def run_full_experiment(X, y, channel_names, test_size=0.3,
                       random_state=42, verbose=True):
    """
    experimento completo del sprint 3
    """
    from sklearn.model_selection import train_test_split

    if verbose:
        print("=" * 70)
        print("EXPERIMENTO COMPLETO - SPRINT 3")
        print("=" * 70)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print(f"\nDatos: {len(X)} users ({len(X_train)} train / {len(X_test)} test)")

    # 1. grid search
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 1: Grid Search")
        print("-" * 70)

    param_grid = {
        "population_size": [30, 50, 100],
        "generations": [50, 100],
        "cxpb": [0.6, 0.8],
        "mutpb": [0.1, 0.2],
    }

    gs = grid_search(X_train, y_train, X_test, y_test,
                    param_grid=param_grid, n_runs=1, verbose=verbose)

    # 2. cross validation
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 2: Cross Validation (5-fold)")
        print("-" * 70)

    cv = cross_validate(X, y, k_folds=5, optimizer_params=gs["best_params"],
                       verbose=verbose)

    # 3. convergencia
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 3: Analisis de Convergencia")
        print("-" * 70)

    # crear wrapper para analyze_convergence
    best = gs["best_result"]
    conv = analyze_convergence({
        "history": best["history"],
    })

    if verbose:
        print(f"  Mejora total: {conv['improvement_pct']:.1f}%")
        print(f"  Gen 90% mejora: {conv['gen_90_improvement']}")
        print(f"  Convergencia en: {conv['convergence_gen']}")

    # 4. baselines
    if verbose:
        print("\n" + "-" * 70)
        print("FASE 4: Comparacion con Baselines")
        print("-" * 70)

    comparison = compare_baselines(X_test, y_test,
                                  ga_weights=best["best_weights"],
                                  channel_names=channel_names)

    if verbose:
        print(comparison.to_string(index=False))

    # 5. pesos finales
    if verbose:
        print("\n" + "-" * 70)
        print("PESOS DE ATRIBUCION")
        print("-" * 70)
        weights = best["best_weights"]
        for ch, w in sorted(zip(channel_names, weights), key=lambda x: -x[1]):
            bar = "#" * int(w * 40)
            print(f"  {ch:20s} {w:.4f} ({w*100:5.1f}%) {bar}")

    return {
        "grid_search": gs,
        "cross_validation": cv,
        "convergence": conv,
        "baseline_comparison": comparison,
        "best_params": gs["best_params"],
        "best_weights": best["best_weights"],
        "final_rmse": best["rmse_test"],
        "final_pearson": best["pearson_test"],
    }


# tipos para compatibilidad
class GridSearchResult:
    """wrapper para resultados de grid search"""
    def __init__(self, result_dict):
        self.results = result_dict["results"]
        self.best_result = result_dict["best_result"]
        self.best_params = result_dict["best_params"]
        self.results_df = result_dict["results_df"]


class ExperimentResult:
    """wrapper para resultado de experimento"""
    def __init__(self, result_dict):
        for k, v in result_dict.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            **self.params,
            "rmse_train": self.rmse_train,
            "rmse_test": self.rmse_test,
            "pearson_test": self.pearson_test,
            "convergence_gen": self.convergence_gen,
        }
