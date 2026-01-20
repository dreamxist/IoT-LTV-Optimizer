"""
modelos baseline para comparar con el GA
"""

import numpy as np
import pandas as pd

from omnievo.fitness import calculate_rmse, calculate_pearson, predict_ltv, normalize_weights


def uniform_model(n_channels):
    """modelo 1/N - todos los canales iguales"""
    return np.ones(n_channels) / n_channels


def last_touch_model(n_channels, last_idx=-1):
    """todo el credito al ultimo canal"""
    w = np.zeros(n_channels)
    w[last_idx] = 1.0
    return w


def first_touch_model(n_channels):
    """todo el credito al primer canal"""
    w = np.zeros(n_channels)
    w[0] = 1.0
    return w


def random_model(n_channels, seed=None):
    """pesos random (para baseline)"""
    if seed is not None:
        np.random.seed(seed)
    w = np.random.random(n_channels)
    return normalize_weights(w)


def position_decay_model(n_channels, decay=0.5):
    """pesos decrecientes"""
    w = np.array([decay**i for i in range(n_channels)])
    return normalize_weights(w)


def evaluate_baseline(weights, X, y, name="Baseline"):
    """evalua un baseline"""
    y_pred = predict_ltv(X, weights, scale_factor=y.max())
    return {
        "name": name,
        "weights": normalize_weights(weights),
        "rmse": calculate_rmse(y, y_pred),
        "pearson": calculate_pearson(y, y_pred),
    }


def compare_baselines(X, y, ga_weights=None, channel_names=None, random_state=42):
    """
    compara todos los baselines (y opcionalmente el GA)
    """
    n_ch = X.shape[1]
    results = []

    # todos los baselines
    baselines = [
        ("Uniforme (1/N)", uniform_model(n_ch)),
        ("Last-Touch", last_touch_model(n_ch)),
        ("First-Touch", first_touch_model(n_ch)),
        ("Position Decay", position_decay_model(n_ch)),
        ("Random", random_model(n_ch, random_state)),
    ]

    for name, weights in baselines:
        res = evaluate_baseline(weights, X, y, name)
        results.append(res)

    # agregar GA si hay
    if ga_weights is not None:
        ga_res = evaluate_baseline(ga_weights, X, y, "Algoritmo Genetico")
        results.append(ga_res)

    # armar dataframe
    df = pd.DataFrame([
        {"Modelo": r["name"], "RMSE": r["rmse"], "Pearson": r["pearson"]}
        for r in results
    ])

    df = df.sort_values("RMSE").reset_index(drop=True)

    # calcular mejora vs uniforme
    rmse_unif = df[df["Modelo"] == "Uniforme (1/N)"]["RMSE"].values[0]
    df["Mejora vs Uniforme (%)"] = ((rmse_unif - df["RMSE"]) / rmse_unif * 100).round(2)

    return df


def get_baseline_weights_table(n_channels, channel_names=None, random_state=42):
    """tabla con pesos de todos los baselines"""
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    baselines = {
        "Uniforme": uniform_model(n_channels),
        "Last-Touch": last_touch_model(n_channels),
        "First-Touch": first_touch_model(n_channels),
        "Position Decay": position_decay_model(n_channels),
        "Random": random_model(n_channels, random_state),
    }

    data = []
    for name, w in baselines.items():
        row = {"Modelo": name}
        row.update(dict(zip(channel_names, w)))
        data.append(row)

    return pd.DataFrame(data)


