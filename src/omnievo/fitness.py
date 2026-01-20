"""
funciones de fitness para el GA
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def normalize_weights(weights):
    """normaliza pesos para que sumen 1"""
    w = np.array(weights)
    w = np.maximum(w, 0)  # no negativos

    total = w.sum()
    if total == 0:
        # caso raro pero hay que manejarlo
        return np.ones_like(w) / len(w)

    return w / total


def calculate_rmse(y_true, y_pred):
    """rmse basico"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_pearson(y_true, y_pred):
    """correlacion de pearson"""
    if len(y_true) < 2:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)
    return corr[0, 1]


def predict_ltv(X, weights, scale_factor=None):
    """
    predice LTV con modelo lineal simple:
    LTV_pred = X @ weights * scale
    """
    w = normalize_weights(weights)
    y_pred = X @ w

    if scale_factor is not None:
        y_pred = y_pred * scale_factor

    return y_pred


def calculate_fitness(individual, X, y, metric="rmse"):
    """
    funcion de fitness para DEAP
    retorna tupla porque DEAP lo requiere asi
    """
    weights = normalize_weights(np.array(individual))
    y_pred = predict_ltv(X, weights, scale_factor=y.max())

    if metric == "rmse":
        rmse = calculate_rmse(y, y_pred)
        # negativo porque DEAP maximiza
        return (-rmse,)
    elif metric == "pearson":
        r = calculate_pearson(y, y_pred)
        return (r,)
    else:
        raise ValueError(f"metrica '{metric}' no soportada")


def evaluate_model(weights, X, y):
    """evalua un modelo con varias metricas"""
    w = normalize_weights(weights)
    y_pred = predict_ltv(X, w, scale_factor=y.max())

    return {
        "rmse": calculate_rmse(y, y_pred),
        "pearson": calculate_pearson(y, y_pred),
        "mae": np.mean(np.abs(y - y_pred)),
        "mape": np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100,
        "weights": w.tolist(),
    }
