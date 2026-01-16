"""
Funciones de Fitness para el Algoritmo Genético.

Este módulo contiene las funciones de evaluación que guían
la evolución del AG hacia soluciones óptimas de atribución.
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector de pesos para que sume 1.

    Esta es una restricción crítica del problema: los pesos de atribución
    deben representar una distribución de crédito válida.

    Args:
        weights: Vector de pesos sin normalizar

    Returns:
        Vector normalizado donde sum(weights) = 1
    """
    weights = np.array(weights)
    weights = np.maximum(weights, 0)  # Asegurar no negativos

    total = weights.sum()
    if total == 0:
        # Evitar división por cero: distribución uniforme
        return np.ones_like(weights) / len(weights)

    return weights / total


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Root Mean Square Error entre valores reales y predichos.

    Args:
        y_true: Valores reales de LTV
        y_pred: Valores predichos de LTV

    Returns:
        Valor de RMSE (menor es mejor)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el coeficiente de correlación de Pearson.

    Args:
        y_true: Valores reales de LTV
        y_pred: Valores predichos de LTV

    Returns:
        Coeficiente de Pearson (-1 a 1, mayor es mejor)
    """
    if len(y_true) < 2:
        return 0.0

    correlation_matrix = np.corrcoef(y_true, y_pred)
    return correlation_matrix[0, 1]


def predict_ltv(X: np.ndarray, weights: np.ndarray, scale_factor: float = None) -> np.ndarray:
    """
    Predice el LTV usando el modelo de atribución lineal.

    LTV_pred = X @ weights * scale_factor

    Args:
        X: Matriz de features (N usuarios x M canales)
        weights: Vector de pesos de atribución (M canales)
        scale_factor: Factor de escala para ajustar al rango del LTV real.
                     Si es None, se usa el máximo de X @ weights

    Returns:
        Vector de LTV predicho (N usuarios)
    """
    weights_norm = normalize_weights(weights)
    y_pred = X @ weights_norm

    if scale_factor is not None:
        y_pred = y_pred * scale_factor

    return y_pred


def calculate_fitness(
    individual: list,
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "rmse",
) -> tuple[float]:
    """
    Función de fitness para el Algoritmo Genético.

    Evalúa la calidad de un individuo (vector de pesos) midiendo
    qué tan bien predice el LTV real.

    Args:
        individual: Vector de pesos candidato [w1, w2, ..., wn]
        X: Matriz de features normalizadas (N x M)
        y: Vector de LTV real (N,)
        metric: Métrica a optimizar ("rmse" o "pearson")

    Returns:
        Tupla con el valor de fitness (formato DEAP).
        - Para RMSE: retorna -RMSE (negativo porque DEAP maximiza)
        - Para Pearson: retorna correlación directamente
    """
    # Normalizar pesos
    weights = normalize_weights(np.array(individual))

    # Calcular predicciones
    y_pred = predict_ltv(X, weights, scale_factor=y.max())

    if metric == "rmse":
        rmse = calculate_rmse(y, y_pred)
        # Retornamos negativo porque DEAP maximiza y queremos minimizar RMSE
        return (-rmse,)

    elif metric == "pearson":
        pearson = calculate_pearson(y, y_pred)
        return (pearson,)

    else:
        raise ValueError(f"Métrica no soportada: {metric}. Use 'rmse' o 'pearson'")


def evaluate_model(
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Evalúa un modelo de atribución completo con múltiples métricas.

    Args:
        weights: Vector de pesos de atribución
        X: Matriz de features
        y: Vector de LTV real

    Returns:
        Diccionario con métricas de evaluación
    """
    weights_norm = normalize_weights(weights)
    y_pred = predict_ltv(X, weights_norm, scale_factor=y.max())

    return {
        "rmse": calculate_rmse(y, y_pred),
        "pearson": calculate_pearson(y, y_pred),
        "mae": np.mean(np.abs(y - y_pred)),
        "mape": np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100,  # Porcentaje
        "weights": weights_norm.tolist(),
    }
