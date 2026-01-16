"""
Modelos Baseline para Comparación.

Este módulo implementa modelos de atribución tradicionales
que sirven como referencia para evaluar el desempeño del AG.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omnievo.fitness import calculate_rmse, calculate_pearson, predict_ltv, normalize_weights


@dataclass
class BaselineResult:
    """Resultado de evaluación de un modelo baseline."""

    name: str
    weights: np.ndarray
    rmse: float
    pearson: float

    def __repr__(self) -> str:
        return f"{self.name}: RMSE={self.rmse:.4f}, r={self.pearson:.4f}"


def uniform_model(n_channels: int) -> np.ndarray:
    """
    Modelo Uniforme: asigna peso igual a todos los canales (1/N).

    Este es el modelo más simple y representa la hipótesis de que
    todos los canales contribuyen igualmente al LTV.

    Args:
        n_channels: Número de canales

    Returns:
        Vector de pesos uniformes
    """
    return np.ones(n_channels) / n_channels


def last_touch_model(n_channels: int, last_channel_idx: int = -1) -> np.ndarray:
    """
    Modelo Last-Touch: asigna todo el crédito al último canal.

    Este modelo asume que solo el último punto de contacto
    antes de la conversión es relevante.

    Args:
        n_channels: Número de canales
        last_channel_idx: Índice del canal "último" (default: último)

    Returns:
        Vector de pesos one-hot
    """
    weights = np.zeros(n_channels)
    weights[last_channel_idx] = 1.0
    return weights


def first_touch_model(n_channels: int) -> np.ndarray:
    """
    Modelo First-Touch: asigna todo el crédito al primer canal.

    Este modelo asume que solo el primer punto de contacto
    (awareness) es relevante.

    Args:
        n_channels: Número de canales

    Returns:
        Vector de pesos one-hot
    """
    weights = np.zeros(n_channels)
    weights[0] = 1.0
    return weights


def random_model(n_channels: int, random_state: int = None) -> np.ndarray:
    """
    Modelo Random: asigna pesos aleatorios normalizados.

    Este modelo sirve como control negativo para verificar
    que el AG está aprendiendo algo significativo.

    Args:
        n_channels: Número de canales
        random_state: Semilla aleatoria

    Returns:
        Vector de pesos aleatorios normalizados
    """
    if random_state is not None:
        np.random.seed(random_state)

    weights = np.random.random(n_channels)
    return normalize_weights(weights)


def position_decay_model(n_channels: int, decay_rate: float = 0.5) -> np.ndarray:
    """
    Modelo Position Decay: pesos decrecientes por posición.

    Asigna más peso a los canales iniciales y menos a los finales.

    Args:
        n_channels: Número de canales
        decay_rate: Factor de decaimiento (0-1)

    Returns:
        Vector de pesos con decaimiento
    """
    weights = np.array([decay_rate**i for i in range(n_channels)])
    return normalize_weights(weights)


def evaluate_baseline(
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    name: str = "Baseline",
) -> BaselineResult:
    """
    Evalúa un modelo baseline dado sus pesos.

    Args:
        weights: Vector de pesos del modelo
        X: Matriz de features
        y: Vector de LTV real
        name: Nombre del modelo

    Returns:
        BaselineResult con métricas de evaluación
    """
    y_pred = predict_ltv(X, weights, scale_factor=y.max())

    return BaselineResult(
        name=name,
        weights=normalize_weights(weights),
        rmse=calculate_rmse(y, y_pred),
        pearson=calculate_pearson(y, y_pred),
    )


def compare_baselines(
    X: np.ndarray,
    y: np.ndarray,
    ga_weights: np.ndarray = None,
    channel_names: list[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compara múltiples modelos baseline con el AG (si se proporciona).

    Args:
        X: Matriz de features
        y: Vector de LTV real
        ga_weights: Pesos del AG (opcional)
        channel_names: Nombres de los canales
        random_state: Semilla aleatoria

    Returns:
        DataFrame con comparación de todos los modelos
    """
    n_channels = X.shape[1]
    results = []

    # Modelos baseline
    baselines = [
        ("Uniforme (1/N)", uniform_model(n_channels)),
        ("Last-Touch", last_touch_model(n_channels)),
        ("First-Touch", first_touch_model(n_channels)),
        ("Position Decay", position_decay_model(n_channels)),
        ("Random", random_model(n_channels, random_state)),
    ]

    for name, weights in baselines:
        result = evaluate_baseline(weights, X, y, name)
        results.append(result)

    # Agregar AG si se proporcionan pesos
    if ga_weights is not None:
        ga_result = evaluate_baseline(ga_weights, X, y, "Algoritmo Genético")
        results.append(ga_result)

    # Crear DataFrame de resultados
    df = pd.DataFrame([
        {
            "Modelo": r.name,
            "RMSE": r.rmse,
            "Pearson": r.pearson,
        }
        for r in results
    ])

    # Ordenar por RMSE (menor es mejor)
    df = df.sort_values("RMSE").reset_index(drop=True)

    # Calcular mejora porcentual vs Uniforme
    rmse_uniforme = df[df["Modelo"] == "Uniforme (1/N)"]["RMSE"].values[0]
    df["Mejora vs Uniforme (%)"] = ((rmse_uniforme - df["RMSE"]) / rmse_uniforme * 100).round(2)

    return df


def get_baseline_weights_table(
    n_channels: int,
    channel_names: list[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Genera una tabla con los pesos de todos los modelos baseline.

    Args:
        n_channels: Número de canales
        channel_names: Nombres de los canales
        random_state: Semilla aleatoria

    Returns:
        DataFrame con pesos por modelo y canal
    """
    if channel_names is None:
        channel_names = [f"Canal_{i}" for i in range(n_channels)]

    baselines = {
        "Uniforme": uniform_model(n_channels),
        "Last-Touch": last_touch_model(n_channels),
        "First-Touch": first_touch_model(n_channels),
        "Position Decay": position_decay_model(n_channels),
        "Random": random_model(n_channels, random_state),
    }

    data = []
    for name, weights in baselines.items():
        row = {"Modelo": name}
        row.update(dict(zip(channel_names, weights)))
        data.append(row)

    return pd.DataFrame(data)
