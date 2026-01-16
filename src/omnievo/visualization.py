"""
Visualizaciones para OmniEvo.

Este módulo proporciona funciones para generar gráficos
de convergencia, comparación de modelos y análisis de resultados.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from omnievo.genetic import OptimizationResult


# Configuración de estilo
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#2ecc71",
    "secondary": "#3498db",
    "accent": "#9b59b6",
    "warning": "#e74c3c",
    "neutral": "#95a5a6",
}


def plot_convergence(
    result: OptimizationResult,
    figsize: tuple = (10, 5),
    show_std: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera gráfica de convergencia del Algoritmo Genético.

    Args:
        result: Resultado de la optimización
        figsize: Tamaño de la figura
        show_std: Mostrar banda de desviación estándar
        save_path: Ruta para guardar la figura

    Returns:
        Objeto Figure de matplotlib
    """
    generations = [s.generation for s in result.history]
    best_fitness = [-s.best_fitness for s in result.history]  # Convertir a RMSE positivo
    avg_fitness = [-s.avg_fitness for s in result.history]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        generations,
        best_fitness,
        label="Mejor RMSE",
        linewidth=2,
        color=COLORS["primary"],
    )
    ax.plot(
        generations,
        avg_fitness,
        label="RMSE Promedio",
        linewidth=2,
        color=COLORS["secondary"],
        alpha=0.7,
    )

    if show_std:
        std_fitness = [s.std_fitness for s in result.history]
        avg_array = np.array(avg_fitness)
        std_array = np.array(std_fitness)
        ax.fill_between(
            generations,
            avg_array - std_array,
            avg_array + std_array,
            alpha=0.2,
            color=COLORS["secondary"],
        )

    ax.set_xlabel("Generación", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Convergencia del Algoritmo Genético", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_weights(
    weights: np.ndarray,
    channel_names: list[str],
    figsize: tuple = (8, 5),
    horizontal: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera gráfica de barras con los pesos de atribución.

    Args:
        weights: Vector de pesos normalizados
        channel_names: Nombres de los canales
        figsize: Tamaño de la figura
        horizontal: Si True, barras horizontales
        save_path: Ruta para guardar la figura

    Returns:
        Objeto Figure de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ordenar por peso
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [channel_names[i] for i in sorted_idx]

    # Colores por tipo de canal (si se puede inferir)
    colors = [COLORS["accent"]] * len(weights)

    if horizontal:
        bars = ax.barh(sorted_names, sorted_weights, color=colors, alpha=0.8)
        ax.set_xlabel("Peso de Atribución", fontsize=11)

        # Añadir etiquetas de valor
        for bar, weight in zip(bars, sorted_weights):
            ax.text(
                weight + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{weight:.3f} ({weight*100:.1f}%)",
                va="center",
                fontsize=10,
            )
    else:
        bars = ax.bar(sorted_names, sorted_weights, color=colors, alpha=0.8)
        ax.set_ylabel("Peso de Atribución", fontsize=11)
        plt.xticks(rotation=45, ha="right")

        for bar, weight in zip(bars, sorted_weights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                weight + 0.01,
                f"{weight:.2f}",
                ha="center",
                fontsize=10,
            )

    ax.set_title("Pesos de Atribución (Algoritmo Genético)", fontsize=12, fontweight="bold")
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison(
    comparison_df: pd.DataFrame,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera gráfica comparativa de RMSE entre modelos.

    Args:
        comparison_df: DataFrame de compare_baselines()
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura

    Returns:
        Objeto Figure de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = comparison_df["Modelo"].tolist()
    rmse_values = comparison_df["RMSE"].tolist()

    # Colores: destacar AG si existe
    colors = []
    for model in models:
        if "Genético" in model:
            colors.append(COLORS["primary"])
        elif "Uniforme" in model:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["neutral"])

    bars = ax.bar(models, rmse_values, color=colors, alpha=0.8)

    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Comparación de Modelos de Atribución", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Rotar etiquetas si son largas
    plt.xticks(rotation=30, ha="right")

    # Añadir valores sobre las barras
    for bar, rmse in zip(bars, rmse_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rmse + 0.3,
            f"{rmse:.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    correlation: float = None,
    figsize: tuple = (7, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera scatter plot de LTV predicho vs real.

    Args:
        y_true: Valores reales de LTV
        y_pred: Valores predichos de LTV
        correlation: Coeficiente de Pearson (calculado si no se provee)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura

    Returns:
        Objeto Figure de matplotlib
    """
    if correlation is None:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        color=COLORS["secondary"],
        edgecolors="black",
        s=50,
    )

    # Línea de predicción perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Predicción perfecta",
    )

    ax.set_xlabel("LTV Real", fontsize=11)
    ax.set_ylabel("LTV Predicho", fontsize=11)
    ax.set_title(f"Predicción vs Realidad (r = {correlation:.3f})", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_evolution_dashboard(
    result: OptimizationResult,
    comparison_df: pd.DataFrame,
    channel_names: list[str],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera un dashboard completo con todas las visualizaciones.

    Args:
        result: Resultado de la optimización
        comparison_df: DataFrame de comparación de modelos
        channel_names: Nombres de los canales
        y_test: Valores reales de LTV (test set)
        y_pred: Valores predichos de LTV
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura

    Returns:
        Objeto Figure de matplotlib
    """
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Convergencia
    ax1 = fig.add_subplot(2, 2, 1)
    generations = [s.generation for s in result.history]
    best_fitness = [-s.best_fitness for s in result.history]
    avg_fitness = [-s.avg_fitness for s in result.history]

    ax1.plot(generations, best_fitness, label="Mejor RMSE", linewidth=2, color=COLORS["primary"])
    ax1.plot(
        generations, avg_fitness, label="RMSE Promedio", linewidth=2, color=COLORS["secondary"], alpha=0.7
    )
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Convergencia del AG", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Subplot 2: Comparación de modelos
    ax2 = fig.add_subplot(2, 2, 2)
    models = comparison_df["Modelo"].tolist()
    rmse_values = comparison_df["RMSE"].tolist()
    colors = [COLORS["primary"] if "Genético" in m else COLORS["neutral"] for m in models]
    ax2.bar(models, rmse_values, color=colors, alpha=0.8)
    ax2.set_ylabel("RMSE")
    ax2.set_title("Comparación de Modelos", fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Scatter predicción
    ax3 = fig.add_subplot(2, 2, 3)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    ax3.scatter(y_test, y_pred, alpha=0.6, color=COLORS["secondary"], edgecolors="black", s=30)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax3.set_xlabel("LTV Real")
    ax3.set_ylabel("LTV Predicho")
    ax3.set_title(f"Predicción (r = {correlation:.3f})", fontweight="bold")
    ax3.grid(alpha=0.3)

    # Subplot 4: Pesos finales
    ax4 = fig.add_subplot(2, 2, 4)
    sorted_idx = np.argsort(result.best_weights)[::-1]
    sorted_weights = result.best_weights[sorted_idx]
    sorted_names = [channel_names[i] for i in sorted_idx]
    ax4.barh(sorted_names, sorted_weights, color=COLORS["accent"], alpha=0.8)
    ax4.set_xlabel("Peso")
    ax4.set_title("Pesos de Atribución", fontweight="bold")
    ax4.grid(axis="x", alpha=0.3)

    plt.suptitle("OmniEvo - Resultados de Optimización", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
