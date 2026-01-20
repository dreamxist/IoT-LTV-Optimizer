"""
graficas y visualizaciones
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# estilo
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("seaborn-whitegrid")  # versiones viejas

# colores
COLORS = {
    "primary": "#2ecc71",
    "secondary": "#3498db",
    "accent": "#9b59b6",
    "warning": "#e74c3c",
    "neutral": "#95a5a6",
}


def plot_convergence(result, figsize=(10, 5), show_std=True, save_path=None):
    """grafica de convergencia del GA"""
    # manejar dict o objeto
    if isinstance(result, dict):
        history = result["history"]
        if isinstance(history[0], dict):
            gens = [h["gen"] for h in history]
            best = [-h["best"] for h in history]
            avg = [-h["avg"] for h in history]
            std = [h.get("std", 0) for h in history]
        else:
            gens = [h.generation for h in history]
            best = [-h.best_fitness for h in history]
            avg = [-h.avg_fitness for h in history]
            std = [h.std_fitness for h in history]
    else:
        gens = [h.generation for h in result.history]
        best = [-h.best_fitness for h in result.history]
        avg = [-h.avg_fitness for h in result.history]
        std = [h.std_fitness for h in result.history]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(gens, best, label="Mejor RMSE", lw=2, color=COLORS["primary"])
    ax.plot(gens, avg, label="RMSE Promedio", lw=2, color=COLORS["secondary"], alpha=0.7)

    if show_std and any(std):
        avg_arr = np.array(avg)
        std_arr = np.array(std)
        ax.fill_between(gens, avg_arr - std_arr, avg_arr + std_arr,
                       alpha=0.2, color=COLORS["secondary"])

    ax.set_xlabel("Generacion")
    ax.set_ylabel("RMSE")
    ax.set_title("Convergencia del GA", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_weights(weights, channel_names, figsize=(8, 5), horizontal=True, save_path=None):
    """barras con pesos de atribucion"""
    fig, ax = plt.subplots(figsize=figsize)

    # ordenar por peso
    idx = np.argsort(weights)[::-1]
    sorted_w = weights[idx]
    sorted_names = [channel_names[i] for i in idx]

    colors = [COLORS["accent"]] * len(weights)

    if horizontal:
        bars = ax.barh(sorted_names, sorted_w, color=colors, alpha=0.8)
        ax.set_xlabel("Peso")

        # etiquetas
        for bar, w in zip(bars, sorted_w):
            ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{w:.3f} ({w*100:.1f}%)", va="center", fontsize=10)
    else:
        bars = ax.bar(sorted_names, sorted_w, color=colors, alpha=0.8)
        ax.set_ylabel("Peso")
        plt.xticks(rotation=45, ha="right")

        for bar, w in zip(bars, sorted_w):
            ax.text(bar.get_x() + bar.get_width()/2, w + 0.01,
                   f"{w:.2f}", ha="center", fontsize=10)

    ax.set_title("Pesos de Atribucion (GA)", fontweight="bold")
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison(comparison_df, figsize=(10, 5), save_path=None):
    """comparacion de RMSE entre modelos"""
    fig, ax = plt.subplots(figsize=figsize)

    models = comparison_df["Modelo"].tolist()
    rmses = comparison_df["RMSE"].tolist()

    # colores
    colors = []
    for m in models:
        if "Genet" in m or "GA" in m:
            colors.append(COLORS["primary"])
        elif "Uniforme" in m:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["neutral"])

    bars = ax.bar(models, rmses, color=colors, alpha=0.8)

    ax.set_ylabel("RMSE")
    ax.set_title("Comparacion de Modelos", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=30, ha="right")

    # valores
    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, rmse + 0.3,
               f"{rmse:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_prediction_scatter(y_true, y_pred, correlation=None,
                           figsize=(7, 6), save_path=None):
    """scatter de prediccion vs real"""
    if correlation is None:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.6, color=COLORS["secondary"],
              edgecolors="black", s=50)

    # linea de prediccion perfecta
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfecto")

    ax.set_xlabel("LTV Real")
    ax.set_ylabel("LTV Predicho")
    ax.set_title(f"Prediccion vs Realidad (r={correlation:.3f})", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_evolution_dashboard(result, comparison_df, channel_names,
                            y_test, y_pred, figsize=(16, 10), save_path=None):
    """dashboard completo"""
    fig = plt.figure(figsize=figsize)

    # extraer datos
    if isinstance(result, dict):
        history = result["history"]
        weights = result["best_weights"]
        if isinstance(history[0], dict):
            gens = [h["gen"] for h in history]
            best = [-h["best"] for h in history]
            avg = [-h["avg"] for h in history]
        else:
            gens = [h.generation for h in history]
            best = [-h.best_fitness for h in history]
            avg = [-h.avg_fitness for h in history]
    else:
        gens = [h.generation for h in result.history]
        best = [-h.best_fitness for h in result.history]
        avg = [-h.avg_fitness for h in result.history]
        weights = result.best_weights

    # 1. convergencia
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(gens, best, label="Mejor", lw=2, color=COLORS["primary"])
    ax1.plot(gens, avg, label="Promedio", lw=2, color=COLORS["secondary"], alpha=0.7)
    ax1.set_xlabel("Gen")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Convergencia", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. comparacion modelos
    ax2 = fig.add_subplot(2, 2, 2)
    models = comparison_df["Modelo"].tolist()
    rmses = comparison_df["RMSE"].tolist()
    colors = [COLORS["primary"] if "Genet" in m else COLORS["neutral"] for m in models]
    ax2.bar(models, rmses, color=colors, alpha=0.8)
    ax2.set_ylabel("RMSE")
    ax2.set_title("Modelos", fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    # 3. scatter
    ax3 = fig.add_subplot(2, 2, 3)
    r = np.corrcoef(y_test, y_pred)[0, 1]
    ax3.scatter(y_test, y_pred, alpha=0.6, color=COLORS["secondary"],
               edgecolors="black", s=30)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax3.set_xlabel("LTV Real")
    ax3.set_ylabel("LTV Pred")
    ax3.set_title(f"Prediccion (r={r:.3f})", fontweight="bold")
    ax3.grid(alpha=0.3)

    # 4. pesos
    ax4 = fig.add_subplot(2, 2, 4)
    idx = np.argsort(weights)[::-1]
    sorted_w = weights[idx]
    sorted_n = [channel_names[i] for i in idx]
    ax4.barh(sorted_n, sorted_w, color=COLORS["accent"], alpha=0.8)
    ax4.set_xlabel("Peso")
    ax4.set_title("Atribucion", fontweight="bold")
    ax4.grid(axis="x", alpha=0.3)

    plt.suptitle("OmniEvo - Resultados", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# para compatibilidad - importar OptimizationResult
try:
    from omnievo.genetic import OptimizationResult
except ImportError:
    OptimizationResult = dict
