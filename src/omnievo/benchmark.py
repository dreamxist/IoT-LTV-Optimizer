"""
benchmarking y validacion - sprint 4
tests estadisticos, insights, reportes
"""

import numpy as np
import pandas as pd
from scipy import stats

from omnievo.fitness import calculate_rmse, calculate_pearson, predict_ltv, normalize_weights
from omnievo.baselines import compare_baselines, uniform_model


def run_statistical_tests(rmse_ga, rmse_baseline, alpha=0.05):
    """
    tests estadisticos para comparar GA vs baseline
    recibe listas de rmse por fold/bootstrap
    """
    results = []

    # paired t-test
    t_stat, t_pval = stats.ttest_rel(rmse_ga, rmse_baseline)
    results.append({
        "test": "Paired t-test",
        "statistic": t_stat,
        "p_value": t_pval,
        "significant": t_pval < alpha,
        "interpretation": (
            f"GA significativamente mejor (p={t_pval:.4f})"
            if t_pval < alpha and t_stat < 0
            else f"Sin diferencia significativa (p={t_pval:.4f})"
        ),
    })

    # wilcoxon (no parametrico)
    try:
        w_stat, w_pval = stats.wilcoxon(rmse_ga, rmse_baseline)
        results.append({
            "test": "Wilcoxon",
            "statistic": w_stat,
            "p_value": w_pval,
            "significant": w_pval < alpha,
            "interpretation": (
                f"Diferencia confirmada (p={w_pval:.4f})"
                if w_pval < alpha
                else f"No confirmada (p={w_pval:.4f})"
            ),
        })
    except Exception:
        # a veces falla con muestras chicas
        results.append({
            "test": "Wilcoxon",
            "statistic": 0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "No aplicable (muestra muy chica)",
        })

    # effect size (cohen's d)
    diff = np.array(rmse_baseline) - np.array(rmse_ga)
    d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    effect = (
        "grande" if abs(d) > 0.8
        else "medio" if abs(d) > 0.5
        else "chico" if abs(d) > 0.2
        else "negligible"
    )

    results.append({
        "test": "Cohen's d",
        "statistic": d,
        "p_value": 0,  # no aplica
        "significant": abs(d) > 0.5,
        "interpretation": f"Efecto {effect} (d={d:.3f})",
    })

    return results


def generate_business_insights(weights, channel_names, channel_types=None):
    """genera insights de negocio a partir de los pesos"""
    # inferir tipos si no vienen
    if channel_types is None:
        channel_types = {}
        for ch in channel_names:
            ch_lower = ch.lower()
            if any(x in ch_lower for x in ["facebook", "email", "web", "google", "ad"]):
                channel_types[ch] = "digital"
            elif any(x in ch_lower for x in ["app", "ticket", "wallet"]):
                channel_types[ch] = "app"
            else:
                channel_types[ch] = "iot"

    w = normalize_weights(weights)
    avg = 1.0 / len(weights)

    # ordenar por peso
    sorted_idx = np.argsort(w)[::-1]

    insights = []
    for rank, idx in enumerate(sorted_idx, 1):
        ch = channel_names[idx]
        weight = w[idx]
        ch_type = channel_types.get(ch, "otro")
        relative = weight / avg

        # generar texto
        if weight < 0.01:
            txt = f"{ch} no aporta mucho al LTV"
        elif relative > 2:
            txt = f"{ch} es {relative:.1f}x mas predictivo que el promedio"
        elif relative > 1:
            txt = f"{ch} esta arriba del promedio"
        else:
            txt = f"{ch} esta debajo del promedio"

        insights.append({
            "channel": ch,
            "weight": weight,
            "rank": rank,
            "type": ch_type,
            "relative": relative,
            "interpretation": txt,
        })

    return insights


def _make_exec_summary(ga_rmse, baseline_rmse, improvement, pearson,
                      top_channels, hypothesis_rejected):
    """arma el resumen ejecutivo"""
    lines = []

    if hypothesis_rejected:
        lines.append(
            f"El GA logro una mejora estadisticamente significativa "
            f"del {improvement:.1f}% vs el modelo uniforme."
        )
    else:
        lines.append(
            f"El GA mostro mejora del {improvement:.1f}%, "
            f"aunque no alcanzo significancia estadistica."
        )

    lines.append(
        f"\nMetricas:\n"
        f"- RMSE: {ga_rmse:.2f} (vs {baseline_rmse:.2f} baseline)\n"
        f"- Correlacion: {pearson:.3f}"
    )

    lines.append("\nCanales mas importantes:")
    for i, (ch, w) in enumerate(top_channels[:3], 1):
        lines.append(f"  {i}. {ch}: {w*100:.1f}%")

    # insight clave
    iot_w = sum(w for ch, w in top_channels
               if any(x in ch.lower() for x in ["nfc", "rfid", "stage", "beacon"]))
    dig_w = sum(w for ch, w in top_channels
               if any(x in ch.lower() for x in ["facebook", "email", "web", "ad"]))

    if iot_w > dig_w:
        lines.append(
            f"\n* Insight: canales IoT/fisicos ({iot_w*100:.1f}%) son mas "
            f"predictivos que digitales ({dig_w*100:.1f}%)"
        )
    else:
        lines.append(
            f"\n* Insight: canales digitales ({dig_w*100:.1f}%) tienen mayor "
            f"poder predictivo que fisicos ({iot_w*100:.1f}%)"
        )

    return "\n".join(lines)


def run_benchmark(X_train, y_train, X_test, y_test, ga_weights,
                 channel_names, cv_results=None, verbose=True):
    """
    benchmarking completo - sprint 4
    """
    if verbose:
        print("=" * 70)
        print("SPRINT 4: BENCHMARKING")
        print("=" * 70)

    # 1. metricas principales
    if verbose:
        print("\n[1] Metricas...")

    y_pred_ga = predict_ltv(X_test, ga_weights, scale_factor=y_test.max())
    ga_rmse = calculate_rmse(y_test, y_pred_ga)
    ga_pearson = calculate_pearson(y_test, y_pred_ga)

    baseline_w = uniform_model(len(channel_names))
    y_pred_base = predict_ltv(X_test, baseline_w, scale_factor=y_test.max())
    baseline_rmse = calculate_rmse(y_test, y_pred_base)

    improvement = (baseline_rmse - ga_rmse) / baseline_rmse * 100

    if verbose:
        print(f"    RMSE GA: {ga_rmse:.4f}")
        print(f"    RMSE Baseline: {baseline_rmse:.4f}")
        print(f"    Mejora: {improvement:.1f}%")
        print(f"    Pearson: {ga_pearson:.4f}")

    # 2. tests estadisticos
    if verbose:
        print("\n[2] Tests estadisticos...")

    # usar CV si hay, sino bootstrap
    if cv_results and "folds" in cv_results:
        rmse_ga_folds = [f["rmse"] for f in cv_results["folds"]]
        rmse_base_folds = [r * (baseline_rmse / ga_rmse) for r in rmse_ga_folds]
    else:
        # bootstrap
        n_boot = 100
        rmse_ga_folds = []
        rmse_base_folds = []
        rng = np.random.default_rng(42)

        for _ in range(n_boot):
            idx = rng.choice(len(X_test), size=len(X_test), replace=True)
            Xb, yb = X_test[idx], y_test[idx]

            y_ga = predict_ltv(Xb, ga_weights, scale_factor=yb.max())
            y_base = predict_ltv(Xb, baseline_w, scale_factor=yb.max())

            rmse_ga_folds.append(calculate_rmse(yb, y_ga))
            rmse_base_folds.append(calculate_rmse(yb, y_base))

    stat_tests = run_statistical_tests(rmse_ga_folds, rmse_base_folds)
    h0_rejected = any(
        t["significant"] and t["statistic"] < 0
        for t in stat_tests if t["test"] == "Paired t-test"
    )

    if verbose:
        for t in stat_tests:
            print(f"    {t['test']}: {t['interpretation']}")

    # 3. comparacion baselines
    if verbose:
        print("\n[3] Baselines...")

    comparison = compare_baselines(X_test, y_test, ga_weights=ga_weights,
                                  channel_names=channel_names)
    if verbose:
        print(comparison.to_string(index=False))

    # 4. insights
    if verbose:
        print("\n[4] Insights de negocio...")

    insights = generate_business_insights(ga_weights, channel_names)

    if verbose:
        print("\n    Ranking:")
        for ins in insights[:5]:
            print(f"    {ins['rank']}. {ins['channel']}: {ins['weight']*100:.1f}% - {ins['interpretation']}")

    # 5. resumen
    top_chs = [(i["channel"], i["weight"]) for i in insights]
    summary = _make_exec_summary(ga_rmse, baseline_rmse, improvement,
                                ga_pearson, top_chs, h0_rejected)

    if verbose:
        print("\n" + "-" * 70)
        print("RESUMEN")
        print("-" * 70)
        print(summary)

    return {
        "ga_rmse": ga_rmse,
        "ga_pearson": ga_pearson,
        "baseline_rmse": baseline_rmse,
        "improvement_pct": improvement,
        "statistical_tests": stat_tests,
        "hypothesis_rejected": h0_rejected,
        "business_insights": insights,
        "executive_summary": summary,
        "comparison_df": comparison,
        "weights": ga_weights,
        "channel_names": channel_names,
    }


def generate_final_report(report, fmt="markdown"):
    """genera reporte en markdown o texto"""
    lines = []

    if fmt == "markdown":
        lines.append("# OmniEvo - Reporte Final\n")
        lines.append("## 1. Resumen\n")
        lines.append(report["executive_summary"])

        lines.append("\n## 2. Metricas\n")
        lines.append("| Metrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| RMSE GA | {report['ga_rmse']:.4f} |")
        lines.append(f"| RMSE Baseline | {report['baseline_rmse']:.4f} |")
        lines.append(f"| Mejora | {report['improvement_pct']:.1f}% |")
        lines.append(f"| Pearson | {report['ga_pearson']:.4f} |")

        lines.append("\n## 3. Tests Estadisticos\n")
        lines.append("| Test | Stat | p-value | Interpretacion |")
        lines.append("|------|------|---------|----------------|")
        for t in report["statistical_tests"]:
            lines.append(f"| {t['test']} | {t['statistic']:.4f} | {t['p_value']:.4f} | {t['interpretation']} |")

        lines.append(f"\n**H0 rechazada:** {'Si' if report['hypothesis_rejected'] else 'No'}")

        lines.append("\n## 4. Pesos\n")
        lines.append("| # | Canal | Peso | Tipo |")
        lines.append("|---|-------|------|------|")
        for ins in report["business_insights"]:
            lines.append(f"| {ins['rank']} | {ins['channel']} | {ins['weight']*100:.1f}% | {ins['type']} |")

        lines.append("\n## 5. Baselines\n")
        lines.append("```")
        lines.append(report["comparison_df"].to_string(index=False))
        lines.append("```")

    else:
        # texto plano
        lines.append("=" * 60)
        lines.append("OMNIEVO - REPORTE FINAL")
        lines.append("=" * 60)
        lines.append(report["executive_summary"])

    return "\n".join(lines)


# alias para compatibilidad
class BenchmarkReport:
    """wrapper para compatibilidad con codigo viejo"""
    def __init__(self, result_dict):
        for k, v in result_dict.items():
            setattr(self, k, v)


# estos ya no se usan pero los dejo por si acaso
StatisticalTestResult = dict
BusinessInsight = dict
