# OmniEvo: Optimización de Atribución Omnicanal con Algoritmos Genéticos

> **Proyecto de Soft Computing** - Optimización metaheurística de modelos de atribución omnicanal en entornos IoT para la maximización de la precisión del Customer Lifetime Value (LTV).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DEAP](https://img.shields.io/badge/DEAP-1.4+-green.svg)](https://deap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](#)

---

## Paper Base

Este proyecto extiende y mejora la metodología propuesta en:

> **Pramono, P. P., Surjandari, I., & Laoh, E.** (2019). *"Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method"*. IEEE ICSSSM 2019. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/8887704/)

### Innovación de OmniEvo

| Aspecto | Paper Original | OmniEvo |
|---------|----------------|---------|
| **Variables** | LRFM (4 variables) | 9 canales omnicanal |
| **Determinación de pesos** | Fuzzy AHP (expertos) | **Algoritmo Genético** (data-driven) |
| **Objetivo** | Segmentación de clientes | Predicción de LTV |
| **Fuente de datos** | Transacciones | Digital + App + **IoT** |

**Contribución:** Reemplazamos la determinación subjetiva de pesos (Fuzzy AHP) por un proceso evolutivo que optimiza los pesos directamente desde los datos.

---

## Problema

En marketing híbrido (digital + físico), determinar qué canal realmente impulsa el valor del cliente es un desafío conocido como el **Problema de Atribución**. Los modelos tradicionales como "Last Click" o "Linear" son arbitrarios y no capturan la complejidad de las interacciones omnicanal.

## Solución

Utilizar un **Algoritmo Genético** para evolucionar pesos de atribución óptimos que minimicen el error entre el LTV predicho y el LTV real:

```
Datos Omnicanal → Algoritmo Genético → Pesos Óptimos → Predicción LTV
     [9 canales]    [50 ind × 100 gen]    [Σwᵢ = 1]      [RMSE ↓]
```

---

## Escenario de Simulación: Smart Music Festival

Generación de datos sintéticos integrando tres capas de datos:

| Capa | Canales | Tecnología |
|------|---------|------------|
| **Digital** | Facebook Ads, Email Opens, Web Visits | AdTech |
| **App** | App Opens, Ticket Purchase, Wallet Top-up | Backend |
| **Física (IoT)** | RFID Entry, Stage Dwell Time, NFC Purchases | Sensores |

---

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd IoT-LTV-Optimizer

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e .

# Ejecutar tests
pytest tests/ -v
```

## Uso Rápido

```python
from omnievo import DataGenerator, GeneticOptimizer, compare_baselines
from sklearn.model_selection import train_test_split

# 1. Generar datos sintéticos
generator = DataGenerator(n_users=1000, random_state=42)
df = generator.generate()
channels = generator.get_channel_names()

# 2. Preparar datos
X = df[channels].values
y = df['LTV_real'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. Optimizar con Algoritmo Genético
optimizer = GeneticOptimizer(
    population_size=50,
    generations=100,
    random_state=42
)
result = optimizer.fit(X_train, y_train)

# 4. Ver pesos optimizados
print("Pesos de Atribución:")
for ch, w in result.get_weights_dict(channels).items():
    print(f"  {ch}: {w:.3f} ({w*100:.1f}%)")

# 5. Comparar con baselines
comparison = compare_baselines(X_test, y_test, ga_weights=result.best_weights)
print(comparison)
```

---

## Estructura del Proyecto

```
.
├── src/omnievo/           # Código fuente modular
│   ├── data_generator.py  # Simulador de datos (Sprint 1)
│   ├── genetic.py         # Core del Algoritmo Genético (Sprint 2)
│   ├── fitness.py         # Funciones de fitness (RMSE, Pearson)
│   ├── baselines.py       # Modelos baseline (Uniforme, Last-Touch, etc.)
│   └── visualization.py   # Gráficas y dashboards
├── notebooks/             # Experimentos interactivos
├── docs/                  # Documentación del proyecto
│   ├── init.md            # Definición del problema
│   ├── paper_base.md      # Análisis del paper de referencia
│   ├── roadmap.md         # Plan de desarrollo
│   └── GA.md              # Guía de Algoritmos Genéticos
├── tests/                 # Tests unitarios (pytest)
└── results/               # Outputs de experimentos
```

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [Definición del Problema](docs/init.md) | Contexto, formulación matemática e hipótesis |
| [Análisis del Paper Base](docs/paper_base.md) | Comparación con metodología LRFM + Fuzzy AHP |
| [Roadmap de Desarrollo](docs/roadmap.md) | Plan de sprints detallado |
| [Guía de Algoritmos Genéticos](docs/GA.md) | Referencia técnica de AG |

---

## Técnica de Resolución

### Algoritmo Genético

| Componente | Implementación | Parámetros |
|------------|----------------|------------|
| **Codificación** | Vector de floats | n = número de canales |
| **Restricción** | Normalización | Σwᵢ = 1, wᵢ ≥ 0 |
| **Fitness** | -RMSE (minimizar error) | |
| **Selección** | Tournament | k=3 |
| **Cruce** | Blend Crossover | α=0.5 |
| **Mutación** | Gaussian | σ=0.1, indpb=0.3 |
| **Elitismo** | Top-k preservado | k=2 |

### Modelos Baseline

- **Uniforme (1/N):** Distribución equitativa
- **Last-Touch:** 100% al último canal
- **First-Touch:** 100% al primer canal
- **Position Decay:** Pesos decrecientes
- **Random:** Control negativo

---

## Resultados Preliminares

```
============================================================
COMPARACIÓN DE MODELOS (Test Set, 500 usuarios)
============================================================
            Modelo      RMSE   Pearson  Mejora vs Uniforme
Algoritmo Genético    27.24     0.88           15.4%
    Uniforme (1/N)    32.22     0.89            0.0%
            Random    32.42     0.85           -0.6%
        Last-Touch    32.86     0.83           -2.0%
============================================================
```

El AG logra una **mejora del 15.4%** en RMSE sobre el modelo uniforme.

---

## Roadmap

| Sprint | Objetivo | Estado |
|--------|----------|--------|
| Sprint 1 | Generador de datos sintéticos | ✅ Completado |
| Sprint 2 | Core del Algoritmo Genético | ✅ Completado |
| Sprint 3 | Experimentación y tuning | 🔄 En progreso |
| Sprint 4 | Validación y benchmarking | ⏳ Pendiente |

---

## Referencias

1. **Paper Base:** Pramono, P. P., Surjandari, I., & Laoh, E. (2019). *Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method*. IEEE ICSSSM 2019. [[DOI]](https://ieeexplore.ieee.org/document/8887704/)

2. **LRFM Model:** Alvandi, M., Fazli, S., & Abdoli, F. (2012). *K-Mean Clustering Method For Analysis Customer Lifetime Value With LRFM Relationship Model*.

3. **Algoritmos Genéticos:** Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing*. Springer.

4. **DEAP:** Fortin, F. A., et al. (2012). *DEAP: Evolutionary Algorithms Made Easy*. [[Docs]](https://deap.readthedocs.io/)

---

## Licencia

Proyecto académico - Soft Computing / Computación Evolutiva
