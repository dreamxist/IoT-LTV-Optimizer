# OmniEvo: Optimización de Atribución Omnicanal con Algoritmos Genéticos

Optimización metaheurística de modelos de atribución omnicanal en entornos IoT para la maximización de la precisión del Customer Lifetime Value (LTV).

## Problema

En marketing híbrido (digital + físico), determinar qué canal realmente impulsa el valor del cliente es un desafío conocido como el **Problema de Atribución**. Los modelos tradicionales como "Last Click" o "Linear" son arbitrarios y no capturan la complejidad de las interacciones omnicanal.

## Solución

Utilizar un **Algoritmo Genético** para evolucionar pesos de atribución óptimos que minimicen el error entre el LTV predicho y el LTV real, superando los modelos heurísticos tradicionales.

## Escenario de Simulación: Smart Music Festival

Generación de datos sintéticos integrando tres capas de datos:

| Capa | Canales | Tecnología |
|------|---------|------------|
| **Digital** | Facebook Ads, Email, Google Search | AdTech |
| **App** | App Opens, Ticket Purchase, Wallet Top-up | Backend |
| **Física (IoT)** | RFID Entry, Stage Presence, NFC Purchases | Sensores |

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd IoT-LTV-Optimizer

# Instalar dependencias
pip install -e .
```

## Uso Rápido

```python
from omnievo import DataGenerator, GeneticOptimizer, compare_baselines

# Generar datos sintéticos
generator = DataGenerator(n_users=1000)
df = generator.generate()

# Optimizar pesos con Algoritmo Genético
optimizer = GeneticOptimizer(
    population_size=50,
    generations=100
)
best_weights = optimizer.fit(df)

# Comparar con baselines
results = compare_baselines(df, best_weights)
```

## Estructura del Proyecto

```
.
├── src/omnievo/           # Código fuente modular
│   ├── data_generator.py  # Simulador de datos (Sprint 1)
│   ├── genetic.py         # Core del Algoritmo Genético (Sprint 2)
│   ├── fitness.py         # Funciones de fitness
│   ├── baselines.py       # Modelos baseline
│   └── visualization.py   # Gráficas y reportes
├── notebooks/             # Experimentos interactivos
├── docs/                  # Documentación del proyecto
├── tests/                 # Tests unitarios
└── results/               # Outputs de experimentos
```

## Documentación

- [Definición del Problema](docs/init.md) - Contexto y formulación matemática
- [Roadmap de Desarrollo](docs/roadmap.md) - Plan de sprints
- [Guía de Algoritmos Genéticos](docs/GA.md) - Referencia técnica

## Técnica de Resolución

- **Codificación**: Vector de pesos reales normalizados (Σwᵢ = 1)
- **Fitness**: Minimización del RMSE entre LTV predicho y LTV real
- **Selección**: Tournament Selection (k=3)
- **Cruce**: Blend Crossover (BLX-α)
- **Mutación**: Gaussian Mutation

## Roadmap

- [x] Sprint 1: Generador de datos sintéticos
- [x] Sprint 2: Core del Algoritmo Genético
- [ ] Sprint 3: Experimentación y tuning de hiperparámetros
- [ ] Sprint 4: Validación y benchmarking

## Referencias

- Paper Base: "Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method"
- Eiben & Smith (2015) - "Introduction to Evolutionary Computing"
- DEAP Documentation: https://deap.readthedocs.io/

## Licencia

Proyecto académico - Soft Computing
