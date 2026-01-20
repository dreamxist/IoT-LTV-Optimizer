# OmniEvo

Optimizacion de atribucion omnicanal usando algoritmos geneticos.

Proyecto de Soft Computing - usa GA para encontrar los pesos optimos de atribucion que predicen el Customer Lifetime Value (LTV) en entornos con datos digitales + IoT.

## Que es esto?

En marketing omnicanal, el problema de atribucion es: cual canal realmente genera valor? Los modelos clasicos (last-click, linear, etc) son arbitrarios. Este proyecto usa un algoritmo genetico para aprender los pesos directamente de los datos.

```
Datos Omnicanal -> GA -> Pesos Optimos -> Prediccion LTV
   [9 canales]    [evoluciona]  [suman 1]     [minimiza error]
```

## Basado en

Paper de Pramono et al (2019) sobre segmentacion con LRFM + Fuzzy AHP, pero cambiamos:
- Fuzzy AHP (subjetivo) -> **Algoritmo Genetico** (data-driven)
- 4 variables LRFM -> 9 canales omnicanal (digital + app + IoT)

## Setup

```bash
git clone <repo>
cd IoT-LTV-Optimizer

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# correr tests
pytest tests/ -v
```

## Uso rapido

```python
from omnievo import DataGenerator, GeneticOptimizer, compare_baselines
from sklearn.model_selection import train_test_split

# generar datos
gen = DataGenerator(n_users=1000, random_state=42)
df = gen.generate()
channels = gen.get_channel_names()

# preparar
X = df[channels].values
y = df['LTV_real'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# optimizar
opt = GeneticOptimizer(population_size=50, generations=100)
result = opt.fit(X_train, y_train)

# ver pesos
print("Pesos:")
for ch, w in zip(channels, result["best_weights"]):
    print(f"  {ch}: {w:.3f}")

# comparar con baselines
print(compare_baselines(X_test, y_test, ga_weights=result["best_weights"]))
```

## Datos simulados: Smart Music Festival

Tres capas de datos:

| Capa | Canales |
|------|---------|
| Digital | facebook_ads, email_opens, web_visits |
| App | app_opens, ticket_purchase, wallet_topup |
| IoT | rfid_entry, stage_dwell_time, nfc_purchases |

## Estructura

```
src/omnievo/
  data_generator.py   # genera datos sinteticos
  genetic.py          # algoritmo genetico (DEAP)
  fitness.py          # funciones de fitness
  baselines.py        # modelos baseline para comparar
  experiments.py      # grid search, cross validation
  benchmark.py        # tests estadisticos, reportes
  visualization.py    # graficas

notebooks/            # ejemplos interactivos
tests/               # pytest
docs/                # documentacion
```

## El GA

| Componente | Config |
|------------|--------|
| Codificacion | vector de floats |
| Restriccion | pesos suman 1, no negativos |
| Fitness | -RMSE (minimizar) |
| Seleccion | tournament (k=3) |
| Cruce | blend crossover |
| Mutacion | gaussiana |
| Elitismo | top 2 |

## Baselines

- Uniforme (1/N)
- Last-Touch
- First-Touch
- Position Decay
- Random

## Resultados

El GA mejora ~15-23% vs el modelo uniforme en RMSE.

```
Modelo              RMSE    Mejora
----------------------------------
GA                 23.99    23.0%
Uniforme           31.17     0.0%
Last-Touch         45.23   -45.1%
```

## Notebooks

- `01_intro.ipynb` - ejemplo basico
- `02_tuning.ipynb` - grid search y cross validation
- `03_validacion.ipynb` - tests estadisticos y benchmark

## Docs

- `docs/init.md` - definicion del problema
- `docs/paper_base.md` - analisis del paper original
- `docs/GA.md` - guia de algoritmos geneticos
- `docs/roadmap.md` - plan de desarrollo

## Referencias

- Pramono et al (2019) - LRFM + Fuzzy AHP para segmentacion
- DEAP - libreria de algoritmos evolutivos

