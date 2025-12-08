# Algoritmos Genéticos - Guía y Ejemplos

Repositorio con documentación completa y ejemplos prácticos de **Algoritmos Genéticos (GA)** implementados en Python usando DEAP.

## Contenido

- **[GA.md](GA.md)**: Guía completa de Algoritmos Genéticos
  - Conceptos fundamentales (fenotipo, genotipo, fitness)
  - Operadores genéticos (selección, cruzamiento, mutación)
  - Algoritmo paso a paso
  - Ejemplo: Problema de las N-Reinas
  - Implementación con DEAP
  - Ventajas y limitaciones
  - Referencias y ejercicios

## Requisitos

```bash
pip install deap
```

## Inicio Rápido

El documento GA.md incluye código completo y ejecutable del problema de las N-Reinas:

```python
from deap import creator, base, tools, algorithms
import random

# Ver GA.md para la implementación completa
```

## Temas Cubiertos

1. Introducción a Algoritmos Genéticos
2. Representación (genotipos y fenotipos)
3. Función de Fitness
4. Operadores Genéticos:
   - Selección (Ruleta, Torneos, Ranking)
   - Cruzamiento (1-punto, 2-puntos, Uniforme)
   - Mutación (Bit-flip, Swap, Uniform Int)
   - Reemplazo y Elitismo
5. Implementación práctica con DEAP
6. Ejemplos y ejercicios

## Estructura del Repositorio

```
.
├── README.md          # Este archivo
├── GA.md             # Guía completa de Algoritmos Genéticos
└── examples/         # (Próximamente) Ejemplos de código
```

## Recursos Adicionales

- [DEAP Documentation](https://deap.readthedocs.io/)
- [DEAP GitHub Examples](https://github.com/DEAP/deap/tree/master/examples)

## Referencias

- **"Introduction to Evolutionary Computing"** - A.E. Eiben & J.E. Smith (2015)
- **"Genetic Algorithms in Search, Optimization, and Machine Learning"** - David Goldberg (1989)

## Licencia

Material educativo para aprendizaje de Algoritmos Genéticos.
