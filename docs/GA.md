# Algoritmos Genéticos (Genetic Algorithms)

## Introducción

Los **Algoritmos Genéticos (GA)** son metaheurísticas bio-inspiradas basadas en la **Teoría de la Evolución y Selección Natural** de Darwin, desarrollados por John Holland en 1973 y popularizados por De Jong en 1975.

### ¿Qué problemas resuelven?

Los GA se utilizan para resolver problemas de optimización combinatoria:

**Tipos de Problemas:**
- **FOP (Free Optimisation Problem)**: Optimiza función sin restricciones
- **CSP (Constraint Satisfaction Problem)**: Cumple restricciones sin optimizar
- **CSOP/COP (Constraint Optimisation Problem)**: Optimiza Y cumple restricciones

### Características principales
- Búsqueda con múltiples candidatos simultáneos (población)
- Operadores genéticos: selección, cruzamiento y mutación
- Balance entre exploración y explotación del espacio de búsqueda

---

## Conceptos Fundamentales

### Representación

| Concepto | Descripción |
|----------|-------------|
| **Fenotipo** | Solución en el mundo real |
| **Genotipo** | Solución en el mundo evolutivo (cromosoma) |
| **Gen** | Elemento/variable de la solución |
| **Alelo** | Valor en un gen |
| **Codificar** | Mundo real → mundo evolutivo |
| **Decodificar** | Mundo evolutivo → mundo real |

**En GA clásicos:**
- Representación típica: **Bit-vector** (cadena de bits)
- También se pueden usar: vectores de enteros, reales, permutaciones

### Función de Fitness
- Mide calidad de genotipos
- Mide adaptación de individuos al ambiente
- Se evalúa decodificando al espacio de fenotipos
- Guía el proceso de selección

### Población
- Multiconjunto de candidatos a solución
- Individuos estáticos, poblaciones dinámicas
- Tamaño generalmente constante
- **Diversidad**: Medida de dispersión de soluciones diferentes
- Mayor diversidad → mayor exploración del espacio de búsqueda

---

## Operadores Genéticos

### 1. Selección de Padres
Filtrar miembros según calidad para reproducción.

**Métodos comunes en GA:**
- **Ruleta (Roulette-Wheel)**: Probabilidad proporcional al fitness (método clásico)
- **Torneos**: Competencia entre k individuos aleatorios
- **Ranking**: Selección basada en orden, no en fitness absoluto
- **Aleatoria**: Sin preferencia (baja presión selectiva)

### 2. Cruzamiento/Recombinación
- Mezcla información genética de **dos padres** → uno o dos hijos
- **Operador principal en GA** (más importante que mutación)
- Estocástica (probabilidad de cruzamiento: típicamente 0.6-0.9)

**Tipos de cruzamiento:**
- **1 punto**: Corte en una posición, intercambio de segmentos (clásico GA)
- **2 puntos**: Dos cortes, intercambio del segmento central
- **Uniforme**: Cada gen se hereda de un padre con cierta probabilidad
- **n puntos**: Múltiples puntos de corte

### 3. Mutación
- Perturbación aplicada a **un individuo** → modifica alelos
- **Operador secundario en GA** (exploración adicional)
- Conecta distintos sectores del espacio de búsqueda
- Probabilidad de mutación baja: típicamente 0.01-0.1

**Tipos para GA:**
- **Bit-flip**: Invertir bits (0→1, 1→0) para representación binaria
- **Swap**: Intercambiar posiciones (para permutaciones)
- **Uniform Int**: Cambio aleatorio dentro de rango (vectores de enteros)

### 4. Reemplazo/Supervivencia
**En GA clásico:**
- **Supervivencia por edad**: Los hijos reemplazan a los padres completamente
- Generacional: Se reemplaza toda la población en cada iteración

**Variantes modernas:**
- **Elitismo**: Mantener los mejores individuos de generación en generación
- **Steady-state**: Reemplazar solo algunos individuos por iteración

---

## Algoritmo General de GA

```
1. Inicializar población aleatoria de tamaño N
2. Evaluar fitness de cada individuo
3. MIENTRAS no se cumpla criterio de término:
   a. Seleccionar padres (ej: Roulette-Wheel)
   b. Aplicar cruzamiento (ej: 1-punto) con probabilidad Pc
   c. Aplicar mutación (ej: Bit-flip) con probabilidad Pm
   d. Evaluar fitness de hijos
   e. Reemplazar población (ej: reemplazo generacional)
   f. Incrementar contador de generación
4. Retornar mejor individuo encontrado
```

### Criterios de Término

- **Convergencia**: % población con mismo fitness
- Tiempo sin mejoras
- Total de evaluaciones fitness
- Número de generaciones/iteraciones
- Tiempo de ejecución
- Fitness objetivo alcanzado

### Parámetros típicos

- **Tamaño de población**: 50-200 individuos
- **Probabilidad de cruzamiento (Pc)**: 0.6-0.9
- **Probabilidad de mutación (Pm)**: 0.01-0.1 (1/longitud_cromosoma)
- **Presión selectiva**: Depende del método de selección

---

## Ejemplo: Problema de las N-Reinas con GA

> **Objetivo:** Ubicar n reinas en tablero n×n sin que hagan jaque entre ellas.

### Representación
- **Vector de enteros** de tamaño n
- Gen i: columna i del tablero
- Alelo: fila donde está la reina en esa columna
- **Ventaja**: Elimina automáticamente checks en columnas (1 reina por columna)

### Modelado del problema

**Como FOP (Free Optimization Problem):**
- Función fitness: Maximizar número de reinas libres de jaque
- Solución óptima: f(s) = n

**Como CSOP (Constraint Optimization Problem):**
- Restricciones implícitas: Una reina por fila y columna
- Función fitness: Minimizar número de jaques en diagonales

### Función de evaluación
```python
def evaluar_nreinas(individual):
    """
    Cuenta el número de pares de reinas en jaque
    Objetivo: minimizar (0 = solución óptima)
    """
    n = len(individual)
    checks = 0

    # Contar jaques en filas (reinas en misma fila)
    for i in range(n):
        for j in range(i+1, n):
            if individual[i] == individual[j]:
                checks += 1

    # Contar jaques en diagonales
    for i in range(n):
        for j in range(i+1, n):
            if abs(individual[i] - individual[j]) == abs(i - j):
                checks += 1

    return (checks,)  # DEAP requiere tupla
```

---

## Implementación con DEAP (Python)

### ¿Qué es DEAP?

**DEAP** (Distributed Evolutionary Algorithms in Python) es una biblioteca de computación evolutiva que:
- Soporta GA, GP, ES y otros algoritmos evolutivos
- Es altamente configurable y extensible
- Tiene excelente documentación
- Es la más popular para GA en Python

### Instalación

```bash
pip install deap
```

### Estructura básica de un GA en DEAP

#### 1. Configuración inicial

```python
import random
from deap import creator, base, tools, algorithms

# Configurar el problema (minimizar)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # -1.0 = minimizar
# Para maximizar usar: weights=(1.0,)

# Crear clase Individuo basada en lista
creator.create("Individual", list, fitness=creator.FitnessMin)
```

#### 2. Registrar operadores en Toolbox

```python
N_QUEENS = 8  # Tamaño del tablero

toolbox = base.Toolbox()

# Generador de genes: enteros aleatorios entre 0 y N-1
toolbox.register("attr_int", random.randint, 0, N_QUEENS-1)

# Generador de individuos: lista de N genes
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=N_QUEENS)

# Generador de población
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación (fitness)
toolbox.register("evaluate", evaluar_nreinas)

# Operadores genéticos
toolbox.register("mate", tools.cxTwoPoint)  # Cruzamiento de 2 puntos
toolbox.register("mutate", tools.mutUniformInt, low=0, up=N_QUEENS-1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)  # Selección por torneo
```

#### 3. Ejecutar el algoritmo

```python
def main():
    # Parámetros del GA
    POPULATION_SIZE = 100
    GENERATIONS = 100
    CXPB = 0.7   # Probabilidad de cruzamiento
    MUTPB = 0.2  # Probabilidad de mutación

    # Crear población inicial
    population = toolbox.population(n=POPULATION_SIZE)

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum([i[0] for i in x]) / len(x))
    stats.register("min", lambda x: min([i[0] for i in x]))

    # Ejecutar GA
    for generation in range(GENERATIONS):
        # Seleccionar la próxima generación
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Aplicar cruzamiento
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Aplicar mutación
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluar individuos con fitness inválido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Reemplazar población
        population[:] = offspring

        # Mostrar estadísticas
        record = stats.compile(population)
        print(f"Gen {generation}: Min={record['min']:.2f}, Avg={record['avg']:.2f}")

        # Criterio de término: solución encontrada
        if record['min'] == 0:
            print(f"¡Solución encontrada en generación {generation}!")
            break

    # Retornar mejor solución
    best_ind = tools.selBest(population, 1)[0]
    return best_ind

if __name__ == "__main__":
    resultado = main()
    print(f"Mejor solución: {resultado}")
    print(f"Fitness: {resultado.fitness.values[0]}")
```

### Usando el algoritmo eaSimple de DEAP

DEAP también proporciona algoritmos evolutivos predefinidos:

```python
from deap import algorithms

def main_simple():
    population = toolbox.population(n=100)

    # Configurar estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum([i[0] for i in x]) / len(x))
    stats.register("min", lambda x: min([i[0] for i in x]))

    # Ejecutar GA estándar
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,    # Probabilidad de cruzamiento
        mutpb=0.2,   # Probabilidad de mutación
        ngen=100,    # Número de generaciones
        stats=stats,
        verbose=True
    )

    return tools.selBest(population, 1)[0]
```

---

## Ventajas y Limitaciones de GA

### Ventajas
- No requiere información del gradiente
- Exploran múltiples regiones simultáneamente
- Robustos ante ruido y multimodalidad
- Aplicables a diversos tipos de problemas
- Fáciles de paralelizar

### Limitaciones
- No garantizan encontrar el óptimo global
- Requieren ajuste de parámetros (tunning)
- Pueden converger prematuramente
- Costosos computacionalmente para problemas grandes
- La representación del problema es crítica

---

## Referencias y Recursos

### Libros
- **"Introduction to Evolutionary Computing"** - A.E. Eiben & J.E. Smith (2015)
- **"Genetic Algorithms in Search, Optimization, and Machine Learning"** - David Goldberg (1989)

### Documentación
- DEAP Documentation: https://deap.readthedocs.io/
- DEAP Examples: https://github.com/DEAP/deap/tree/master/examples

### Bibliotecas alternativas
- **PyGAD**: https://pygad.readthedocs.io/ (más simple para principiantes)
- **GAFT**: https://github.com/PytLab/gaft (sintaxis moderna)
- **NEAT-Python**: Para neuroevolución

---

## Ejercicios Propuestos

1. Modificar el GA de N-Reinas para usar selección por ruleta en vez de torneos
2. Implementar un GA para el problema del viajante (TSP) usando permutaciones
3. Experimentar con diferentes probabilidades de cruzamiento y mutación
4. Agregar elitismo: mantener los mejores 10% de individuos en cada generación
5. Implementar un GA para optimizar una función matemática continua (ej: Rastrigin)
