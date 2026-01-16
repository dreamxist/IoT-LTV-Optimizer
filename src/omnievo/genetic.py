"""
Motor del Algoritmo Genético para Optimización de Atribución.

Este módulo implementa el core del AG usando DEAP, incluyendo:
- Configuración de individuos y población
- Operadores genéticos (selección, cruce, mutación)
- Loop evolutivo con monitoreo de convergencia
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from deap import algorithms, base, creator, tools

from omnievo.fitness import calculate_fitness, normalize_weights


@dataclass
class EvolutionStats:
    """Estadísticas de la evolución por generación."""

    generation: int
    best_fitness: float
    avg_fitness: float
    std_fitness: float
    best_individual: list = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Resultado de la optimización genética."""

    best_weights: np.ndarray
    best_fitness: float
    history: list[EvolutionStats]
    generations_run: int
    converged: bool = False

    def get_weights_dict(self, channel_names: list[str]) -> dict[str, float]:
        """Retorna los pesos como diccionario con nombres de canales."""
        return dict(zip(channel_names, self.best_weights))


class GeneticOptimizer:
    """
    Optimizador basado en Algoritmos Genéticos para atribución omnicanal.

    Usa DEAP para evolucionar vectores de pesos que minimicen el error
    entre el LTV predicho y el LTV real.

    Attributes:
        n_channels: Número de canales (dimensión del cromosoma)
        population_size: Tamaño de la población
        generations: Número máximo de generaciones
        cxpb: Probabilidad de cruzamiento
        mutpb: Probabilidad de mutación
        tournsize: Tamaño del torneo para selección
        elite_size: Número de individuos élite a preservar
        metric: Métrica de fitness ("rmse" o "pearson")
        random_state: Semilla para reproducibilidad
    """

    def __init__(
        self,
        n_channels: int = 4,
        population_size: int = 50,
        generations: int = 100,
        cxpb: float = 0.7,
        mutpb: float = 0.2,
        tournsize: int = 3,
        elite_size: int = 2,
        mutation_sigma: float = 0.1,
        mutation_indpb: float = 0.3,
        metric: str = "rmse",
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.n_channels = n_channels
        self.population_size = population_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.elite_size = elite_size
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        # Estado interno
        self._toolbox = None
        self._setup_complete = False

    def _setup_deap(self, X: np.ndarray, y: np.ndarray):
        """Configura DEAP con el problema específico."""
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Limpiar creators anteriores si existen
        if "FitnessMax" in creator.__dict__:
            del creator.FitnessMax
        if "Individual" in creator.__dict__:
            del creator.Individual

        # Crear tipos de fitness e individuo
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Configurar toolbox
        self._toolbox = base.Toolbox()

        # Generador de genes: floats aleatorios [0, 1]
        self._toolbox.register("attr_float", random.random)

        # Generador de individuos
        self._toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self._toolbox.attr_float,
            n=self.n_channels,
        )

        # Generador de población
        self._toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self._toolbox.individual,
        )

        # Función de fitness
        self._toolbox.register(
            "evaluate",
            calculate_fitness,
            X=X,
            y=y,
            metric=self.metric,
        )

        # Operadores genéticos
        self._toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self._toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=self.mutation_sigma,
            indpb=self.mutation_indpb,
        )
        self._toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.tournsize,
        )

        self._setup_complete = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """
        Ejecuta la optimización genética.

        Args:
            X: Matriz de features (N usuarios x M canales)
            y: Vector de LTV real (N usuarios)

        Returns:
            OptimizationResult con los mejores pesos encontrados
        """
        # Detectar número de canales del input
        self.n_channels = X.shape[1]

        # Configurar DEAP
        self._setup_deap(X, y)

        # Crear población inicial
        population = self._toolbox.population(n=self.population_size)

        # Evaluar población inicial
        fitnesses = list(map(self._toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Tracking de evolución
        history = []
        best_ever = None
        best_fitness_ever = float("-inf")

        # Loop evolutivo
        for gen in range(self.generations):
            # Seleccionar élite
            elite = tools.selBest(population, self.elite_size)
            elite = list(map(self._toolbox.clone, elite))

            # Seleccionar y clonar para offspring
            offspring = self._toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self._toolbox.clone, offspring))

            # Aplicar cruzamiento
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self._toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Aplicar mutación
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self._toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluar individuos con fitness inválido
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self._toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Reemplazar población (élite + offspring)
            population[:] = elite + offspring

            # Calcular estadísticas
            fits = [ind.fitness.values[0] for ind in population]
            best_fitness = max(fits)
            avg_fitness = sum(fits) / len(fits)
            std_fitness = np.std(fits)

            # Actualizar mejor global
            current_best = tools.selBest(population, 1)[0]
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_ever = current_best[:]

            # Guardar estadísticas
            stats = EvolutionStats(
                generation=gen + 1,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                std_fitness=std_fitness,
                best_individual=current_best[:],
            )
            history.append(stats)

            # Log de progreso
            if self.verbose and (gen + 1) % 10 == 0:
                print(
                    f"Gen {gen + 1:03d}: "
                    f"Best = {best_fitness:.4f} | "
                    f"Avg = {avg_fitness:.4f} | "
                    f"Std = {std_fitness:.4f}"
                )

        # Normalizar pesos finales
        best_weights = normalize_weights(np.array(best_ever))

        return OptimizationResult(
            best_weights=best_weights,
            best_fitness=best_fitness_ever,
            history=history,
            generations_run=self.generations,
            converged=self._check_convergence(history),
        )

    def _check_convergence(
        self,
        history: list[EvolutionStats],
        window: int = 10,
        threshold: float = 0.001,
    ) -> bool:
        """Verifica si el algoritmo convergió."""
        if len(history) < window:
            return False

        recent = history[-window:]
        fitness_values = [s.best_fitness for s in recent]
        variation = np.std(fitness_values) / (np.abs(np.mean(fitness_values)) + 1e-8)

        return variation < threshold


def run_optimization(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 50,
    generations: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Función de conveniencia para ejecutar optimización con parámetros por defecto.

    Args:
        X: Matriz de features
        y: Vector de LTV real
        population_size: Tamaño de población
        generations: Número de generaciones
        random_state: Semilla aleatoria
        verbose: Mostrar progreso

    Returns:
        OptimizationResult con los resultados
    """
    optimizer = GeneticOptimizer(
        n_channels=X.shape[1],
        population_size=population_size,
        generations=generations,
        random_state=random_state,
        verbose=verbose,
    )

    return optimizer.fit(X, y)
