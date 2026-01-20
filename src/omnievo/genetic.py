"""
Motor del algoritmo genetico - usa DEAP
"""

import random
import numpy as np
from deap import base, creator, tools

from omnievo.fitness import calculate_fitness, normalize_weights


class GeneticOptimizer:
    """
    Optimizador con algoritmos geneticos para encontrar
    los pesos de atribucion que minimicen el error
    """

    def __init__(self, n_channels=4, population_size=50, generations=100,
                 cxpb=0.7, mutpb=0.2, tournsize=3, elite_size=2,
                 mutation_sigma=0.1, mutation_indpb=0.3,
                 metric="rmse", random_state=42, verbose=True):

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

        self._toolbox = None
        self._ready = False

    def _setup(self, X, y):
        """configura DEAP - esto es medio feo pero funciona"""
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # limpiar si ya existe (DEAP es raro con esto)
        for name in ["FitnessMax", "Individual"]:
            if name in creator.__dict__:
                del creator.__dict__[name]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        tb = base.Toolbox()
        tb.register("attr_float", random.random)
        tb.register("individual", tools.initRepeat, creator.Individual,
                   tb.attr_float, n=self.n_channels)
        tb.register("population", tools.initRepeat, list, tb.individual)

        # fitness
        tb.register("evaluate", calculate_fitness, X=X, y=y, metric=self.metric)

        # operadores
        tb.register("mate", tools.cxBlend, alpha=0.5)
        tb.register("mutate", tools.mutGaussian, mu=0,
                   sigma=self.mutation_sigma, indpb=self.mutation_indpb)
        tb.register("select", tools.selTournament, tournsize=self.tournsize)

        self._toolbox = tb
        self._ready = True

    def fit(self, X, y):
        """ejecuta la optimizacion"""
        self.n_channels = X.shape[1]
        self._setup(X, y)

        pop = self._toolbox.population(n=self.population_size)

        # evaluar poblacion inicial
        fits = list(map(self._toolbox.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        history = []
        best_ever = None
        best_fit = float("-inf")

        # loop principal
        for gen in range(self.generations):
            # elite
            elite = tools.selBest(pop, self.elite_size)
            elite = list(map(self._toolbox.clone, elite))

            # seleccion + offspring
            offspring = self._toolbox.select(pop, len(pop) - self.elite_size)
            offspring = list(map(self._toolbox.clone, offspring))

            # cruce
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self._toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # mutacion
            for mut in offspring:
                if random.random() < self.mutpb:
                    self._toolbox.mutate(mut)
                    del mut.fitness.values

            # evaluar los que cambiaron
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = map(self._toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            # nueva poblacion
            pop[:] = elite + offspring

            # stats
            all_fits = [ind.fitness.values[0] for ind in pop]
            gen_best = max(all_fits)
            gen_avg = sum(all_fits) / len(all_fits)
            gen_std = np.std(all_fits)

            curr_best = tools.selBest(pop, 1)[0]
            if gen_best > best_fit:
                best_fit = gen_best
                best_ever = curr_best[:]

            history.append({
                "gen": gen + 1,
                "best": gen_best,
                "avg": gen_avg,
                "std": gen_std,
                "individual": curr_best[:],
            })

            if self.verbose and (gen + 1) % 10 == 0:
                print(f"Gen {gen+1:03d}: Best={gen_best:.4f} Avg={gen_avg:.4f}")

        # normalizar al final
        weights = normalize_weights(np.array(best_ever))

        return {
            "best_weights": weights,
            "best_fitness": best_fit,
            "history": history,
            "generations_run": self.generations,
            "converged": self._check_conv(history),
        }

    def _check_conv(self, history, window=10, thresh=0.001):
        """checa si convergio"""
        if len(history) < window:
            return False
        recent = [h["best"] for h in history[-window:]]
        var = np.std(recent) / (abs(np.mean(recent)) + 1e-8)
        return var < thresh


def run_optimization(X, y, population_size=50, generations=100,
                    random_state=42, verbose=True):
    """funcion helper para correr rapido"""
    opt = GeneticOptimizer(
        n_channels=X.shape[1],
        population_size=population_size,
        generations=generations,
        random_state=random_state,
        verbose=verbose,
    )
    return opt.fit(X, y)


# wrapper para compatibilidad con el codigo viejo
class OptimizationResult:
    """para que el codigo viejo siga funcionando"""
    def __init__(self, result_dict):
        self.best_weights = result_dict["best_weights"]
        self.best_fitness = result_dict["best_fitness"]
        self.generations_run = result_dict["generations_run"]
        self.converged = result_dict["converged"]

        # convertir history al formato viejo
        self.history = []
        for h in result_dict["history"]:
            self.history.append(_EvolutionStats(
                h["gen"], h["best"], h["avg"], h["std"], h["individual"]
            ))

    def get_weights_dict(self, channel_names):
        return dict(zip(channel_names, self.best_weights))


class _EvolutionStats:
    """stats por generacion - formato viejo"""
    def __init__(self, gen, best, avg, std, ind):
        self.generation = gen
        self.best_fitness = best
        self.avg_fitness = avg
        self.std_fitness = std
        self.best_individual = ind


