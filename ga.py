from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


FitnessFn = Callable[[np.ndarray], Tuple[float, Dict]]


@dataclass
class GAConfig:
    pop_size: int = 40
    crossover_rate: float = 0.9
    mutation_sigma: float = 0.5
    tournament_k: int = 3
    elite_size: int = 2
    evaluation_budget: int = 2000
    bounds: Tuple[float, float] = (-5.0, 5.0)


class GeneticAlgorithm:
    """Real-valued Genetic Algorithm with tournament selection and arithmetic crossover."""

    def __init__(self, num_genes: int, fitness_fn: FitnessFn, config: GAConfig, rng: np.random.Generator | None = None):
        self.num_genes = num_genes
        self.fitness_fn = fitness_fn
        self.config = config
        self.rng = rng or np.random.default_rng()

    def _initialize_population(self) -> np.ndarray:
        low, high = self.config.bounds
        return self.rng.uniform(low, high, size=(self.config.pop_size, self.num_genes))

    def _tournament(self, fitness: np.ndarray) -> int:
        contenders = self.rng.choice(len(fitness), size=self.config.tournament_k, replace=False)
        best_idx = contenders[np.argmin(fitness[contenders])]
        return int(best_idx)

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self.rng.uniform(0.0, 1.0, size=self.num_genes)
        child1 = alpha * parent_a + (1.0 - alpha) * parent_b
        child2 = (1.0 - alpha) * parent_a + alpha * parent_b
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual + self.rng.normal(0.0, self.config.mutation_sigma, size=self.num_genes)
        low, high = self.config.bounds
        return np.clip(mutated, low, high)

    def _evaluate_population(self, population: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        fitness_values = []
        infos: List[Dict] = []
        for individual in population:
            fit, info = self.fitness_fn(individual)
            fitness_values.append(fit)
            infos.append(info)
        return np.asarray(fitness_values, dtype=float), infos

    def run(self) -> Tuple[np.ndarray, float, Dict]:
        pop = self._initialize_population()
        fitness, infos = self._evaluate_population(pop)
        evaluations = len(pop)

        best_idx = int(np.argmin(fitness))
        best_vector = pop[best_idx].copy()
        best_score = float(fitness[best_idx])
        best_info = infos[best_idx]

        while evaluations < self.config.evaluation_budget:
            new_pop: List[np.ndarray] = []
            elite_count = min(self.config.elite_size, self.config.pop_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            for idx in elite_indices:
                new_pop.append(pop[idx].copy())

            while len(new_pop) < self.config.pop_size:
                pa_idx = self._tournament(fitness)
                pb_idx = self._tournament(fitness)
                parent_a = pop[pa_idx]
                parent_b = pop[pb_idx]

                if self.rng.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent_a, parent_b)
                else:
                    child1, child2 = parent_a.copy(), parent_b.copy()

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_pop.append(child1)
                if len(new_pop) < self.config.pop_size:
                    new_pop.append(child2)

            pop = np.asarray(new_pop[: self.config.pop_size])
            fitness, infos = self._evaluate_population(pop)
            evaluations += len(pop)

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_score:
                best_score = float(fitness[gen_best_idx])
                best_vector = pop[gen_best_idx].copy()
                best_info = infos[gen_best_idx]

        return best_vector, best_score, best_info
