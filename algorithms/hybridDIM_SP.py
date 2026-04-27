import sys
import os
import random
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))
from problem.scenarioM    import get_scenario
from problem.FitnessFinal import (compute_fitness, initialise_demand_proportional,
                                   initialise_urgency_biased, initialise_random)
from problem.constraint   import repair
from algorithms.ga        import DisasterReliefGA
from algorithms.pso       import PSO, LinearInertia

RANDOM_SEED            = 42
TOTAL_GENERATIONS      = 100
EPOCH_INTERVAL         = 20
ISLAND_SIZE            = 50
SIGMA                  = 1.0
MIN_CLUSTER_SIZE       = 10
EXTINCTION_THRESHOLD   = 10
DIVERSITY_PERTURBATION = 0.2
MAX_ISLANDS = ISLAND_SIZE // MIN_CLUSTER_SIZE

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

#Spectral-clustering helpers:
def build_similarity_matrix(population, sigma=SIGMA):
    N, dim = population.shape
    W = np.zeros((N, N))
    two_sig_sq_dim = 2.0 * sigma ** 2 * dim
    for i in range(N):
        for j in range(i + 1, N):
            diff    = population[i] - population[j]
            w_ij    = np.exp(-float(diff @ diff) / two_sig_sq_dim)
            W[i, j] = w_ij
            W[j, i] = w_ij
        W[i, i] = 1.0
    return W


def spectral_cluster(population, fitnesses, max_k, sigma=SIGMA,
                     min_size=MIN_CLUSTER_SIZE, scenario=None):
    N = len(population)
    if N < 2 * min_size:
        return [population]

    W          = build_similarity_matrix(population, sigma)
    d          = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-10)))
    L_norm     = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    idx_sort     = np.argsort(eigenvalues)
    eigenvalues  = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    upper  = min(max_k + 1, N - 1)
    gaps   = np.diff(eigenvalues[1:upper + 1])
    best_k = int(np.argmax(gaps)) + 2
    best_k = max(2, min(best_k, max_k, N // min_size))

    embedding = eigenvectors[:, :best_k]
    norms     = np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-10)
    embedding = embedding / norms

    labels  = _kmeans(embedding, best_k)
    islands = []
    for c in range(best_k):
        mask    = labels == c
        members = population[mask].copy()
        if len(members) == 0:
            continue
        if len(members) < min_size and scenario is not None:# If cluster is too small, pad it by mutating the best individual
            best_ind = population[int(np.argmin(fitnesses))]
            while len(members) < min_size:
                noise  = np.random.normal(0, 0.3, size=best_ind.shape)
                padded = repair(best_ind + noise, scenario).flatten(order='F')
                members = np.vstack([members, padded[None, :]])
        islands.append(members)
    return islands if islands else [population]


def _kmeans(X, k, max_iters=50):# Custom K-means clustering (used after spectral embedding)
    N       = len(X)
    centers = [X[np.random.randint(N)]]
    for _ in range(k - 1):
        dists = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X])
        probs = dists / dists.sum()
        centers.append(X[np.random.choice(N, p=probs)])
    centers = np.array(centers)

    labels = np.zeros(N, dtype=int)
    for _ in range(max_iters):
        dists      = np.array([[np.sum((x - c) ** 2) for c in centers] for x in X])
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = X[mask].mean(axis=0)
    return labels

#Island class:
class Island:
    def __init__(self, population, scenario, operator,
                 island_id=0, init_strategy="Random", seed=RANDOM_SEED,
                 global_step_offset=0, total_steps=TOTAL_GENERATIONS):
        self.scenario       = scenario
        self.island_id      = island_id
        self.population     = np.array(population, dtype=float)
        self.n              = len(self.population)
        self.dim            = scenario["dim"]
        self.operator       = operator
        self.seed           = seed
        self.init_strategy  = init_strategy
        self._global_step   = global_step_offset
        self._total_steps   = total_steps

        self.best_history       = []
        self.generations_without_improvement = 0

        self._ga_helper = DisasterReliefGA(
            scenario_data   = scenario,
            config_type     = "baseline",
            init_strategy   = init_strategy,
            max_generations = total_steps,
            population_size = self.n,
            seed            = seed,
        )
        self._generation = global_step_offset


        self._pso = None
        self._init_pso_from_population()

        self.best_score    = float('inf')
        self.best_solution = None
        self._refresh_best()

    def _init_pso_from_population(self):
        rng_seed = (self.seed * 1000 + self.island_id) if self.seed is not None else None
        n_neighbors = min(4, max(2, self.n // 3))
        self._pso = PSO(
            scenario                = self.scenario,
            num_particles           = self.n,
            max_iterations          = self._total_steps,
            c1                      = 1.5,
            c2                      = 1.5,
            inertia                 = LinearInertia(w_start=0.9, w_end=0.4),
            bare                    = False,
            ring                    = True,
            neighbors               = n_neighbors,
            seed                    = rng_seed,
            initialization_strategy = self.init_strategy
        )
        self._pso.pos  = self.population.copy()
        self._pso.vel  = np.zeros((self.n, self.dim))
        fits = self._eval_all()
        self._pso.pbest_x = self.population.copy()
        self._pso.pbest_f = fits.copy()
        best_idx = int(np.argmin(fits))
        self._pso._gbest_x = self.population[best_idx].copy()
        self._pso._gbest_f = float(fits[best_idx])

    def _eval_all(self):# Evaluate fitness for all individuals in the island
        return np.array([
            compute_fitness(ind, self.scenario)[0]
            for ind in self.population
        ])

    def _refresh_best(self):# Update best solution and track improvement history
        fits = self._eval_all()
        idx  = int(np.argmin(fits))
        if fits[idx] < self.best_score:
            self.best_score    = float(fits[idx])
            self.best_solution = self.population[idx].copy()
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        self.best_history.append(self.best_score)
        return fits
    #Extinction event:
    def _extinction_event(self):
        fits = self._eval_all()
        sorted_idx = np.argsort(fits)   # ascending: best first
        n_keep    = max(1, int(0.4 * self.n))
        keep_idx  = sorted_idx[:n_keep]
        replace_idx = sorted_idx[n_keep:]
        n_replace = len(replace_idx)
        fresh = initialise_random(len(replace_idx), self.scenario,
                                  seed=np.random.randint(int(1e6)))
        for k, idx in enumerate(replace_idx):
            self.population[idx] = fresh[k]

        if self.operator == "PSO" and self._pso is not None:
            self._pso.pos = self.population.copy()
            # Reset velocities for replaced particles
            self._pso.vel[replace_idx] = np.random.uniform(
                -0.5, 0.5, size=(n_replace, self.dim))
            for idx in replace_idx:
                self._pso.pbest_x[idx] = self.population[idx].copy()
                self._pso.pbest_f[idx] = compute_fitness(
                    self.population[idx], self.scenario)[0]
            elite_fits = np.array([
                compute_fitness(self.population[idx], self.scenario)[0]
                for idx in keep_idx
            ])
            best_elite_idx = keep_idx[int(np.argmin(elite_fits))]
            self._pso._gbest_x = self.population[best_elite_idx].copy()
            self._pso._gbest_f = float(min(elite_fits))

        self.generations_without_improvement = 0

    def _ga_step(self, n_gens):
        for _ in range(n_gens):
            scores    = self._eval_all()
            parents   = self._tourn_select(scores)
            fake      = _FakeGA(self.population, self._generation)
            offspring = self._ga_helper.blx(parents, (self.n, self.dim), fake)
            offspring = self._ga_helper.nonuniform_mutate(offspring, fake)
            offspring[0] = self.population[int(np.argmin(scores))]  # elitism
            self.population  = offspring
            self._generation += 1
            self._refresh_best()

            if self.generations_without_improvement >= EXTINCTION_THRESHOLD:
                self._extinction_event()

    def _tourn_select(self, scores, k=3):
        n, parents = len(self.population), []
        for _ in range(n):
            c      = np.random.choice(n, min(k, n), replace=False)
            winner = c[int(np.argmin(scores[c]))]
            parents.append(self.population[winner])
        return np.array(parents)

    def _pso_step(self, n_steps):
        for _ in range(n_steps):
            w = self._pso.inertia.get(self._global_step, self._total_steps)
            self._pso._canonical_step(w)
            # Evaluate and update memories (pbest / gbest)
            fits = self._pso._evaluate_all(self._pso.pos)
            self._pso._update_memory(fits)

            # Sync population from PSO positions
            self.population = self._pso.pos.copy()
            self._global_step += 1
            self._refresh_best()

            if self.generations_without_improvement >= EXTINCTION_THRESHOLD:
                self._extinction_event()
                if self._pso is not None:
                    self._pso.pos = self.population.copy()

    def evolve(self, n_steps):
        if self.operator == "PSO":
            self._pso_step(n_steps)
        else:
            self._ga_step(n_steps)
        self._refresh_best()


class _FakeGA:
    def __init__(self, population, generations_completed):
        self.population            = population
        self.generations_completed = generations_completed


class DIMSPHybrid:
    def __init__( # Initialize the hybrid system with one initial island
        self,
        scenario,
        total_generations = TOTAL_GENERATIONS,
        epoch_interval    = EPOCH_INTERVAL,
        island_size       = ISLAND_SIZE,
        init_strategy     = "Random",
        # cluster_strategy  = "large_psi",
        seed              = RANDOM_SEED,
        verbose           = False,
    ):
        self.scenario          = scenario
        self.total_generations = total_generations
        self.epoch_interval    = epoch_interval
        self.island_size       = island_size
        # self.cluster_strategy  = cluster_strategy
        self.seed              = seed
        self.verbose           = verbose
        self._init_strategy    = init_strategy
        self.max_islands = max(2, island_size // MIN_CLUSTER_SIZE)

        if init_strategy == "Demand_Proportional":
            pop = initialise_demand_proportional(island_size, scenario, seed)
        elif init_strategy == "Urgency_Biased":
            pop = initialise_urgency_biased(island_size, scenario, seed)
        else:
            pop = initialise_random(island_size, scenario, seed)

        self.islands = [Island(
            pop, scenario, operator="PSO", island_id=0,
            init_strategy   = init_strategy,
            seed            = seed,
            global_step_offset = 0,
            total_steps     = total_generations,
        )]

        self.convergence       = []
        self.island_count_hist = []
        self.best_solution     = None
        self.best_score        = np.inf

        # Track global generation for step offsets
        self._current_gen = 0

    # def _determine_operator(self, cluster_size, median_size):
    #     if self.cluster_strategy == "large_psi":
    #         return "PSO" if cluster_size >= median_size else "GA"
    #     elif self.cluster_strategy == "small_psi":
    #         return "GA"  if cluster_size >= median_size else "PSO"
    #     else:
    #         return np.random.choice(["PSO", "GA"])
    def _determine_operator(self, cluster_size, median_size):
        """Small clusters → PSO  |  Large clusters → GA  (small-psi strategy)"""
        return "GA" if cluster_size >= median_size else "PSO"

    def _run_epoch(self, current_gen):
        Pa       = np.vstack([isl.population for isl in self.islands])
        fits_all = np.array([
            compute_fitness(ind, self.scenario)[0] for ind in Pa
        ])
        new_pops = spectral_cluster(
            Pa, fits_all,
            max_k    = self.max_islands,
            sigma    = SIGMA,
            min_size = MIN_CLUSTER_SIZE,
            scenario = self.scenario,
        )
        median_size = np.median([len(p) for p in new_pops])
        new_islands = []
        for idx, pop in enumerate(new_pops):
            op  = self._determine_operator(len(pop), median_size)
            isl = Island(
                pop, self.scenario, operator=op, island_id=idx,
                init_strategy      = self._init_strategy,
                seed               = self.seed,
                global_step_offset = current_gen,
                total_steps        = self.total_generations,
            )
            new_islands.append(isl)
        self.islands = new_islands

    def _update_global_best(self):
        for isl in self.islands:
            if isl.best_score < self.best_score:
                self.best_score    = isl.best_score
                self.best_solution = isl.best_solution.copy()
    #Main loop:
    def run(self):
        n_epochs        = self.total_generations // self.epoch_interval
        steps_per_epoch = self.epoch_interval
        for epoch in range(n_epochs):
            if epoch > 0:
                self._run_epoch(self._current_gen)
            for isl in self.islands:
                isl.evolve(n_steps=steps_per_epoch)
            self._current_gen += steps_per_epoch
            self._update_global_best()
            self.convergence.append(self.best_score)
            self.island_count_hist.append(len(self.islands))
        best_solution = self.best_solution.copy()
        best_score, _ = compute_fitness(best_solution, self.scenario)

        return best_solution, best_score, {
            "hybrid_convergence": self.convergence,
            "island_count":       self.island_count_hist,
        }