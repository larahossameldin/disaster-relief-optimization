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
MAX_ISLANDS            = ISLAND_SIZE // MIN_CLUSTER_SIZE

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

def _initialise_population(strategy, size, scenario, seed):
    if strategy == "Demand_Proportional":
        return initialise_demand_proportional(size, scenario, seed)
    elif strategy == "Urgency_Biased":
        return initialise_urgency_biased(size, scenario, seed)
    else:
        return initialise_random(size, scenario, seed)

#Spectral-clustering helpers:
def build_similarity_matrix(population, sigma=SIGMA):
    N, dim        = population.shape
    W             = np.zeros((N, N))
    two_sig_sq_dim = 2.0 * sigma ** 2 * dim
    for i in range(N):
        for j in range(i + 1, N):
            diff    = population[i] - population[j]
            w_ij    = np.exp(-float(diff @ diff) / two_sig_sq_dim)
            W[i, j] = w_ij
            W[j, i] = w_ij
        W[i, i] = 1.0
    return W

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
        labels  = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = X[mask].mean(axis=0)
    return labels

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
        # Pad undersized clusters by perturbing the global best
        if len(members) < min_size and scenario is not None:# If cluster is too small, pad it by mutating the best individual
            best_ind = population[int(np.argmin(fitnesses))]
            while len(members) < min_size:
                noise  = np.random.normal(0, 0.3, size=best_ind.shape)
                padded = repair(best_ind + noise, scenario).flatten(order='F')
                members = np.vstack([members, padded[None, :]])
        islands.append(members)
    return islands if islands else [population]

#Island class:
class _FakeGA:
    def __init__(self, population, generations_completed):
        self.population            = population
        self.generations_completed = generations_completed

class Island:
    def __init__(self, population, scenario, operator,
                 island_id=0, init_strategy="Random", seed=RANDOM_SEED):
        self.scenario      = scenario
        self.island_id     = island_id
        self.population    = np.array(population, dtype=float)
        self.n             = len(self.population)
        self.dim           = scenario["dim"]
        self.operator      = operator
        self.seed          = seed
        self.init_strategy = init_strategy

        self.best_history                    = []
        self.generations_without_improvement = 0
        self._generation                     = 0

        self._ga_helper = DisasterReliefGA(
            scenario_data   = scenario,
            config_type     = "baseline",
            init_strategy   = init_strategy,
            max_generations = TOTAL_GENERATIONS,
            population_size = self.n,
            seed            = seed,
        )

        self._pso = PSO(
            scenario                = scenario,
            num_particles           = self.n,
            max_iterations          = 1,          # we drive the loop manually
            c1                      = 1.5,
            c2                      = 1.5,
            inertia                 = LinearInertia(0.9, 0.4),
            bare                    = False,
            ring                    = False,
            seed                    = seed,
            initialization_strategy = init_strategy,
        )

        self._pso.pos     = self.population.copy()
        self._pso.vel     = np.zeros_like(self.population)
        fits              = self._eval_all()
        self._pso.pbest_x = self.population.copy()
        self._pso.pbest_f = fits.copy()
        best_idx          = int(np.argmin(fits))
        self._pso._gbest_x = self.population[best_idx].copy()
        self._pso._gbest_f = float(fits[best_idx])

        self.best_score    = self._pso._gbest_f
        self.best_solution = self._pso._gbest_x.copy()
        self.best_history.append(self.best_score)

    #Evaluation:
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
        fits        = self._eval_all()
        sorted_idx  = np.argsort(fits)    
        n_keep      = max(1, int(0.4 * self.n))
        keep_idx    = sorted_idx[:n_keep]
        replace_idx = sorted_idx[n_keep:]

        fresh = initialise_random(len(replace_idx), self.scenario,
                                  seed=np.random.randint(int(1e6)))
        for k, idx in enumerate(replace_idx):
            self.population[idx] = fresh[k]

        if self.operator == "PSO":
            # Reset velocities for replaced particles
            self._pso.vel[replace_idx] = np.random.uniform(
                -0.5, 0.5, size=(len(replace_idx), self.dim))

            # Reset personal bests for replaced particles
            for idx in replace_idx:
                self._pso.pbest_x[idx] = self.population[idx].copy()
                self._pso.pbest_f[idx] = compute_fitness(
                    self.population[idx], self.scenario)[0]
                
            elite_fits          = np.array([
                compute_fitness(self.population[idx], self.scenario)[0]
                for idx in keep_idx
            ])
            best_elite_idx      = keep_idx[int(np.argmin(elite_fits))]
            self._pso._gbest_x  = self.population[best_elite_idx].copy()
            self._pso._gbest_f  = float(np.min(elite_fits))

            # Mirror population back into PSO
            self._pso.pos = self.population.copy()

        self.generations_without_improvement = 0

    #GA step:
    def _ga_step(self, n_gens):
        for _ in range(n_gens):
            scores   = self._eval_all()
            parents  = self._tourn_select(scores)
            fake     = _FakeGA(self.population, self._generation)
            offspring = self._ga_helper.blx(parents, (self.n, self.dim), fake)
            offspring = self._ga_helper.nonuniform_mutate(offspring, fake)
            offspring[0] = self.population[int(np.argmin(scores))]  # elitism
            self.population  = offspring
            self._generation += 1
            self._refresh_best()

            if self.generations_without_improvement >= EXTINCTION_THRESHOLD:
                self._extinction_event()

    def _tourn_select(self, scores, k=3):# Select parents using tournament selection
        n, parents = len(self.population), []
        for _ in range(n):
            candidates = np.random.choice(n, min(k, n), replace=False)
            winner     = candidates[int(np.argmin(scores[candidates]))]
            parents.append(self.population[winner])
        return np.array(parents)

    #PSO step:
    def _pso_step(self, n_steps):
        for t in range(n_steps):
            w = self._pso.inertia.get(t, n_steps)

            # Delegate movement and repair entirely to PSO
            self._pso._canonical_step(w)

            # Evaluate and update memories (pbest / gbest)
            fits = self._pso._evaluate_all(self._pso.pos)
            self._pso._update_memory(fits)

            # Mirror PSO's authoritative position back to the island
            self.population = self._pso.pos.copy()
            self._refresh_best()

            if self.generations_without_improvement >= EXTINCTION_THRESHOLD:
                self._extinction_event()

    def evolve(self, n_steps):
        if self.operator == "PSO":
            self._pso_step(n_steps)
        else:
            self._ga_step(n_steps)
        self._refresh_best()

#DIM-SP Hybrid:
class DIMSPHybrid:
    def __init__( # Initialize the hybrid system with one initial island
        self,
        scenario,
        total_generations = TOTAL_GENERATIONS,
        epoch_interval    = EPOCH_INTERVAL,
        island_size       = ISLAND_SIZE,
        init_strategy     = "Random",
        seed              = RANDOM_SEED,
        verbose           = False,
    ):
        self.scenario          = scenario
        self.total_generations = total_generations
        self.epoch_interval    = epoch_interval
        self.island_size       = island_size
        self.seed              = seed
        self.verbose           = verbose
        self._init_strategy    = init_strategy
        self.max_islands       = max(2, island_size // MIN_CLUSTER_SIZE)

        initial_pop = _initialise_population(init_strategy, island_size,
                                             scenario, seed) 
        self.islands = [
            Island(initial_pop, scenario, operator="PSO", island_id=0,
                   init_strategy=init_strategy, seed=seed)
        ]

        self.convergence       = []
        self.island_count_hist = []
        self.best_solution     = None
        self.best_score        = np.inf

    #Operator assignment:
    def _determine_operator(self, cluster_size, median_size):
        """Small clusters → PSO  |  Large clusters → GA  (small-psi strategy)"""
        return "GA" if cluster_size >= median_size else "PSO"

    #Epoch: re-cluster and reassign operators
    def _run_epoch(self): 
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
        self.islands = [
            Island(pop, self.scenario,
                   operator  = self._determine_operator(len(pop), median_size),
                   island_id = idx,
                   init_strategy = self._init_strategy,
                   seed      = self.seed)
            for idx, pop in enumerate(new_pops)
        ]

    def _update_global_best(self): # Track best solution across all islands
        for isl in self.islands:
            if isl.best_score < self.best_score:
                self.best_score    = isl.best_score
                self.best_solution = isl.best_solution.copy()

    #Main loop:
    def run(self):
        n_epochs        = self.total_generations // self.epoch_interval
        steps_per_epoch = self.epoch_interval

        for epoch in range(n_epochs):
            if epoch > 0:                   # first epoch: use the initial single island
                self._run_epoch()
            for isl in self.islands:
                isl.evolve(n_steps=steps_per_epoch)
            self._update_global_best()
            self.convergence.append(self.best_score)
            self.island_count_hist.append(len(self.islands))

            if self.verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} | "
                      f"Islands: {len(self.islands)} | "
                      f"Best: {self.best_score:.4f}")

        best_solution = repair(self.best_solution, self.scenario).flatten(order='F')
        best_score, _ = compute_fitness(best_solution, self.scenario)

        return best_solution, best_score, {
            "hybrid_convergence": self.convergence,
            "island_count":       self.island_count_hist,
        }