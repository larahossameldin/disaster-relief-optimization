import sys 
import os
import numpy as np
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))

from problem.scenarioM    import get_scenario
from problem.FitnessFinal import (compute_fitness, initialise_demand_proportional,
                                   initialise_urgency_biased, initialise_random)
from problem.constraint   import repair
from algorithms.ga  import DisasterReliefGA
from algorithms.pso import PSO, LinearInertia

TOTAL_GENERATIONS = 100
EPOCH_INTERVAL    = 20
MAX_ISLANDS       = 5
ISLAND_SIZE       = 50
SIGMA             = 1.0
MIN_CLUSTER_SIZE  = 10
STAGNATION_THRESHOLD = 3
DIVERSITY_PERTURBATION = 0.2

def build_similarity_matrix(population, sigma=SIGMA): # Builds similarity graph using Gaussian kernel for spectral clustering
    N, dim = population.shape
    W = np.zeros((N, N))
    two_sig_sq_dim = 2.0 * sigma**2 * dim
    
    for i in range(N):
        for j in range(i + 1, N):
            diff = population[i] - population[j]
            dist_sq = float(diff @ diff)
            w_ij = np.exp(-dist_sq / two_sig_sq_dim)
            W[i, j] = w_ij
            W[j, i] = w_ij
        W[i, i] = 1.0
    return W

def spectral_cluster(population, fitnesses, max_k=MAX_ISLANDS,
                     sigma=SIGMA, min_size=MIN_CLUSTER_SIZE, scenario=None): # Performs spectral clustering to dynamically partition population into islands
    N = len(population)
    if N < 2 * min_size:
        return [population]
    
    W = build_similarity_matrix(population, sigma)
    
    d = W.sum(axis=1)
    d_safe = np.maximum(d, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
    L_norm = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    idx_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]
    
    upper = min(max_k + 1, N - 1)
    gaps = np.diff(eigenvalues[1:upper + 1])
    best_k = int(np.argmax(gaps)) + 2
    best_k = max(2, min(best_k, max_k, N // min_size))
    
    embedding = eigenvectors[:, :best_k]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embedding = embedding / norms
    
    labels = _kmeans(embedding, best_k)
    
    islands = []
    for c in range(best_k):
        mask = labels == c
        members = population[mask].copy()
        
        if len(members) == 0:
            continue
        
        if len(members) < min_size and scenario is not None:
            best_global_idx = int(np.argmin(fitnesses))
            best_ind = population[best_global_idx]
            while len(members) < min_size:
                noise = np.random.normal(0, 0.3, size=best_ind.shape)
                padded = repair(best_ind + noise, scenario).flatten(order='F')
                members = np.vstack([members, padded[None, :]])
        
        islands.append(members)
    
    return islands if islands else [population]

def _kmeans(X, k, max_iters=50):  # Applies k-means clustering on spectral embedding space
    N = len(X)
    centers = [X[np.random.randint(N)]]
    for _ in range(k - 1):
        dists = np.array([min(np.sum((x - c)**2) for c in centers) for x in X])
        probs = dists / dists.sum()
        centers.append(X[np.random.choice(N, p=probs)])
    centers = np.array(centers)
    
    labels = np.zeros(N, dtype=int)
    for _ in range(max_iters):
        dists = np.array([[np.sum((x - c)**2) for c in centers] for x in X])
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = X[mask].mean(axis=0)
    return labels

class Island:
    def __init__(self, population, scenario, operator, island_id=0):
        self.scenario = scenario
        self.island_id = island_id
        self.population = np.array(population, dtype=float)
        self.n = len(self.population)
        self.dim = scenario["dim"]
        self.operator = operator
        
        self.best_history = []
        self.stagnation_counter = 0
        
        self._ga_helper = DisasterReliefGA(
            scenario_data=scenario,
            config_type="baseline",
            init_strategy="Demand_Proportional",
            max_generations=TOTAL_GENERATIONS,
            population_size=self.n,
        )
        self._generation = 0
        
        self._vel = np.zeros_like(self.population)
        fits = self._eval_all()
        self._pbest_x = self.population.copy()
        self._pbest_f = fits.copy()
        best_idx = int(np.argmin(fits))
        self._gbest_x = self.population[best_idx].copy()
        self._gbest_f = float(fits[best_idx])
        self._inertia = LinearInertia(0.9, 0.4)
        
        self.best_score = float(np.min(fits))
        self.best_solution = self.population[int(np.argmin(fits))].copy()
        self.best_history.append(self.best_score)
    
    def _eval_all(self):  # Evaluates fitness for all individuals in island population
        return np.array([compute_fitness(ind, self.scenario)[0]
                         for ind in self.population])
    
    def _refresh_best(self): # Updates island best solution and tracks stagnation progress
        fits = self._eval_all()
        idx = int(np.argmin(fits))
        if fits[idx] < self.best_score:
            self.best_score = float(fits[idx])
            self.best_solution = self.population[idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.best_history.append(self.best_score)
        return fits
    
    def _inject_diversity(self): # Injects diversity when stagnation is detected to avoid local optima
        if self.stagnation_counter >= STAGNATION_THRESHOLD:
            fits = self._eval_all()
            worst_indices = np.argsort(fits)[-int(0.3 * self.n):]
            
            for idx in worst_indices:
                noise = np.random.normal(0, DIVERSITY_PERTURBATION, size=self.best_solution.shape)
                new_individual = repair(self.best_solution + noise, self.scenario).flatten(order='F')
                self.population[idx] = new_individual
            
            if self.operator == "PSO":
                self._vel[worst_indices] = np.random.uniform(-0.5, 0.5, size=(len(worst_indices), self.dim))
            
            self.stagnation_counter = 0
    
    def _ga_step(self, n_gens): # Creates a lightweight GA-like context to enable reuse of crossover and mutation operators inside the island model without running a full GA pipeline
        for _ in range(n_gens):
            scores = self._eval_all()
            parents = self._tourn_select(scores)
            fake = _FakeGA(self.population, self._generation)
            offspring = self._ga_helper.blx(parents, (self.n, self.dim), fake)
            offspring = self._ga_helper.nonuniform_mutate(offspring, fake)
            best_old = int(np.argmin(scores))
            offspring[0] = self.population[best_old]
            self.population = offspring
            self._generation += 1
            self._refresh_best()
            self._inject_diversity()
    
    def _tourn_select(self, scores, k=3):
        n, parents = len(self.population), []
        for _ in range(n):
            c = np.random.choice(n, min(k, n), replace=False)
            winner = c[int(np.argmin(scores[c]))]
            parents.append(self.population[winner])
        return np.array(parents)
    
    def _pso_step(self, n_steps):
        c1, c2 = 1.5, 1.5
        for t in range(n_steps):
            w = self._inertia.get(t, n_steps)
            for i in range(self.n):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self._vel[i] = (w * self._vel[i]
                                + c1 * r1 * (self._pbest_x[i] - self.population[i])
                                + c2 * r2 * (self._gbest_x - self.population[i]))
                self.population[i] = repair(
                    self.population[i] + self._vel[i], self.scenario
                ).flatten(order='F')
            fits = self._eval_all()
            for i in range(self.n):
                if fits[i] < self._pbest_f[i]:
                    self._pbest_f[i] = fits[i]
                    self._pbest_x[i] = self.population[i].copy()
                if fits[i] < self._gbest_f:
                    self._gbest_f = fits[i]
                    self._gbest_x = self.population[i].copy()
            
            self._refresh_best()
            self._inject_diversity()
    
    def evolve(self, n_steps): # Executes either GA or PSO evolution depending on island operator
        if self.operator == "PSO":
            self._pso_step(n_steps)
        else:
            self._ga_step(n_steps)
        self._refresh_best()

class _FakeGA:
    def __init__(self, population, generations_completed):
        self.population = population
        self.generations_completed = generations_completed

class DIMSPHybrid:
    def __init__(
        self,
        scenario,
        total_generations=TOTAL_GENERATIONS,
        epoch_interval=EPOCH_INTERVAL,
        island_size=ISLAND_SIZE,
        max_islands=MAX_ISLANDS,
        init_strategy="Demand_Proportional",
        verbose=False,
    ):
        self.scenario = scenario
        self.total_generations = total_generations
        self.epoch_interval = epoch_interval
        self.island_size = island_size
        self.max_islands = max_islands
        self.verbose = verbose
        
        if init_strategy == "Demand_Proportional":
            pop = initialise_demand_proportional(island_size, scenario)
        elif init_strategy == "Urgency_Biased":
            pop = initialise_urgency_biased(island_size, scenario)
        else:
            pop = initialise_random(island_size, scenario)
        
        self.islands = [Island(pop, scenario, operator="PSO", island_id=0)]
        
        self.convergence = []
        self.island_count_hist = []
        self.best_solution = None
        self.best_score = np.inf
    
    def _run_epoch(self, current_gen): # Re-clusters population and dynamically assigns GA or PSO to new islands
        Pa = np.vstack([isl.population for isl in self.islands])
        fits_all = np.array([compute_fitness(ind, self.scenario)[0] for ind in Pa])
        
        new_pops = spectral_cluster(
            Pa, fits_all,
            max_k=self.max_islands,
            sigma=SIGMA,
            min_size=MIN_CLUSTER_SIZE,
            scenario=self.scenario,
        )
        
        median_size = np.median([len(p) for p in new_pops]) # Computes median cluster size to classify islands into large (PSO) and small (GA) for adaptive hybrid optimization
        new_islands = []
        for idx, pop in enumerate(new_pops):
            op = "PSO" if len(pop) >= median_size else "GA"
            isl = Island(pop, self.scenario, operator=op, island_id=idx)
            new_islands.append(isl)
        
        self.islands = new_islands
    
    def _update_global_best(self): # Tracks global best solution across all islands
        for isl in self.islands:
            if isl.best_score < self.best_score:
                self.best_score = isl.best_score
                self.best_solution = isl.best_solution.copy()
    
    def run(self): # Main execution loop controlling evolution, clustering, and optimization cycles
        n_epochs = self.total_generations // self.epoch_interval
        steps_per_epoch = self.epoch_interval
        
        for epoch in range(n_epochs):
            current_gen = epoch * steps_per_epoch
            
            if epoch > 0:
                self._run_epoch(current_gen)
            
            for isl in self.islands:
                isl.evolve(n_steps=steps_per_epoch)
            
            self._update_global_best()
            self.convergence.append(self.best_score)
            self.island_count_hist.append(len(self.islands))
        
        best_solution = repair(self.best_solution, self.scenario).flatten(order='F')
        best_score, _ = compute_fitness(best_solution, self.scenario)
        
        return best_solution, best_score, {
            "hybrid_convergence": self.convergence,
            "island_count": self.island_count_hist,
        }

if __name__ == "__main__": # Compares hybrid model performance against standalone GA and PSO
    scenario = get_scenario()
    
    dimsp = DIMSPHybrid(scenario, total_generations=100, epoch_interval=20,
                        island_size=50, max_islands=5, verbose=False)
    sol_h, score_h, hist_h = dimsp.run()
    
    ga = DisasterReliefGA(scenario_data=scenario, config_type="baseline",
                          init_strategy="Demand_Proportional",
                          max_generations=100, population_size=50)
    _, score_g, _, _ = ga.run()
    
    pso = PSO(scenario=scenario, num_particles=50, max_iterations=100,
              ring=True, neighbors=4,
              initialization_strategy="demand_proportional")
    score_p, _, _ = pso.optimize()
    
    best_baseline = min(score_g, score_p)
    improvement = (best_baseline - score_h) / best_baseline * 100
    
    print("FINAL RESULTS:")
    print(f"GA Standalone Score:      {score_g:.4f}")
    print(f"PSO Standalone Score:     {score_p:.4f}")
    print(f"Hybrid (GA+PSO) Score:    {score_h:.4f}")
    print(f"Best Baseline Score:      {best_baseline:.4f}")
    print(f"Improvement from Hybrid:  {improvement:.2f}%")
    
    if improvement > 0:
        print(f"\nSUCCESS: Hybrid improved by {improvement:.2f}% over the best baseline")
        if score_h < score_g and score_h < score_p:
            print("Hybrid outperformed BOTH GA and PSO individually")
        elif score_h < best_baseline:
            print("Hybrid outperformed the best individual algorithm")
    else:
        print(f"\nHybrid was {abs(improvement):.2f}% worse than the best baseline")