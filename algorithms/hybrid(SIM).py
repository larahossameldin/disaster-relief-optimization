import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))

from problem.scenarioM    import get_scenario
from problem.FitnessFinal import compute_fitness, initialise_demand_proportional
from problem.constraint   import repair
from algorithms.ga  import DisasterReliefGA
from algorithms.pso import PSO

TOTAL_GENERATIONS = 100
EXCHANGE_INTERVAL = 10
ISLAND_SIZE       = 50
GA_CONFIG         = "config1"
INIT_STRATEGY     = "Demand_Proportional"

class StaticIslandHybrid:
    def __init__(self, scenario):
        self.scenario = scenario
        self.total_generations = TOTAL_GENERATIONS
        self.exchange_interval = EXCHANGE_INTERVAL
        self.island_size = ISLAND_SIZE
        self.dim = scenario["dim"]
        
        self.ga_population = initialise_demand_proportional(self.island_size, self.scenario)
        self.pso_population = initialise_demand_proportional(self.island_size, self.scenario)
        
        self.ga_best_solution = None
        self.ga_best_score = np.inf
        self.ga_generation = 0
        
        self.pso_best_solution = None
        self.pso_best_score = np.inf
        self.pso_iteration = 0
        
        self.global_best_solution = None
        self.global_best_score = np.inf
        
        self.ga = DisasterReliefGA(
            scenario_data=scenario,
            config_type=GA_CONFIG,
            init_strategy=INIT_STRATEGY,
            max_generations=TOTAL_GENERATIONS,
            population_size=ISLAND_SIZE,
        )
        self.ga.population = self.ga_population.copy()
        
        self.pso = PSO(
            scenario=scenario,
            num_particles=ISLAND_SIZE,
            max_iterations=TOTAL_GENERATIONS,
            ring=True,
            neighbors=4,
            initialization_strategy="demand_proportional"
        )
        self.pso.particles = self.pso_population.copy()
        
        self.pso_velocities = np.random.uniform(-1, 1, self.pso.particles.shape)
        self.pso_pbest = self.pso.particles.copy()
        self.pso_pbest_scores = self._get_fitnesses(self.pso_pbest)
        
        best_idx = np.argmin(self.pso_pbest_scores)
        self.pso_island_best = self.pso_pbest[best_idx].copy()
        self.pso_island_best_score = self.pso_pbest_scores[best_idx]
        
        self._update_ga_best()
        self._update_pso_best()
        self._update_global_best()
    
    def _get_fitnesses(self, population):
        return np.array([compute_fitness(ind, self.scenario)[0] for ind in population]) 
    
    def _update_ga_best(self):
        fits = self._get_fitnesses(self.ga_population)
        best_idx = np.argmin(fits)
        self.ga_best_score = fits[best_idx]
        self.ga_best_solution = self.ga_population[best_idx].copy()
    
    def _update_pso_best(self):
        fits = self._get_fitnesses(self.pso_population)
        best_idx = np.argmin(fits)
        self.pso_best_score = fits[best_idx]
        self.pso_best_solution = self.pso_population[best_idx].copy()
    
    def _update_global_best(self):
        if self.ga_best_score < self.global_best_score:
            self.global_best_score = self.ga_best_score
            self.global_best_solution = self.ga_best_solution.copy()
        if self.pso_best_score < self.global_best_score:
            self.global_best_score = self.pso_best_score
            self.global_best_solution = self.pso_best_solution.copy()
    
    def ga_step(self):
        scores = self._get_fitnesses(self.ga.population)
        n = self.island_size
        
        parents = []
        for _ in range(n):
            candidates = np.random.choice(n, size=5, replace=False)
            winner = candidates[np.argmin(scores[candidates])]
            parents.append(self.ga.population[winner])
        parents = np.array(parents)
        
        fake_ga = type('obj', (), {'population': self.ga.population, 
                                  'generations_completed': self.ga_generation})()
        offspring = self.ga.blx(parents, (n, self.dim), fake_ga)
        offspring = self.ga.nonuniform_mutate(offspring, fake_ga)
        
        best_indices = np.argsort(scores)[:4]
        for i in range(4):
            offspring[i] = self.ga.population[best_indices[i]].copy()
        
        self.ga.population = offspring
        self.ga_population = offspring.copy()
        self.ga_generation += 1
        self._update_ga_best()
    
    def pso_step(self):
        w = 0.95 - 0.7 * (self.pso_iteration / self.total_generations)
        c1 = 2.0 - 0.5 * (self.pso_iteration / self.total_generations)
        c2 = 0.5 + 1.5 * (self.pso_iteration / self.total_generations)
        
        current_scores = self._get_fitnesses(self.pso.particles)
        
        improved_mask = current_scores < self.pso_pbest_scores
        self.pso_pbest[improved_mask] = self.pso.particles[improved_mask].copy()
        self.pso_pbest_scores[improved_mask] = current_scores[improved_mask]
        
        current_best_idx = np.argmin(current_scores)
        if current_scores[current_best_idx] < self.pso_island_best_score:
            self.pso_island_best_score = current_scores[current_best_idx]
            self.pso_island_best = self.pso.particles[current_best_idx].copy()
        
        for i in range(len(self.pso.particles)):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = c1 * r1 * (self.pso_pbest[i] - self.pso.particles[i])
            social = c2 * r2 * (self.pso_island_best - self.pso.particles[i])
            self.pso_velocities[i] = w * self.pso_velocities[i] + cognitive + social
            max_velocity = 3.0 * (1 - self.pso_iteration / self.total_generations) + 0.5
            self.pso_velocities[i] = np.clip(self.pso_velocities[i], -max_velocity, max_velocity)
            new_position = self.pso.particles[i] + self.pso_velocities[i]
            self.pso.particles[i] = repair(new_position, self.scenario).flatten(order='F')
        
        self.pso_population = self.pso.particles.copy()
        self.pso_iteration += 1
        self._update_pso_best()
    
    def exchange_solutions(self):
        ga_fits = self._get_fitnesses(self.ga_population)
        pso_fits = self._get_fitnesses(self.pso_population)
        
        ga_worst_idx = np.argmax(ga_fits)
        pso_worst_idx = np.argmax(pso_fits)
        
        if self.ga_best_score < pso_fits[pso_worst_idx]:
            self.pso_population[pso_worst_idx] = self.ga_best_solution.copy()
        
        if self.pso_best_score < ga_fits[ga_worst_idx]:
            self.ga_population[ga_worst_idx] = self.pso_best_solution.copy()
        
        self.ga.population = self.ga_population.copy()
        self.pso.particles = self.pso_population.copy()
        
        if self.ga_best_score < pso_fits[pso_worst_idx]:
            new_particle_score = self.ga_best_score
            if new_particle_score < self.pso_pbest_scores[pso_worst_idx]:
                self.pso_pbest[pso_worst_idx] = self.ga_best_solution.copy()
                self.pso_pbest_scores[pso_worst_idx] = new_particle_score
        
        self._update_ga_best()
        self._update_pso_best()
    
    def run(self):
        for gen in range(self.total_generations):
            self.ga_step()
            self.pso_step()
            self._update_global_best()
            
            if (gen + 1) % self.exchange_interval == 0:
                self.exchange_solutions()
                self._update_global_best()
        
        final_solution = repair(self.global_best_solution, self.scenario).flatten(order='F')
        final_score, _ = compute_fitness(final_solution, self.scenario)
        
        return final_solution, final_score

if __name__ == "__main__":
    scenario = get_scenario()
    
    hybrid = StaticIslandHybrid(scenario)
    sol_h, score_h = hybrid.run()
    
    ga = DisasterReliefGA(scenario_data=scenario, config_type=GA_CONFIG,
                          init_strategy=INIT_STRATEGY, max_generations=100, 
                          population_size=50)
    best_sol, score_g, conv_g, _ = ga.run()
    
    pso = PSO(scenario=scenario, num_particles=50, max_iterations=100,
              ring=True, neighbors=4, initialization_strategy="demand_proportional")
    score_p, _, _ = pso.optimize()

    print("FINAL RESULTS")
    print(f"GA Score:      {score_g:.4f}")
    print(f"PSO Score:     {score_p:.4f}")
    print(f"Hybrid Score:  {score_h:.4f}")
    
    best_baseline = min(score_g, score_p)
    improvement = (best_baseline - score_h) / best_baseline * 100
    print(f"\nImprovement from Hybrid: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\nSUCCESS: Hybrid improved by {improvement:.2f}% over the best baseline")
        if score_h < score_g and score_h < score_p:
            print("Hybrid outperformed BOTH GA and PSO individually")
        elif score_h < best_baseline:
            print("Hybrid outperformed the best individual algorithm")
    else:
        print(f"\nHybrid was {abs(improvement):.2f}% worse than the best baseline")