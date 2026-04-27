import sys
import os
import numpy as np
import pygad
import io
import random
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))
from problem.scenarioM import get_scenario
from problem.FitnessFinal import compute_fitness, initialise_demand_proportional, initialise_urgency_biased,initialise_random
from problem.constraint import repair as repair_solution



class DisasterReliefGA:
    def __init__(self, scenario_data, config_type, init_strategy, max_generations=100, population_size=50 , crossover_prob=0.9, seed=None ,  f1_mode="asymmetric" , sigma_share = 14.218593557890934 , alpha_sharing =  1.102497 , K_tourn = 8):

        self.crossover_prob = crossover_prob
        self.scenario_data = scenario_data
        self.config_type = config_type
        self.init_strategy = init_strategy
        self.max_generations = max_generations
        self.population_size = population_size
        self.ga_instance = None
        self.seed = seed
        self.ga_instance = None
        self.f1_mode = f1_mode
        self.sigma_share = sigma_share
        self.alpha_sharing = alpha_sharing
        self.K_tourn = K_tourn

    def initialize_population(self, seed):

        if self.init_strategy == "Demand_Proportional":
            return initialise_demand_proportional(self.population_size, self.scenario_data, seed)
        elif self.init_strategy == "Urgency_Biased":
            return initialise_urgency_biased(self.population_size, self.scenario_data, seed)
        else:
            return initialise_random(self.population_size, self.scenario_data, seed)

    def evaluate(self, ga_inst, solution, solution_idx):
        rawScore, _ = compute_fitness(solution, self.scenario_data , self.f1_mode)
        if ga_inst.population is not None and len(ga_inst.population) > 0:
            sigma_share = self.sigma_share
            alpha_sharing = self.alpha_sharing
            niche_count = 0.0
            for other_solution in ga_inst.population:
                distance = np.linalg.norm(solution - other_solution)
                if distance <= sigma_share:
                    sh_d = 1.0 - ((distance / sigma_share) ** alpha_sharing)
                    niche_count += sh_d
            niche_count = max(1.0, niche_count)
            shared_score = rawScore * niche_count
            return -shared_score
        return -rawScore

    def blx(self, parents, offspring_size, ga_instance):
        offspring = []
        idx = 0
        # alpha=0.3 restricts exploration to 30% beyond the [min_val, max_val] interval
        # reducing the chance of generating infeasible values that require heavy repair
        alpha = 0.3
        prob = self.crossover_prob
        while len(offspring) < offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :]
            parent2 = parents[(idx + 1) % parents.shape[0], :]
            if np.random.rand() < prob:
                child = np.empty_like(parent1)
                for i in range(len(parent1)):
                    min_val = min(parent1[i], parent2[i])
                    max_val = max(parent1[i], parent2[i])
                    u = np.random.rand()
                    gamma = (1 + 2 * alpha) * u - alpha
                    zi = (1 - gamma) * min_val + gamma * max_val
                    child[i] = zi
                offspring.append(child)
            else:
                offspring.append(parent1.copy())
            idx += 2

        return np.array(offspring)

    def nonuniform_mutate(self, offspring, ga_instance):
        # we need ONE best allocation plan not a fit population
        # so we pick the upper bound (1/chr_len) for better search space coverage
        chromosome_length = offspring.shape[1]
        mutation_rate = 1 / chromosome_length
        initial_sigma = 50
        decay_factor = 1.0 - (ga_instance.generations_completed / self.max_generations)
        dynamic_sigma = initial_sigma * decay_factor     # step size shrinks as generations progress ( wide exploration early , fine-tuning near the end )
        for i in range(offspring.shape[0]):
            for j in range(offspring.shape[1]):
                region_index = j // 3
                resource_index = j % 3
                # upper bound per gene can't exceed region capacity or resource budget
                Ui = min(self.scenario_data["capacity"][region_index], self.scenario_data["budget_array"][resource_index])
                #align lower bound with problem-defined minimum allocation
                #this reduces constraint violations at the source and minimizes
                Li = self.scenario_data["minimums"][region_index][resource_index]
                r = np.random.rand()
                if r < mutation_rate:
                    mutation_size = np.random.normal(0, dynamic_sigma)
                    mutated_val = offspring[i, j] + mutation_size
                    if mutated_val < Li:
                        offspring[i, j] = Li
                    elif mutated_val > Ui:
                        offspring[i, j] = Ui
                    else:
                        offspring[i, j] = mutated_val
        return np.array(offspring)

    def uniform_mutate(self, offspring, ga_instance):
        chromosome_length = offspring.shape[1]
        mutation_rate = 1 / chromosome_length

        for i in range(offspring.shape[0]):
            for j in range(offspring.shape[1]):
                region_index = j // 3
                resource_index = j % 3
                Ui = min(self.scenario_data["capacity"][region_index],self.scenario_data["budget_array"][resource_index])
                Li = self.scenario_data["minimums"][region_index][resource_index]
                r = np.random.rand()
                if r < mutation_rate:
                    offspring[i, j] = np.random.uniform(Li, Ui)

        return np.array(offspring)

    # i kept repair_population separate so it can be reused anywhere
    # on_generation_complete is just a PyGAD hook that calls it after each generation
    # this keeps the code cleaner and avoids repeating the repair logic

    def repair_population(self, population_array):
        # it go through each solution in the population and fix any constraint violations
        for i in range(population_array.shape[0]):
            population_array[i] = repair_solution(population_array[i], self.scenario_data).flatten(order='F')
        return population_array

    def on_generation_complete(self, ga_instance):
        # this runs after each generation to make sure all solutions are valid before moving on
        ga_instance.population = self.repair_population(ga_instance.population)


    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        population = self.initialize_population(self.seed)

        #Baseline Configuration
        parent_selection = "tournament"
        K_tourn = self.K_tourn
        crossover_type = self.blx
        mutation_type = self.nonuniform_mutate
        elitism = 2

        if self.config_type == "baseline":
            pass
        elif self.config_type == "rws":
            parent_selection = "rws"
            K_tourn = None
        elif self.config_type == "uniform_crossover":
            crossover_type = "uniform"
        elif self.config_type == "uniform_mutation":
            mutation_type = self.uniform_mutate
        elif self.config_type == "generational":
            elitism = 0

        self.ga_instance = pygad.GA(
            num_generations=self.max_generations,
            num_parents_mating=self.population_size // 2,
            initial_population=population,
            fitness_func=self.evaluate,
            parent_selection_type=parent_selection,
            K_tournament=K_tourn,
            crossover_type=crossover_type,
            crossover_probability=self.crossover_prob,
            mutation_type=mutation_type,
            keep_elitism=elitism,
            stop_criteria=["saturate_20"],
            on_generation=self.on_generation_complete,
            random_seed=self.seed
        )

        self.ga_instance.run()
        return self._finalize()

    def _finalize(self):

        FinalPopulation = self.ga_instance.population

        if self.config_type == "generational":
            final_fitnesses = self.ga_instance.last_generation_fitness
            best_idx = np.argmax(final_fitnesses)
            best_solution = FinalPopulation[best_idx]
        else :
            best_position, _, _ = self.ga_instance.best_solution()
            best_solution = best_position

        final_repaired_solution = repair_solution(best_solution, self.scenario_data).flatten(order='F')
        final_score, _ = compute_fitness(final_repaired_solution, self.scenario_data,self.f1_mode)
        convergence_history = [-fit for fit in self.ga_instance.best_solutions_fitness]

        return final_repaired_solution, final_score, convergence_history, FinalPopulation


if __name__ == "__main__":
    scenario = get_scenario()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    configs_dict = {
        "baseline": "Baseline (Tournament, BLX, Non-Uniform, Elitism=2)",
        "generational": "Generational (Elitism=0)",
        "rws": "Roulette Wheel Selection",
        "uniform_crossover": "Uniform Crossover",
        "uniform_mutation": "Uniform Mutation"
    }

    final_results = {}

    for config_key, config_name in configs_dict.items():
        ga_optimizer = DisasterReliefGA(
            scenario_data=scenario,
            config_type=config_key,
            init_strategy="Demand_Proportional",
            seed=42
        )

        sol, score, hist, pop = ga_optimizer.run()
        final_results[config_name] = score

    baseline_name = configs_dict["baseline"]
    baseline_score = final_results[baseline_name]

    print("Results (difference from baseline):\n")

    for name, score in final_results.items():
        diff = baseline_score - score

        print(f"{name.ljust(50)} {score:.4f} | Δ = {diff:+.4f}")