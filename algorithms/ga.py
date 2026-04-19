import sys
import os
import numpy as np
import pygad

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))

from problem.scenarioM import get_scenario
from problem.FitnessFinal import compute_fitness, initialise_demand_proportional, initialise_urgency_biased,initialise_random
from problem.constraint import repair as repair_solution



class DisasterReliefGA:
    def __init__(self, scenario_data, config_type, init_strategy, max_generations=100, population_size=50):

        self.scenario_data = scenario_data
        self.config_type = config_type
        self.init_strategy = init_strategy
        self.max_generations = max_generations
        self.population_size = population_size
        self.ga_instance = None

    def initialize_population(self):

        if self.init_strategy == "Demand_Proportional":
            return initialise_demand_proportional(self.population_size, self.scenario_data)
        elif self.init_strategy == "Urgency_Biased":
            return initialise_urgency_biased(self.population_size, self.scenario_data)
        else:
            return initialise_random(self.population_size, self.scenario_data)

    def evaluate(self, ga_inst, solution, solution_idx):
        rawScore, _ = compute_fitness(solution, self.scenario_data)
        if ga_inst.population is not None and len(ga_inst.population) > 0:
            sigma_share = 150.0  # Niche size
            alpha = 1.0  # Sharing level
            niche_count = 0.0
            for other_solution in ga_inst.population:
                distance = np.linalg.norm(solution - other_solution)
                if distance <= sigma_share:
                    sh_d = 1.0 - ((distance / sigma_share) ** alpha)
                    niche_count += sh_d
            shared_score = rawScore / niche_count
            return -shared_score
        return -rawScore

    def crossover(self, parents, offspring_size, ga_instance):
        offspring = []
        idx = 0
        alpha = 0.5
        Li = 0

        while len(offspring) < offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :]
            parent2 = parents[(idx + 1) % parents.shape[0], :]

            child = np.empty_like(parent1)
            for i in range(len(parent1)):
                min_val = min(parent1[i], parent2[i])
                max_val = max(parent1[i], parent2[i])
                u = np.random.rand()
                gamma = (1 + 2 * alpha) * u - alpha
                zi = (1 - gamma) * min_val + gamma * max_val

                if zi < Li:
                    child[i] = Li
                else:
                    child[i] = zi

            offspring.append(child)
            idx += 2

        return np.array(offspring)

    def mutate(self, offspring, ga_instance):
        mutation_rate = 0.1
        initial_sigma = 50
        decay_factor = 1.0 - (ga_instance.generations_completed / self.max_generations)
        dynamic_sigma = initial_sigma * decay_factor
        Li = 0
        for i in range(offspring.shape[0]):
            for j in range(offspring.shape[1]):
                Ui = self.scenario_data["capacity"][j % self.scenario_data["n_regions"]]     # ?
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
        return self.repair(offspring)

    def repair(self, offspring):
        for i in range(offspring.shape[0]):
            offspring[i] = repair_solution(offspring[i], self.scenario_data).flatten(order='F')
        return offspring

    def run(self):
        population = self.initialize_population()
        if self.config_type == "config1":
            parent_selection = "tournament"
            K_tourn = 3
            crossover_type = self.crossover
            mutation_type = self.mutate
            elitism = 2
        else:
            parent_selection = "rws"
            K_tourn = None
            crossover_type = "uniform"
            mutation_type = "random"
            elitism = 0
        self.ga_instance = pygad.GA(
            num_generations=self.max_generations,
            num_parents_mating=self.population_size // 2,
            initial_population=population,
            fitness_func=self.evaluate,
            parent_selection_type=parent_selection,
            K_tournament=K_tourn,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            keep_elitism=elitism,
            stop_criteria=["saturate_20"]
        )

        self.ga_instance.run()
        return self._finalize()

    def _finalize(self):

        FinalPopulation = self.ga_instance.population

        if self.config_type == "config1":
            Best_Position, _, _ = self.ga_instance.best_solution()
            BestSolution = Best_Position

        elif self.config_type == "config2":
            final_fitnesses = self.ga_instance.last_generation_fitness
            best_idx = np.argmax(final_fitnesses)
            BestSolution = FinalPopulation[best_idx]

        Final_Repaired_Solution = repair_solution(BestSolution, self.scenario_data).flatten(order='F')
        Final_Score, _ = compute_fitness(Final_Repaired_Solution, self.scenario_data)
        Convergence_History = [-fit for fit in self.ga_instance.best_solutions_fitness]

        return Final_Repaired_Solution, Final_Score, Convergence_History, FinalPopulation


if __name__ == "__main__":
    scenario = get_scenario()

    print("Config 1 (Proposed Method)")
    ga_optimizer_1 = DisasterReliefGA(scenario_data=scenario, config_type="config1",init_strategy="Demand_Proportional")
    sol1, score1, hist1, pop1 = ga_optimizer_1.run()
    print(f"Config 1 Score: {score1:.4f}\n")

    print("Config 2 (Baseline)")
    ga_optimizer_2 = DisasterReliefGA(scenario_data=scenario, config_type="config2",init_strategy="Demand_Proportional")
    sol2, score2, hist2, pop2 = ga_optimizer_2.run()
    print(f"Config 2 Score: {score2:.4f}")