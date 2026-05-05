#!/usr/bin/env python3
"""
Feasibility Checker for Disaster Relief Algorithms (GA, PSO, Hybrid-DIMSP)
Runs each configuration across 3 random seeds; marks configuration feasible if any run yields a feasible solution.
Counts number of feasible scenarios per algorithm and ranks them.
"""

import sys
import os
import numpy as np
import random

# Add paths to allow imports from problem/ and algorithms/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'algorithms')))
from problem.scenarioM import get_scenario
from problem.FitnessFinal import compute_fitness
from algorithms.ga import DisasterReliefGA
from algorithms.pso import PSO, LinearInertia
from hybridDIM_SP import DIMSPHybrid

# ----------------------------------------------------------------------
# Parameters
SEEDS = [42, 123, 456]          # three different seeds per config
FEASIBILITY_TOL = 1e-6          # penalty below this -> feasible

# ----------------------------------------------------------------------
# Helper: run a single GA configuration and return (best_fitness, best_solution, details)
def run_ga(scenario, init_strategy, config_type, seed, max_generations=100, population_size=105):
    ga = DisasterReliefGA(
        scenario_data=scenario,
        init_strategy=init_strategy,
        config_type=config_type,
        max_generations=max_generations,
        population_size=population_size,
        seed=seed
    )
    best_solution, best_score, _, _, details = ga.run()
    return best_score, best_solution, details

# Helper: run a single PSO configuration (default canonical, global, balanced, linear inertia)
def run_pso(scenario, seed, num_particles=30, max_iterations=100):
    inertia = LinearInertia(w_start=0.9, w_end=0.5)
    pso = PSO(
        scenario=scenario,
        num_particles=num_particles,
        max_iterations=max_iterations,
        c1=1.5,
        c2=1.5,
        inertia=inertia,
        bare=False,
        ring=False,
        seed=seed,
        initialization_strategy='demand_proportional'
    )
    best_fitness, best_solution, _ = pso.optimize()
    # recompute details for feasibility check
    _, details = compute_fitness(best_solution, scenario)
    return best_fitness, best_solution, details

# Helper: run a single Hybrid-DIMSP configuration
def run_hybrid(scenario, seed, total_generations=100, island_size=50):
    hybrid = DIMSPHybrid(
        scenario=scenario,
        total_generations=total_generations,
        epoch_interval=10,
        island_size=island_size,
        init_strategy="Demand_Proportional",
        seed=seed,
        verbose=False
    )
    best_solution, best_score, info = hybrid.run()
    details = info["details"]
    return best_score, best_solution, details

# ----------------------------------------------------------------------
# Define configurations for each algorithm
# GA configurations (7 scenarios)
ga_configs = [
    {"name": "GA-Scenario-Default",          "scenario": get_scenario(),                "init_strategy": "Demand_Proportional", "config_type": "baseline"},
    {"name": "GA-Scenario-Epidemic",         "scenario": get_scenario("Epidemic"),     "init_strategy": "Demand_Proportional", "config_type": "baseline"},
    {"name": "GA-Scenario-Floods",           "scenario": get_scenario("Floods"),       "init_strategy": "Demand_Proportional", "config_type": "baseline"},
    {"name": "GA-Scenario-LargeDisaster",    "scenario": get_scenario("Large Disaster"),"init_strategy": "Demand_Proportional", "config_type": "baseline"},
    {"name": "GA-Scenario-ResourceShortage", "scenario": get_scenario("Resource Shortage"),"init_strategy": "Demand_Proportional", "config_type": "baseline"},
    {"name": "GA-Scenario-WorstCase",        "scenario": get_scenario("Worst Case"),   "init_strategy": "Demand_Proportional", "config_type": "baseline"},
    {"name": "GA-WorstCase-UrgencyInit",     "scenario": get_scenario("Worst Case"),   "init_strategy": "Urgency_Biased",      "config_type": "baseline"}
]

# PSO configurations (same 7 scenarios, using default PSO)
pso_configs = [
    {"name": "PSO-Scenario-Default",          "scenario": get_scenario()},
    {"name": "PSO-Scenario-Epidemic",         "scenario": get_scenario("Epidemic")},
    {"name": "PSO-Scenario-Floods",           "scenario": get_scenario("Floods")},
    {"name": "PSO-Scenario-LargeDisaster",    "scenario": get_scenario("Large Disaster")},
    {"name": "PSO-Scenario-ResourceShortage", "scenario": get_scenario("Resource Shortage")},
    {"name": "PSO-Scenario-WorstCase",        "scenario": get_scenario("Worst Case")},
    {"name": "PSO-WorstCase-UrgencyInit",     "scenario": get_scenario("Worst Case")}   # identical scenario, just for count
]

# Hybrid configurations (same 7 scenarios)
hybrid_configs = [
    {"name": "Hybrid-Scenario-Default",          "scenario": get_scenario()},
    {"name": "Hybrid-Scenario-Epidemic",         "scenario": get_scenario("Epidemic")},
    {"name": "Hybrid-Scenario-Floods",           "scenario": get_scenario("Floods")},
    {"name": "Hybrid-Scenario-LargeDisaster",    "scenario": get_scenario("Large Disaster")},
    {"name": "Hybrid-Scenario-ResourceShortage", "scenario": get_scenario("Resource Shortage")},
    {"name": "Hybrid-Scenario-WorstCase",        "scenario": get_scenario("Worst Case")},
    {"name": "Hybrid-WorstCase-UrgencyInit",     "scenario": get_scenario("Worst Case")}
]

# ----------------------------------------------------------------------
def evaluate_algorithm(configs, run_func):
    """
    For each configuration in configs, run it with seeds from SEEDS.
    Mark config feasible if any run gives penalty < FEASIBILITY_TOL.
    Returns a dict: {config_name: (feasible_bool, best_fitness_over_seeds, best_details)}
    """
    results = {}
    for cfg in configs:
        name = cfg["name"]
        scenario = cfg["scenario"]
        feasible = False
        best_fitness = np.inf
        best_details = None
        
        for seed in SEEDS:
            print(f"  Running {name} with seed={seed} ...")
            try:
                # Call the appropriate run function with correct arguments
                # run_func signature depends on algorithm
                if run_func.__name__ == "run_ga":
                    fitness, _, details = run_func(scenario, cfg["init_strategy"], cfg["config_type"], seed)
                else:
                    fitness, _, details = run_func(scenario, seed)
                
                penalty = details.get("penalty", 1e10)
                if penalty < FEASIBILITY_TOL:
                    feasible = True
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_details = details
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        results[name] = {
            "feasible": feasible,
            "best_fitness": best_fitness,
            "best_penalty": best_details["penalty"] if best_details else None,
            "details": best_details
        }
    return results

# ----------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("FEASIBILITY CHECKER FOR DISASTER RELIEF ALGORITHMS")
    print("="*70 + "\n")

    # 1. Evaluate GA
    print("--- Evaluating GA ---")
    ga_results = evaluate_algorithm(ga_configs, run_ga)
    
    # 2. Evaluate PSO
    print("\n--- Evaluating PSO ---")
    pso_results = evaluate_algorithm(pso_configs, run_pso)
    
    # 3. Evaluate Hybrid
    print("\n--- Evaluating Hybrid ---")
    hybrid_results = evaluate_algorithm(hybrid_configs, run_hybrid)
    
    # Count feasible configurations per algorithm
    ga_feasible_count = sum(1 for v in ga_results.values() if v["feasible"])
    pso_feasible_count = sum(1 for v in pso_results.values() if v["feasible"])
    hybrid_feasible_count = sum(1 for v in hybrid_results.values() if v["feasible"])
    
    # Ranking
    ranking = sorted([
        ("GA", ga_feasible_count),
        ("PSO", pso_feasible_count),
        ("Hybrid", hybrid_feasible_count)
    ], key=lambda x: x[1], reverse=True)
    
    # Output detailed results
    print("\n" + "="*70)
    print("DETAILED FEASIBILITY RESULTS")
    print("="*70)
    
    print("\nGA:")
    for name, res in ga_results.items():
        print(f"  {name:40} Feasible: {res['feasible']:5}  Best fitness: {res['best_fitness']:.4f}  Penalty: {res['best_penalty']:.2e}")
    
    print("\nPSO:")
    for name, res in pso_results.items():
        print(f"  {name:40} Feasible: {res['feasible']:5}  Best fitness: {res['best_fitness']:.4f}  Penalty: {res['best_penalty']:.2e}")
    
    print("\nHybrid:")
    for name, res in hybrid_results.items():
        print(f"  {name:40} Feasible: {res['feasible']:5}  Best fitness: {res['best_fitness']:.4f}  Penalty: {res['best_penalty']:.2e}")
    
    # Summary ranking
    print("\n" + "="*70)
    print("ALGORITHM RANKING (by number of feasible configurations)")
    print("="*70)
    for i, (algo, count) in enumerate(ranking, 1):
        print(f"{i}. {algo:10} -> {count} / {len(ga_configs)} configurations feasible")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()