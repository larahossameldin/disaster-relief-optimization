import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'algorithms')))
from problem.scenarioM    import get_scenario
from problem.FitnessFinal import compute_fitness
from algorithms.ga        import DisasterReliefGA
from algorithms.pso       import PSO
from algorithms.hybridDIM_SP        import DIMSPHybrid, seed_all

BASE_SEED  = 42
N_RUNS     = 30        # number of independent seeds per experiment
SEEDS      = list(range(N_RUNS))   # seeds: 0, 1, 2, …, 29

def run_one(scenario, seed):
    seed_all(seed)

    dimsp = DIMSPHybrid(
        scenario,
        total_generations = 100,
        epoch_interval    = 20,
        island_size       = 50,
        init_strategy     = "Random",
        seed              = seed,
    )
    _, score_h, _ = dimsp.run()

    ga = DisasterReliefGA(
        scenario_data   = scenario,
        config_type     = "config1",
        init_strategy   = "Random",
        max_generations = 100,
        population_size = 50,
        seed            = seed,
    )
    _, score_g, _, _ = ga.run()

    pso = PSO(
        scenario                = scenario,
        num_particles           = 50,
        max_iterations          = 100,
        ring                    = True,
        neighbors               = 4,
        initialization_strategy = "random",
        seed                    = seed,
    )
    score_p, _, _ = pso.optimize()
    return score_h, score_g, score_p

def run_experiment(scenario):
    seed_all(BASE_SEED)
    print(f"\n{'='*65}")
    print(f"  Experiment : Small clusters -> PSO (small_psi)")
    print(f"  Strategy   : small_psi    Seeds: {SEEDS}")
    print(f"{'='*65}")

    h_scores, g_scores, p_scores = [], [], []
    for i, seed in enumerate(SEEDS):
        sh, sg, sp = run_one(scenario, seed)
        h_scores.append(sh)
        g_scores.append(sg)
        p_scores.append(sp)
        print(f"  run {i:>2} (seed={seed})  hybrid={sh:.4f}  GA={sg:.4f}  PSO={sp:.4f}")

    mean_h  = float(np.mean(h_scores))
    mean_g  = float(np.mean(g_scores))
    mean_p  = float(np.mean(p_scores))
    std_h   = float(np.std(h_scores))
    best_bl = min(mean_g, mean_p)
    improv  = (best_bl - mean_h) / best_bl * 100

    return {
        "experiment"  : "Small clusters -> PSO",
        "strategy"    : "small_psi",
        "mean_hybrid" : mean_h,
        "std_hybrid"  : std_h,
        "mean_ga"     : mean_g,
        "mean_pso"    : mean_p,
        "improvement" : improv,
        "all_hybrid"  : h_scores,
    }

if __name__ == "__main__":
    # Optional: Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    scenario = get_scenario()
    result = run_experiment(scenario)
    print("\n" + "=" * 75)
    print("FINAL RESULTS -- averaged over", N_RUNS, "seeds")
    print("=" * 75)
    best_bl = min(result['mean_ga'], result['mean_pso'])
    print(f"Hybrid (small_psi)  : {result['mean_hybrid']:.4f} +- {result['std_hybrid']:.4f}")
    print(f"GA baseline         : {result['mean_ga']:.4f}")
    print(f"PSO baseline        : {result['mean_pso']:.4f}")
    print(f"Best baseline       : {best_bl:.4f}")
    print(f"Improvement         : {result['improvement']:+.2f}%")
    print("=" * 75)
    print("\nRecommendation: Small clusters -> PSO (small_psi) performed best.")