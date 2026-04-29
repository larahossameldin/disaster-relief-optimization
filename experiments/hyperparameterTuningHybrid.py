import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'algorithms')))
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from problem.scenarioM       import get_scenario
import algorithms.hybridDIM_SP as h  # import the module to modify its globals
from algorithms.hybridDIM_SP import DIMSPHybrid, seed_all

#Setup 
SCENARIO = get_scenario()
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fixed (match test.py baselines so comparison is fair)
TOTAL_GENERATIONS = 100
ISLAND_SIZE       = 50

# 5 seeds per trial — fast enough for 50 trials, diverse enough for reliability
TUNING_SEEDS = [0, 1, 2, 3, 4]

#BEST_PARAMS placeholder
BEST_PARAMS = {}   # will be filled after tuning

#Helper: run one trial with overridden globals
def run_hybrid(trial_params, seeds, trial_number):
    # Override module‑level hyperparameters before creating the hybrid
    h.SIGMA                  = trial_params["sigma"]
    h.MIN_CLUSTER_SIZE       = trial_params["min_cluster_size"]
    h.C1                     = trial_params["c1"]
    h.C2                     = trial_params["c2"]
    h.INERTIA_TYPE           = trial_params["inertia_type"]
    h.W_START                = trial_params.get("w_start", 0.9)
    h.W_END                  = trial_params.get("w_end", 0.4)
    h.RING                   = trial_params["ring"]
    h.NEIGHBORS              = trial_params["neighbors"]
    h.BARE                   = trial_params["bare"]
    h.BARE_PROB              = trial_params.get("bare_prob", 0.5)
    h.EXTINCTION_THRESHOLD   = trial_params["extinction_threshold"]
    h.EXTINCTION_KEEP_RATIO  = trial_params["extinction_keep_ratio"]

    fitnesses = []
    for seed in seeds:
        seed_all(seed)
        hybrid = DIMSPHybrid(
            scenario              = SCENARIO,
            total_generations     = TOTAL_GENERATIONS,
            island_size           = ISLAND_SIZE,
            epoch_interval        = trial_params["epoch_interval"],
            init_strategy         = trial_params["init_strategy"],
            seed                  = seed,
        )
        _, best_f, _ = hybrid.run()
        fitnesses.append(best_f)
    return float(np.mean(fitnesses))

#Optuna objective 
def objective(trial):
    bare          = trial.suggest_categorical("bare",  [True, False])
    ring          = trial.suggest_categorical("ring",  [True, False])
    inertia_type  = trial.suggest_categorical("inertia_type", ["linear", "random"])

    params = {
        "c1"          : trial.suggest_float("c1",  0.3, 3.0),
        "c2"          : trial.suggest_float("c2",  0.3, 3.0),
        "inertia_type": inertia_type,
        "bare"        : bare,
        "ring"        : ring,
        "epoch_interval"   : trial.suggest_int(  "epoch_interval",    10,   30),
        "init_strategy"    : trial.suggest_categorical(
                                 "init_strategy",
                                 ["Random", "Demand_Proportional", "Urgency_Biased"]
                             ),
        "sigma"            : trial.suggest_float("sigma",             0.5,   3.0),
        "min_cluster_size" : trial.suggest_int(  "min_cluster_size",   5,    20),
        "extinction_threshold"  : trial.suggest_int(  "extinction_threshold",  5,  30),
        "extinction_keep_ratio" : trial.suggest_float("extinction_keep_ratio", 0.2, 0.8),
    }

    if inertia_type == "linear":
        params["w_start"] = trial.suggest_float("w_start", 0.6, 1.0)
        params["w_end"]   = trial.suggest_float("w_end",   0.1, 0.5)
    else:
        params["w_start"] = 0.9   # dummy, will be ignored in random mode
        params["w_end"]   = 0.4

    if bare:
        params["bare_prob"] = trial.suggest_float("bare_prob", 0.3, 0.95)
    else:
        params["bare_prob"] = 0.5   # dummy, not used when bare=False

    if ring:
        params["neighbors"] = trial.suggest_int("neighbors", 2, 8)
    else:
        params["neighbors"] = 4

    return run_hybrid(params, seeds=TUNING_SEEDS, trial_number=trial.number)

#Logging callback
def log_callback(study, trial):
    best    = study.best_value
    current = trial.value
    tag     = " NEW BEST" if current == best else ""
    print(
        f"[Trial {trial.number + 1:>3}] "
        f"Fitness = {current:.6f}   |   Best = {best:.6f}   {tag}"
    )

# Main
if __name__ == "__main__":
    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(
            n_startup_trials = 10,
            seed             = 42,
        ),
    )

    print("\n" + "=" * 50)
    print("   DIMSPHybrid TPE Hyperparameter Search Started")
    print("=" * 50 + "\n")

    study.optimize(objective, n_trials=50, callbacks=[log_callback])

    print("\n" + "=" * 50)
    print("   FINAL RESULTS")
    print("=" * 50)

    # Build final parameter dict with all values (including those set in module)
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params["w_start"] = best_trial.params.get("w_start", 0.9)
    best_params["w_end"]   = best_trial.params.get("w_end", 0.4)
    best_params["bare_prob"] = best_trial.params.get("bare_prob", 0.5)
    best_params["neighbors"] = best_trial.params.get("neighbors", 4)

    print(f"\nBest Fitness: {study.best_value:.6f}")
    print(f"Best Params : {best_params}")

    # Results table
    results      = []
    best_so_far  = []
    current_best = float("inf")
    for t in study.trials:
        if t.value is not None:
            row = {"trial": t.number + 1, "fitness": t.value}
            row.update(t.params)
            results.append(row)
            current_best = min(current_best, t.value)
            best_so_far.append(current_best)

    df = pd.DataFrame(results).sort_values("fitness")
    print("\nTop 5 Results:")
    print(df.head(5).to_string(index=False))

    # Convergence plot
    plt.figure(figsize=(8, 4))
    plt.plot(best_so_far)
    plt.xlabel("Trial")
    plt.ylabel("Best Fitness So Far")
    plt.title("DIMSPHybrid TPE Convergence")
    plt.grid()
    save_path = os.path.join(PLOTS_DIR, "tpe_hybrid_convergence.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    print("\nCopy the dictionary printed above into BEST_PARAMS at the top of this file.")