import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))

import numpy as np
import optuna
import pandas as pd #for results table
import matplotlib.pyplot as plt

from problem.scenarioM import get_scenario
from algorithms.pso import PSO, LinearInertia, RandomInertia


#SETUP
SCENARIO = get_scenario()

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


#INERTIA
def make_inertia(name, seed):
    if name == "linear":
        return LinearInertia(w_start=0.9, w_end=0.4)
    elif name == "random":
        return RandomInertia(np.random.default_rng(seed))


#RUN PSO
def run_pso(params, n_runs, trial_number):
    fitnesses = []

    for i in range(n_runs):
        seed = trial_number * 100 + i # Unique seed for each trial and run inside it

        pso = PSO(
            scenario=SCENARIO,
            num_particles=params["num_particles"],
            max_iterations=params["max_iterations"],
            c1=params["c1"],
            c2=params["c2"],
            inertia=make_inertia(params["inertia"], seed),
            bare=params["bare"],
            bare_prob=params["bare_prob"],
            ring=params["ring"],
            neighbors=params["neighbors"],
            initialization_strategy=params["initialization_strategy"],
            seed=seed
        )

        best_f, _, _ = pso.optimize()
        fitnesses.append(best_f)

    return float(np.mean(fitnesses)) #return average fitness across runs for this trial


# OBJECTIVE
def objective(trial):
    #These are sampled first because the next parameters depend on them conditionally
    bare = trial.suggest_categorical("bare", [True, False])
    ring = trial.suggest_categorical("ring", [True, False])

    params = {
        "num_particles": trial.suggest_int("num_particles", 10, 80),
        "max_iterations": trial.suggest_int("max_iterations", 50, 200),
        "c1": trial.suggest_float("c1", 0.3, 3.0),
        "c2": trial.suggest_float("c2", 0.3, 3.0),
        "inertia": trial.suggest_categorical("inertia", ["linear", "random"]),
        "bare": bare,
        "ring": ring,
        "initialization_strategy": trial.suggest_categorical(
            "initialization_strategy",
            ["random", "demand_proportional", "urgency_biased"]
        )
    }
#Conditional parameters
    if bare:
        params["bare_prob"] = trial.suggest_float("bare_prob", 0.3, 0.95)
    else:
        params["bare_prob"] = 0.5

    if ring:
        params["neighbors"] = trial.suggest_int("neighbors", 2, 8)
    else:
        params["neighbors"] = 4

    return run_pso(params, n_runs=3, trial_number=trial.number)



#LOGGING
def log_callback(study, trial):
    best = study.best_value
    current = trial.value

    tag = " NEW BEST" if current == best else ""
    print(
        f"[Trial {trial.number+1:>3}] "
        f"Fitness = {current:.6f}   |   Best = {best:.6f}   {tag}"
        # f"   |   Params: {trial.params}"
    )


# MAIN
#RUNS ONLY IF U RUN THIS FILE DIRECTLY
if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10, #TPE needs some initial data to build its model, so we start with RANDOM SAMPLING for a few trials
            seed=42 #makes the TPE sampling deterministic, so we get the same sequence of hyperparameters each time we run the script
        )
    )

    print("\n" + "="*50)
    print("   TPE Hyperparameter Search Started")
    print("="*50 + "\n")

    study.optimize(objective, n_trials=50, callbacks=[log_callback])
    #Optuna calls objective(trial) 50 times,
    # each time with a new trial object whose suggest_* calls are guided by TPE.
    # After each trial completes, log_callback is called. 
    # The study internally tracks all trial results and uses them to inform the TPE sampler for future trials.
    print("\n" + "="*50)
    print("   FINAL RESULTS")
    print("="*50)

    print(f"\nBest Fitness: {study.best_value:.6f}")
    print(f"Best Params : {study.best_params}")

    # RESULTS TABLE
    results = [] #list of dicts, where each dict is a row in the final results table
    best_so_far = [] #to track the best fitness value found up to each trial, for convergence plotting

    current_best = float("inf") #minimizing

    for t in study.trials:
        if t.value is not None:
            row = {"trial": t.number+1, "fitness": t.value}
            row.update(t.params) #merges the t.params dict into the row dict
            results.append(row) 

            current_best = min(current_best, t.value)
            best_so_far.append(current_best)

    df = pd.DataFrame(results).sort_values("fitness")

    print("\nTop 5 Results:")
    print(df.head(5).to_string(index=False))

    # PLOT: CONVERGENCE
    plt.figure(figsize=(8, 4))
    plt.plot(best_so_far)
    plt.xlabel("Trial")
    plt.ylabel("Best Fitness So Far")
    plt.title("TPE Convergence")
    plt.grid()

    save_path = os.path.join(PLOTS_DIR, "tpe_pso_convergence.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight") #dpi dots per inch, bbox_inches="tight" trims whitespace around the figure
    print(f"\nPlot saved to: {save_path}")
    plt.show()