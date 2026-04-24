import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))

import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from problem.scenarioM import get_scenario
from algorithms.ga import DisasterReliefGA


SCENARIO = get_scenario()
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

F1_MODE = "asymmetric"  # Fixed — not tuned

def run_ga(params, n_runs, trial_number):
    fitnesses = []
    for i in range(n_runs):
        seed = trial_number * 100 + i

        ga = DisasterReliefGA(
            scenario_data=SCENARIO,
            config_type="baseline",
            init_strategy="Demand_Proportional",
            seed=seed,
            f1_mode=F1_MODE,
            sigma_share=params["sigma_share"],
            alpha_sharing=params["alpha_sharing"],
            K_tourn=params["K_tourn"],
            population_size=params["population_size"]
        )

        _, best_f, _, _ = ga.run()
        fitnesses.append(best_f)

    return float(np.mean(fitnesses))

def objective(trial):
    params = {
        "sigma_share":     trial.suggest_float("sigma_share",     5.0,  100.0),
        "alpha_sharing":   trial.suggest_float("alpha_sharing",   0.5,    2.0),
        "K_tourn":         trial.suggest_int(  "K_tourn",           2,     10),
        "population_size": trial.suggest_int(  "population_size",  50,    150, step=5),
    }

    return run_ga(params, n_runs=3, trial_number=trial.number)

def log_callback(study, trial):
    best    = study.best_value
    current = trial.value

    tag = " NEW BEST" if current == best else ""
    print(
        f"[Trial {trial.number + 1:>3}] "
        f"Fitness = {current:.6f}   |   Best = {best:.6f}   {tag}"
    )

if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            seed=42
        )
    )

    print("\n" + "=" * 50)
    print("   TPE Hyperparameter Search Started")
    print("=" * 50 + "\n")

    study.optimize(objective, n_trials=50, callbacks=[log_callback])

    print("\n" + "=" * 50)
    print("   FINAL RESULTS")
    print("=" * 50)

    print(f"\nBest Fitness : {study.best_value:.6f}")
    print(f"Best Params  : {study.best_params}")
    print(f"f1_mode      : {F1_MODE} (fixed)")

    results     = []
    best_so_far = []
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

    plt.figure(figsize=(8, 4))
    plt.plot(best_so_far)
    plt.xlabel("Trial")
    plt.ylabel("Best Fitness So Far")
    plt.title("TPE Convergence")
    plt.grid()

    save_path = os.path.join(PLOTS_DIR, "tpe_ga_convergence.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    plt.show()