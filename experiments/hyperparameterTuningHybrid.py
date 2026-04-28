import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'algorithms')))

import numpy as np
import optuna
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — never blocks the terminal
import matplotlib.pyplot as plt

from problem.scenarioM       import get_scenario
from algorithms.hybridDIM_SP import DIMSPHybrid, seed_all

# ── Setup ──────────────────────────────────────────────────────────────────────
SCENARIO = get_scenario()
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TOTAL_GENERATIONS = 100
ISLAND_SIZE       = 50
TUNING_SEEDS      = [0, 1, 2, 3, 4]

# ── BEST_PARAMS ────────────────────────────────────────────────────────────────
# Imported by test.py.
# Before tuning: sensible defaults. After tuning: auto-updated in this file.
BEST_PARAMS = {
    "epoch_interval"         : 10,
    "init_strategy"          : "Demand_Proportional",
    "sigma"                  : 2.51814268,
    "min_cluster_size"       : 17,
    "c1"                     : 1.95620561,
    "c2"                     : 1.38233751,
    "inertia_type"           : "random",
    "w_start"                : 0.90000000,
    "w_end"                  : 0.40000000,
    "ring"                   : True,
    "neighbors"              : 3,
    "bare"                   : True,
    "bare_prob"              : 0.30121215,
    "extinction_threshold"   : 15,
    "extinction_keep_ratio"  : 0.35674329,
}


def _resolve_params(raw):
    """Fill in conditional defaults that Optuna omits for inactive branches."""
    p = dict(raw)
    if not p.get("bare", False):
        p.setdefault("bare_prob", 0.5)
    if not p.get("ring", True):
        p.setdefault("neighbors", 4)
    if p.get("inertia_type", "linear") == "random":
        p.setdefault("w_start", 0.9)
        p.setdefault("w_end", 0.4)
    return p


# ── Run one trial ──────────────────────────────────────────────────────────────
def run_hybrid(params, seeds, trial_number):
    """Run DIMSPHybrid once per seed, return mean fitness. Mirrors run_pso()."""
    fitnesses = []
    for seed in seeds:
        seed_all(seed)
        hybrid = DIMSPHybrid(
            scenario              = SCENARIO,
            total_generations     = TOTAL_GENERATIONS,
            island_size           = ISLAND_SIZE,
            epoch_interval        = params["epoch_interval"],
            init_strategy         = params["init_strategy"],
            sigma                 = params["sigma"],
            min_cluster_size      = params["min_cluster_size"],
            c1                    = params["c1"],
            c2                    = params["c2"],
            inertia_type          = params["inertia_type"],
            w_start               = params["w_start"],
            w_end                 = params["w_end"],
            ring                  = params["ring"],
            neighbors             = params["neighbors"],
            bare                  = params["bare"],
            bare_prob             = params["bare_prob"],
            extinction_threshold  = params["extinction_threshold"],
            extinction_keep_ratio = params["extinction_keep_ratio"],
            seed                  = seed,
        )
        _, best_f, _ = hybrid.run()
        fitnesses.append(best_f)
    return float(np.mean(fitnesses))


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(trial):
    bare         = trial.suggest_categorical("bare",         [True, False])
    ring         = trial.suggest_categorical("ring",         [True, False])
    inertia_type = trial.suggest_categorical("inertia_type", ["linear", "random"])

    params = {
        "c1"                    : trial.suggest_float("c1",  0.3, 3.0),
        "c2"                    : trial.suggest_float("c2",  0.3, 3.0),
        "inertia_type"          : inertia_type,
        "bare"                  : bare,
        "ring"                  : ring,
        "epoch_interval"        : trial.suggest_int(  "epoch_interval",        10,  30),
        "init_strategy"         : trial.suggest_categorical(
                                      "init_strategy",
                                      ["Random", "Demand_Proportional", "Urgency_Biased"]
                                  ),
        "sigma"                 : trial.suggest_float("sigma",                  0.5,  3.0),
        "min_cluster_size"      : trial.suggest_int(  "min_cluster_size",        5,   20),
        "extinction_threshold"  : trial.suggest_int(  "extinction_threshold",    5,   30),
        "extinction_keep_ratio" : trial.suggest_float("extinction_keep_ratio",  0.2,  0.8),
    }

    # Conditional parameters — same pattern as hyperparameterTuningPSO.py
    if inertia_type == "linear":
        params["w_start"] = trial.suggest_float("w_start", 0.6, 1.0)
        params["w_end"]   = trial.suggest_float("w_end",   0.1, 0.5)
    else:
        params["w_start"] = 0.9
        params["w_end"]   = 0.4

    if bare:
        params["bare_prob"] = trial.suggest_float("bare_prob", 0.3, 0.95)
    else:
        params["bare_prob"] = 0.5

    if ring:
        params["neighbors"] = trial.suggest_int("neighbors", 2, 8)
    else:
        params["neighbors"] = 4

    return run_hybrid(params, seeds=TUNING_SEEDS, trial_number=trial.number)


# ── Logging callback ───────────────────────────────────────────────────────────
def log_callback(study, trial):
    best    = study.best_value
    current = trial.value
    tag     = " NEW BEST" if current == best else ""
    print(
        f"[Trial {trial.number + 1:>3}] "
        f"Fitness = {current:.6f}   |   Best = {best:.6f}   {tag}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
    )

    print("\n" + "=" * 50)
    print("   DIMSPHybrid TPE Hyperparameter Search Started")
    print("=" * 50 + "\n")

    study.optimize(objective, n_trials=50, callbacks=[log_callback])

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   FINAL RESULTS")
    print("=" * 50)

    best_found = _resolve_params(study.best_params)
    print(f"\nBest Fitness: {study.best_value:.6f}")
    print(f"Best Params : {best_found}")

    # ── Results table ─────────────────────────────────────────────────────────
    results, best_so_far, current_best = [], [], float("inf")
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

    # ── Persist best params into BEST_PARAMS and save into this file ──────────
    # Done BEFORE the plot so the script finishes even if no display is available.
    BEST_PARAMS.update(best_found)

    this_file = os.path.abspath(__file__)
    try:
        with open(this_file, "r", encoding="utf-8") as fh:
            source = fh.read()

        key_order = [
            "epoch_interval", "init_strategy", "sigma", "min_cluster_size",
            "c1", "c2", "inertia_type", "w_start", "w_end",
            "ring", "neighbors", "bare", "bare_prob",
            "extinction_threshold", "extinction_keep_ratio",
        ]
        lines = ["BEST_PARAMS = {"]
        for k in key_order:
            v   = BEST_PARAMS.get(k, "")
            pad = " " * max(1, 25 - len(k))
            if isinstance(v, str):
                lines.append(f'    "{k}"{pad}: "{v}",')
            elif isinstance(v, bool):
                lines.append(f'    "{k}"{pad}: {v},')
            elif isinstance(v, int):
                lines.append(f'    "{k}"{pad}: {v},')
            else:
                lines.append(f'    "{k}"{pad}: {v:.8f},')
        lines.append("}")
        new_block = "\n".join(lines)

        import re
        new_source = re.sub(r"BEST_PARAMS\s*=\s*\{[^}]*\}", new_block,
                            source, flags=re.DOTALL)
        with open(this_file, "w", encoding="utf-8") as fh:
            fh.write(new_source)

        print(f"\nBEST_PARAMS updated in {os.path.basename(this_file)}.")
        print("test.py will use the tuned values automatically on the next run.")

    except Exception as exc:
        print(f"\n[Warning] Could not auto-update source: {exc}")
        print("Paste this into BEST_PARAMS manually:\n")
        print(new_block)

    # ── Convergence plot — saved to file, never blocks ────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(best_so_far)
    plt.xlabel("Trial")
    plt.ylabel("Best Fitness So Far")
    plt.title("DIMSPHybrid TPE Convergence")
    plt.grid()
    save_path = os.path.join(PLOTS_DIR, "tpe_hybrid_convergence.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()   # release memory; no plt.show() so the script exits immediately
    print(f"\nPlot saved to: {save_path}")
    print("[Done]")