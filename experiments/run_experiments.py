"""
run_experiments.py  —  Disaster Relief Optimization
Place inside the experiments/ folder.

Run one :  python experiments\run_experiments.py --exp 1
Run all :  python experiments\run_experiments.py
Seeds   :  python experiments\run_experiments.py --seeds 30
"""

import sys, os, argparse, json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "problem"))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))

from problem.scenarioM    import get_scenario, DEFAULT_REGIONS
from problem.FitnessFinal import compute_fitness
from problem.constraint   import is_feasible
from algorithms.ga        import DisasterReliefGA
from algorithms.pso       import PSO, LinearInertia, RandomInertia, build_all_configs
from algorithms.hybridDIM_SP import DIMSPHybrid

SCENARIO  = get_scenario()
PLOTS_DIR = os.path.join(HERE, "experiment_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
SEEDS_30  = list(range(30))

# =============================================================================
#  SCENARIOS
# =============================================================================

# def get_all_scenarios():
#     import copy
#     high_urg = copy.deepcopy(DEFAULT_REGIONS)
#     for r in high_urg:
#         r["urgency"]    = min(r["urgency"] * 1.3, 1.0)
#         r["need_score"] = min(r["need_score"] * 1.2, 1.0)

#     remote = copy.deepcopy(DEFAULT_REGIONS)
#     for r in remote:
#         r["access_difficulty"] = min(r["access_difficulty"] * 1.4, 1.0)

#     mass = copy.deepcopy(DEFAULT_REGIONS)
#     for r in mass:
#         r["population"] = int(r["population"] * 2)

#     return {
#         "Baseline":      get_scenario(),
#         "High Urgency":  get_scenario(regions=high_urg),
#         "Remote Access": get_scenario(regions=remote),
#         "Mass Casualty": get_scenario(regions=mass,
#                          budgets={"food": 1200, "water": 900, "medicine": 700}),
#     }

# =============================================================================
#  UTILITIES
# =============================================================================

def _save_seeds():
    with open(os.path.join(HERE, "seeds.json"), "w") as f:
        json.dump({"seeds": SEEDS_30}, f, indent=2)

def _stats(values):
    a = np.array(values, dtype=float)
    return {"mean": a.mean(), "std": a.std(), "min": a.min(), "max": a.max()}

def _print_table(headers, rows):
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))

def _feasible(solution):
    sol = np.asarray(solution)
    X   = sol.reshape(SCENARIO["n_regions"], SCENARIO["n_resources"]) if sol.ndim == 1 else sol
    ok, _ = is_feasible(X, SCENARIO)
    return ok

def _save(name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")

def _avg_curve(results):
    hists = [r["history"] for r in results if r["history"]]
    if not hists:
        return []
    L = min(len(h) for h in hists)
    return np.array([h[:L] for h in hists]).mean(axis=0)

def _std_curve(results):
    hists = [r["history"] for r in results if r["history"]]
    if not hists:
        return []
    L = min(len(h) for h in hists)
    return np.array([h[:L] for h in hists]).std(axis=0)


#  RUNNERS  

def run_ga(config="baseline", init="Demand_Proportional", f1_mode="asymmetric",
           seeds=None, scenario=None, **kw):
    sc, out = scenario or SCENARIO, []
    for s in (seeds or [42]):
        ga = DisasterReliefGA(scenario_data=sc, config_type=config,
                              init_strategy=init, seed=s, f1_mode=f1_mode, **kw)
        sol, score, hist, pop = ga.run()
        out.append({"score": score, "history": hist, "sol": sol})
    return out

def run_pso(init="random", seeds=None, scenario=None, **kw):
    sc, out = scenario or SCENARIO, []
    for s in (seeds or [42]):
        pso = PSO(scenario=sc, seed=s, initialization_strategy=init, **kw)
        score, sol, hist = pso.optimize()
        out.append({"score": score, "history": hist["convergence"], "sol": sol})
    return out

def run_hybrid(init="Demand_Proportional", seeds=None, scenario=None, **kw):
    sc, out = scenario or SCENARIO, []
    for s in (seeds or [42]):
        h = DIMSPHybrid(scenario=sc, seed=s, init_strategy=init, **kw)
        sol, score, info = h.run()
        out.append({"score": score, "history": info.get("hybrid_convergence", []), "sol": sol})
    return out

def _pso_single(seed, inertia_type, kw):
    inertia = (RandomInertia(np.random.default_rng(seed))
               if inertia_type == "random" else LinearInertia(0.9, 0.4))
    pso = PSO(scenario=SCENARIO, seed=seed, inertia=inertia, max_iterations=200, **kw)
    score, sol, hist = pso.optimize()
    return {"score": score, "history": hist["convergence"], "sol": sol}


#  EXP-1  GA Component Study

def exp1_ga_component_study(seeds):
    print("\n── EXP-1: GA Component Study ──")
    configs = {
        "baseline":          "Baseline (Tournament+BLX+NonUniform+Elitism=2)",
        "rws":               "Roulette Wheel Selection",
        "uniform_crossover": "Uniform Crossover",
        "uniform_mutation":  "Uniform Mutation",
        "generational":      "Generational (Elitism=0)",
    }
    rows, curves = [], {}
    for i, (key, label) in enumerate(configs.items()):
        print(f"  [{i+1}/{len(configs)}] {label}")
        res    = run_ga(config=key, seeds=seeds)
        scores = [r["score"] for r in res]
        st     = _stats(scores)
        curves[label] = _avg_curve(res)
        rows.append([label[:40], f"{st['mean']:.4f}", f"{st['std']:.4f}",
                     f"{st['min']:.4f}", f"{st['max']:.4f}"])

    print(); _print_table(["Config", "Mean", "Std", "Min", "Max"], rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    for label, curve in curves.items():
        a1.plot(curve, label=label[:25], linewidth=1.6)
    a1.set(title="EXP-1 Convergence", xlabel="Generation", ylabel="Fitness")
    a1.legend(fontsize=7); a1.grid(alpha=.3)

    a2.bar([r[0][:20] for r in rows], [float(r[1]) for r in rows],
           yerr=[float(r[2]) for r in rows], capsize=5, edgecolor="black", lw=0.7)
    a2.set(title="EXP-1 Mean ± Std per Config", ylabel="Mean Fitness")
    plt.sca(a2); plt.xticks(rotation=25, ha="right", fontsize=8)
    a2.grid(axis="y", alpha=.3)
    plt.tight_layout(); _save("exp1_ga_component_study")


#  EXP-2  PSO Parameter Study  — uses build_all_configs() from pso.py

def exp2_pso_configs(seeds):
    print("\n── EXP-2: PSO Parameter Study ──")
    all_configs = build_all_configs()
    rows, curves = [], {}

    for i, (label, kwargs) in enumerate(all_configs):
        print(f"  [{i+1}/{len(all_configs)}] {label}")
        kw           = dict(kwargs)
        inertia_obj  = kw.pop("inertia", None)
        inertia_type = "random" if isinstance(inertia_obj, RandomInertia) else "linear"

        scores, hists = [], []
        for s in seeds:
            res = _pso_single(s, inertia_type, kw)
            scores.append(res["score"]); hists.append(res["history"])

        st = _stats(scores)
        L  = min(len(h) for h in hists)
        curves[label] = np.array([h[:L] for h in hists]).mean(axis=0)
        rows.append([label, f"{st['mean']:.4f}", f"{st['std']:.4f}", f"{st['min']:.4f}"])

    rows_sorted = sorted(rows, key=lambda r: float(r[1]))
    print(); _print_table(["Config", "Mean", "Std", "Min"], rows_sorted[:10])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, curve in list(curves.items())[:6]:
        a1.plot(curve, label=label[:22], linewidth=1.6)
    a1.set(title="EXP-2 Top-6 Convergence", xlabel="Iteration", ylabel="Fitness")
    a1.legend(fontsize=7); a1.grid(alpha=.3)

    means_s  = [float(r[1]) for r in rows_sorted]
    labels_s = [r[0][:24] for r in rows_sorted]
    a2.barh(range(len(means_s)), means_s)
    a2.set_yticks(range(len(labels_s))); a2.set_yticklabels(labels_s, fontsize=7)
    a2.set(title="EXP-2 All PSO Configs Ranked", xlabel="Mean Fitness")
    a2.grid(axis="x", alpha=.3)
    plt.tight_layout(); _save("exp2_pso_parameter_study")


#  EXP-3  Scenario Comparison

# def exp3_scenario_comparison(seeds):
#    print("\n── EXP-3: Scenario Comparison ──")
#    scenarios = get_all_scenarios()
#    rows = []
#    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))

#    for i, (sc_name, sc) in enumerate(scenarios.items()):
#        print(f"  [{i+1}/{len(scenarios)}] {sc_name}")
#        res    = run_ga(seeds=seeds, scenario=sc)
#        scores = [r["score"] for r in res]
#        st     = _stats(scores)
#        feas   = sum(_feasible(r["sol"]) for r in res) / len(res) * 100
#        rows.append([sc_name, f"{st['mean']:.4f}", f"{st['std']:.4f}",
#                     f"{st['min']:.4f}", f"{feas:.0f}%"])
#        a1.plot(_avg_curve(res), label=sc_name, linewidth=1.6)

#     print(); _print_table(["Scenario", "Mean", "Std", "Min", "Feasible%"], rows)
#     a1.set(title="EXP-3 Convergence per Scenario", xlabel="Generation", ylabel="Fitness")
#     a1.legend(); a1.grid(alpha=.3)
#     a2.bar([r[0] for r in rows], [float(r[1]) for r in rows],
#            yerr=[float(r[2]) for r in rows], capsize=5, edgecolor="black", lw=0.7)
#     a2.set(title="EXP-3 Mean ± Std per Scenario", ylabel="Mean Fitness")
#     plt.sca(a2); plt.xticks(rotation=15, ha="right")
#     a2.grid(axis="y", alpha=.3)
#     plt.tight_layout(); _save("exp3_scenario_comparison")

#  EXP-4  Initialization Strategy  (GA vs PSO vs Hybrid per strategy)

def exp4_init_strategy(seeds):
    print("\n── EXP-4: Initialization Strategy ──")
    ga_strats  = ["Demand_Proportional", "Urgency_Biased", "Random"]
    pso_strats = ["demand_proportional", "urgency_biased", "random"]

    all_data = {}
    for i, (g_strat, p_strat) in enumerate(zip(ga_strats, pso_strats)):
        print(f"  [{i+1}/{len(ga_strats)}] {g_strat}")
        print("    GA:");     ga_res  = run_ga(init=g_strat, seeds=seeds)
        print("    PSO:");    pso_res = run_pso(init=p_strat, seeds=seeds,
                                                num_particles=30, max_iterations=200,
                                                c1=1.5, c2=1.5)
        print("    Hybrid:"); hyb_res = run_hybrid(init=g_strat, seeds=seeds)
        all_data[g_strat] = {"GA": ga_res, "PSO": pso_res, "Hybrid": hyb_res}

    rows = []
    for strat, algos in all_data.items():
        row = [strat]
        for res in algos.values():
            st = _stats([r["score"] for r in res])
            row += [f"{st['mean']:.4f}", f"{st['std']:.4f}"]
        rows.append(row)
    print()
    _print_table(["Init", "GA Mean", "GA Std",
                  "PSO Mean", "PSO Std", "Hyb Mean", "Hyb Std"], rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (strat, algos) in zip(axes, all_data.items()):
        for algo, res in algos.items():
            ax.plot(_avg_curve(res), label=algo, linewidth=1.6)
        ax.set(title=f"Init: {strat}", xlabel="Gen/Iter", ylabel="Fitness")
        ax.legend(fontsize=9); ax.grid(alpha=.3)

    plt.suptitle("EXP-4: Initialization Strategy — GA vs PSO vs Hybrid",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); _save("exp4_init_strategy")


#  EXP-5  f1 Mode Sensitivity  (one convergence plot per mode)

def exp5_f1_modes(seeds):
    print("\n── EXP-5: f1 Mode Sensitivity ──")
    modes   = ["asymmetric", "absolute", "squared", "relative"]
    rows    = []
    all_res = {}

    for i, mode in enumerate(modes):
        print(f"  [{i+1}/{len(modes)}] {mode}")
        res    = run_ga(f1_mode=mode, seeds=seeds)
        scores = [r["score"] for r in res]
        st     = _stats(scores)
        rows.append([mode, f"{st['mean']:.4f}", f"{st['std']:.4f}",
                     f"{st['min']:.4f}", f"{st['max']:.4f}"])
        all_res[mode] = res

    print(); _print_table(["f1 Mode", "Mean", "Std", "Min", "Max"], rows)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (mode, res) in zip(axes, all_res.items()):
        curve = _avg_curve(res)
        std   = _std_curve(res)
        ax.plot(curve, linewidth=2, label=mode)
        ax.fill_between(range(len(curve)),
                        np.array(curve) - std, np.array(curve) + std, alpha=0.15)
        ax.set(title=f"f1 = {mode}", xlabel="Generation", ylabel="Fitness")
        ax.legend(fontsize=8); ax.grid(alpha=.3)

    plt.suptitle("EXP-5: f1 Mode Sensitivity", fontsize=13, fontweight="bold")
    plt.tight_layout(); _save("exp5_f1_modes")

#  EXP-6  Selection Operators

def exp6_selection_operators(seeds):
    print("\n── EXP-6: Selection Operators ──")
    configs = {
        "Tournament + Elitism":     "baseline",
        "Roulette Wheel + Elitism": "rws",
        "Tournament + Generational":"generational",
    }
    rows, curves = [], {}
    for i, (label, cfg) in enumerate(configs.items()):
        print(f"  [{i+1}/{len(configs)}] {label}")
        res    = run_ga(config=cfg, seeds=seeds)
        scores = [r["score"] for r in res]
        st     = _stats(scores)
        rows.append([label, f"{st['mean']:.4f}", f"{st['std']:.4f}", f"{st['min']:.4f}"])
        curves[label] = _avg_curve(res)

    print(); _print_table(["Configuration", "Mean", "Std", "Min"], rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for label, curve in curves.items():
        a1.plot(curve, label=label[:28], linewidth=1.6)
    a1.set(title="EXP-6 Convergence by Selection", xlabel="Generation", ylabel="Fitness")
    a1.legend(fontsize=8); a1.grid(alpha=.3)

    a2.bar([r[0][:22] for r in rows], [float(r[1]) for r in rows],
           yerr=[float(r[2]) for r in rows], capsize=5, edgecolor="black", lw=0.7)
    a2.set(title="EXP-6 Mean ± Std by Selection", ylabel="Mean Fitness")
    plt.sca(a2); plt.xticks(rotation=15, ha="right")
    a2.grid(axis="y", alpha=.3)
    plt.tight_layout(); _save("exp6_selection_operators")

#  EXP-7  Algorithm Comparison  GA vs PSO vs Hybrid

def exp7_algorithm_comparison(seeds):
    print("\n── EXP-7: Algorithm Comparison (GA vs PSO vs Hybrid) ──")
    print("  [1/3] GA");     ga_res  = run_ga(seeds=seeds)
    print("  [2/3] PSO");    pso_res = run_pso(seeds=seeds, num_particles=30,
                                               max_iterations=200, c1=1.5, c2=1.5,
                                               init="demand_proportional")
    print("  [3/3] Hybrid"); hyb_res = run_hybrid(seeds=seeds)

    algos = {"GA": ga_res, "PSO": pso_res, "Hybrid": hyb_res}
    rows  = []
    for name, res in algos.items():
        scores = [r["score"] for r in res]
        feas   = sum(_feasible(r["sol"]) for r in res) / len(res) * 100
        st     = _stats(scores)
        rows.append([name, f"{st['mean']:.4f}", f"{st['std']:.4f}",
                     f"{st['min']:.4f}", f"{st['max']:.4f}", f"{feas:.0f}%"])

    print(); _print_table(["Algorithm", "Mean", "Std", "Min", "Max", "Feasible%"], rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, res in algos.items():
        curve = _avg_curve(res)
        std   = _std_curve(res)
        axes[0].plot(curve, label=name, linewidth=1.8)
        axes[0].fill_between(range(len(curve)),
                             np.array(curve) - std, np.array(curve) + std, alpha=0.12)
    axes[0].set(title="EXP-7 Convergence ± std", xlabel="Gen/Iter", ylabel="Fitness")
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].boxplot([[r["score"] for r in res] for res in algos.values()],
                    labels=list(algos.keys()))
    axes[1].set(title="EXP-7 Score Distribution", ylabel="Fitness")
    axes[1].grid(axis="y", alpha=.3)

    x = np.arange(3)
    f1s, f2s, f3s = [], [], []
    for res in algos.values():
        best = min(res, key=lambda r: r["score"])
        _, d = compute_fitness(np.asarray(best["sol"]).flatten(), SCENARIO)
        f1s.append(d["f1"]); f2s.append(d["f2"]); f3s.append(d["f3"])
    axes[2].bar(x - .25, f1s, .25, label="f1 Suffering")
    axes[2].bar(x,       f2s, .25, label="f2 Waste")
    axes[2].bar(x + .25, f3s, .25, label="f3 Delivery")
    axes[2].set_xticks(x); axes[2].set_xticklabels(list(algos.keys()))
    axes[2].set(title="EXP-7 Sub-objectives (best solution)")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=.3)
    plt.tight_layout()
    # allocation heatmap — best overall solution
    best_algo = min(algos.items(),
                    key=lambda x: min(r["score"] for r in x[1]))
    #plot_allocation_heatmap(best_algo[1],title=f"Best Allocation ({best_algo[0]})")
    #_save("exp7_allocation_heatmap")

    # radar chart
    #plot_subobjective_radar(algos)_save("exp7_subobjective_radar")

    # diversity over time
    plot_diversity_over_time(algos)
    _save("exp7_diversity_over_time")


#  EXP-8  Diversity Preservation

def exp8_diversity(seeds):
    print("\n── EXP-8: Diversity Preservation ──")

    print("  [1/5] GA + Fitness Sharing")
    sharing_res  = run_ga(config="baseline", seeds=seeds)
    print("  [2/5] GA – No Sharing")
    no_share_res = run_ga(config="baseline", seeds=seeds, sigma_share=0.0001)
    print("  [3/5] Hybrid Island Model")
    hybrid_res   = run_hybrid(seeds=seeds)
    print("  [4/5] PSO Topology (Global vs Ring)")
    pso_global   = run_pso(seeds=seeds, num_particles=30, max_iterations=200,
                           c1=1.5, c2=1.5, ring=False)
    pso_ring     = run_pso(seeds=seeds, num_particles=30, max_iterations=200,
                           c1=1.5, c2=1.5, ring=True, neighbors=4)
    print("  [5/5] PSO Inertia (Linear vs Random)")
    linear_res, random_res = [], []
    for s in seeds:
        r = _pso_single(s, "linear", dict(num_particles=30, c1=1.5, c2=1.5, ring=False))
        linear_res.append(r)
        r = _pso_single(s, "random", dict(num_particles=30, c1=1.5, c2=1.5, ring=False))
        random_res.append(r)

    ga_configs      = {"GA + Fitness Sharing": sharing_res,
                       "GA – No Sharing":      no_share_res,
                       "Hybrid Island Model":  hybrid_res}
    topo_configs    = {"PSO Global":   pso_global, "PSO Ring k=4": pso_ring}
    inertia_configs = {"Linear Inertia": linear_res, "Random Inertia": random_res}

    def _table(configs):
        rows = []
        for label, res in configs.items():
            st = _stats([r["score"] for r in res])
            rows.append([label, f"{st['mean']:.4f}", f"{st['std']:.4f}", f"{st['min']:.4f}"])
        _print_table(["Method", "Mean", "Std", "Min"], rows)

    print("\n  GA Diversity:"); _table(ga_configs)
    print("\n  PSO Topology:"); _table(topo_configs)
    print("\n  PSO Inertia:");  _table(inertia_configs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for label, res in ga_configs.items():
        curve = _avg_curve(res); std = _std_curve(res)
        axes[0].plot(curve, label=label, linewidth=1.8)
        axes[0].fill_between(range(len(curve)),
                             np.array(curve)-std, np.array(curve)+std, alpha=0.12)
    axes[0].set(title="EXP-8a GA Diversity", xlabel="Gen/Iter", ylabel="Fitness")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    for label, res in topo_configs.items():
        curve = _avg_curve(res); std = _std_curve(res)
        axes[1].plot(curve, label=label, linewidth=1.8)
        axes[1].fill_between(range(len(curve)),
                             np.array(curve)-std, np.array(curve)+std, alpha=0.12)
    axes[1].set(title="EXP-8b PSO Topology", xlabel="Iteration", ylabel="Fitness")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

    for label, res in inertia_configs.items():
        curve = _avg_curve(res); std = _std_curve(res)
        axes[2].plot(curve, label=label, linewidth=1.8)
        axes[2].fill_between(range(len(curve)),
                             np.array(curve)-std, np.array(curve)+std, alpha=0.12)
    axes[2].set(title="EXP-8c PSO Inertia", xlabel="Iteration", ylabel="Fitness")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)

    plt.suptitle("EXP-8: Diversity Preservation", fontsize=13, fontweight="bold")
    plt.tight_layout(); _save("exp8_diversity_preservation")


#  EXTRA PLOTS

def plot_subobjective_radar(results_dict):
    """
    Radar chart comparing f1/f2/f3 across algorithms or configs.
    results_dict = {"GA": ga_res, "PSO": pso_res, "Hybrid": hyb_res}
    """
    labels = ["f1 Suffering", "f2 Waste", "f3 Delivery"]
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name, res in results_dict.items():
        best    = min(res, key=lambda r: r["score"])
        _, d    = compute_fitness(np.asarray(best["sol"]).flatten(), SCENARIO)
        values  = [d["f1"], d["f2"], d["f3"]]
        # normalise so all three fit on the same radar
        max_val = max(values) if max(values) > 0 else 1
        values  = [v / max_val for v in values]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Sub-objective Radar (best solution per algo)", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()


def plot_diversity_over_time(results_dict):
    """
    Diversity over time = std across seeds at each generation.
    results_dict = {"GA": ga_res, "PSO": pso_res, ...}
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, res in results_dict.items():
        hists = [r["history"] for r in res if r["history"]]
        if not hists:
            continue
        L   = min(len(h) for h in hists)
        arr = np.array([h[:L] for h in hists])   # shape (n_seeds, L)
        # diversity = std of fitness values across seeds at each step
        diversity = arr.std(axis=0)
        ax.plot(diversity, label=name, linewidth=1.8)

    ax.set(title="Diversity Over Time (std across seeds)",
           xlabel="Generation / Iteration", ylabel="Std of Fitness Across Seeds")
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()

#  MAIN

EXP_MAP = {
    #1: exp1_ga_component_study,
    #2: exp2_pso_configs,
    #3: exp3_scenario_comparison,
    4: exp4_init_strategy,
    #5: exp5_f1_modes,
    #6: exp6_selection_operators,
    7: exp7_algorithm_comparison,
    8: exp8_diversity,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   type=int, default=None,
                        help="experiment 1-8 (omit to run all)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds (30 for full report run)")
    args   = parser.parse_args()
    seeds  = SEEDS_30[: args.seeds]

    _save_seeds()
    print(f"Seeds : {seeds}")
    print(f"Plots → {PLOTS_DIR}\n")

    if args.exp:
        EXP_MAP[args.exp](seeds)
    else:
        for fn in EXP_MAP.values():
            fn(seeds)

    print("\nAll done.")