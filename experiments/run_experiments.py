"""
Run one  :  python experiments/run_experiments.py --exp 1
Run all  :  python experiments/run_experiments.py

Seeds    :  python experiments/run_experiments.py --seeds 30  
            estakhdmo da ya gma3a fy elterminal just copy paste it into the terminal 
            bas boso ta7t fy akhr elcode fy elmain w 23mlo uncomment 
            le elexperiments elly hat3mlolha run bas 

  :  python experiments/run_experiments.py --exp 7 --seeds 10 
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
 
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "problem"))
sys.path.insert(0, os.path.join(ROOT, "algorithms"))
 
from problem.scenarioM import get_scenario
from algorithms.ga import DisasterReliefGA
from algorithms.pso import PSO, LinearInertia, RandomInertia, build_all_configs
from algorithms.hybridDIM_SP import DIMSPHybrid
from problem.FitnessFinal import compute_fitness, compute_norm_constants
 
# CONFIG
 
SCENARIO = get_scenario()
PLOTS_DIR = os.path.join(HERE, "experiment_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
SEEDS_30 = list(range(30))
VERBOSE = False
 
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
 
plt.rcParams.update({                        # Update global matplotlib plotting style settings
    "figure.facecolor": "white",             # Set figure background color
    "axes.facecolor": "white",               # Set axes background color
    "axes.edgecolor": "black",               # Set border color of axes
    "axes.grid": True,                       # Enable grid by default
    "grid.color": "#cccccc",                 # Grid line color
    "grid.linewidth": 0.6,                   # Grid line thickness
    "lines.linewidth": 1.8,                  # Default line thickness for plots
    "font.size": 9,                          # Default font size
    "axes.titlesize": 10,                    # Title font size
    "axes.labelsize": 9,                     # Axis label font size
    "legend.fontsize": 8,                    # Legend font size
    "savefig.dpi": 150,                      # Resolution when saving figures
})
 
# UTILITIES

def _save(name):
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}.png"))   # Save current plot as PNG in plots directory
    plt.close()                                           # Close current figure to free memory
 
 
def curve_mean(results):
    hists = [r["history"] for r in results if r.get("history")]
 
    if not hists:
        return np.array([])
 
    # FORCE SAME LENGTH ACROSS ALL RUNS
    L = min(len(h) for h in hists)
 
    arr = np.array([h[:L] for h in hists])
    return arr.mean(axis=0)
 
def create_table(data, title, filename, headers ,  highlight_min_cols=None):
    fig_width  = max(len(headers) * 1.5, 8)
    fig_height = max(len(data) * 0.5 + 1.5, 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    table = ax.table(
        cellText=[headers] + data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1.2, 1.8)
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLORS[0])
        table[(0, i)].set_text_props(weight='bold', color='white')
                                     
    if highlight_min_cols:
        for col_idx in highlight_min_cols:
            col_vals = []
            for row_idx in range(len(data)):
                try:
                    col_vals.append(float(data[row_idx][col_idx]))
                except:
                    col_vals.append(float('inf'))
            min_val = min(col_vals)
            for row_idx, val in enumerate(col_vals):
                    if val == min_val:
                        table[(row_idx + 1 , col_idx)].set_facecolor('#90EE90')


    plt.title(title, pad=20, wrap=True)
    plt.tight_layout()
    _save(filename)


def extra_stats(res, scenario):
    norm = compute_norm_constants(scenario)
    f1s = [r["f1"] for r in res]
    f2s = [r["f2"] for r in res]
    f3s = [r["f3"] for r in res]
    iters = [len(r["history"]) for r in res if r.get("history")]
    mean_f1 = np.mean(f1s)
    mean_f2 = np.mean(f2s)
    mean_f3 = np.mean(f3s)
    dist_eff = (1 - mean_f1 / norm["f1_max"]) * 100
    res_utils = (1 - mean_f2 / norm["f2_max"]) * 100
    avg_iter = np.mean(iters) if iters else 0
    avg_time = np.mean([r["time"] for r in res])
    return mean_f1 ,dist_eff , mean_f2 , res_utils , mean_f3 , avg_iter , avg_time
 
# GENERIC RUNNER
 
def run_algo(algo_func, seeds):
    return [algo_func(seed=seed) for seed in seeds] # Run the given algorithm once for each seed and collect all results
 
 
# ALGORITHM WRAPPERS : # Configure algorithm once, then run multiple times with different random seeds to get performance distribution
 
def make_ga(scenario=None, config="baseline", init="Demand_Proportional", f1_mode="asymmetric", **kw):
    sc = scenario or SCENARIO
    def _fn(seed): # Create GA instance with chosen configuration
        ga = DisasterReliefGA(scenario_data=sc, config_type=config, init_strategy=init, 
                              seed=seed, f1_mode=f1_mode, **kw)
        start = time.time()
        sol, score, hist, _, details = ga.run()
        elapsed = time.time() - start
        return {"score": score, "history": hist, "sol": sol, "f1": details["f1"], "f2": details["f2"], "f3": details["f3"], "time": elapsed}
    return _fn # Return wrapper function so it can be called later with different seeds
 
def make_pso(
    scenario=None,num_particles=30,max_iterations=199,c1=1.5,c2=1.5,
    init="demand_proportional",inertia=None,ring=False,neighbors=4,bare=False,**kw):
 
    sc = scenario or SCENARIO
    def _fn(seed):
        # Convert inertia setting into proper object
        if inertia == "random":
            inertia_obj = RandomInertia(np.random.default_rng(seed))
        elif inertia == "linear":
            inertia_obj = LinearInertia(0.9, 0.4)
        else:
            inertia_obj = inertia
 
        for key in ['max_iterations', 'bare_prob', 'initialization_strategy', 'init']:
            kw.pop(key, None)
 
        pso = PSO(
            scenario=sc,seed=seed,num_particles=num_particles,max_iterations=max_iterations,c1=c1,c2=c2,
            initialization_strategy=init,inertia=inertia_obj,ring=ring,neighbors=neighbors,bare=bare,**kw)

        start = time.time()
        score, sol, hist = pso.optimize()
        elapsed = time.time() - start
 
        return { "score": score,"history": hist["convergence"], "f1": hist["f1_history"][-1] if hist.get("f1_history") else 0,
        "f2": hist["f2_history"][-1] if hist.get("f2_history") else 0, "f3": hist["f3_history"][-1] if hist.get("f3_history") else 0,
        "sol": sol,"time": elapsed
    }
 
    return _fn
 
def make_hybrid(scenario=None, init="Demand_Proportional", epoch_interval=10, island_size=50, **kw):
    sc = scenario or SCENARIO
    def _fn(seed):
        h = DIMSPHybrid(scenario=sc, seed=seed, init_strategy=init, 
                        epoch_interval=epoch_interval, island_size=island_size, **kw)
        start = time.time()
        sol, score, info = h.run()
        elapsed = time.time() - start
        details = info["details"]
        return {"score": score, "history": info.get("hybrid_convergence", []),
                "island_count": info.get("island_count", []), "sol": sol,
                "f1": details["f1"], "f2": details["f2"], "f3": details["f3"],
                "time": elapsed}
    return _fn


# PLOTTING HELPERS
def plot_curves(ax, data, title=None): # Plot mean curve
    for i, (label, results) in enumerate(data.items()): # Loop through each algorithm/configuration in the data dictionary
        mean = curve_mean(results)  # Compute average convergence curve across all runs
        if len(mean):
            ax.plot(mean, label=label, color=COLORS[i % len(COLORS)], linewidth=1.8)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Generation/Iteration")
    ax.set_ylabel("Fitness")
    ax.legend()
 
 
def plot_bars(ax, labels, means, title=None): # Plot bar chart
    x = np.arange(len(labels))
    ax.bar(x, means, color=COLORS[0], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if title:
        ax.set_title(title)
    ax.set_ylabel("Mean Fitness")
 
 
# EXPERIMENTS
def exp1_ga_components(seeds):
    configs = {
        "Baseline (Tournament+BLX+NonUniform+Elite=2)": "baseline",
        "Roulette Wheel": "rws",
        "Uniform Crossover": "uniform_crossover",
        "Uniform Mutation": "uniform_mutation",
        "Generational (Elite=0)": "generational"
    }
 
    results = {}   # Store full results for each configuration
    rows = []      # Store summary statistics for table/bar chart
 
    # Run all GA configurations
    for name, cfg in configs.items():
        print(f"  Running {name}")
        res = run_algo(make_ga(config=cfg), seeds)
        results[name] = res
        scores = [r["score"] for r in res] # Extract final score from each run
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter , avg_time = extra_stats(res, SCENARIO)
        rows.append([name, np.mean(scores), np.min(scores), np.max(scores), mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    # Save summary table
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
          f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
          f"{r[8]:.4f}", f"{r[9]:.1f}" , f"{r[10]:.2f}s"] for r in rows],
        "EXP-1: GA Component Study",
        "exp1_table.png",
        ["Config", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
          "f2", "Resource Utilization%", "f3", "Avg Iter" , "Avg Time(s)"],
        highlight_min_cols=[1, 9]  # Mean fitness and Avg Iter
    )
 
    baseline_name = "Baseline (Tournament+BLX+NonUniform+Elite=2)"
    baseline_mean = curve_mean(results[baseline_name])
 
    variants = [name for name in configs if name != baseline_name]
 
    # Convergence comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
    for ax, variant, color in zip(axes.flatten(), variants, COLORS):
 
        ax.plot(
            baseline_mean,
            color='gray',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label='Baseline'
        )
 
        ax.plot(
            curve_mean(results[variant]),
            color=color,
            linewidth=2,
            label=variant.split('(')[0].strip()
        )
 
        ax.set_title(variant.split('(')[0].strip(), fontsize=11, fontweight='bold')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    plt.figtext(
        0.5,
        0.02,
        "Note: Baseline = Tournament Selection + BLX-alpha Crossover + Non-uniform Mutation + Elite=2",
        ha='center',
        fontsize=10,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black')
    )
 
    plt.suptitle(
        "EXP-1: GA Component Study - Each operator vs Baseline",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
 
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save("exp1_ga_convergence")
 
    # Bar chart
    means = [r[1] for r in rows]
    labels = [
        "Baseline",
        "Roulette",
        "Uniform XO",
        "Uniform Mut",
        "Generational"
    ]
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    bars = ax.bar(
        np.arange(len(means)),
        means,
        color=COLORS[:len(means)],
        edgecolor='black',
        linewidth=1.5
    )
 
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Fitness")
    ax.set_title("EXP-1: Mean Fitness Comparison", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
 
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
 
    plt.tight_layout()
    _save("exp1_ga_bar_chart")
 
# PSO EXPERIMENTS
def filter_configs(configs, selected_labels): #Filter configs by keywords in their labels
    return [cfg for cfg in configs if cfg[0] in selected_labels] #(label, parameters)
 
 
def run_pso_group(seeds, group_name, group_keywords):
    all_configs = build_all_configs()
    group_configs = filter_configs(all_configs, group_keywords)
    results = {}
    rows = []
 
    for name, kwargs in group_configs:
        print(f"  Running {name}...")
        res = run_algo(make_pso(**kwargs), seeds)
        results[name] = res
        scores = [r["score"] for r in res]
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter , avg_time = extra_stats(res, SCENARIO)
        rows.append([name, np.mean(scores), np.min(scores), np.max(scores), mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
        f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
        f"{r[8]:.4f}", f"{r[9]:.1f}", f"{r[10]:.2f}s"] for r in rows],
        group_name,
        f"{group_name}_table.png",
        ["Config", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
        "f2", "Resource Utilization%", "f3", "Avg Iter", "Avg Time(s)"],
        highlight_min_cols=[1, 9]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    ax1 = axes[0]
    for i, (name, res) in enumerate(results.items()):
        curve = curve_mean(res)
        ax1.plot(
            curve,
            color=COLORS[i % len(COLORS)],
            linewidth=2,
            label=name,
            alpha=0.7)
 
    ax1.set_title("Convergence")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    names = list(results.keys())
 
    def safe_curve(res):
        hists = [r["history"] for r in res if r.get("history")]
        if not hists:
            return np.array([])
        L = min(len(h) for h in hists)
        arr = np.array([h[:L] for h in hists])
        return arr.mean(axis=0)
 
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = safe_curve(results[names[i]])
            b = safe_curve(results[names[j]])
            L = min(len(a), len(b))     
            diff = np.max(np.abs(a[:L] - b[:L]))
            print(names[i], "vs", names[j], "max diff:", diff)
 
    ax2 = axes[1]
    labels = list(results.keys())
    means = [np.mean([r["score"] for r in results[name]]) for name in labels]
    bars = ax2.bar(
        range(len(means)),
        means,
        color=COLORS[:len(means)],
        edgecolor='black',
        linewidth=1.5)
 
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax2.set_ylabel("Mean Fitness")
    ax2.set_title(f"{group_name}: Fitness Comparison", fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
 
    for bar, val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontsize=9)
 
    plt.tight_layout()
    _save(f"{group_name}_summary")
 
 
def exp2_pso_update_rules(seeds): 
    run_pso_group(seeds, "EXP-2a_UpdateRules",["Canonical-Global", "BareBones-p0.5-Global", "BareBones-p0.9-Global"])
 
def exp2_pso_topology(seeds):
    run_pso_group(seeds, "EXP-2b_Topology", ["Canonical-Ring-k2", "Canonical-Ring-k4", "BareBones-Ring-k2" , "BareBones-Ring-k4" ])
 
def exp2_pso_balance(seeds):
    run_pso_group(seeds, "EXP-2c_Balance", ["Balanced-c1.5-c1.5", "Cognitive-c2.5-c0.5", "Social-c0.5-c2.5", "Equal-c1.49-c1.49"])
 
def exp2_pso_inertia(seeds):
    run_pso_group(seeds, "EXP-2d_Inertia", ["Linear-Inertia", "Random-Inertia"])
 
def exp2_pso_swarm_size(seeds):
    run_pso_group(seeds, "EXP-2e_SwarmSize", ["Swarm-10", "Swarm-30", "Swarm-50", "Swarm-100"])
 
def exp2_pso_bonus(seeds):
    run_pso_group(seeds, "EXP-2f_interesting_CrossCombinations", ["Ring-k4-Cognitive-Large", "BareBones-Ring-k2-UrgencyInit", "Social-RandomInertia-DemandInit"],)


def exp3_scenarios(seeds):
    scenarios = ["Default", "Epidemic", "Floods", "Large Disaster", "Resource Shortage", "Worst Case"]
    algos_list = ["GA", "PSO", "Hybrid"]
 
    def get_scores(res):
        return [r["score"] for r in res]
 
    def stats(res):
        s = get_scores(res)
        return np.mean(s)
 
    # RUN ALL SCENARIOS
    all_data = {}
 
    for name in scenarios:
        sc = get_scenario() if name == "Default" else get_scenario(name)
        all_data[name] = {
            "GA": run_algo(make_ga(scenario=sc), seeds),
            "PSO": run_algo(make_pso(scenario=sc), seeds),
            "Hybrid": run_algo(make_hybrid(scenario=sc), seeds)}
 
    # TABLE
    rows = []
    for sc_name, algos in all_data.items():
        row = [sc_name]
        for algo in algos_list:
            mean = stats(algos[algo])
            avg_iter = np.mean([len(r["history"]) for r in algos[algo] if r.get("history")])
            avg_time = np.mean([r["time"] for r in algos[algo]])
            row += [f"{mean:.4f}", f"{avg_iter:.1f}", f"{avg_time:.2f}s"]
        rows.append(row)

    create_table(
        rows,
        "EXP-3: Scenario Results",
        "exp3_table.png",
        ["Scenario", "GA Mean", "GA Iter", "GA Time", "PSO Mean", "PSO Iter", "PSO Time", "Hyb Mean", "Hyb Iter", "Hyb Time"],
        highlight_min_cols=[1, 2, 4, 5, 7, 8]
    )
 
    # CONVERGENCE PLOTS
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten() # 1D axes
 
    for ax, (name, algos) in zip(axes, all_data.items()):
        for i, algo in enumerate(algos_list):
            ax.plot(
                curve_mean(algos[algo]),
                label=algo,
                color=COLORS[i % len(COLORS)],
                linewidth=2,
                alpha = 0.7,
            )
 
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    plt.suptitle("EXP-3: Convergence per Scenario")
    plt.tight_layout()
    _save("exp3_scenario_convergence")
 
 
def exp4_init_strategies(seeds):
    strategies = [("Demand_Proportional", "demand_proportional"), ("Urgency_Biased", "urgency_biased"), ("Random", "random")]
    results = {}
    
    for ga_s, pso_s in strategies:
        results[ga_s] = {
            "GA": run_algo(make_ga(init=ga_s), seeds),
            "PSO": run_algo(make_pso(init=pso_s), seeds),
            "Hybrid": run_algo(make_hybrid(init=ga_s), seeds) # Run Hybrid using GA-style initialization
        }
    
    rows = []
    for strat, algos in results.items():
        row = [strat]
        for algo in ["GA", "PSO", "Hybrid"]:
            scores = [r["score"] for r in algos[algo]]
            avg_iter = np.mean([len(r["history"]) for r in algos[algo] if r.get("history")])
            avg_time = np.mean([r["time"] for r in algos[algo]])
            row += [f"{np.mean(scores):.4f}", f"{avg_iter:.1f}", f"{avg_time:.2f}s"]
        rows.append(row)

    create_table(
        rows,
        "EXP-4: Initialization Results",
        "exp4_table.png",
        ["Init", "GA Mean", "GA Iter", "GA Time", "PSO Mean", "PSO Iter", "PSO Time", "Hyb Mean", "Hyb Iter", "Hyb Time"],
        highlight_min_cols=[1, 2, 4, 5, 7, 8])
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (strat, algos) in zip(axes, results.items()):
        plot_curves(ax, algos, f"Init: {strat}")
    plt.suptitle("EXP-4: Initialization Strategies")
    plt.tight_layout()
    _save("exp4_init_strategy")
 
 
def exp5_f1_modes(seeds):
    modes = ["asymmetric", "absolute", "squared", "relative"]
    results = {}
    rows = []
    
    for mode in modes:
        res = run_algo(make_ga(f1_mode=mode), seeds)
        results[mode] = res
        scores = [r["score"] for r in res]
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time = extra_stats(res, SCENARIO)
        rows.append([mode, np.mean(scores), np.min(scores), np.max(scores),mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
        f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
        f"{r[8]:.4f}", f"{r[9]:.1f}", f"{r[10]:.2f}s"] for r in rows],
        "EXP-5: f1 Mode Results", "exp5_table.png",
        ["Config", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
        "f2", "Resource Utilization%", "f3", "Avg Iter", "Avg Time(s)"],
        highlight_min_cols=[1, 9]
    )
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (mode, res) in zip(axes, results.items()):
        mean = curve_mean(res)
        ax.plot(mean, color=COLORS[0], linewidth=2)
        ax.set_title(f"f1 = {mode}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
    plt.suptitle("EXP-5: f1 Mode Sensitivity")
    plt.tight_layout()
    _save("exp5_f1_modes")
 
 
def exp6_algorithm_comparison(seeds):
    algos = {
        "GA": make_ga(),
        "PSO": make_pso(),
        "Hybrid": make_hybrid()
    }
    results = {name: run_algo(fn, seeds) for name, fn in algos.items()}
 
    # TABLE DATA
    rows = []
    for name, res in results.items():
        scores = [r["score"] for r in res]
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter , avg_time = extra_stats(res, SCENARIO)
        rows.append([name, np.mean(scores), np.min(scores), np.max(scores), mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
        f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
        f"{r[8]:.4f}", f"{r[9]:.1f}", f"{r[10]:.2f}s"] for r in rows],
        "EXP-6: Algorithm Comparison",
        "exp6_table.png",
        ["Algorithm", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
        "f2", "Resource Utilization%", "f3", "Avg Iter", "Avg Time(s)"],
        highlight_min_cols=[1, 9]
    )
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    #  CONVERGENCE PLOT
    for i, (name, res) in enumerate(results.items()):
        curve = curve_mean(res)
 
        ax1.plot(
            curve,
            label=name,
            color=COLORS[i],
            linewidth=2,
            alpha=0.8
        )
 
    ax1.set_title("Convergence Comparison")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    #  BAR CHART 
    names = list(results.keys())
    means = [np.mean([r["score"] for r in results[name]]) for name in names]
 
    bars = ax2.bar(
        names,
        means,
        color=COLORS[:len(names)],
        edgecolor='black'
    )
 
    ax2.set_title("Mean Fitness Comparison")
    ax2.set_ylabel("Mean Fitness")
    ax2.grid(True, axis='y', alpha=0.3)
 
    # value labels on bars
    for bar, val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha='center',
            va='bottom'
        )
 
    plt.suptitle("EXP-6: GA vs PSO vs Hybrid Comparison")
    plt.tight_layout()
    _save("exp6_combined")
 
def exp7_diversity(seeds):
 
    results = {
        "GA + Sharing": run_algo(make_ga(), seeds),
        "GA – No Sharing": run_algo(make_ga(sigma_share=0.0001), seeds),
        "Hybrid": run_algo(make_hybrid(), seeds)
    }
 
    # TABLE
    rows = []
    for name, res in results.items():
        scores = [r["score"] for r in res]
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter , avg_time = extra_stats(res, SCENARIO)
        rows.append([name, np.mean(scores), np.min(scores), np.max(scores), mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
        f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
        f"{r[8]:.4f}", f"{r[9]:.1f}", f"{r[10]:.2f}s"] for r in rows],
        "EXP-7: Diversity Results",
        "exp7_table.png",
        ["Method", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
        "f2", "Resource Utilization%", "f3", "Avg Iter", "Avg Time(s)"],
        highlight_min_cols=[1, 9]
    )
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    # Convergence 
    for i, (name, res) in enumerate(results.items()):
        ax1.plot(
            curve_mean(res),
            label=name,
            color=COLORS[i % len(COLORS)],
            linewidth=2,
            alpha=0.8
        )
 
    ax1.set_title("Convergence (Diversity Study)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    # Bar Chart 
    names = list(results.keys())
    means = [np.mean([r["score"] for r in results[n]]) for n in names]
 
    ax2.bar(
        names,
        means,
        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        edgecolor='black'
    )
 
    ax2.set_title("Mean Fitness Comparison")
    ax2.set_ylabel("Fitness")
    ax2.grid(True, axis='y', alpha=0.3)
 
    for i, v in enumerate(means):
        ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
 
    plt.suptitle("EXP-7: Diversity Preservation Analysis")
    plt.tight_layout()
    _save("exp7_diversity_fixed")
 
def exp8_hybrid_islands(seeds):
    epoch_configs = {
        "Epoch=5": 5,
        "Epoch=10": 10,
        "Epoch=20": 20
    }
 
    size_configs = {
        "Size=30": 30,
        "Size=50": 50,
        "Size=70": 70
    }
 
    epoch_res = {
        f"E={v}": run_algo(make_hybrid(epoch_interval=v), seeds)
        for v in epoch_configs.values()
    }
 
    size_res = {
        f"S={v}": run_algo(make_hybrid(island_size=v), seeds)
        for v in size_configs.values()
    }
 
    epoch_rows = []
    for k, res in epoch_res.items():
        scores = [r["score"] for r in res]
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter , avg_time = extra_stats(res, SCENARIO)
        epoch_rows.append([k, np.mean(scores), np.min(scores), np.max(scores), mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
        f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
        f"{r[8]:.4f}", f"{r[9]:.1f}", f"{r[10]:.2f}s"] for r in epoch_rows],
        "EXP-8a: Epoch Variation",
        "exp8a_table.png",
        ["Config", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
        "f2", "Resource Utilization%", "f3", "Avg Iter", "Avg Time(s)"],
        highlight_min_cols=[1, 9]
    )
 
    size_rows = []
    for k, res in size_res.items():
        scores = [r["score"] for r in res]
        mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter , avg_time = extra_stats(res, SCENARIO)
        size_rows.append([k, np.mean(scores), np.min(scores), np.max(scores), mean_f1, dist_eff, mean_f2, res_util, mean_f3, avg_iter, avg_time])
 
    create_table(
        [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
        f"{r[4]:.4f}", f"{r[5]:.2f}%", f"{r[6]:.4f}", f"{r[7]:.2f}%",
        f"{r[8]:.4f}", f"{r[9]:.1f}", f"{r[10]:.2f}s"] for r in size_rows],
        "EXP-8a: Epoch Variation",
        "exp8a_table.png",
        ["Config", "Mean fitness", "Best fitness", "Worst fitness", "f1", "Distribution efficiency%",
        "f2", "Resource Utilization%", "f3", "Avg Iter", "Avg Time(s)"],
        highlight_min_cols=[1, 9]
    )
 
    #EPOCH Interval
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    #  Convergence 
    for i, (label, res) in enumerate(epoch_res.items()):
        hists = [r["history"] for r in res if r.get("history")]
 
        if hists:
            L = min(len(h) for h in hists)
            arr = np.array([h[:L] for h in hists])
            median_curve = np.median(arr, axis=0)
 
            ax1.plot(
                median_curve,
                label=label,
                color=COLORS[i % len(COLORS)],
                linewidth=2
            )
 
    ax1.set_title("EXP-8a: Epoch Variation (Convergence)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    # Bar chart (mean fitness) 
    names = list(epoch_res.keys())
    means = [np.mean([r["score"] for r in epoch_res[n]]) for n in names]
 
    ax2.bar(
        names,
        means,
        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        edgecolor='black'
    )
 
    ax2.set_title("Mean Fitness (Epoch Variation)")
    ax2.set_ylabel("Fitness")
    ax2.grid(True, axis='y', alpha=0.3)
 
    for i, v in enumerate(means):
        ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
 
    plt.suptitle("EXP-8a: Hybrid Island Dynamics - Epoch")
    plt.tight_layout()
    _save("exp8a_hybrid_epoch")
 
    # Island SIZE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    # Convergence
    for i, (label, res) in enumerate(size_res.items()):
        hists = [r["history"] for r in res if r.get("history")]
 
        if hists:
            L = min(len(h) for h in hists)
            arr = np.array([h[:L] for h in hists])
            median_curve = np.median(arr, axis=0)
 
            ax1.plot(
                median_curve,
                label=label,
                color=COLORS[i % len(COLORS)],
                linewidth=2
            )
 
    ax1.set_title("EXP-8b: Island Size Variation (Convergence)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    # Bar chart
    names = list(size_res.keys())
    means = [np.mean([r["score"] for r in size_res[n]]) for n in names]
 
    ax2.bar(
        names,
        means,
        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        edgecolor='black')
 
    ax2.set_title("Mean Fitness (Island Size Variation)")
    ax2.set_ylabel("Fitness")
    ax2.grid(True, axis='y', alpha=0.3)
 
    for i, v in enumerate(means):
        ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
 
    plt.suptitle("EXP-8b: Hybrid Island Dynamics -Island Size")
    plt.tight_layout()
    _save("exp8b_hybrid_size")
 
 
def exp9_population_size(seeds):
    sizes = [20, 50, 100, 150]
 
    results = {}
    for size in sizes:
        results[size] = {
            "GA": run_algo(make_ga(population_size=size), seeds),
            "PSO": run_algo(make_pso(num_particles=size), seeds),
            "Hybrid": run_algo(make_hybrid(island_size=size), seeds)
        }
 
        #  TABLE 
    rows = []
    for size in sizes:
        row = [str(size)]
        for algo in ["GA", "PSO", "Hybrid"]:
            scores = [r["score"] for r in results[size][algo]]
            avg_iter = np.mean([len(r["history"]) for r in results[size][algo] if r.get("history")])
            avg_time = np.mean([r["time"] for r in results[size][algo]])
            row += [f"{np.mean(scores):.4f}", f"{avg_iter:.1f}", f"{avg_time:.2f}s"]
        rows.append(row)

    create_table(
        rows,
        "Population Size Study",
        "exp9_table.png",
        ["Size", "GA Mean", "GA Iter", "GA Time", "PSO Mean", "PSO Iter", "PSO Time", "Hyb Mean", "Hyb Iter", "Hyb Time"],
        highlight_min_cols=[1, 2, 4 , 5, 7 , 8]
    )
 
    algos = ["GA", "PSO", "Hybrid"]
 
    for algo in algos:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
        #  CONVERGENCE 
        for i, size in enumerate(sizes):
            res = results[size][algo]
 
            hists = [r["history"] for r in res if r.get("history")]
 
            if hists:
                L = min(len(h) for h in hists)
                arr = np.array([h[:L] for h in hists])
 
                curve = np.median(arr, axis=0)
 
                ax1.plot(
                    curve,
                    label=f"Size={size}",
                    color=COLORS[i % len(COLORS)],
                    linewidth=2
                )
 
        ax1.set_title(f"{algo}: Convergence vs Population Size")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Fitness")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
 
        #  BAR CHART 
        means = [
            np.mean([r["score"] for r in results[s][algo]])
            for s in sizes
        ]
 
        ax2.bar(
            [str(s) for s in sizes],
            means,
            color=[COLORS[i % len(COLORS)] for i in range(len(sizes))],
            edgecolor='black'
        )
 
        ax2.set_title(f"{algo}: Mean Fitness vs Population Size")
        ax2.set_ylabel("Fitness")
        ax2.grid(True, axis='y', alpha=0.3)
 
        for i, v in enumerate(means):
            ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
 
        plt.suptitle(f"EXP-9: Population Size Effect ({algo})")
        plt.tight_layout()
        _save(f"exp9_population_size_{algo.lower()}")
 
# MAIN
"""
 uncomment the exp you will run from EXP_MAP and EXP_DESCRIPTIONS, then run the script 
 law 3mlto run w 3ayzen t3mlo tany lnafs el exp mslan ems7o el imgs mn elfolder la2no msh by overwrite  hy3od yrun 3la elfady w msh haynzl haga  
 """
EXP_MAP = {
    # Mirna
    # 1: exp1_ga_components,
    # 2: exp2_pso_update_rules,
    # 3: exp2_pso_topology,

    #Ahmed
    # 4: exp2_pso_balance,
    # 5: exp2_pso_inertia,
    # 6: exp2_pso_swarm_size,

    # Lara 
    # 7: exp2_pso_bonus,
    # 8: exp3_scenarios, 
    # 9: exp4_init_strategies,

    # Mariam
    # 10: exp5_f1_modes,
    # 11: exp6_algorithm_comparison, 
    # 12: exp7_diversity,

    # Nour 
    # 13: exp8_hybrid_islands, 
    # 14: exp9_population_size, 
 }
 
EXP_DESCRIPTIONS = {
    # Mirna
    # 1: "GA Component Study",
    # 2: "PSO Update Rules (Canonical vs Bare-bones)",
    # 3: "PSO Topology (Global vs Ring)",

    #Ahmed
    # 4: "PSO Cognitive vs Social Balance (c1/c2)",
    # 5: "PSO Inertia Schedule (Linear vs Random)",
    # 6: "PSO Swarm Size Study",

    # Lara 
    # 7: "PSO Bonus Combinations",
    # 8: "Scenario Comparison (6 disaster scenarios)",
    # 9: "Initialization Strategies (GA/PSO/Hybrid)",

    # Mariam
    # 10: "f1 Mode Sensitivity (asymmetric, absolute, squared, relative)",
    # 11: "Algorithm Comparison (GA vs PSO vs Hybrid)",
    # 12: "Diversity Preservation (Fitness Sharing)",

    # Nour 
    # 13: "Hybrid Island Dynamics (Epoch interval & Island size)",
    # 14: "Population Size Study (GA, PSO, Hybrid)",
}
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, help=f"Experiment number (1-{len(EXP_MAP)})")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    
    if args.list:
        for k, d in EXP_DESCRIPTIONS.items():
            print(f"  {k:2d}  {d}")
        sys.exit(0)
    
    VERBOSE = args.verbose
    seeds = SEEDS_30[:args.seeds]
    
    print(f"\nSeeds: {seeds}\nPlots → {PLOTS_DIR}\n")
    
    if args.exp:
        if args.exp in EXP_MAP:
            EXP_MAP[args.exp](seeds)
        else:
            print(f"Experiment {args.exp} not found. Valid experiments: {list(EXP_MAP.keys())}")
    else:
        for k, fn in EXP_MAP.items():
            print(f"\nEXP-{k}: {EXP_DESCRIPTIONS[k]}")
            fn(seeds)
        print(f"\nAll done → {PLOTS_DIR}")