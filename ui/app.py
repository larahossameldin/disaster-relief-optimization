"""
HOW TO RUN (from the project ROOT folder):
    python -m streamlit run ui/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

USE_MOCK = False

if not USE_MOCK:
    from problem.scenarioM import get_scenario
    from algorithms.pso import PSO
    from algorithms.ga import DisasterReliefGA
    from algorithms.hybridDIM_SP import DIMSPHybrid

SCENARIO_OPTIONS = [
    "Baseline",
    "Epidemic",
    "Floods",
    "Large Disaster",
    "Resource Shortage",
    "Worst Case",
]

SCENARIO_DESCRIPTIONS = {
    "Baseline":          "Reference scenario — default 8-region setup with standard budgets.",
    "Epidemic":          "Urgency ×1.3, need ×1.2 — algorithm must prioritise critical regions.",
    "Floods":            "Access difficulty ×1.4 — all regions are harder to reach.",
    "Large Disaster":    "Population ×2, budget cut to 60% — severe constraint pressure.",
    "Resource Shortage": "Water budget ×0.5, medicine budget ×0.3 — resource-specific scarcity.",
    "Worst Case":        "Urgency ×1.4, access ×1.5, population ×1.3, budget ×0.5 — everything worsened.",
}


def load_scenario(scenario_name="Baseline"):
    """Returns the scenario dict for the given scenario name."""
    if not USE_MOCK:
        name = None if scenario_name == "Baseline" else scenario_name
        return get_scenario(name)

    # Hardcoded mock — matches the structure of get_scenario() from scenarioM.py
    names = ["Region A", "Region B", "Region C", "Region D",
             "Region E", "Region F", "Region G", "Region H"]

    demand = np.array([
        [120, 95, 60],
        [80,  65, 40],
        [100, 80, 55],
        [60,  50, 30],
        [110, 90, 65],
        [45,  35, 20],
        [130, 105, 75],
        [55,  45, 25],
    ], dtype=float)

    return {
        "n_regions":      8,
        "n_resources":    3,
        "dim":            24,
        "names":          names,
        "resource_order": ["food", "water", "medicine"],
        "budget_array":   np.array([1000.0, 800.0, 600.0]),
        "demand":         demand,
        "minimums":       demand * 0.10,
        "capacity":       np.array([350, 230, 290, 170, 320, 130, 380, 160], dtype=float),
        "need":           np.array([0.90, 0.60, 0.80, 0.40, 0.70, 0.50, 1.00, 0.30]),
        "urgency":        np.array([0.85, 0.55, 0.75, 0.35, 0.65, 0.45, 0.95, 0.25]),
        "access":         np.array([0.30, 0.50, 0.20, 0.70, 0.40, 0.90, 0.60, 0.15]),
    }


# =============================================================================
# SECTION 2 — MOCK ALGORITHM RUNNERS
# Produce realistic-looking fake results so the UI works without real code.
# Each mock mirrors the exact return signature of the real function.
# =============================================================================

def _mock_allocation(sc, rng):
    """Fake (8x3) matrix where each column sums to its budget."""
    matrix = np.zeros((8, 3))
    for j, budget in enumerate(sc["budget_array"]):
        raw = rng.random(8)
        matrix[:, j] = raw / raw.sum() * budget
    return matrix


def mock_run_pso(sc, num_particles, max_iterations, seed, **kwargs):
    """Returns (best_fitness, best_matrix, history) — same as real pso.optimize()"""
    rng = np.random.default_rng(seed)
    best_matrix  = _mock_allocation(sc, rng)
    best_fitness = float(rng.uniform(280, 380))

    val = best_fitness * rng.uniform(2.5, 3.5)
    convergence = []
    for i in range(max_iterations + 1):
        decay = 0.95 if i < max_iterations * 0.5 else 0.99
        val = max(val * (decay + rng.uniform(-0.01, 0.01)), best_fitness)
        convergence.append(float(val))

    history = {
        "convergence": convergence,
        "f1_history":  [v * 0.6 for v in convergence],
        "f2_history":  [v * 0.2 for v in convergence],
        "f3_history":  [v * 0.2 for v in convergence],
    }
    return best_fitness, best_matrix, history


def mock_run_ga(sc, max_generations, population_size, seed, **kwargs):
    """Returns (solution, score, history, final_pop) — same as real ga.run()"""
    rng = np.random.default_rng(seed)
    best_matrix = _mock_allocation(sc, rng)
    solution    = best_matrix.flatten(order='F')
    score       = float(rng.uniform(300, 420))

    val = score * rng.uniform(2.0, 3.0)
    history = []
    for i in range(max_generations):
        decay = 0.94 if i < max_generations * 0.6 else 0.99
        val = max(val * (decay + rng.uniform(-0.01, 0.01)), score)
        history.append(float(val))

    final_pop = rng.random((population_size, 24))
    return solution, score, history, final_pop


def mock_run_hybrid(sc, total_generations, island_size, seed, epoch_interval=20, **kwargs):
    """Returns (best_solution, best_score, metadata) — same as real h.run()"""
    rng          = np.random.default_rng(seed)
    best_matrix  = _mock_allocation(sc, rng)
    best_solution = best_matrix.flatten(order='F')
    best_score   = float(rng.uniform(240, 330))

    n_epochs = max(1, total_generations // epoch_interval)
    val = best_score * rng.uniform(2.5, 3.5)
    hybrid_convergence = []
    island_count = []
    for i in range(n_epochs):
        decay = 0.88 if i < n_epochs * 0.5 else 0.97
        val = max(val * (decay + rng.uniform(-0.02, 0.02)), best_score)
        hybrid_convergence.append(float(val))
        island_count.append(max(1, min(i + 1, 4)))

    metadata = {"hybrid_convergence": hybrid_convergence, "island_count": island_count}
    return best_solution, best_score, metadata


# =============================================================================
# SECTION 3 — UNIFIED DISPATCHER
# The UI always calls run_algorithm() — this picks mock vs real automatically
# and returns a standardised dict so the rest of the UI never has to branch.
# =============================================================================

def run_algorithm(sc, algo, params):
    """
    Returns a standardised dict:
        best_matrix  : np.ndarray (8, 3)
        best_fitness : float
        convergence  : list[float]
        metadata     : dict  (algorithm-specific extras)
    """
    if algo == "PSO":
        if USE_MOCK:
            fitness, matrix, history = mock_run_pso(sc, **params)
        else:
            pso = PSO(sc, **params)
            fitness, matrix, history = pso.optimize()

        return {
            "best_matrix":  matrix,
            "best_fitness": fitness,
            "convergence":  history["convergence"],
            "metadata":     {k: v for k, v in history.items() if k != "convergence"},
        }

    elif algo == "GA":
        if USE_MOCK:
            solution, score, history, final_pop = mock_run_ga(sc, **params)
        else:
            ga = DisasterReliefGA(scenario_data=sc, **params)
            solution, score, history, final_pop = ga.run()

        matrix = solution.reshape(8, 3, order='F')
        return {
            "best_matrix":  matrix,
            "best_fitness": score,
            "convergence":  history,
            "metadata":     {"final_pop": final_pop},
        }

    elif algo == "Hybrid (DIM-SP)":
        if USE_MOCK:
            best_solution, best_score, meta = mock_run_hybrid(sc, **params)
        else:
            h = DIMSPHybrid(sc, **params)
            best_solution, best_score, meta = h.run()

        matrix = best_solution.reshape(8, 3, order='F')
        return {
            "best_matrix":  matrix,
            "best_fitness": best_score,
            "convergence":  meta["hybrid_convergence"],
            "metadata":     meta,
        }


# =============================================================================
# SECTION 4 — PLOT FUNCTIONS
# Each returns a matplotlib Figure rendered with st.pyplot(fig).
# =============================================================================

def plot_need_scores(sc):
    """Horizontal bar chart of need scores, colour-coded by urgency."""
    need   = sc["need"]
    names  = sc["names"]
    colors = ["#c0392b" if v >= 0.8 else "#e67e22" if v >= 0.55 else "#27ae60"
              for v in need]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(names, need, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Need Score  (0 = no need, 1 = critical)")
    ax.set_title("Region Need Scores", fontsize=13, pad=10)
    ax.set_xlim(0, 1.18)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    from matplotlib.patches import Patch
    legend = [Patch(color="#c0392b", label="Critical (≥ 0.8)"),
              Patch(color="#e67e22", label="Moderate (0.55–0.8)"),
              Patch(color="#27ae60", label="Low (< 0.55)")]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_convergence(convergence, algo, x_label="Iteration"):
    """Line plot: fitness improvement over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(convergence, color="#2980b9", linewidth=2, zorder=2)
    ax.fill_between(range(len(convergence)), convergence,
                    alpha=0.12, color="#2980b9")
    ax.set_title(f"{algo} — Convergence Curve", fontsize=13, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Best Fitness  (lower = better)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_allocation_bars(matrix, sc):
    """Grouped bar chart: Food / Water / Medicine per region. matrix: (8, 3)"""
    names     = sc["names"]
    resources = sc["resource_order"]
    colors    = ["#e67e22", "#2980b9", "#c0392b"]
    x         = np.arange(len(names))
    width     = 0.25

    fig, ax = plt.subplots(figsize=(11, 4))
    for j, (res, col) in enumerate(zip(resources, colors)):
        ax.bar(x + (j - 1) * width, matrix[:, j], width,
               label=res.capitalize(), color=col, alpha=0.88)

    ax.set_title("Resource Allocation per Region", fontsize=13, pad=10)
    ax.set_xlabel("Region")
    ax.set_ylabel("Units Allocated")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_heatmap(matrix, sc):
    """Heatmap: rows = resources, cols = regions. matrix: (8,3) transposed to (3,8)"""
    resources = [r.capitalize() for r in sc["resource_order"]]
    fig, ax = plt.subplots(figsize=(11, 3))
    sns.heatmap(
        matrix.T,
        ax=ax,
        xticklabels=sc["names"],
        yticklabels=resources,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Allocation Heatmap  (darker = more resources)", fontsize=13, pad=10)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_island_count(metadata):
    """(Hybrid only) Bar chart of active islands per epoch."""
    island_count = metadata.get("island_count", [])
    if not island_count:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(1, len(island_count) + 1), island_count, color="#8e44ad", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Islands")
    ax.set_title("DIM-SP — Island Count per Epoch", fontsize=13, pad=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_comparison_fitness(comp_results):
    """
    Bar chart comparing the best fitness of PSO, GA, and Hybrid.
    comp_results : dict  { algo_name: result_dict }
    """
    algos    = list(comp_results.keys())
    fitnesses = [comp_results[a]["best_fitness"] for a in algos]
    colors   = ["#2980b9", "#27ae60", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(algos, fitnesses, color=colors, alpha=0.87, width=0.45, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10, fontweight="bold")
    ax.set_title("Algorithm Comparison — Best Fitness (lower = better)",
                 fontsize=13, pad=10)
    ax.set_ylabel("Best Fitness")
    ax.set_ylim(0, max(fitnesses) * 1.18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Highlight the winner
    best_idx = int(np.argmin(fitnesses))
    bars[best_idx].set_edgecolor("#f39c12")
    bars[best_idx].set_linewidth(2.5)
    ax.annotate("🏆 Best", xy=(best_idx, fitnesses[best_idx]),
                xytext=(best_idx, fitnesses[best_idx] * 1.06),
                ha="center", fontsize=10, color="#f39c12", fontweight="bold")

    fig.tight_layout()
    return fig


def plot_comparison_convergence(comp_results):
    """
    Overlay convergence curves for all three algorithms on a normalised x-axis.
    Each curve is normalised to [0, 1] on the x-axis so different step counts
    can be compared side by side.
    comp_results : dict  { algo_name: result_dict }
    """
    palette = {"PSO": "#2980b9", "GA": "#27ae60", "Hybrid (DIM-SP)": "#8e44ad"}
    labels  = {"PSO": "PSO", "GA": "GA", "Hybrid (DIM-SP)": "Hybrid DIM-SP"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for algo, result in comp_results.items():
        conv = result["convergence"]
        xs   = np.linspace(0, 1, len(conv))
        ax.plot(xs, conv,
                color=palette.get(algo, "#7f8c8d"),
                linewidth=2,
                label=labels.get(algo, algo))

    ax.set_title("Convergence Comparison (normalised x-axis)", fontsize=13, pad=10)
    ax.set_xlabel("Progress  (0 = start, 1 = end)")
    ax.set_ylabel("Best Fitness  (lower = better)")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_comparison_heatmap(comp_results, sc):
    """
    Side-by-side allocation heatmaps for each algorithm.
    comp_results : dict  { algo_name: result_dict }
    """
    n_algos   = len(comp_results)
    resources = [r.capitalize() for r in sc["resource_order"]]
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 3.5), sharey=True)
    if n_algos == 1:
        axes = [axes]

    # Shared colour scale across all three heatmaps for fair comparison
    all_vals  = np.concatenate([r["best_matrix"].flatten() for r in comp_results.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for ax, (algo, result) in zip(axes, comp_results.items()):
        sns.heatmap(
            result["best_matrix"].T,
            ax=ax,
            xticklabels=sc["names"],
            yticklabels=resources if ax == axes[0] else False,
            annot=True, fmt=".0f",
            cmap="YlOrRd",
            vmin=vmin, vmax=vmax,
            linewidths=0.4, linecolor="white",
            cbar=(ax == axes[-1]),
            cbar_kws={"shrink": 0.8} if ax == axes[-1] else {},
        )
        ax.set_title(algo, fontsize=12, pad=8)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Allocation Heatmaps — All Algorithms  (same colour scale)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# SECTION 5 — PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Disaster Relief Optimizer",
    page_icon="🆘",
    layout="wide",
)

st.title("🆘 Disaster Relief Resource Allocation")
st.markdown(
    "Distributing **Food · Water · Medicine** across 8 disaster regions "
    "using Evolutionary Algorithms — **PSO**, **GA**, and **Hybrid DIM-SP**."
)

st.divider()


# =============================================================================
# SECTION 6 — SIDEBAR  (scenario + algorithm selector + parameter sliders)
# =============================================================================

with st.sidebar:
    st.header("⚙️ Controls")

    # ── Scenario selector ─────────────────────────────────────────────────────
    st.subheader("🌍 Scenario")
    selected_scenario = st.selectbox(
        "Problem Scenario",
        SCENARIO_OPTIONS,
        index=0,
        help="Choose a pre-built disaster scenario. Baseline is the default."
    )
    st.caption(SCENARIO_DESCRIPTIONS[selected_scenario])

    st.divider()

    # ── Algorithm selector ────────────────────────────────────────────────────
    st.subheader("🤖 Algorithm")
    algo = st.selectbox(
        "Algorithm",
        ["PSO", "GA", "Hybrid (DIM-SP)"],
        help="PSO = Particle Swarm | GA = Genetic Algorithm | Hybrid = DIM-SP island model"
    )

    st.subheader("Parameters")

    seed = st.number_input(
        "Random Seed", min_value=0, max_value=99999, value=42, step=1,
        help="Same seed → same result every time."
    )

    # ── PSO parameters ────────────────────────────────────────────────────────
    if algo == "PSO":
        num_particles  = st.slider("Swarm Size", 10, 100, 30, 5)
        max_iterations = st.slider("Max Iterations", 50, 500, 200, 50)

        bare = st.toggle(
            "Bare-bones PSO",
            help="Positions sampled from N(pbest, gbest) instead of velocity update."
        )
        ring = st.toggle(
            "Ring Topology",
            help="Each particle sees only k nearest neighbours (slower, more robust)."
        )

        # ── Conditional sub-parameters ──────────────────────────────────────
        if bare and ring:
            # Both enabled: show bare_prob + neighbors
            st.caption("**Bare-bones options**")
            bare_prob = st.slider(
                "Bare Probability", 0.0, 1.0, 0.5, 0.05,
                help="Probability of sampling from N(pbest, gbest) vs. staying at pbest."
            )
            st.caption("**Ring topology options**")
            neighbors = st.slider(
                "Neighbors (k)", 1, max(1, num_particles - 1), min(7, num_particles - 1), 1,
                help="Number of ring neighbours each particle can see. Must be < swarm size."
            )
            algo_params = dict(
                num_particles=num_particles, max_iterations=max_iterations,
                seed=seed, bare=bare, bare_prob=bare_prob,
                ring=ring, neighbors=neighbors,
            )

        elif bare:
            # Only bare-bones enabled: show bare_prob
            st.caption("**Bare-bones options**")
            bare_prob = st.slider(
                "Bare Probability", 0.0, 1.0, 0.5, 0.05,
                help="Probability of sampling from N(pbest, gbest) vs. staying at pbest."
            )
            algo_params = dict(
                num_particles=num_particles, max_iterations=max_iterations,
                seed=seed, bare=bare, bare_prob=bare_prob, ring=False,
            )

        elif ring:
            # Only ring topology enabled: show neighbors slider
            st.caption("**Ring topology options**")
            neighbors = st.slider(
                "Neighbors (k)", 1, max(1, num_particles - 1), min(7, num_particles - 1), 1,
                help="Number of ring neighbours each particle can see. Must be < swarm size."
            )
            algo_params = dict(
                num_particles=num_particles, max_iterations=max_iterations,
                seed=seed, bare=False, ring=ring, neighbors=neighbors,
            )

        else:
            # Neither enabled: show C1 and C2 sliders with C1+C2 ≤ 4 constraint
            st.caption("**Acceleration coefficients  (C1 + C2 ≤ 4)**")
            c1 = st.slider(
                "C1 — Cognitive", 0.1, 3.9, 1.5, 0.1,
                help="Weight of attraction toward a particle's own personal best."
            )
            # Dynamically cap C2 so the sum never exceeds 4.0
            c2_max    = max(0.1, round(4.0 - c1, 1))
            c2_default = min(1.5, c2_max)
            c2 = st.slider(
                "C2 — Social", 0.1, c2_max, c2_default, 0.1,
                help="Weight of attraction toward the global (or neighbourhood) best."
            )
            st.caption(f"C1 + C2 = **{c1 + c2:.1f}** / 4.0")
            algo_params = dict(
                num_particles=num_particles, max_iterations=max_iterations,
                seed=seed, bare=False, ring=False, c1=c1, c2=c2,
            )

        # Init strategy is always visible for PSO
        init_strat = st.selectbox(
            "Init Strategy",
            ["random", "demand_proportional", "urgency_biased"],
            key="pso_init"
        )
        algo_params["initialization_strategy"] = init_strat
        x_label = "Iteration"

    # ── GA parameters ─────────────────────────────────────────────────────────
    elif algo == "GA":
        max_generations = st.slider("Max Generations", 50, 500, 150, 50)
        population_size = st.slider("Population Size", 20, 200, 100, 5)

        init_strat = st.selectbox(
            "Init Strategy",
            ["Demand_Proportional", "Urgency_Biased", "Random"],
            key="ga_init"
        )

        with st.expander("⚙️ Advanced settings", expanded=False):

            # ── Crossover ──────────────────────────────────────────────────
            st.markdown("**Crossover**")
            crossover = st.radio(
                "crossover_type", ["BLX-α", "Uniform"],
                key="ga_crossover", label_visibility="collapsed",
                help="BLX-α = blend crossover (recommended). Uniform = gene-wise random mix."
            )
            crossover_val = "blx" if crossover == "BLX-α" else "uniform"

            # ── Mutation ───────────────────────────────────────────────────
            st.markdown("**Mutation**")
            mutation = st.radio(
                "mutation_type", ["Non-uniform", "Uniform"],
                key="ga_mutation", label_visibility="collapsed",
                help="Non-uniform = step size shrinks over generations. Uniform = random reset."
            )
            mutation_val = "nonuniform" if mutation == "Non-uniform" else "uniform"

            # ── Selection ──────────────────────────────────────────────────
            st.markdown("**Selection**")
            selection = st.radio(
                "selection_type", ["Tournament", "Roulette wheel"],
                key="ga_selection", label_visibility="collapsed",
                help="Tournament = deterministic pressure. Roulette = fitness-proportionate."
            )
            selection_val = "tournament" if selection == "Tournament" else "rws"

            # ── K (tournament size) — only when Tournament is active ────────
            if selection_val == "tournament":
                k_tourn = st.slider(
                    "K (tournament size)", 2, 20, 9, 1,
                    help="Number of candidates drawn per tournament round. "
                         "Higher K = more selection pressure."
                )
            else:
                k_tourn = 9   # irrelevant for RWS but keep default

            # ── Elitism ────────────────────────────────────────────────────
            elitism = st.slider(
                "Elitism", 0, 10, 2, 1,
                help="Number of best solutions carried unchanged into the next generation. "
                     "0 = generational model (no elitism)."
            )

            # ── Crossover rate ─────────────────────────────────────────────
            crossover_prob = st.number_input(
                "Crossover rate  *ideal 0.6 – 0.9*",
                min_value=0.10, max_value=1.00, value=0.90, step=0.05,
                format="%.2f",
                help="Probability that two parents actually recombine. "
                     "Below this threshold, parent1 is copied unchanged."
            )

            # ── Mutation rate ──────────────────────────────────────────────
            mutation_rate_input = st.number_input(
                "Mutation rate  *ideal 0.01 – 0.0417*",
                min_value=0.001, max_value=0.500,
                value=round(1 / 24, 4),   # auto default = 1/24 ≈ 0.0417
                step=0.001,
                format="%.4f",
                help="Per-gene mutation probability. "
                     "Default 1/24 ≈ 0.0417 (one gene per chromosome on average)."
            )
            mutation_rate = float(mutation_rate_input)

        algo_params = dict(
            max_generations=max_generations,
            population_size=population_size,
            config_type=None,           # individual operator params used instead
            init_strategy=init_strat,
            seed=seed,
            selection=selection_val,
            crossover=crossover_val,
            mutation=mutation_val,
            elitism=elitism,
            K_tourn=k_tourn,
            crossover_prob=crossover_prob,
            mutation_rate=mutation_rate,
        )
        x_label = "Generation"

    # ── Hybrid (DIM-SP) parameters ────────────────────────────────────────────
    elif algo == "Hybrid (DIM-SP)":
        total_generations = st.slider("Total Generations", 40, 400, 100, 20)
        island_size       = st.slider("Island Size", 20, 150, 50, 10)
        epoch_interval    = st.slider(
            "Epoch Interval", 5, 50, 20, 5,
            help="How often (in generations) islands are re-clustered."
        )
        init_strat = st.selectbox(
            "Init Strategy",
            ["Random", "Demand_Proportional", "Urgency_Biased"],
            key="hybrid_init"
        )
        algo_params = dict(
            total_generations=total_generations,
            island_size=island_size,
            epoch_interval=epoch_interval,
            init_strategy=init_strat,
            seed=seed,
        )
        x_label = "Epoch"

    st.divider()

    run_clicked     = st.button("▶ Run",             type="primary",    use_container_width=True)
    compare_clicked = st.button("⚖️ Compare All",    type="secondary",  use_container_width=True,
                                help="Runs all three algorithms with default parameters "
                                     "and compares results side by side.")
    st.caption(f"Mock mode: {'ON ✅' if USE_MOCK else 'OFF — real algorithms'}")


# =============================================================================
# SECTION 7 — SCENARIO INFO (always visible)
# =============================================================================

# Reload scenario when the selection changes
@st.cache_data
def cached_scenario(scenario_name):
    return load_scenario(scenario_name)

sc = cached_scenario(selected_scenario)

# Scenario banner
scenario_color = {
    "Baseline":          "#2980b9",
    "Epidemic":          "#c0392b",
    "Floods":            "#1a6091",
    "Large Disaster":    "#e74c3c",
    "Resource Shortage": "#e67e22",
    "Worst Case":        "#922b21",
}
banner_col = scenario_color.get(selected_scenario, "#2980b9")
st.markdown(
    f'<div style="background:{banner_col};color:white;padding:8px 14px;'
    f'border-radius:6px;font-size:0.95rem;margin-bottom:12px;">'
    f'<b>Active Scenario:</b> {selected_scenario} — {SCENARIO_DESCRIPTIONS[selected_scenario]}'
    f'</div>',
    unsafe_allow_html=True,
)

st.subheader("📋 Disaster Scenario — 8 Regions")

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    scenario_df = pd.DataFrame({
        "Region":       sc["names"],
        "Need":         [f"{v:.2f}" for v in sc["need"]],
        "Urgency":      [f"{v:.2f}" for v in sc["urgency"]],
        "Access Diff.": [f"{v:.2f}" for v in sc["access"]],
        "Capacity":     sc["capacity"].astype(int),
    })
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    budgets = sc["budget_array"]
    res     = sc["resource_order"]
    st.caption(
        f"**Budgets →**  "
        f"{res[0].capitalize()}: {budgets[0]:.0f}  |  "
        f"{res[1].capitalize()}: {budgets[1]:.0f}  |  "
        f"{res[2].capitalize()}: {budgets[2]:.0f}"
    )

with col_right:
    st.pyplot(plot_need_scores(sc))

st.divider()


# =============================================================================
# SECTION 8 — RESULTS 
# =============================================================================

if run_clicked:

    with st.spinner(f"Running {algo}... please wait ⏳"):
        result = run_algorithm(sc, algo, algo_params)

    best_matrix  = result["best_matrix"]
    best_fitness = result["best_fitness"]
    convergence  = result["convergence"]
    metadata     = result["metadata"]

    st.success(f"✅ {algo} done!")

    # ── Summary metric cards ──────────────────────────────────────────────────
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Fitness", f"{best_fitness:.4f}",
                help="Combined weighted score. Lower = better allocation.")
    pct = convergence[-1] / convergence[0] * 100 if convergence[0] != 0 else 100
    col2.metric("📉 Final / Initial fitness", f"{pct:.1f}%",
                help="Lower % = more improvement from start to finish.")
    col3.metric("🔁 Iterations / Epochs", len(convergence))

    st.divider()

    # ── Convergence curve ─────────────────────────────────────────────────────
    st.subheader("📉 Convergence Curve")
    st.markdown("A fast early drop = algorithm is learning quickly. Flat tail = converged.")
    st.pyplot(plot_convergence(convergence, algo, x_label))

    # ── Hybrid-only: island activity ──────────────────────────────────────────
    if algo == "Hybrid (DIM-SP)":
        st.divider()
        st.subheader("🏝️ Island Activity  (Hybrid-only)")
        h1, h2 = st.columns(2)
        h1.metric("Epochs completed",   len(convergence))
        h1.metric("Epoch interval",     f"every {algo_params['epoch_interval']} generations")
        with h2:
            fig_islands = plot_island_count(metadata)
            if fig_islands:
                st.pyplot(fig_islands)

    st.divider()

    # ── Allocation charts ─────────────────────────────────────────────────────
    st.subheader("📦 Resource Allocations")
    tab1, tab2 = st.tabs(["📊 Bar Chart", "🌡️ Heatmap"])

    with tab1:
        st.markdown(
            "Each cluster of 3 bars = one region.  "
            "Critical regions (high need) should receive more resources."
        )
        st.pyplot(plot_allocation_bars(best_matrix, sc))

    with tab2:
        st.markdown("Darker cell = more resources. Compare rows to spot imbalances.")
        st.pyplot(plot_heatmap(best_matrix, sc))

    st.divider()

    # ── Raw numbers ───────────────────────────────────────────────────────────
    with st.expander("🔢 Raw allocation numbers"):
        alloc_df = pd.DataFrame(
            best_matrix,
            columns=[r.capitalize() for r in sc["resource_order"]],
            index=sc["names"]
        ).round(2)
        alloc_df.index.name = "Region"
        st.dataframe(alloc_df, use_container_width=True)

        st.caption(
            "**Totals →**  "
            + "  |  ".join(
                f"{res[j].capitalize()}: {best_matrix[:, j].sum():.1f} / {budgets[j]:.0f}"
                for j in range(3)
            )
        )


# =============================================================================
# SECTION 9 — COMPARISON  
# =============================================================================

elif compare_clicked:

    st.subheader("⚖️ Algorithm Comparison")
    st.markdown(
        "Running **PSO**, **GA**, and **Hybrid DIM-SP** with default parameters "
        f"on the **{selected_scenario}** scenario (seed = {seed}). "
        "All three use identical seeds for a fair comparison."
    )

    # Default parameters for each algorithm — sensible, fast defaults
    default_params = {
        "PSO": dict(
            num_particles=30, max_iterations=200,
            seed=seed, bare=False, ring=False, c1=1.5, c2=1.5,
            initialization_strategy="urgency_biased",
        ),
        "GA": dict(
            max_generations=150, population_size=50,
            config_type="baseline", init_strategy="Demand_Proportional",
            seed=seed, mutation_rate=None,
        ),
        "Hybrid (DIM-SP)": dict(
            total_generations=100, island_size=50,
            epoch_interval=20, init_strategy="Random", seed=seed,
        ),
    }

    comp_results = {}
    progress_bar = st.progress(0, text="Starting comparison...")

    for i, (name, params) in enumerate(default_params.items()):
        progress_bar.progress(
            int((i / 3) * 100),
            text=f"Running {name}..."
        )
        comp_results[name] = run_algorithm(sc, name, params)

    progress_bar.progress(100, text="Done!")
    st.success("✅ All three algorithms finished!")

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("📋 Results Summary")

    rows = []
    for name, result in comp_results.items():
        conv = result["convergence"]
        improvement = (
            (conv[0] - conv[-1]) / conv[0] * 100
            if conv[0] != 0 else 0.0
        )
        rows.append({
            "Algorithm":        name,
            "Best Fitness":     round(result["best_fitness"], 4),
            "Initial Fitness":  round(conv[0], 4),
            "Improvement (%)":  round(improvement, 1),
            "Steps Taken":      len(conv),
        })

    summary_df = pd.DataFrame(rows).set_index("Algorithm")

    # Highlight the best fitness row
    best_algo = summary_df["Best Fitness"].idxmin()

    def highlight_best(row):
        return ["background-color: #d4efdf; font-weight: bold"
                if row.name == best_algo else "" for _ in row]

    st.dataframe(
        summary_df.style.apply(highlight_best, axis=1).format({
            "Best Fitness":    "{:.4f}",
            "Initial Fitness": "{:.4f}",
            "Improvement (%)": "{:.1f}%",
        }),
        use_container_width=True,
    )
    st.caption(f"🏆 **Winner:** {best_algo}  "
               f"(fitness = {summary_df.loc[best_algo, 'Best Fitness']:.4f})")

    st.divider()

    # ── Fitness bar chart ─────────────────────────────────────────────────────
    st.subheader("🏆 Best Fitness Comparison")
    st.markdown("Gold outline = algorithm with the lowest (best) fitness score.")
    st.pyplot(plot_comparison_fitness(comp_results))

    st.divider()

    # ── Overlay convergence ───────────────────────────────────────────────────
    st.subheader("📉 Convergence Comparison")
    st.markdown(
        "X-axis is normalised to [0, 1] so algorithms with different step counts "
        "can be compared on the same plot. A steeper early drop = faster learning."
    )
    st.pyplot(plot_comparison_convergence(comp_results))

    st.divider()

    # ── Side-by-side allocation heatmaps ─────────────────────────────────────
    st.subheader("🌡️ Allocation Heatmaps — Side by Side")
    st.markdown(
        "All three heatmaps share the **same colour scale** so darker always means "
        "the same amount, regardless of which algorithm produced it."
    )
    st.pyplot(plot_comparison_heatmap(comp_results, sc))

    st.divider()

    # ── Per-algorithm raw numbers ─────────────────────────────────────────────
    with st.expander("🔢 Raw allocation numbers (all algorithms)"):
        for name, result in comp_results.items():
            st.markdown(f"**{name}**")
            alloc_df = pd.DataFrame(
                result["best_matrix"],
                columns=[r.capitalize() for r in sc["resource_order"]],
                index=sc["names"],
            ).round(2)
            alloc_df.index.name = "Region"
            st.dataframe(alloc_df, use_container_width=True)

else:
    st.info(
        "👈 Choose your scenario and algorithm in the sidebar, "
        "then click **▶ Run** for a single algorithm "
        "or **⚖️ Compare All** to benchmark all three at once.",
        icon="ℹ️"
    )