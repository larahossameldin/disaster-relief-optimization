"""
app.py — Disaster Relief Resource Allocation  (Member 6 — Streamlit UI)
========================================================================

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
    
    # pso = PSO(sc, num_particles=30, max_iterations=200, seed=42)
    # best_fitness, best_matrix, history = pso.optimize()
    # history["convergence"] -> list, length = max_iterations + 1
    from algorithms.pso import PSO

    # ga = DisasterReliefGA(scenario_data=sc, config_type="baseline",
    #                       init_strategy="Demand_Proportional",
    #                       max_generations=200, population_size=50)
    # solution, score, history, final_pop = ga.run()
    # solution is flat (24,) array; history is list[float]
    from algorithms.ga import DisasterReliefGA

    # h = DIMSPHybrid(sc, total_generations=100, island_size=50, seed=42)
    # best_solution, best_score, metadata = h.run()
    # metadata["hybrid_convergence"] -> list (one value per epoch)
    # metadata["island_count"]       -> list of ints
    from algorithms.hybridDIM_SP import DIMSPHybrid


# =============================================================================
# SECTION 1 — SCENARIO DATA
# =============================================================================

def load_scenario():
    """Returns the scenario dict the UI needs."""
    if not USE_MOCK:
        return get_scenario()

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
    for i in range(max_iterations + 1):       # length = max_iterations + 1 (matches real)
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
    solution    = best_matrix.flatten(order='F')   # flat 24-element array
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
            # best_matrix from real PSO is already (8,3) — no reshape needed

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

        # solution is flat (24,) — reshape to (8,3) column-major order
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


@st.cache_data
def cached_scenario():
    """Cache the scenario so it only loads once per session."""
    return load_scenario()

sc = cached_scenario()


# =============================================================================
# SECTION 6 — SIDEBAR  (algorithm selector + parameter sliders)
# =============================================================================

with st.sidebar:
    st.header("⚙️ Controls")

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

    # Show different sliders depending on which algorithm is selected
    if algo == "PSO":
        num_particles  = st.slider("Swarm Size", 10, 100, 30, 5)
        max_iterations = st.slider("Max Iterations", 50, 500, 200, 50)
        bare = st.toggle("Bare-bones PSO",
                         help="Positions sampled from N(pbest, gbest) instead of velocity update.")
        ring = st.toggle("Ring Topology",
                         help="Each particle sees only k nearest neighbours (slower, more robust).")
        init_strat = st.selectbox("Init Strategy",
                                  ["random", "demand_proportional", "urgency_biased"])
        algo_params = dict(num_particles=num_particles, max_iterations=max_iterations,
                           seed=seed, bare=bare, ring=ring, initialization_strategy=init_strat)
        x_label = "Iteration"

    elif algo == "GA":
        max_generations = st.slider("Max Generations", 50, 500, 150, 50)
        population_size = st.slider("Population Size", 20, 200, 100, 10)
        config_type = st.selectbox(
            "Config Type",
            ["baseline", "rws", "uniform_crossover", "uniform_mutation", "generational"],
            help="baseline = Tournament + BLX + Non-Uniform (recommended)"
        )
        init_strat = st.selectbox("Init Strategy",
                                  ["Demand_Proportional", "Urgency_Biased", "Random"])
        algo_params = dict(max_generations=max_generations, population_size=population_size,
                           config_type=config_type, init_strategy=init_strat, seed=seed)
        x_label = "Generation"

    elif algo == "Hybrid (DIM-SP)":
        total_generations = st.slider("Total Generations", 40, 400, 100, 20)
        island_size       = st.slider("Island Size", 20, 150, 50, 10)
        epoch_interval    = st.slider("Epoch Interval", 5, 50, 20, 5,
                                      help="How often islands are re-clustered.")
        init_strat = st.selectbox("Init Strategy",
                                  ["Random", "Demand_Proportional", "Urgency_Biased"])
        algo_params = dict(total_generations=total_generations, island_size=island_size,
                           epoch_interval=epoch_interval, init_strategy=init_strat, seed=seed)
        x_label = "Epoch"

    st.divider()
    run_clicked = st.button("▶ Run", type="primary", use_container_width=True)
    st.caption(f"Mock mode: {'ON ✅' if USE_MOCK else 'OFF — real algorithms'}")


# =============================================================================
# SECTION 7 — SCENARIO INFO  
# =============================================================================

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
# SECTION 8 — RESULTS  (only shown after clicking Run)
# =============================================================================

if run_clicked:

    with st.spinner(f"Running {algo}... please wait ⏳"):
        result = run_algorithm(sc, algo, algo_params)

    best_matrix  = result["best_matrix"]    # (8, 3) ndarray
    best_fitness = result["best_fitness"]   # float
    convergence  = result["convergence"]    # list[float]
    metadata     = result["metadata"]

    st.success(f"✅ {algo} done!")

    # ── Summary metric cards ──────────────────────────────────────────────
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Fitness", f"{best_fitness:.4f}",
                help="Combined weighted score. Lower = better allocation.")
    pct = convergence[-1] / convergence[0] * 100 if convergence[0] != 0 else 100
    col2.metric("📉 Final / Initial fitness", f"{pct:.1f}%",
                help="Lower % = more improvement from start to finish.")
    col3.metric("🔁 Iterations / Epochs", len(convergence))

    st.divider()

    # ── Convergence curve ─────────────────────────────────────────────────
    st.subheader("📉 Convergence Curve")
    st.markdown("A fast early drop = algorithm is learning quickly. Flat tail = converged.")
    st.pyplot(plot_convergence(convergence, algo, x_label))

    # ── Hybrid-only: island activity ──────────────────────────────────────
    if algo == "Hybrid (DIM-SP)":
        st.divider()
        st.subheader("🏝️ Island Activity  (Hybrid-only)")
        h1, h2 = st.columns(2)
        h1.metric("Epochs completed",    len(convergence))
        h1.metric("Epoch interval",      f"every {algo_params['epoch_interval']} generations")
        with h2:
            fig_islands = plot_island_count(metadata)
            if fig_islands:
                st.pyplot(fig_islands)

    st.divider()

    # ── Allocation charts ─────────────────────────────────────────────────
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

    # ── Raw numbers ───────────────────────────────────────────────────────
    with st.expander("🔢 Raw allocation numbers"):
        res = sc["resource_order"]
        alloc_df = pd.DataFrame(
            best_matrix,
            columns=[r.capitalize() for r in res],
            index=sc["names"]
        ).round(2)
        alloc_df.index.name = "Region"
        st.dataframe(alloc_df, use_container_width=True)

        # Quick sanity check: do columns sum to their budgets?
        budgets = sc["budget_array"]
        st.caption(
            "**Totals →**  "
            + "  |  ".join(
                f"{res[j].capitalize()}: {best_matrix[:, j].sum():.1f} / {budgets[j]:.0f}"
                for j in range(3)
            )
        )

else:
    st.info(
        "👈 Choose your algorithm and parameters in the sidebar, then click **▶ Run**.",
        icon="ℹ️"
    )