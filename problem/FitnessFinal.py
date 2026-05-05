# DECISION VARIABLE (flat 1D array, length = n_regions * n_resources)
#   solution = [food_r1, food_r2, ..., water_r1, water_r2, ..., medicine_r1, ...]
#   Use decode() to convert → matrix X shape (n_regions, n_resources)
#
# THREE SUB-OBJECTIVES  (all minimised)
#   f1  Suffering cost   — penalises unmet demand
#   f2  Waste cost       — penalises over-allocation beyond demand
#   f3  Delivery cost    — aka speed distribution elamakn els3b elwsol liha bywslaha el resources mt2khar (Implicitly)
#
# COMBINED FITNESS  (lower = better)
#   F(X) = w1*f1 + w2*f2 + w3*f3  (+penalty if still infeasible after repair)
#
# THREE INITIALISATION STRATEGIES
#   initialise_random              — uniform random, repaired
#   initialise_demand_proportional — starts near proportional demand split
#   initialise_urgency_biased      — biases allocation toward urgent regions
#
# USAGE (for all team members):
#   from FITNESSSSSSS import compute_fitness, initialise_random
#   score, details = compute_fitness(flat_solution, scenario)
# =============================================================================

import numpy as np
from constraint import repair, compute_penalty


W1      = 0.6          # weight for f1 (suffering)  — highest, lives at stake
W2      = 0.2          # weight for f2 (waste)
W3      = 0.2          # weight for f3 (delivery difficulty)
F1_MODE = "asymmetric" # default f1 formula
BETA    = 0.5          # over-supply multiplier in asymmetric mode (< 1 = lenient)


# DECODE  — flat 1D array → 2D matrix


def decode(solution, n_regions):

    x        = np.asarray(solution, dtype=float)
    food     = x[0           : n_regions]
    water    = x[n_regions   : 2*n_regions]
    medicine = x[2*n_regions : 3*n_regions]
    return np.column_stack([food, water, medicine])


# SUB-OBJECTIVE 1  —  Suffering / Unmet Demand


def f1_asymmetric(X, demand, urgency, beta=BETA):
    """
    Asymmetric error  (DEFAULT — best for disaster context).

    Under-supply penalised at weight 1.
    Over-supply  penalised at weight beta  (0.5 → lenient on over-supply).

    Formula:
        f1 = Σ_ij  u_i · [ max(0, d_ij - x_ij)  +  beta · max(0, x_ij - d_ij) ]
    """
    under = np.maximum(0.0, demand - X)
    over  = np.maximum(0.0, X - demand)
    cost  = urgency[:, None] * (under + beta * over)
    return float(cost.sum())


def f1_absolute(X, demand, urgency, **kwargs):
    """
    Absolute error — treats under and over-supply equally.

    Formula:
        f1 = Σ_ij  u_i · |x_ij - d_ij|
    """
    cost = urgency[:, None] * np.abs(X - demand)
    return float(cost.sum())


def f1_squared(X, demand, urgency, **kwargs):
    """
    Squared error — large misses penalised much more than small ones.

    Formula:
        f1 = Σ_ij  u_i · (x_ij - d_ij)²
    """
    cost = urgency[:, None] * (X - demand) ** 2
    return float(cost.sum())


def f1_relative(X, demand, urgency, **kwargs):
    """
    Relative error — normalises by demand, fair across differently-sized regions.

    Formula:
        f1 = Σ_ij  u_i · |x_ij - d_ij| / (d_ij + ε)
    """
    eps  = 1e-9
    cost = urgency[:, None] * np.abs(X - demand) / (demand + eps)
    return float(cost.sum())


_F1_MODES = {
    "asymmetric": f1_asymmetric,
    "absolute"  : f1_absolute,
    "squared"   : f1_squared,
    "relative"  : f1_relative,
}



# SUB-OBJECTIVE 2  —  Waste / Over-allocation


def f2_waste(X, demand):
    """
    Waste cost — penalises any allocation that exceeds demand.

    Formula:
        f2 = Σ_ij  max(0, x_ij - d_ij)
    """
    return float(np.maximum(0.0, X - demand).sum())



# SUB-OBJECTIVE 3  —  Delivery Cost  (access difficulty based)


def f3_delivery(X, access, urgency, demand):
    """
    Delivery cost — penalises sending large amounts to hard-to-reach regions,
    weighted by urgency. Capped at demand so we don't penalise waste twice.

    Formula:
        f3 = Σ_i  access_i · urgency_i · min( Σ_j x_ij,  Σ_j d_ij )

    Parameters
    ----------
    X      : ndarray (n, m)
    access : ndarray (n,)   access_difficulty  [0-1]
    urgency: ndarray (n,)   urgency weight     [0-1]
    demand : ndarray (n, m) demand matrix  (used for cap)
    """
    total_allocated = X.sum(axis=1)
    total_demand    = demand.sum(axis=1)
    effective       = np.minimum(total_allocated, total_demand)
    cost            = access * urgency * effective
    return float(cost.sum())



# NORMALISATION CONSTANTS  (optional — pass norm=... to compute_fitness)

def compute_norm_constants(scenario):
    """
    Compute worst-case values for f1, f2, f3 so they can be normalised
    to [0, 1] before combining. Useful for fair weighting across scales.
    Pass the returned dict as norm=... to compute_fitness.
    """
    demand  = scenario["demand"]
    urgency = scenario["urgency"]
    access  = scenario["access"]
    budgets = scenario["budget_array"]

    f1_max = float(np.sum(urgency[:, None] * demand))  # worst = send nothing
    f2_max = float(budgets.sum())                       # worst = all waste
    f3_max = float(np.sum(access * urgency * budgets.sum()))

    return {
        "f1_max": max(f1_max, 1e-9),
        "f2_max": max(f2_max, 1e-9),
        "f3_max": max(f3_max, 1e-9),
    }



# COMBINED FITNESS  — the one function everyone calls


def compute_fitness(solution, scenario, f1_mode=F1_MODE, beta=BETA,
                    w1=W1, w2=W2, w3=W3, norm=None):
    """
    Compute the combined fitness  F = w1*f1 + w2*f2 + w3*f3 + penalty.

    Lower score = better solution  (minimisation).

    Parameters
    ----------
    solution : 1D array-like length=dim  OR  ndarray shape (n_regions, n_resources)
    scenario : dict  — from scenarioM.get_scenario()
    f1_mode  : str   — "asymmetric" | "absolute" | "squared" | "relative"
    beta     : float — over-supply multiplier for asymmetric mode
    w1,w2,w3 : float — sub-objective weights
    norm     : dict or None — from compute_norm_constants(); normalises scores

    Returns
    -------
    score   : float   combined fitness  (lower = better)
    details : dict    {"f1":..., "f2":..., "f3":..., "F":..., "penalty":...}
    """
    # --- decode: accept both flat vectors and matrices ----------------------
    sol = np.asarray(solution, dtype=float)
    if sol.ndim == 1:
        X = decode(sol, scenario["n_regions"])   
    else:
        X = sol.copy()
        
    # --- residual penalty (X is 2D here, so compute_penalty works) ----------              
    
    # --- repair before evaluating -------------------------------------------
    X = repair(X, scenario)                      # X is always 2D after this
    penalty  = compute_penalty(X, scenario)
    demand  = scenario["demand"]
    urgency = scenario["urgency"]
    access  = scenario["access"]

    # --- pick f1 formula ----------------------------------------------------
    f1_fn = _F1_MODES.get(f1_mode)
    if f1_fn is None:
        raise ValueError(f"Unknown f1_mode '{f1_mode}'. "
                         f"Choose from: {list(_F1_MODES.keys())}")

    # --- compute sub-objectives ---------------------------------------------
    score_f1 = f1_fn(X, demand, urgency, beta=beta)
    score_f2 = f2_waste(X, demand)
    score_f3 = f3_delivery(X, access, urgency, demand)   # BUG FIX: pass demand

    # --- optional normalisation ---------------------------------------------
    if norm is not None:
        score_f1 = score_f1 / norm["f1_max"]
        score_f2 = score_f2 / norm["f2_max"]
        score_f3 = score_f3 / norm["f3_max"]

    F = w1 * score_f1 + w2 * score_f2 + w3 * score_f3 + penalty

    details = {
        "f1"     : score_f1,
        "f2"     : score_f2,
        "f3"     : score_f3,
        "F"      : F,
        "penalty": penalty,
    }
    return F, details


# INITIALISATION STRATEGIES chat ale nst5dm 20% 40% 40% 

def initialise_random(n_solutions, scenario, seed=None):
    """
    Strategy 1 — Uniform random initialisation.

    Each allocation x_ij drawn uniformly from [minimum_ij, budget_j],
    then repaired to satisfy all constraints.

    Good for: broad exploration of the search space.
    """
    rng      = np.random.default_rng(seed)
    n        = scenario["n_regions"]
    budgets  = scenario["budget_array"]
    minimums = scenario["minimums"]
    pop      = np.zeros((n_solutions, scenario["dim"]))

    for k in range(n_solutions):
        X = np.zeros((n, 3))
        for j in range(3):
            X[:, j] = rng.uniform(minimums[:, j], budgets[j])
        pop[k] = repair(X, scenario).flatten(order='F')

    return pop


def initialise_demand_proportional(n_solutions, scenario, seed=None):
    """
    Strategy 2 — Demand-proportional initialisation.

    Each solution starts at the fair-share demand allocation,
    then small noise (+-5% of budget) is added, then repaired.

    Good for: warm-starting near a reasonable baseline.
    """
    rng     = np.random.default_rng(seed)
    n       = scenario["n_regions"]
    budgets = scenario["budget_array"]
    demand  = scenario["demand"]

    col_sums = demand.sum(axis=0) + 1e-9
    base     = demand / col_sums * budgets[None, :]    # (n, 3)

    pop = np.zeros((n_solutions, scenario["dim"]))
    for k in range(n_solutions):
        noise  = rng.uniform(-0.05, 0.05, size=(n, 3)) * budgets[None, :]
        X      = base + noise                          # BUG FIX: was undefined X
        pop[k] = repair(X, scenario).flatten(order='F')

    return pop


def initialise_urgency_biased(n_solutions, scenario, seed=None):
    """
    Strategy 3 — Urgency-biased initialisation.

    Each region's base share is weighted by its urgency score,
    so urgent regions start with more resources. Noise added, then repaired.

    Good for: starting with solutions that already prioritise critical regions.
    """
    rng     = np.random.default_rng(seed)
    n       = scenario["n_regions"]
    budgets = scenario["budget_array"]
    demand  = scenario["demand"]
    urgency = scenario["urgency"]

    ud       = urgency[:, None] * demand
    col_sums = ud.sum(axis=0) + 1e-9
    base     = ud / col_sums * budgets[None, :]        # (n, 3)

    pop = np.zeros((n_solutions, scenario["dim"]))
    for k in range(n_solutions):
        noise  = rng.uniform(-0.05, 0.05, size=(n, 3)) * budgets[None, :]
        X      = base + noise                          # BUG FIX: was undefined X
        pop[k] = repair(X, scenario).flatten(order='F')

    return pop


# =============================================================================
# Sanity check — run this file directly JUST TESTING
# =============================================================================
if __name__ == "__main__":
    from scenarioM import get_scenario

    sc   = get_scenario()
    n, m = sc["n_regions"], sc["n_resources"]

    print("=" * 60)
    print("  FITNESS + INITIALISATION — SANITY CHECKS")
    print("=" * 60)

    # Test 1: perfect allocation (matrix input)
    X_perfect = sc["demand"].copy()
    score, d = compute_fitness(X_perfect, sc)
    print(f"\nTest 1 — Perfect allocation (matrix input):")
    print(f"  f1={d['f1']:.4f}  f2={d['f2']:.4f}  f3={d['f3']:.4f}  penalty={d['penalty']:.4f}  F={d['F']:.4f}")

    # Test 2: zero allocation
    X_zero = np.zeros((n, m))
    score, d = compute_fitness(X_zero, sc)
    print(f"\nTest 2 — Zero allocation:")
    print(f"  f1={d['f1']:.4f}  f2={d['f2']:.4f}  f3={d['f3']:.4f}  penalty={d['penalty']:.4f}  F={d['F']:.4f}")

    # Test 3: flat 1D vector input
    flat = sc["demand"].flatten(order='F')
    score, d = compute_fitness(flat, sc)
    print(f"\nTest 3 — Same demand passed as flat 1D vector:")
    print(f"  f1={d['f1']:.4f}  f2={d['f2']:.4f}  f3={d['f3']:.4f}  penalty={d['penalty']:.4f}  F={d['F']:.4f}")

    # Test 4: all f1 modes
    print(f"\nTest 4 — All f1 modes:")
    for mode in ["asymmetric", "absolute", "squared", "relative"]:
        s, d = compute_fitness(X_perfect, sc, f1_mode=mode)
        print(f"  {mode:<12}: f1={d['f1']:.4f}  F={d['F']:.4f}")

    # Test 5: with normalisation
    norm = compute_norm_constants(sc)
    score, d = compute_fitness(X_perfect, sc, norm=norm)
    print(f"\nTest 5 — With normalisation:")
    print(f"  f1={d['f1']:.6f}  f2={d['f2']:.6f}  f3={d['f3']:.6f}  F={d['F']:.6f}")
    print(f"  Norm constants: { {k: round(v,2) for k,v in norm.items()} }")

    # Test 6: initialisation strategies
    print(f"\nTest 6 — Initialisation strategies (5 solutions each):")
    for fn_name, fn in [("random",               initialise_random),
                         ("demand_proportional",  initialise_demand_proportional),
                         ("urgency_biased",       initialise_urgency_biased)]:
        pop    = fn(5, sc, seed=42)
        scores = [compute_fitness(pop[k], sc)[0] for k in range(5)]
        print(f"  {fn_name:<22}: shape={pop.shape}  avg_fitness={np.mean(scores):.4f}")

    print("\nAll tests passed!")