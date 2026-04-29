#   CONSTRAINTS
#   C1 — Supply limit  : Σ_i x_ij  <=  budget_j       for each resource j
#   C2 — Capacity limit: Σ_j x_ij  <=  capacity_i     for each region i
#   C3 — Non-negative  : x_ij >= 0                     for all i, j
#   C4 — Minimum alloc : x_ij >= minimum_ij            for all i, j
#
# OUR APPROACH  (two-stage)
#   Stage 1 — REPAIR:
#       Step 1: clip to [minimum_ij, ∞)    → satisfies C3 and C4
#       Step 2: for each region i, if row sum > capacity_i → scale row down
#       Step 3: for each resource j, if col sum > budget_j → scale column down
#       Steps 2 & 3 are repeated until fully feasible (usually 1–2 passes).
#
#   Stage 2 — PENALTY:
#       Any residual violation after repair (floating-point edge cases) adds
#       a large penalty to the fitness score.
#
# USAGE (for all team members):
#   from constraints import repair, compute_penalty, is_feasible
#
#   X_fixed  = repair(X, scenario)
#   penalty  = compute_penalty(X_fixed, scenario)   # should be ~0.0

import numpy as np
PENALTY_COEFF = 10.0
MAX_REPAIR_ITERS = 10
FEASIBILITY_TOL = 1e-6
def repair(X, scenario, max_iters=MAX_REPAIR_ITERS):
    minimums = scenario["minimums"]   # (n, m)
    capacity = scenario["capacity"]   # (n,)
    budget   = scenario["budget_array"]  # (m,)
    n, m     = scenario["n_regions"], scenario["n_resources"]
    original_shape = X.shape
    if X.ndim == 1:
        X = X.reshape(scenario["n_regions"], scenario["n_resources"] , order='F')
    X_r = X.copy()
 
    for _ in range(max_iters):
        changed = False
        for i in range(n):
            row_sum = X_r[i, :].sum()
            if row_sum > capacity[i] + FEASIBILITY_TOL:
                scale   = capacity[i] / row_sum
                X_r[i, :] = np.maximum(X_r[i, :] * scale, minimums[i, :])
                changed = True
        for j in range(m):
            col_sum = X_r[:, j].sum()
            if col_sum > budget[j] + FEASIBILITY_TOL:
                scale      = budget[j] / col_sum
                X_r[:, j]  = np.maximum(X_r[:, j] * scale, minimums[:, j])
                changed = True
        if not changed:
            break
    # since the 1D chromosome was reshaped into a 2D matrix with ('F') order, it must also be flattened back using 'F' order to preserve the original mapping
    if len(original_shape) == 1:
        return X_r.flatten(order='F')
    return X_r

def compute_penalty(X, scenario, coeff=PENALTY_COEFF):
    budget   = scenario["budget_array"]   # (m,)
    capacity = scenario["capacity"]        # (n,)
    minimums = scenario["minimums"]
    total_violation = 0.0
    # Violation: supply budget exceeded
    for j in range(scenario["n_resources"]):
        excess = X[:, j].sum() - budget[j]
        if excess > 0:
            total_violation += excess
    # Violation: region capacity exceeded
    for i in range(scenario["n_regions"]):
        excess = X[i, :].sum() - capacity[i]
        if excess > 0:
            total_violation += excess
    # Violation: below minimum allocation
    below_min = np.maximum(0.0, minimums - X).sum()
    total_violation += below_min
 
    return coeff * total_violation


def is_feasible(X, scenario, tol=FEASIBILITY_TOL):
    budget   = scenario["budget_array"]
    capacity = scenario["capacity"]
    minimums = scenario["minimums"]
    violations = {}
     # Check supply budgets
    for j, res in enumerate(scenario["resource_order"]):
        total_j = X[:, j].sum()
        if total_j > budget[j] + tol:
            violations[f"budget_{res}_exceeded"] = float(total_j - budget[j])
 
    # Check region capacities
    for i, name in enumerate(scenario["names"]):
        total_i = X[i, :].sum()
        if total_i > capacity[i] + tol:
            violations[f"capacity_{name}_exceeded"] = float(total_i - capacity[i])
 
    # Check minimums
    below = np.sum(np.maximum(0.0, minimums - X))
    if below > tol:
        violations["below_minimum_total"] = float(below)
 
    feasible = len(violations) == 0
    return feasible, violations

def deb_compare(sol1, sol2, scenario, fit1, fit2):
    """
    Returns True if sol1 is better than sol2 based on Deb's rules
    """
    X1 = sol1.reshape(scenario["n_regions"], scenario["n_resources"])
    X2 = sol2.reshape(scenario["n_regions"], scenario["n_resources"])

    feas1, _ = is_feasible(X1, scenario)
    feas2, _ = is_feasible(X2, scenario)

    pen1 = compute_penalty(X1, scenario)
    pen2 = compute_penalty(X2, scenario)

    # Rule 1: feasible > infeasible
    if feas1 and not feas2:
        return True
    if feas2 and not feas1:
        return False

    # Rule 2: both infeasible → lower violation
    if not feas1 and not feas2:
        return pen1 < pen2

    # Rule 3: both feasible → compare fitness
    return fit1 < fit2   # minimization