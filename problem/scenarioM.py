import numpy as np
import copy

DEFAULT_BUDGETS = {
    "food":     1000,
    "water":    800,
    "medicine": 600,
}
RESOURCE_ORDER = ["food", "water", "medicine"]
MIN_FRACTION = 0.1
DEFAULT_REGIONS = [
    {
        "name":             "Region A",
        "population":       8000,
        "need_score":       0.90,
        "urgency":          0.95,
        "access_difficulty":0.30,
        "capacity":         400,
        "resource_weights": {"food": 0.5, "water": 0.3, "medicine": 0.2},
    },
    {
        "name":             "Region B",
        "population":       3000,
        "need_score":       0.70,
        "urgency":          0.60,
        "access_difficulty":0.80,
        "capacity":         200,
        "resource_weights": {"food": 0.4, "water": 0.4, "medicine": 0.2},
    },
    {
        "name":             "Region C",
        "population":       6000,
        "need_score":       0.85,
        "urgency":          0.90,
        "access_difficulty":0.50,
        "capacity":         350,
        "resource_weights": {"food": 0.5, "water": 0.3, "medicine": 0.2},
    },
    {
        "name":             "Region D",
        "population":       2000,
        "need_score":       0.40,
        "urgency":          0.30,
        "access_difficulty":0.90,
        "capacity":         150,
        "resource_weights": {"food": 0.3, "water": 0.5, "medicine": 0.2},
    },
    {
        "name":             "Region E",
        "population":       7000,
        "need_score":       0.95,
        "urgency":          0.98,
        "access_difficulty":0.20,
        "capacity":         450,
        "resource_weights": {"food": 0.4, "water": 0.3, "medicine": 0.3},
    },
    {
        "name":             "Region F",
        "population":       4000,
        "need_score":       0.60,
        "urgency":          0.50,
        "access_difficulty":0.70,
        "capacity":         250,
        "resource_weights": {"food": 0.5, "water": 0.3, "medicine": 0.2},
    },
    {
        "name":             "Region G",
        "population":       5000,
        "need_score":       0.75,
        "urgency":          0.70,
        "access_difficulty":0.40,
        "capacity":         300,
        "resource_weights": {"food": 0.4, "water": 0.4, "medicine": 0.2},
    },
    {
        "name":             "Region H",
        "population":       1500,
        "need_score":       0.50,
        "urgency":          0.40,
        "access_difficulty":0.95,
        "capacity":         100,
        "resource_weights": {"food": 0.3, "water": 0.4, "medicine": 0.3},
    },
]


ASYMMETRIC_REGIONS = copy.deepcopy(DEFAULT_REGIONS)
ASYMMETRIC_REGIONS[3]["urgency"]           = 1.00
ASYMMETRIC_REGIONS[3]["access_difficulty"] = 0.98
ASYMMETRIC_REGIONS[3]["need_score"]        = 0.95
BUDGET_MILD   = DEFAULT_BUDGETS
BUDGET_SEVERE = {res: round(v * 0.6, 2) for res, v in DEFAULT_BUDGETS.items()}

def compute_demand(regions, budgets): 
    'dij​=need_scorei​×populationi​×resource_weightij​×scale_factor'
    'scale_factorj​=Sj/∑i​need_scorei​×populationi​×resource_weightij​​'
    n= len(regions)
    raw=np.zeros((n,3))
    for i, r in enumerate(regions):
        for j, res in enumerate(RESOURCE_ORDER):
            raw[i, j] = (r["need_score"]
                         * r["population"]
                         * r["resource_weights"][res])
    demand = np.zeros_like(raw)
    for j, res in enumerate(RESOURCE_ORDER):
        col_sum = raw[:, j].sum()
        if col_sum > 0:
            scale = budgets[res] / col_sum
        else:
            scale = 1.0
        demand[:, j] = raw[:, j] * scale

    return demand

def compute_minimums(demand, fraction=MIN_FRACTION): 
    'mij'
    """
    Minimum allocation per region per resource = fraction x demand.
    Default fraction = 0.1 → every region gets at least 10% of what it needs.
    """
    return demand * fraction


def get_scenario(regions=None, budgets=None, fraction=MIN_FRACTION): 
    """
    est5dmoha 3shan t3rfo el scenarios yargala w elheta de so AI (yarit mhdsh y2ra el3bat da bs its just there)
    Build and return the full scenario dictionary.

    Parameters
    ----------
    regions      : list of region dicts, or None → uses DEFAULT_REGIONS
    budgets      : dict {"food": ..., "water": ..., "medicine": ...}
                   or None → uses DEFAULT_BUDGETS
    min_fraction : minimum allocation fraction of demand (default 0.1)

    Returns
    -------
    dict with keys:
        regions       - list of region dicts
        budgets       - resource budget dict
        n_regions     - int (8 by default)
        n_resources   - int (3)
        dim           - int (n_regions * n_resources = 24 by default)
        resource_order- list of resource names in order
        demand        - np.ndarray (n_regions, 3)
        minimums      - np.ndarray (n_regions, 3)
        need          - np.ndarray (n_regions,)   — need scores
        urgency       - np.ndarray (n_regions,)   — urgency scores
        access        - np.ndarray (n_regions,)   — access difficulty
        capacity      - np.ndarray (n_regions,)   — max absorb per region
        budget_array  - np.ndarray (3,)            — budgets as array
        names         - list of region name strings
    """
    if regions is None:
        regions = DEFAULT_REGIONS
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    n   = len(regions)
    m   = len(RESOURCE_ORDER)
    dim = n * m

    demand   = compute_demand(regions, budgets)
    minimums = compute_minimums(demand, fraction=fraction)

    scenario = {
        "regions":       regions,
        "budgets":       budgets,
        "n_regions":     n,
        "n_resources":   m,
        "dim":           dim,
        "resource_order":RESOURCE_ORDER,
        "demand":        demand,
        "minimums":      minimums,
        "need":     np.array([r["need_score"]        for r in regions]),
        "urgency":  np.array([r["urgency"]            for r in regions]),
        "access":   np.array([r["access_difficulty"]  for r in regions]),
        "capacity": np.array([r["capacity"]           for r in regions]),
        "budget_array": np.array([budgets[res] for res in RESOURCE_ORDER],
                                  dtype=float),
        "names":    [r["name"] for r in regions],
    }
    return scenario

'TEST TEST TEST TEST'
if __name__ == "__main__":
    sc = get_scenario()
    print(f"Regions   : {sc['n_regions']}")
    print(f"Dim       : {sc['dim']}")
    print(f"Budgets   : {sc['budgets']}")
    print(f"\nDemand matrix (regions x resources):\n{sc['demand'].round(2)}")
    print(f"\nDemand totals : {sc['demand'].sum(axis=0).round(2)}")
    print(f"Budget array  : {sc['budget_array']}")
    print(f"\nMinimums (10%):\n{sc['minimums'].round(2)}")
    print("\nscenario.py OK")