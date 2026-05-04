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




def compute_minimums(demand, fraction, budget_array=None):
    mins = demand * fraction
    if budget_array is not None:
        for j in range(3):
            if mins[:, j].sum() > budget_array[j]:
                mins[:, j] *= budget_array[j] / mins[:, j].sum() * 0.95
    return mins

# true_need = compute_true_need(regions)  #hashof lsa h7ot elcode da feen btw this makes hard scenarios ele fihom budget change compute by true need msh el demand bs el flood w epidemic w klam da zai mhwa bdal ma ahot flag f kol scenario by2ol hwa true need wla da elnyla eltanya
# # after computing both:
# if true_need.sum() > sum(budgets.values()) * 1.05:       
#     scenario["demand"] = true_need   # scarcity detected, use true need
# else:
#     scenario["demand"] = demand      # easy scenario, original behaviour


def get_scenario(scenario_name=None, regions=None, budgets=None, fraction=MIN_FRACTION):
    
    if scenario_name == "Epidemic":
        regions = copy.deepcopy(DEFAULT_REGIONS)
        for r in regions:
            r["urgency"]    = min(r["urgency"] * 1.3, 1.0)
            r["need_score"] = min(r["need_score"] * 1.2, 1.0)

    elif scenario_name == "Floods":
        regions = copy.deepcopy(DEFAULT_REGIONS)
        for r in regions:
            r["access_difficulty"] = min(r["access_difficulty"] * 1.4, 1.0)

    elif scenario_name == "Large Disaster":
        regions = copy.deepcopy(DEFAULT_REGIONS)
        for r in regions:
            r["population"] = int(r["population"] * 2)
        budgets = {res: round(v * 0.6, 2) for res, v in DEFAULT_BUDGETS.items()}

    elif scenario_name == "Resource Shortage":
        budgets = {"food": DEFAULT_BUDGETS["food"], "water": round(DEFAULT_BUDGETS["water"] * 0.5, 2), "medicine": round(DEFAULT_BUDGETS["medicine"] * 0.3, 2)}

    elif scenario_name == "Worst Case":
        regions = copy.deepcopy(DEFAULT_REGIONS)
        for r in regions:
            r["urgency"]           = min(r["urgency"] * 1.4, 1.0)
            r["access_difficulty"] = min(r["access_difficulty"] * 1.5, 1.0)
            r["population"]        = int(r["population"] * 1.3)
        budgets = {res: round(v * 0.5, 2) for res, v in DEFAULT_BUDGETS.items()}

    if regions is None:
        regions = DEFAULT_REGIONS
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    n   = len(regions)
    m   = len(RESOURCE_ORDER)
    dim = n * m

    budget_array = np.array([budgets[res] for res in RESOURCE_ORDER], dtype=float)
    true_need    = compute_demand(regions, DEFAULT_BUDGETS)  # always on default scale
    scaled_demand = compute_demand(regions, budgets)         # scaled to actual budget

    # Use true need only when budget was actually cut (per-resource check)
    use_true_need = any(
        true_need[:, j].sum() > budget_array[j] * 1.01
        for j in range(m)
    )
    
    demand = true_need if use_true_need else scaled_demand
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
        "budget_array": np.array([budgets[res] for res in RESOURCE_ORDER], dtype=float),
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