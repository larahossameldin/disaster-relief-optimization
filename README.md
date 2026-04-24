# disaster-relief-optimization
---

## DISCLAIMER: these are just loose guidelines, prone to change as we move through the project
## After you are done with your task, change the part where your task is written here with a summary of what you actually did

----
## General Notes for Team

- Always pull from main before starting a session: `git pull origin main`
- Work on your own branch, never directly on main
- Member 1 should merge first — everyone else depends on their files
- After Member 1 merges, run `git pull origin main` to get their files before you start coding
- If you are stuck, ask the team before spending hours on something
---
## Member Tasks
### Member 1 — Problem Setup, Fitness Function & Constraint Handler
## AI420 Project 5: Disaster Relief Resource Allocation
 
---
 
## What This Folder Contains (Member 1's Files)
 
| File | Purpose |
|---|---|
| `scenarioM.py` | Defines the disaster scenario: 8 regions, 3 resources, budgets, demand, minimums |
| `constraint.py` | Constraint handler: repair function, penalty function, feasibility checker |
| `FitnessFinal.py` | Fitness function + 3 initialisation strategies |
 
---
 
## Are the Files Correctly Linked? ✅ YES
 
The dependency chain is:
 
```
scenarioM.py  ──┐
                ├──► FitnessFinal.py
constraint.py ──┘
```
 
- `scenarioM.py` → standalone, no imports from our files
- `constraint.py` → standalone, no imports from our files
- `FitnessFinal.py` → imports from BOTH:
  ```python
  from constraint import repair, compute_penalty
  # and at the bottom test block:
  from scenarioM import get_scenario
  ```
 
No circular imports. No missing links. All three files must be in the **same folder** when running.
 
---
 
## Quick-Start for All Teammates

 
### Step 1 — Install dependencies
```bash
pip install numpy
```
That's it. No other libraries needed for Member 1's files.
 
---
 
## How to Load the Scenario (Everyone needs this)
 
```python
from scenarioM import get_scenario
 
sc = get_scenario()   # loads default 8-region scenario
```
 
### What `sc` gives you (the scenario dictionary)
 
| Key | Type | Description |
|---|---|---|
| `sc["n_regions"]` | int | Number of regions (8) |
| `sc["n_resources"]` | int | Number of resources (3: food, water, medicine) |
| `sc["dim"]` | int | Total decision variables = 8 × 3 = **24** |
| `sc["demand"]` | ndarray (8, 3) | How much each region needs of each resource |
| `sc["minimums"]` | ndarray (8, 3) | Minimum allocation per region per resource (10% of demand) |
| `sc["budget_array"]` | ndarray (3,) | Total budget per resource `[1000, 800, 600]` |
| `sc["capacity"]` | ndarray (8,) | Max total resources each region can absorb |
| `sc["need"]` | ndarray (8,) | Need scores per region (0–1) |
| `sc["urgency"]` | ndarray (8,) | Urgency scores per region (0–1) |
| `sc["access"]` | ndarray (8,) | Access difficulty per region (0–1, higher = harder) |
| `sc["names"]` | list[str] | `["Region A", ..., "Region H"]` |
| `sc["resource_order"]` | list[str] | `["food", "water", "medicine"]` |
 
---
 
## Solution Representation (IMPORTANT — Read Carefully)
 
A solution is a **flat 1D array of length 24**:
 
```
[food_A, food_B, ..., food_H,   water_A, ..., water_H,   medicine_A, ..., medicine_H]
  index 0–7                      index 8–15                index 16–23
```
 
This is called **column-major (Fortran) order**. To convert between flat and matrix:
 
```python
import numpy as np
 
# Flat → Matrix (8 rows = regions, 3 cols = resources)
X_matrix = solution.reshape(8, 3, order='F')    # shape (8, 3)
 
# Matrix → Flat
solution = X_matrix.flatten(order='F')           # shape (24,)
```
 
`FitnessFinal.py` has a helper for this:
```python
from FitnessFinal import decode
X_matrix = decode(solution, n_regions=8)
```
 
---
 
## How to Evaluate Fitness (Member 2, 3, 4 — use this every generation)
 
```python
from FitnessFinal import compute_fitness
from scenarioM import get_scenario
 
sc = get_scenario()
 
# solution can be flat 1D array (length 24) OR matrix (8, 3)
score, details = compute_fitness(solution, sc)
 
print(score)           # combined fitness — LOWER IS BETTER
print(details["f1"])   # suffering/unmet demand cost
print(details["f2"])   # waste cost
print(details["f3"])   # delivery difficulty cost
print(details["penalty"])  # constraint violation penalty (should be ~0 after repair)
```
 
### Fitness Formula
```
F = 0.6 × f1  +  0.2 × f2  +  0.2 × f3  +  penalty
```
- **f1** (suffering) — penalises unmet demand, weighted by urgency. Highest weight because lives are at stake.
- **f2** (waste) — penalises sending more than needed.
- **f3** (delivery) — penalises sending large amounts to hard-to-reach regions.
- **penalty** — large penalty for any constraint violation that survived repair.
### Optional: Normalised Fitness
If your scores are on very different scales, normalise before combining:
```python
from FitnessFinal import compute_fitness, compute_norm_constants
 
norm = compute_norm_constants(sc)   # compute once, reuse every generation
score, details = compute_fitness(solution, sc, norm=norm)
```
 
### Optional: Change f1 mode
```python
score, details = compute_fitness(solution, sc, f1_mode="absolute")
# Options: "asymmetric" (default), "absolute", "squared", "relative"
```
 
---
 
## How to Repair a Solution (Constraint Handler)
 
**Always repair before evaluating fitness.** The repair function handles all 4 constraints:
- C1: Total allocation per resource ≤ budget
- C2: Total allocation per region ≤ capacity
- C3: All values ≥ 0
- C4: Each allocation ≥ minimum (10% of demand)
```python
from constraint import repair, is_feasible, compute_penalty
 
# Input can be matrix (8,3) — repair always returns same shape
X_repaired = repair(X, sc)
 
# Check if feasible
feasible, violations = is_feasible(X_repaired, sc)
print(feasible)     # True/False
print(violations)   # dict of any remaining violations (should be empty)
 
# Compute penalty (for Deb's rules or debugging)
penalty = compute_penalty(X_repaired, sc)
print(penalty)      # should be ~0.0 after repair
```
 
### Deb's Constraint Comparison (useful for selection)
```python
from constraint import deb_compare
 
# Returns True if sol1 is BETTER than sol2
is_better = deb_compare(sol1, sol2, sc, fit1=score1, fit2=score2)
```
Rules: feasible > infeasible → lower violation → lower fitness.
 
---
 
## How to Initialise a Population (Member 2, 3, 4 — use for your starting population)
 
Three strategies available. **Use all three and compare results** (bonus marks!):
 
```python
from FitnessFinal import (
    initialise_random,
    initialise_demand_proportional,
    initialise_urgency_biased
)
 
sc = get_scenario()
POP_SIZE = 50
 
# Strategy 1: Uniform random (broad exploration)
pop = initialise_random(POP_SIZE, sc, seed=42)
 
# Strategy 2: Demand-proportional (warm start near fair allocation)
pop = initialise_demand_proportional(POP_SIZE, sc, seed=42)
 
# Strategy 3: Urgency-biased (prioritises critical regions from the start)
pop = initialise_urgency_biased(POP_SIZE, sc, seed=42)
 
# pop.shape == (50, 24)  — 50 solutions, each a flat array of length 24
```
 
The project guidelines require at least 2 initialisation strategies compared across 30 runs. Use `seed=RUN_INDEX` to get reproducible but different results per run.
 
---
 
## Constraints Summary (Quick Reference)
 
| Constraint | Formula | Meaning |
|---|---|---|
| C1 Budget | Σᵢ x_ij ≤ budget_j | Can't spend more than available per resource |
| C2 Capacity | Σⱼ x_ij ≤ capacity_i | Can't send more than region can absorb |
| C3 Non-negative | x_ij ≥ 0 | No negative allocations |
| C4 Minimum | x_ij ≥ 0.1 × demand_ij | Every region gets at least 10% of what it needs |
 
---
 
## Running Sanity Checks
 
To verify everything works, run each file directly:
 
```bash
python scenarioM.py      # Should print demand matrix and "scenario.py OK"
python constraint.py     # (no __main__ block — import and test manually)
python FitnessFinal.py   # Runs 6 tests and prints "All tests passed!"
```
 
Expected output from `FitnessFinal.py`:
- Test 1 (perfect allocation): penalty ≈ 0, F > 0 due to f3
- Test 2 (zero allocation): high F due to unmet demand
- Test 6 (initialisation): all three strategies produce valid populations
---
 

 
### Member 2 (PSO)
#### `algorithms/pso.py` — Particle Swarm Optimisation

#### What this file does

This file implements a complete, self-contained PSO optimiser for the disaster relief resource allocation problem. Given a scenario (8 regions, 3 resources — food, water, medicine), it finds the allocation matrix that minimises a weighted combination of three objectives: suffering (unmet demand), waste (over-allocation), and delivery difficulty (access cost).

#### How it works

**Representation:** Each particle is a flat 1D vector of length 24 (= 8 regions × 3 resources, column-major order). The `decode()` function from `FitnessFinal.py` converts it back to an (8×3) allocation matrix.

**Fitness:** Calls `compute_fitness()` from `FitnessFinal.py`, which returns `F = w1·f1 + w2·f2 + w3·f3 + penalty`. Lower is better.

**Constraints:** Every new particle position is passed through `repair()` from `constraint.py` immediately after each move, so the swarm always stays in the feasible region.

**Two update rules** (controlled by `bare=True/False`):
- Canonical PSO — velocity update with inertia, cognitive, and social terms
- Bare-bones PSO — no velocity; each dimension sampled from a normal distribution centred between the particle's personal best and its neighborhood best

**Two topologies** (controlled by `ring=True/False`):
- Global (gbest) — every particle is attracted to the single swarm-wide best. Fast convergence, higher risk of getting stuck.
- Ring (lbest) — each particle only knows its `k` nearest neighbours' bests. Slower but more robust.

**Two inertia schedules:**
- `LinearInertia(w_start, w_end)` — ω decreases linearly over iterations (default 0.9 → 0.5)
- `RandomInertia(rng)` — ω drawn randomly from [0.5, 1) each iteration

**Three initialisation strategies** (controlled by `initialization_strategy`):
- `'random'` — uniform random, broad exploration
- `'demand_proportional'` — starts near the fair-share demand split
- `'urgency_biased'` — biases initial allocation toward the most urgent regions

#### How to use it

```python
from algorithms.pso import PSO, build_all_configs
from problem.scenarioM import get_scenario

sc = get_scenario()

# Basic run
pso = PSO(sc, num_particles=30, max_iterations=200, seed=42)
best_fitness, best_matrix, history = pso.optimize()

# best_fitness  — float, the combined F score (lower = better)
# best_matrix   — ndarray shape (8, 3), the allocation per region per resource
# history       — dict with keys:
#                   "convergence"  list of gbest fitness at each iteration (length = max_iterations + 1)
#                   "f1_history"   suffering sub-objective per iteration
#                   "f2_history"   waste sub-objective per iteration
#                   "f3_history"   delivery sub-objective per iteration
```

All 22 pre-defined configurations are available via:

```python
configs = build_all_configs()   # returns list of (label, kwargs) tuples
for label, kwargs in configs:
    pso = PSO(sc, max_iterations=200, seed=42, **kwargs)
    fitness, matrix, history = pso.optimize()
```

#### For the plotting member

Everything you need is in the `history` dict returned by `optimize()`. Specifically:
- `history["convergence"]` — plot this against iteration index for the convergence curve. Length is always `max_iterations + 1` (index 0 is the initial swarm state before any updates).
- `history["f1_history"]`, `history["f2_history"]`, `history["f3_history"]` — same length, plot these to show how each sub-objective evolved separately.
- `best_matrix` — an (8×3) numpy array. Rows are regions (A–H), columns are food/water/medicine. Use `sc["names"]` for region labels and `sc["resource_order"]` for column labels.
- To compare configs, run each one and collect their `"convergence"` lists — they're all the same length so they can go on the same axes directly.


#### Hyperparameter Tuning — PSO

To find the best PSO configuration for this problem, we use **Optuna** — a hyperparameter optimisation framework that is more flexible than scikit-learn's `GridSearchCV`/`RandomizedSearchCV` because it does not require a sklearn-compatible model. You simply define an objective function that runs your algorithm and returns a score, and Optuna handles the rest.

#### Search Strategy

We use **TPE (Tree-structured Parzen Estimator)**, Optuna's default and most powerful sampler. Rather than searching randomly or exhaustively, TPE builds a probabilistic model from completed trials and uses it to propose configurations more likely to improve on the current best — making it significantly more sample-efficient than random search.

The first 10 trials are random warm-up to seed the model, after which TPE guides the remaining 40 trials intelligently.

### Parameters Tuned

| Parameter | Type | Range / Options |
|---|---|---|
| `num_particles` | int | 10 – 80 |
| `c1` | float | 0.3 – 3.0 |
| `c2` | float | 0.3 – 3.0 |
| `inertia` | categorical | `linear`, `random` |
| `bare` | categorical | `True`, `False` |
| `bare_prob` | float | 0.3 – 0.95 *(only when `bare=True`)* |
| `ring` | categorical | `True`, `False` |
| `neighbors` | int | 2 – 8 *(only when `ring=True`)* |
| `initialization_strategy` | categorical | `random`, `demand_proportional`, `urgency_biased` |
| `f1_mode` | categorical | `asymmetric` | `absolute` | `squared` | `relative` |

`bare_prob` and `neighbors` are **conditionally sampled** — they are only explored when the parameter they depend on is active, keeping the search space tight and avoiding wasted trials on irrelevant combinations.

#### Noise Reduction

Each trial runs PSO **3 times with different random seeds** and returns the mean fitness. This reduces the effect of lucky or unlucky initialisation and gives Optuna a more reliable signal to optimise against.

#### Output

Running `hyperparameterTuningPSO.py` will print a trial log, display the top 5 configurations found, and save a convergence plot to `experiments/plots/tpe_convergence.png`.

```
python experiments/hyperparameterTuningPSO.py
```
# Member 3 (GA with PyGAD)
### *An Evolutionary Optimization Engine for Emergency Resource Allocation*


---

## What Is This?

**DisasterReliefGA** is a Genetic Algorithm that solves a real-world constrained optimization problem: *how do you allocate limited emergency resources across multiple disaster-hit regions as effectively as possible?*

It evolves a population of allocation plans generation by generation, using custom-designed operators built specifically for this problem — smart crossover, bounded mutation, and fitness sharing to avoid premature convergence.

---

## Project Structure

```
disaster-relief-optimization/
├── algorithms/
│   ├── ga.py                      ← Main GA engine (start here)
│   ├── pso.py
│   ├── hybrid(DIM_SP).py
│   └── hybrid(SIM).py
├── experiments/
│   ├── plots/
│   ├── run_experiments.py
│   └── hyperparameterTuningPSO.py
├── problem/
│   ├── scenarioM.py               ← Scenario data loader
│   ├── FitnessFinal.py            ← Fitness function + initialization strategies
│   └── constraint.py              ← Repair function for constraint violations
├── ui/
│   └── app.py
├── .gitignore
└── README.md
```

---

## Quick Start

**Step 1 — Install dependencies**

```bash
pip install numpy pygad
```

**Step 2 — Run the ablation study** *(from the project root)*

```bash
python experiments/run_experiments.py
```

> ⚠️ Run from the **root directory** — not from inside `experiments/`. Otherwise the `problem/` imports will fail.

**To run the GA engine directly:**

```bash
python algorithms/ga.py
```

**To launch the UI:**

```bash
python ui/app.py
```

---

## Running Your Own Experiments

Instantiate `DisasterReliefGA` directly for custom runs:

```python
from algorithms.ga import DisasterReliefGA
from problem.scenarioM import get_scenario

scenario = get_scenario()

optimizer = DisasterReliefGA(
    scenario_data=scenario,
    config_type="baseline",
    init_strategy="Demand_Proportional",
    max_generations=150,
    population_size=100,
    crossover_prob=0.75
)

solution, score, history, final_pop = optimizer.run()
print(f"Final Score: {score:.4f}")
```

### What `.run()` Returns

| Return Value  | Type           | Description                                           |
|---------------|----------------|-------------------------------------------------------|
| `solution`    | `np.ndarray`   | Best allocation plan found (repaired & validated)     |
| `score`       | `float`        | Final fitness score                                   |
| `history`     | `list[float]`  | Best score per generation — use for convergence plots |
| `final_pop`   | `np.ndarray`   | Full population at the last generation                |

---

## Parameters Reference

### `config_type` — Operator Architecture

Defines which combination of genetic operators the algorithm uses.

| Value                 | Selection       | Crossover          | Mutation      | Elitism | Notes                         |
|-----------------------|-----------------|--------------------|---------------|---------|-------------------------------|
| `"baseline"`          | Tournament      | Smart BLX (α=0.3)  | Non-Uniform   | 2       | ✅ Recommended configuration  |
| `"rws"`               | Roulette Wheel  | Smart BLX (α=0.3)  | Non-Uniform   | 2       | Tests selection pressure      |
| `"uniform_crossover"` | Tournament      | Uniform            | Non-Uniform   | 2       | Tests crossover impact        |
| `"uniform_mutation"`  | Tournament      | Smart BLX (α=0.3)  | Uniform       | 2       | Tests mutation impact         |
| `"generational"`      | Tournament      | Smart BLX (α=0.3)  | Non-Uniform   | 0       | Pure generational replacement |

---

### `init_strategy` — Seeding the First Generation

How generation zero is populated before evolution begins.

| Value                   | Behavior                                                            | Best For             |
|-------------------------|---------------------------------------------------------------------|----------------------|
| `"Demand_Proportional"` | Seeds solutions proportional to each region's actual resource needs | ✅ Fair comparisons  |
| `"Urgency_Biased"`      | Prioritizes regions with higher disaster severity scores            | Urgency experiments  |
| `"Random"`              | Fully random initialization within bounds                           | Baseline comparison  |

---

### Hyperparameters

| Parameter          | Default | Type    | Effect                                                                         |
|--------------------|---------|---------|--------------------------------------------------------------------------------|
| `max_generations`  | `100`   | `int`   | Max iterations. Early stopping triggers if no improvement for 20 generations.  |
| `population_size`  | `100`   | `int`   | Number of allocation plans per generation.                                     |
| `crossover_prob`   | `0.9`   | `float` | Probability that two selected parents will produce offspring.                  |

---

## Running 30 Experiments

According to the project requirements, each configuration must be run **30 times independently**, with the seed used in every run stored and submitted alongside the results. This is non-negotiable — it's what makes the comparison statistically valid.

### To compare configurations, only change config_type and keep the rest fixed.
> *"The evolution should be carried out multiple times (optimally, 30 runs per setting). The list of seeds used to initialise the random number generator before each run should be stored & provided."*
> — Course project guidelines

The structure is: **30 runs × 5 `config_type` settings = 150 total runs**. Keep `init_strategy` and `population_size` fixed across all runs so the only variable between settings is the config itself.

Here's the full template with seed tracking built in:

```python
import numpy as np
import json
from algorithms.ga import DisasterReliefGA
from problem.scenarioM import get_scenario

scenario = get_scenario()

configs         = ["baseline", "rws", "uniform_crossover", "uniform_mutation", "generational"]
init_strategy   = "Demand_Proportional"   # keep fixed across all runs
population_size = 100                     # keep fixed across all runs
NUM_RUNS        = 30

# Generate 30 seeds once — same seeds reused across ALL configs for a fair comparison
seeds = [int(s) for s in np.random.randint(0, 100000, size=NUM_RUNS)]
print("Seeds:", seeds)   # save this output — it must be submitted

all_results = {}

for cfg in configs:
    config_scores = []

    for run_idx, seed in enumerate(seeds):
        np.random.seed(seed)

        optimizer = DisasterReliefGA(
            scenario_data=scenario,
            config_type=cfg,
            init_strategy=init_strategy,
            population_size=population_size
        )
        solution, score, history, _ = optimizer.run()
        config_scores.append(score)

        print(f"[{cfg}] run {run_idx+1:02d}/30 | seed={seed} | score={score:.4f}")

    all_results[cfg] = {
        "scores": config_scores,
        "mean":   float(np.mean(config_scores)),
        "std":    float(np.std(config_scores)),
        "best":   float(np.min(config_scores)),   # minimization — lower is better
        "seeds":  seeds
    }

# Save results + seeds to file for submission
with open("experiment_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print summary
print("\n--- Summary (lower = better) ---")
for cfg, res in all_results.items():
    print(f"{cfg:<25}  mean={res['mean']:.4f}  std={res['std']:.4f}  best={res['best']:.4f}")
```

**Why the same seeds across all configs?** Every configuration faces the exact same sequence of random starting points. Any score difference is then purely a product of the operator architecture — not luck.

**What to submit:** `experiment_results.json` contains both the scores and the seeds list. Include it with your report.

---

## Reading the Output

The default ablation study in `ga.py` prints a comparison table against the baseline:

```
Results (difference from baseline):

Baseline (Tournament, BLX, Non-Uniform, Elitism=2)    1250.4500 | Δ = +0.0000
Generational (Elitism=0)                              1320.1000 | Δ = -69.6500
Roulette Wheel Selection                              1290.8000 | Δ = -40.3500
Uniform Crossover                                     1310.2500 | Δ = -59.8000
Uniform Mutation                                      1285.5000 | Δ = -35.0500
```

> **This is a minimization problem.** Lower scores = better allocation plans.
> A *negative* Δ means that configuration performed worse than the baseline.

---

## How the Algorithm Works

```
Generation 0
   │
   ├─ Initialize population  (Demand_Proportional / Urgency_Biased / Random)
   │
   └─ For each generation:
         │
         ├─ Evaluate fitness   →  Fitness Sharing (σ=150) to preserve diversity
         │
         ├─ Select parents     →  Tournament (k=3)  or  Roulette Wheel
         │
         ├─ Crossover          →  Smart BLX-α (α=0.3)  or  Uniform
         │
         ├─ Mutate             →  Non-Uniform (σ shrinks over time)  or  Uniform
         │
         ├─ Repair population  →  Constraint repair runs after every generation
         │
         └─ Survivor selection →  Elitism=2 (baseline)  or  full replacement (generational)
```

*Early stopping:* halts automatically if fitness doesn't improve for 20 consecutive generations.

---

## Experimental Design Notes

*These notes explain why the experiments are structured the way they are — the reasoning behind each methodological choice.*

---

### 1. Decoupling Initialization for Fair Baseline Comparison

`init_strategy` is intentionally kept as a **separate, independent parameter** from `config_type`.

This isolation of variables guarantees that when comparing the **Proposed Method** against the **Baseline**, both start from the exact same initial population (`"Demand_Proportional"`). By giving every configuration an identical starting point, we eliminate any "head start" advantage that a smarter initialization might otherwise introduce.

> Any observed performance difference between configurations is strictly attributable to the architectural design of the operators themselves — *selection, crossover, mutation, and elitism* — not to how lucky generation zero happened to be.

---

### 2. Finalization and Solution Extraction Strategy

The two primary configurations extract their best solution in **deliberately different ways**, and this asymmetry is intentional.

| Configuration                             | Extraction Strategy                                        | Why                                                                                       |
|-------------------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Proposed Method** (`"baseline"`)        | `best_solution()` — best solution *across the entire run* | Elitism guarantees the all-time best is always preserved in the population                |
| **Generational Model** (`"generational"`) | Best solution from the **final generation only**          | Elitism=0 means strong solutions can be lost mid-run; final-gen extraction surfaces this weakness |

Restricting the `"generational"` config to its final generation is a deliberate choice to produce empirically honest results. A standard generational model may discover a strong solution early but lose it by generation 100 due to disruptive crossover or mutation. Evaluating only the last generation accurately captures this failure mode — and provides clear empirical evidence for the value of **Elitism** in the proposed configuration.

---

### 3. Dynamic Mutation Rate Selection

#### Theoretical Background

Course lecture guidelines specify that the recommended range for the mutation rate *p_m* is:

```
1 / population_size  ≤  p_m  ≤  1 / chromosome_length
```

The *upper bound* is preferred when the goal is **search space coverage to find one highly fit individual**, rather than maintaining a broadly fit population.

#### Our Decision

Since the sole objective is to find **one optimal allocation plan**, we selected the upper bound and implemented it dynamically:

```python
mutation_rate = 1.0 / chromosome_length
# i.e.: 1.0 / offspring.shape[1]
```

For the current 24-dimensional problem, this automatically computes to:

```
p_m = 1 / 24 ≈ 0.0417
```

This has two advantages over hardcoding a static decimal:

- **Academic alignment** — the rate is derived directly from the course-recommended formula, not arbitrary tuning.
- **Scalability** — if the scenario expands to more regions or resource types, the mutation rate recalculates itself automatically without any manual edits.

---

## Design Decisions

*These aren't arbitrary choices. Every operator was selected or modified for a specific reason — here's the thinking behind each one.*

---

### Non-Uniform Mutation — The Core Innovation

The lecture pseudocode gives a solid starting point: Gaussian mutation with a fixed `σ` and a static `mutation_rate = 0.1`. It works, but it has a fundamental tension built into it — the same mutation magnitude that's great for *exploring* early on becomes destructive later, when you're close to a good solution and don't want to be thrown across the search space.

To resolve this, we applied a **Deterministic Dynamic Parameter Control** strategy *([Eiben, Hinterding, & Michalewicz, 1999](https://www.researchgate.net/publication/220381158_Parameter_control_in_evolutionary_algorithms))*: instead of a fixed `σ`, we use one that **decays linearly with time**:

```python
decay_factor  = 1.0 - (current_generation / max_generations)
dynamic_sigma = initial_sigma * decay_factor   # starts at 50, approaches 0
```

The result is a mutation operator that *changes its personality* across the run — aggressive and wide-ranging early, surgical and precise toward the end. It bridges Gaussian probability with the non-uniform mutation principle from *(Michalewicz, 1992)*, achieving dynamic behavior without touching the chromosome representation or adding structural complexity.

We also tightened the per-gene bounds using problem-specific constraints:

```python
Ui = min(capacity[region], budget[resource])   # can't exceed either ceiling
Li = minimums[region][resource]                # must meet minimum requirement
```

This keeps every mutation within the feasible space from the start — reducing the repair burden downstream.

**On Uniform Mutation as the comparison baseline:** `uniform_mutate` is kept as a direct foil. Its step size never changes — pure exploration from generation 1 to 100. Running both back-to-back makes the cost of ignoring *exploitation* clearly visible in the scores.

---

### Elitism — Why We Can't Afford to Lose Good Solutions

In most optimization problems, losing a good solution for one generation is annoying. In a *constrained* disaster relief scenario it's worse — finding a valid, high-quality allocation plan is genuinely hard, and the repair step doesn't guarantee the repaired solution is as good as what you started with.

**Elitism** (`keep_elitism=2`) is the insurance policy. It guarantees:

1. **Feasibility is preserved** — once a valid solution is found, it survives.
2. **Monotonic improvement** — the best score can only stay the same or get better, never backslide.

The `"generational"` config (elitism=0) exists specifically to demonstrate what happens without this protection. A strong solution found at generation 40 can get destroyed by a disruptive crossover at generation 41 and never recover. That's the failure mode we're deliberately exposing.

---

### Tournament Selection — Backed by the Literature

We chose **Tournament Selection** (k=3) over **Roulette Wheel Selection** as the primary method based on a comparative study by *[Zhong et al. (2005)](https://www.researchgate.net/publication/221216912_Comparison_of_Performance_between_Different_Selection_Strategies_on_Simple_Genetic_Algorithms)*, which found that Tournament consistently outperforms RWS in two key areas:

- **Convergence speed** — Tournament is computationally leaner; it doesn't need to compute the full fitness distribution at every generation.
- **Diversity preservation** — the tournament structure naturally resists the winner-take-all pressure that makes RWS collapse toward premature convergence in complex, constrained landscapes.

The `"rws"` config lets you verify this empirically. Run both, compare the Δ.

---

### Smart BLX-α with α=0.3

Standard **BLX-α** samples offspring from an interval *extended* beyond the parents by a factor of α. With a high α this is great for exploration — but in a problem where every gene must satisfy capacity and budget constraints, it generates values that immediately need repairing, which is expensive and can erode solution quality.

Setting α=0.3 restricts exploration to 30% beyond the `[min, max]` parent range. Enough to search meaningfully. Not so much that the repair step becomes the most active part of the algorithm.

The **BLX operator is intentionally left unconstrained** — it generates pure mathematical interpolations without checking physical limits. This is a deliberate *separation of concerns*: BLX handles exploration, `repair()` handles feasibility. Forcing BLX to check constraints mid-operation would contradict the mathematical intent of the γ equation and create operator coupling that hurts both.

---

### The Zero-Bound Problem — Why `Li` Matters

An easy mistake in resource allocation GAs is assuming a generic lower bound of `Li = 0` for every gene. In reality, every region has a strictly positive *minimum survival threshold* — a floor below which any allocation is physically meaningless.

When operators generate values starting from zero, they chronically produce infeasible allocations, forcing the repair function into exhaustive scaling loops on nearly every solution. By aligning the lower bound of both mutation and initialization directly with `scenario_data["minimums"]`:

```python
Li = minimums[region][resource]
```

the operators generate values that are already within the feasible range from the start — dramatically reducing the repair function's workload and eliminating an entire class of algorithmic friction.

---

### Hard Boundary Clipping Inside Mutation

Unlike **BLX-α**, which is bounded by the distance between parents, Gaussian mutation is *mathematically unbounded* — it can generate extreme outliers in either direction (very large positives or heavy negatives), especially early in the run when `dynamic_sigma` is at its maximum.

Allowing these extreme values to reach the repair function would corrupt the global budget calculation before the repair even has a chance to act. To prevent this, a strict **localized clipping** step is applied inside the mutation operator itself, immediately after each gene is mutated:

```python
gene ← clip(gene + mutation_size, Li, Ui)
```

This truncates mathematically absurd values at the gene level — before they can propagate into the broader solution and destabilize the repair phase.

---

### Why Repair Runs Every Generation

Crossover and mutation will inevitably produce constraint violations — that's not a bug, it's expected. The question is *when* you fix them.

Repairing only at the end means invalid solutions participate in selection and become parents. Their offspring inherit the violation patterns. By the time you repair the final generation, you've potentially evolved a lineage of structurally broken solutions.

Running `repair_population` inside `on_generation_complete` — after every single generation — ensures the parent pool is always clean. Every solution that gets selected, crossed, and mutated is a valid one. The evolutionary pressure stays focused on *quality*, not on accidentally inheriting feasibility bugs.

---

## Full Algorithm Pseudocode

*Each component of the system is defined as a standalone algorithm. Algorithm 7 is the master loop — it calls all others in sequence.*

---

<details>
<summary><strong>Algorithm 1 — Population Initialization</strong></summary>
<br>

```
INPUT  : N, scenario S, init_strategy ∈ {Demand_Proportional, Urgency_Biased, Random}
OUTPUT : initial population P

1:  if init_strategy = Demand_Proportional then
2:      P ← InitialiseDemandProportional(N, S)
3:  else if init_strategy = Urgency_Biased then
4:      P ← InitialiseUrgencyBiased(N, S)
5:  else
6:      P ← InitialiseRandom(N, S)
7:  end if
8:  return P
```

</details>

---

<details>
<summary><strong>Algorithm 2 — Fitness Evaluation with Fitness Sharing</strong></summary>
<br>

```
INPUT  : solution x, population P, scenario S, σ_share = 150, α = 1
OUTPUT : shared fitness f_shared(x)

 1:  raw_score ← ComputeFitness(x, S)
 2:  if P ≠ ∅ then
 3:      niche_count ← 0
 4:      for each x' ∈ P do
 5:          d ← ‖x − x'‖₂
 6:          if d ≤ σ_share then
 7:              sh(d) ← 1 − (d / σ_share)^α
 8:              niche_count ← niche_count + sh(d)
 9:          end if
10:      end for
11:      f_shared(x) ← raw_score / niche_count
12:  else
13:      f_shared(x) ← raw_score
14:  end if
15:  return f_shared(x)
```

</details>

---

<details>
<summary><strong>Algorithm 3 — BLX-α Crossover</strong></summary>
<br>

```
INPUT  : parents P, offspring size n, α = 0.3, crossover probability p_c
OUTPUT : offspring set C

 1:  C ← ∅
 2:  idx ← 0
 3:  while |C| < n do
 4:      p1 ← P[idx mod |P|]
 5:      p2 ← P[(idx+1) mod |P|]
 6:      if rand() < p_c then
 7:          for i = 1 to dim(p1) do                 ▷ gene-wise blend
 8:              min_i ← min(p1[i], p2[i])
 9:              max_i ← max(p1[i], p2[i])
10:              u  ∼ U(0, 1)
11:              γ  ← (1 + 2α) · u − α
12:              child[i] ← (1 − γ) · min_i + γ · max_i
13:          end for
14:          C ← C ∪ {child}
15:      else
16:          C ← C ∪ {p1}                            ▷ copy parent without crossover
17:      end if
18:      idx ← idx + 2
19:  end while
20:  return C
```

</details>

---

<details>
<summary><strong>Algorithm 4 — Non-Uniform Gaussian Mutation</strong></summary>
<br>

```
INPUT  : offspring C, current generation t, max generations T, σ₀ = 50, scenario S
OUTPUT : mutated offspring C

 1:  p_m ← 1 / L                                     ▷ L = chromosome length
 2:  σ_t ← σ₀ · (1 − t / T)                         ▷ step size decays over generations
 3:  for i = 1 to |C| do
 4:      for j = 1 to L do
 5:          r_k   ← ⌊j / 3⌋                         ▷ region index
 6:          r_res ← j mod 3                          ▷ resource index
 7:          U_j ← min(capacity[r_k], budget[r_res])
 8:          L_j ← minimum[r_k][r_res]
 9:          if rand() < p_m then
10:              δ ∼ N(0, σ_t)
11:              C[i, j] ← clip(C[i, j] + δ, L_j, U_j)
12:          end if
13:      end for
14:  end for
15:  return C
```

</details>

---

<details>
<summary><strong>Algorithm 5 — Uniform Mutation</strong></summary>
<br>

```
INPUT  : offspring C, scenario S
OUTPUT : mutated offspring C

 1:  p_m ← 1 / L
 2:  for i = 1 to |C| do
 3:      for j = 1 to L do
 4:          r_k   ← ⌊j / 3⌋
 5:          r_res ← j mod 3
 6:          U_j ← min(capacity[r_k], budget[r_res])
 7:          L_j ← minimum[r_k][r_res]
 8:          if rand() < p_m then
 9:              C[i, j] ∼ U(L_j, U_j)
10:          end if
11:      end for
12:  end for
13:  return C
```

</details>

---

<details>
<summary><strong>Algorithm 6 — Constraint Repair</strong></summary>
<br>

```
INPUT  : population P, scenario S
OUTPUT : feasible population P

1:  for i = 1 to |P| do
2:      P[i] ← Repair(P[i], S)              ▷ fix constraint violations gene-wise
3:      P[i] ← Flatten(P[i], order = 'F')   ▷ column-major flattening
4:  end for
5:  return P
```

</details>

---

<details>
<summary><strong>Algorithm 7 — Master GA Loop</strong></summary>
<br>

```
INPUT  : scenario S, config_type, init_strategy, N = 100, T = 100, p_c = 0.9
OUTPUT : best solution x*, fitness f*, convergence history H, final population P

 1:  selection ← Tournament
 2:  K_tourn  ← 3                                       ▷ default baseline config
 3:  crossover ← BLX-α
 4:  mutation  ← Non-Uniform Gaussian
 5:  k_e ← 2
 6:  if config_type = rws then
 7:      selection ← RWS
 8:      K_tourn  ← None
 9:  end if
10:  if config_type = uniform_crossover then
11:      crossover ← Uniform
12:  end if
13:  if config_type = uniform_mutation then
14:      mutation ← Uniform
15:  end if
16:  if config_type = generational then
17:      k_e ← 0
18:  end if
19:
20:  P ← Algorithm 1(N, S, init_strategy)
21:  for each x ∈ P do
22:      f(x) ← Algorithm 2(x, P, S)
23:  end for
24:  P ← Algorithm 6(P, S)
25:
26:  t ← 0
27:  stagnation ← 0
28:  H ← []
29:  while t < T and stagnation < 20 do
30:      E       ← top-k_e individuals from P                  ▷ elitism
31:      Parents ← Selection(P, N/2, selection, K_tourn)
32:      C       ← Crossover(Parents, N − k_e, p_c)            ▷ Algorithm 3 or Uniform
33:      C       ← Mutation(C, t, T)                           ▷ Algorithm 4 or Algorithm 5
34:      C       ← Algorithm 6(C, S)                           ▷ repair after mutation
35:      for each x ∈ C do
36:          f(x) ← Algorithm 2(x, C ∪ P, S)
37:      end for
38:      P ← E ∪ C
39:
40:      if max f(P) = max f(P_prev) then
41:          stagnation ← stagnation + 1
42:      else
43:          stagnation ← 0
44:      end if
45:      H ← H ∪ {max f(P)}
46:      t ← t + 1
47:  end while
48:
49:  if config_type = generational then
50:      x* ← P[argmax f(P)]                                   ▷ last generation only
51:  else
52:      x* ← BestSolution(P)                                  ▷ overall best across all generations
53:  end if
54:
55:  x* ← Repair(x*, S)
56:  f* ← ComputeFitness(x*, S)
57:  return x*, f*, H, P
```

</details>

---

## Dependencies

| Package  | Purpose                      |
|----------|------------------------------|
| `numpy`  | Array operations, math       |
| `pygad`  | Genetic algorithm framework  |

---

### Member 4 (Island Model)
#### Hybrid GA+PSO Optimization

Two hybrid algorithms combining Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) for disaster relief resource allocation.

---

#### 1- hybrid_SIM_.py — Static Island Model (GA + PSO Hybrid)

#### Overview

This module implements a **Static Island Hybrid** optimizer that runs a Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) in parallel on two separate "islands." Both algorithms work on the same disaster relief resource allocation problem, occasionally sharing their best solutions with each other to improve convergence.

The **island model** prevents either algorithm from getting stuck in a local optimum by cross-pollinating good solutions every 10 generations — without ever merging or restructuring the two populations.

---

#### Pipeline

1. **Initialize** — Create two separate populations of 50 individuals each (one for GA, one for PSO)
2. **Evolve in parallel** — At every generation, run one GA step and one PSO step independently
3. **Exchange** — Every 10 generations, the best solution from GA is injected into PSO (replacing its worst individual), and vice versa — but only if it actually improves the receiving population
4. **Repeat** steps 2–3 until 100 generations are complete
5. **Return** the best solution found across both islands

> **Key difference from other hybrid approaches:** the two islands never merge or restructure — they stay fixed throughout, just sharing good solutions occasionally.

---

#### Configuration Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `TOTAL_GENERATIONS` | `100` | Total number of iterations both algorithms run |
| `EXCHANGE_INTERVAL` | `10` | Frequency (in generations) at which GA and PSO swap their best solutions |
| `ISLAND_SIZE` | `50` | Population size for each algorithm |
| `GA_CONFIG` | `"config1"` | GA hyperparameter preset to use |
| `INIT_STRATEGY` | `"Demand_Proportional"` | Initializes solutions weighted by demand — higher-demand nodes receive more resources by default |

---

#### Class: `StaticIslandHybrid`

The main class that orchestrates the entire hybrid run.

---

##### `__init__(self, scenario)`

Sets up both islands from scratch.

- Receives a `scenario` dict containing problem data (dimensions, demands, constraints, etc.)
- Creates two separate populations of 50 individuals using the demand-proportional strategy — one for GA, one for PSO
- Instantiates both a `DisasterReliefGA` and a `PSO` object and assigns them their respective starting populations
- Initializes PSO-specific state: velocities (random in `[-1, 1]`), personal bests (`pso_pbest`), and the island's best known position
- Runs initial best-tracking for GA, PSO, and the global best across both islands

---

##### `_get_fitnesses(self, population)`

Evaluates every individual in a population.

- Takes a 2D array where each row is one solution
- Returns a 1D array of fitness scores (lower = better — this is a minimization problem)
- Calls `compute_fitness()` from the problem module for each individual

---

##### `_update_ga_best(self)`

Finds and stores the best solution currently in the GA island.

- Scores every individual in `ga_population`
- Updates `ga_best_score` and `ga_best_solution` if a better one is found

---

##### `_update_pso_best(self)`

Same as above, but for the PSO island.

- Finds the best-scoring particle in `pso_population`
- Updates `pso_best_score` and `pso_best_solution`

---

##### `_update_global_best(self)`

Maintains the single best solution found across both islands combined.

- Compares GA's best and PSO's best against the current global best
- If either island found something better, the global record is updated

---

##### `ga_step(self)`

Runs one generation of the Genetic Algorithm.

1. **Selection** — Tournament selection (size 5): 50 random 5-way tournaments are held; the winner of each becomes a parent
2. **Crossover** — Uses the GA object's crossover method to produce 50 offspring
3. **Mutation** — Applies mutation to the offspring
4. **Elitism** — The 4 best solutions from the previous generation are preserved directly (prevents regression)
5. Updates `ga_population`, increments `ga_generation`, and refreshes the GA best

---

##### `pso_step(self)`

Runs one iteration of Particle Swarm Optimization with **adaptive parameters** that shift over time:

| Parameter | Start | End | Effect |
|---|---|---|---|
| `w` (inertia) | `0.95` | `0.25` | Particles explore broadly early, exploit finely later |
| `c1` (cognitive weight) | `2.0` | `1.5` | Trust in personal best gradually decreases |
| `c2` (social weight) | `0.5` | `2.0` | Trust in swarm best gradually increases |

For each particle each iteration:
1. Updates its **personal best** if the current position is better than its history
2. Updates the **island best** if it beats the current swarm record
3. Computes a new **velocity**: `v = w·v + c1·r1·(personal_best − position) + c2·r2·(island_best − position)`
4. Clamps velocity to a decaying maximum (wider search early, finer search late)
5. Moves the particle and **repairs** it into a valid solution

---

##### `exchange_solutions(self)`

The bridge between the two islands — called every 10 generations.

- Finds the **worst individual** in each population
- Replaces GA's worst with PSO's best — only if PSO's best actually improves it
- Replaces PSO's worst with GA's best — only if GA's best actually improves it
- Updates the PSO personal best record for the replaced particle if the new solution is an improvement
- Syncs the updated populations back into both algorithm objects

> This is what makes the system a *hybrid* rather than two independent algorithms running in parallel.

---

##### `run(self)`

The main execution loop.


---

#### 2- `hybrid_DIM_SP_.py` — Dynamic Island Model

1. **Initialize** — Create one population of 50 individuals using demand-proportional seeding
2. **Evolve** — Run PSO on the population for 20 generations
3. **Re-cluster** — Merge all individuals, apply spectral clustering to split them into up to 5 groups based on similarity
4. **Assign operators** — Large clusters get PSO, small clusters get GA
5. **Evolve each island** — Each cluster evolves independently for another 20 generations
6. **Repeat steps 3–5** until 100 generations are done
7. **Return** the best solution found across all islands

> If an island stops improving for 3 epochs, it automatically injects random perturbations to escape local optima.


#### Overview

This file implements **DIM-SP** (Dynamic Island Model with Spectral Partitioning), a hybrid metaheuristic optimization algorithm designed for **disaster relief supply chain problems**. It combines two classical optimization algorithms — **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** — inside a dynamic island model framework. Islands (sub-populations) are discovered and reorganized at regular intervals using **Spectral Clustering**, which groups individuals by similarity in the solution space. Large islands evolve using PSO; small islands evolve using GA. This adaptive strategy allows the algorithm to escape local optima and balance exploration with exploitation throughout the optimization process.

---

#### High-Level Architecture

```
Initial Population
       ↓
  Single Island (PSO)
       ↓
  [Every EPOCH_INTERVAL generations]
       ↓
  Merge all islands → Re-cluster using Spectral Clustering
       ↓
  Assign PSO to large islands, GA to small islands
       ↓
  Each island evolves independently for EPOCH_INTERVAL steps
       ↓
  Track global best solution across all islands
       ↓
  Repeat until TOTAL_GENERATIONS reached
```

---

#### Global Constants (Hyperparameters)

These constants control the overall behavior of the algorithm. They are defined at the top of the file and can be tuned to balance speed, diversity, and solution quality.

| Constant | Default Value | Role |
|---|---|---|
| `TOTAL_GENERATIONS` | 100 | Total number of evolution steps the algorithm runs |
| `EPOCH_INTERVAL` | 20 | How often (in generations) the population is re-clustered into islands |
| `MAX_ISLANDS` | 5 | Maximum number of islands (sub-populations) allowed at any time |
| `ISLAND_SIZE` | 50 | Total population size at initialization |
| `SIGMA` | 1.0 | Controls the width of the Gaussian kernel used in spectral clustering — higher values make more individuals appear similar |
| `MIN_CLUSTER_SIZE` | 10 | Minimum number of individuals an island must contain; smaller clusters are padded |
| `STAGNATION_THRESHOLD` | 3 | Number of consecutive generations without improvement before diversity injection is triggered |
| `DIVERSITY_PERTURBATION` | 0.2 | Standard deviation of the Gaussian noise injected during diversity recovery |

---

#### Functions

##### `build_similarity_matrix(population, sigma=SIGMA)`

**Purpose:** Constructs a symmetric similarity (affinity) matrix between every pair of individuals in the population. This matrix is the foundation of spectral clustering.

**How it works:** For each pair of individuals `i` and `j`, it computes the squared Euclidean distance between their solution vectors, then maps it through a **Gaussian (RBF) kernel**:

```
W[i,j] = exp( -||x_i - x_j||² / (2 * sigma² * dim) )
```

- A value close to **1** means the two individuals are very similar (close in solution space).
- A value close to **0** means they are very different.
- Diagonal entries are set to 1 (each individual is identical to itself).
- The result is a symmetric matrix of shape `(N, N)`.

**Parameters:**
- `population` — NumPy array of shape `(N, dim)` representing all individuals' solution vectors.
- `sigma` — Bandwidth of the Gaussian kernel. Larger values make more distant solutions appear similar, leading to fewer, bigger clusters. Smaller values make clustering more sensitive to fine differences.

---

##### `spectral_cluster(population, fitnesses, max_k, sigma, min_size, scenario)`

**Purpose:** Partitions the merged population into a dynamic number of sub-populations (islands) using **spectral clustering**. The number of clusters is chosen automatically based on the eigenvalue gap heuristic.

**How it works:**
1. Builds the similarity matrix `W` using `build_similarity_matrix`.
2. Computes the **normalized graph Laplacian** `L = I - D^(-1/2) * W * D^(-1/2)`, where `D` is the degree matrix (diagonal matrix of row sums of `W`). This captures the connectivity structure of the population.
3. Performs **eigen-decomposition** of `L` and selects the number of clusters `k` by finding the largest gap between consecutive small eigenvalues (the "eigengap heuristic"). A large gap means a natural cluster boundary exists.
4. Projects individuals into the `k`-dimensional eigenvector embedding space and normalizes each row to unit length.
5. Runs **k-means clustering** (`_kmeans`) on the embedding to assign cluster labels.
6. If any resulting cluster is smaller than `min_size`, it is padded by adding perturbed copies of the global best individual (repaired to be feasible).

**Parameters:**
- `population` — Array of all individuals from all current islands, merged together.
- `fitnesses` — Array of fitness scores corresponding to each individual (used to identify the global best for padding).
- `max_k` — Maximum number of clusters allowed. The algorithm selects the best `k ≤ max_k`.
- `sigma` — Gaussian kernel bandwidth (same as in `build_similarity_matrix`).
- `min_size` — Minimum cluster size; clusters below this are padded.
- `scenario` — Problem scenario dictionary, required for the `repair` function when padding small clusters.

**Returns:** A list of NumPy arrays, each representing one island's population.

---

##### `_kmeans(X, k, max_iters=50)`

**Purpose:** Applies the **k-means++ clustering algorithm** on the spectral embedding space to assign each individual to one of `k` clusters.

**How it works:**
1. Initializes the first centroid randomly and subsequent centroids with **k-means++ seeding** — each new centroid is chosen with probability proportional to its distance from the nearest existing centroid. This improves convergence stability compared to random initialization.
2. Alternates between assigning each point to its nearest centroid and recomputing centroids as the mean of their assigned points.
3. Stops early if assignments do not change between iterations.

**Parameters:**
- `X` — NumPy array of shape `(N, k)`, the normalized spectral embedding of all individuals.
- `k` — Number of clusters to form.
- `max_iters` — Maximum number of assignment-update cycles (default: 50).

**Returns:** Integer array of length `N` with cluster labels for each individual.

---

#### Classes

##### `Island`

**Purpose:** Represents a single sub-population (island) within the DIM-SP framework. Each island maintains its own population, best solution tracking, stagnation counter, and evolves independently using either GA or PSO.

**Constructor Parameters:**
- `population` — Initial array of individuals for this island.
- `scenario` — Problem scenario dictionary (vehicle counts, demands, distances, etc.).
- `operator` — Either `"GA"` or `"PSO"` — determines which evolution strategy the island uses.
- `island_id` — Integer ID for identification purposes.

**Internal State:**
- `best_solution` / `best_score` — The best individual and its score found so far on this island.
- `best_history` — List tracking the best score at each generation for convergence analysis.
- `stagnation_counter` — Counts how many consecutive generations the best score has not improved.
- `_vel` — PSO velocity matrix (only meaningful when `operator == "PSO"`).
- `_pbest_x` / `_pbest_f` — PSO personal best positions and fitnesses.
- `_gbest_x` / `_gbest_f` — PSO global best position and fitness (island-local).
- `_inertia` — `LinearInertia` object from the PSO module that linearly decreases the inertia weight from 0.9 to 0.4 over the run.

###### `Island._eval_all()`

**Purpose:** Evaluates the fitness of every individual in the current island population.

**How it works:** Calls `compute_fitness(ind, scenario)` for each individual and returns a NumPy array of fitness scores. Lower scores are better (minimization problem).

---

###### `Island._refresh_best()`

**Purpose:** After each generation step, checks whether any individual has beaten the current island best and updates tracking state accordingly.

**How it works:** Evaluates all individuals, finds the minimum fitness, and compares it to `self.best_score`. If improved, resets `stagnation_counter` to 0 and records the new best solution. Otherwise, increments `stagnation_counter`. Appends the current best to `best_history`.

---

###### `Island._inject_diversity()`

**Purpose:** Prevents premature convergence by introducing new genetic material when the island has been stagnant for too long.

**How it works:** Triggered when `stagnation_counter >= STAGNATION_THRESHOLD`. Identifies the worst 30% of individuals by fitness, replaces each with a **repaired, noise-perturbed copy of the current best solution**:

```
new_individual = repair(best_solution + N(0, DIVERSITY_PERTURBATION))
```

For PSO islands, the velocities of replaced individuals are also randomized in `[-0.5, 0.5]` to prevent them from immediately converging back. Resets `stagnation_counter` to 0.

---

###### `Island._ga_step(n_gens)`

**Purpose:** Runs `n_gens` generations of GA evolution on the island population.

**How it works:** For each generation:
1. Evaluates all individuals.
2. Selects parents using **tournament selection** (`_tourn_select`).
3. Creates a lightweight `_FakeGA` context (see below) to reuse the `DisasterReliefGA` crossover and mutation operators without running a full standalone GA.
4. Applies crossover and mutation to generate offspring.
5. Enforces **elitism** by placing the current best individual into position 0 of the offspring (so the best is never lost).
6. Replaces the population with the offspring.
7. Calls `_refresh_best` and `_inject_diversity`.

---

###### `Island._tourn_select(scores, k=3)`

**Purpose:** Implements **tournament selection** for parent selection in GA evolution.

**How it works:** For each of `n` parents needed, randomly samples `k` individuals from the population and selects the one with the lowest fitness score. This provides selection pressure while maintaining diversity (unlike pure rank selection).

**Parameters:**
- `scores` — Fitness array for the current population.
- `k` — Tournament size (default: 3). Larger `k` = stronger selection pressure.

---

###### `Island._pso_step(n_steps)`

**Purpose:** Runs `n_steps` iterations of **Particle Swarm Optimization** on the island population.

**How it works:** For each time step:
1. Computes the inertia weight `w` using `LinearInertia` (decreases linearly from 0.9 → 0.4).
2. For each particle `i`, updates its velocity using the standard PSO velocity equation:

```
v_i = w * v_i
    + c1 * r1 * (pbest_i - x_i)    ← cognitive component
    + c2 * r2 * (gbest   - x_i)    ← social component
```

where `c1 = c2 = 1.5` (cognitive and social acceleration coefficients), and `r1`, `r2` are random vectors in `[0, 1)`.

3. Updates the particle's position and **repairs** it to be feasible.
4. Updates personal bests (`_pbest_x`, `_pbest_f`) and island-level global best (`_gbest_x`, `_gbest_f`).
5. Calls `_refresh_best` and `_inject_diversity`.

**Why c1 = c2 = 1.5?** These values are a balanced choice that gives equal weight to individual memory and social influence. The standard literature recommends values around 2.0, but slightly lower values (1.4–1.6) tend to work better for constrained, high-dimensional problems.

---

###### `Island.evolve(n_steps)`

**Purpose:** Public interface for running evolution on the island. Dispatches to either `_pso_step` or `_ga_step` based on the island's assigned `operator`.

---

##### `_FakeGA`

**Purpose:** A minimal shim that mimics enough of the `DisasterReliefGA` interface to allow `Island._ga_step()` to call the crossover and mutation operators from the GA module without instantiating or running a full GA pipeline.

**Why it exists:** The `DisasterReliefGA.crossover()` and `DisasterReliefGA.mutate()` methods internally reference `self.population` and `self.generations_completed` on a GA object. Rather than duplicating that logic, `_FakeGA` supplies just those two attributes so the island can borrow the GA's operators cleanly.

**Fields:**
- `population` — The island's current population array.
- `generations_completed` — The current generation count, used by the GA's adaptive mutation rate logic.

---

##### `DIMSPHybrid`

**Purpose:** The top-level orchestrator. Manages the full lifecycle of the DIM-SP algorithm: initialization, epoch-based re-clustering, per-island evolution, and global best tracking.

**Constructor Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `scenario` | required | Problem scenario dictionary from `get_scenario()` |
| `total_generations` | 100 | Total number of generations to run |
| `epoch_interval` | 20 | Generations per epoch (between re-clusterings) |
| `island_size` | 50 | Initial population size |
| `max_islands` | 5 | Maximum number of islands after clustering |
| `init_strategy` | `"Demand_Proportional"` | Population initialization method |
| `verbose` | False | Reserved for optional logging output |

**Internal State:**
- `islands` — List of active `Island` objects.
- `convergence` — List recording the global best score at the end of each epoch.
- `island_count_hist` — List recording how many islands were active at each epoch.
- `best_solution` / `best_score` — The globally best solution found across all islands and all epochs.

###### `DIMSPHybrid._run_epoch(current_gen)`

**Purpose:** The re-clustering step. Called at the start of every epoch after the first. Merges all island populations into one pool, re-runs spectral clustering to discover natural groupings, and rebuilds islands with appropriate operators.

**How it works:**
1. Stacks all island populations into one array `Pa`.
2. Evaluates fitness for every individual.
3. Calls `spectral_cluster` to partition `Pa` into new sub-populations.
4. Computes the **median cluster size** across all new clusters.
5. Assigns `operator = "PSO"` to islands at or above the median size (they have more particles and benefit from PSO's swarm dynamics), and `operator = "GA"` to smaller islands (better for fine-tuned local search with crossover/mutation).
6. Replaces `self.islands` with the newly created `Island` objects.

**Why median?** Using the median as a dynamic threshold is more robust than a fixed cutoff — it adapts to how the population naturally partitions at each epoch, so the PSO/GA split is always roughly 50/50 by count.

---

###### `DIMSPHybrid._update_global_best()`

**Purpose:** Scans all active islands and updates the global best solution and score if any island has found a better solution than what has been seen so far.

---

###### `DIMSPHybrid.run()`

**Purpose:** Main execution loop. Runs the full DIM-SP optimization.

**How it works:**
1. Divides `total_generations` into epochs of `epoch_interval` steps.
2. For each epoch:
   - If not the first epoch, calls `_run_epoch` to re-cluster.
   - Evolves each island for `epoch_interval` steps.
   - Updates the global best.
   - Records convergence and island count history.
3. After the loop, repairs and re-evaluates the best solution to ensure feasibility.

**Returns:** `(best_solution, best_score, metadata)` where `metadata` contains `"hybrid_convergence"` (score per epoch) and `"island_count"` (islands per epoch).

---

#### Parameters Borrowed from GA and PSO Modules

##### From `DisasterReliefGA` (GA module)

| Parameter / Method | Where Used | Why |
|---|---|---|
| `crossover(parents, shape, ga_context)` | `Island._ga_step()` | Reuses the GA's crossover operator directly, avoiding code duplication. The crossover logic handles problem-specific chromosome structure. |
| `mutate(offspring, ga_context)` | `Island._ga_step()` | Reuses the GA's mutation operator. The `_FakeGA` object passes `generations_completed` so that the GA's adaptive mutation rate (which typically decreases over time) remains meaningful inside the island. |
| `config_type = "config1"` | `Island.__init__()` | Selects the standard operator configuration from the GA (e.g., specific crossover and mutation rates). `"config1"` is the default well-tuned setup. |
| `init_strategy = "Demand_Proportional"` | `Island.__init__()` | Required to instantiate `DisasterReliefGA` (even though the island does not run the full GA pipeline); the GA helper is used only for its operators. |
| Tournament selection (`k=3`) | `Island._tourn_select()` | Implemented locally but mirrors the GA module's tournament selection logic to maintain consistency. |

##### From `PSO` (PSO module)

| Parameter / Method | Where Used | Why |
|---|---|---|
| `LinearInertia(0.9, 0.4)` | `Island.__init__()` | Linearly decays the PSO inertia weight `w` from 0.9 (more exploration early on) to 0.4 (more exploitation later). This schedule is imported directly from the PSO module so behavior is consistent with the standalone PSO. |
| `c1 = 1.5`, `c2 = 1.5` | `Island._pso_step()` | Cognitive and social acceleration coefficients. Standard PSO uses 2.0; the slightly lower value of 1.5 is used here to reduce velocity instability in the constrained, high-dimensional disaster relief solution space. |
| Velocity clamping (implicit via `repair`) | `Island._pso_step()` | After each velocity update, positions are passed through `repair()` to enforce feasibility constraints. This effectively acts as position clamping — a standard PSO stabilization technique. |
| `ring=True` topology (conceptually) | `Island` model | The island model itself acts as a **ring topology** — each island is a local neighborhood that shares information only at re-clustering epochs, preventing premature global convergence. |

##### From `problem.FitnessFinal`

| Import | Role |
|---|---|
| `compute_fitness` | Evaluates a solution vector and returns its scalar fitness score. Used in `_eval_all()` across all islands. |
| `initialise_demand_proportional` | Default initialization strategy. Allocates supply resources proportional to each demand node's needs, producing feasible starting solutions biased toward realistic relief distributions. |
| `initialise_urgency_biased` | Alternative initialization. Prioritizes nodes with highest urgency scores, useful when time-criticality is paramount. |
| `initialise_random` | Baseline random initialization. Generates uniformly random feasible solutions, offering maximum diversity at the cost of initial solution quality. |

### From `problem.constraint`

| Import | Role |
|---|---|
| `repair(individual, scenario)` | Enforces feasibility on any solution vector. Called after PSO velocity updates, after diversity injection, and at the very end of the `run()` method to guarantee the returned solution is constraint-compliant. |

---

#### `__main__` Block — Benchmark Comparison

When run directly, the script performs a controlled comparison between:

1. **Standalone GA** — `DisasterReliefGA` with `config1`, demand-proportional initialization, 100 generations, 50 individuals.
2. **Standalone PSO** — `PSO` with ring topology, 4 neighbors, demand-proportional initialization, 50 particles, 100 iterations.
3. **DIM-SP Hybrid** — The hybrid model with the same total budget (100 generations, 50 individuals).

The improvement percentage is computed relative to the **best** of the two baselines:

```
improvement = (best_baseline - hybrid_score) / best_baseline * 100
```

A positive value means the hybrid found a better solution than either standalone algorithm. The output also distinguishes between cases where the hybrid beats both algorithms versus only the weaker one.

---

#### Initialization Strategies

The `init_strategy` parameter in `DIMSPHybrid` controls how the initial population is generated. All three strategies produce **feasible** solutions (i.e., they satisfy problem constraints before evolution begins):

| Strategy | Description | Best used when |
|---|---|---|
| `"Demand_Proportional"` | Allocates resources to supply nodes in proportion to demand at each node | General use; good balance of feasibility and realism |
| `"Urgency_Biased"` | Biases allocation toward nodes with highest urgency weights | Time-critical scenarios where some nodes are far more urgent |
| `"Random"` (any other string) | Uniformly random feasible initialization | Benchmarking or maximum diversity experiments |

---

#### Convergence Behavior

- **Early epochs:** One large island evolves with PSO, exploring the solution space broadly.
- **After first re-clustering:** Population splits into specialized islands. Large islands (PSO) continue broad exploration; small islands (GA) exploit promising regions more precisely via crossover.
- **Stagnation injection:** If any island stagnates for 3+ generations, the bottom 30% of its population is replaced with perturbed copies of its local best, keeping the search active.
- **Re-clustering every 20 generations:** Ensures islands reflect the current structure of the solution space rather than becoming isolated silos.

---

#### Dependencies

| Module | Source | Role |
|---|---|---|
| `numpy` | External | All numerical operations |
| `problem.scenarioM.get_scenario` | Local | Loads the disaster relief problem instance |
| `problem.FitnessFinal` | Local | Fitness evaluation and initialization strategies |
| `problem.constraint.repair` | Local | Feasibility enforcement |
| `algorithms.ga.DisasterReliefGA` | Local | GA crossover/mutation operators |
| `algorithms.pso.PSO` | Local | Standalone PSO (used in benchmark comparison) |
| `algorithms.pso.LinearInertia` | Local | Inertia weight schedule for island PSO |

---


### Member 5 — Experiments and Analysis
**Files:** `experiments/run_experiments.py` 

**Your job is to run all the algorithms, collect results, and produce all the plots.**
NOTE: you can write everything in one file or add a sepreate file for plots, where you save the results from expirments in csv files then load them into plots.py for plotting and analysis. whichever you think is neater and easier for you.
**Tasks:**
- Run every algorithm configuration 30 times each using seeds 0 through 29.
- Record the best fitness and convergence speed from each run.
- Generate these plots: convergence curves, box plots, allocation heatmap, allocation bar chart.

**Tips:**
- The 30 runs exist because one run proves nothing — you need the average behaviour across many random starts.
- Convergence speed = the iteration at which the algorithm first got close to its final best result.
- The box plots are your most important plot — they show which algorithm is most reliable across 30 runs.
- In your conclusions, answer: which algorithm performed best? does the hybrid beat PSO and GA alone? does smarter initialisation help?

> **You are completely free to structure and implement this however you think is best. The above is just a starting point.**

---

### Member 6 — Streamlit UI
**File:** `app.py`

**Your job is to build the interactive demo.**

**Tasks:**
- Build a sidebar with controls: algorithm selector, parameter sliders, seed input.
- Show the scenario information (region table and need score chart) when the app loads.
- After the user clicks Run, show: best fitness, convergence curve, allocation bar chart, allocation heatmap.

**Tips:**
- Run Streamlit with: `streamlit run app.py`
- Build the full layout with fake/hardcoded data first. Once the algorithm files are ready, replace the fake data with real calls.
- For the hybrid, show two extra metrics: GA phase best and PSO phase best, so the user can see how much PSO improved on the GA.
- Test every algorithm type through the UI at least once before submitting.

> **You are completely free to structure and implement this however you think is best. The above is just a starting point.**

---
