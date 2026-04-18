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
 
### Step 1 — Put all files in the same directory
```
your_project/
├── scenarioM.py
├── constraint.py
├── FitnessFinal.py
├── your_pso.py        ← Member 2
├── your_ga.py         ← Member 3
├── island_model.py    ← Member 4
├── experiment_runner.py ← Member 5
└── app.py             ← Member 6 (Streamlit)
```
 
### Step 2 — Install dependencies
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

### Member 3 (GA with PyGAD)
- Each **chromosome** = flat array of length 24
- PyGAD fitness function must return a **scalar** (higher = better for PyGAD by default — negate if needed, or set `fitness_batch_size`)
- Since `compute_fitness` returns lower = better, negate: `return -score`
- Call `repair` inside your fitness function before evaluating
- Use `initialise_demand_proportional` for warm starting
### Member 4 (Island Model)
- Each island is an independent PSO or GA population (use Member 2 or 3's code)
- Migration: extract best solutions as flat arrays, insert into target island population
- All islands share the same `sc` scenario object (it's read-only, no thread issues)
- Use different seeds per island for diversity
### Member 5 (Experiment Runner)
- Import `get_scenario` once, pass `sc` to all runs
- Save `details` dict from `compute_fitness` to CSV (f1, f2, f3, F, penalty per generation)
- Use seeds list: `SEEDS = list(range(30))` and pass `seed=SEEDS[run]`
### Member 6 (Streamlit UI)
- `get_scenario()` is fast — safe to call on app startup
- Display `sc["names"]`, `sc["demand"]`, `sc["budgets"]` as tables
- Show `details["f1"]`, `details["f2"]`, `details["f3"]` as bar charts after optimisation
- `is_feasible(X, sc)` can show a green/red feasibility indicator
---
 
## Scenario Data at a Glance
 
| Region | Population | Need | Urgency | Access Difficulty | Capacity |
|---|---|---|---|---|---|
| A | 8,000 | 0.90 | 0.95 | 0.30 (easy) | 400 |
| B | 3,000 | 0.70 | 0.60 | 0.80 (hard) | 200 |
| C | 6,000 | 0.85 | 0.90 | 0.50 | 350 |
| D | 2,000 | 0.40 | 0.30 | 0.90 (very hard) | 150 |
| E | 7,000 | 0.95 | 0.98 | 0.20 (easy) | 450 |
| F | 4,000 | 0.60 | 0.50 | 0.70 (hard) | 250 |
| G | 5,000 | 0.75 | 0.70 | 0.40 | 300 |
| H | 1,500 | 0.50 | 0.40 | 0.95 (very hard) | 100 |
 
**Budgets:** Food = 1000, Water = 800, Medicine = 600
 
---

### Member 2 — PSO
**File:** `algorithms/pso.py`

**Your job is to implement the particle swarm optimisation algorithm.**

**Tasks:**
- Implement Standard PSO — particles move through the solution space guided by their personal best and the global best.
- Implement APSO — same as PSO but the inertia weight shrinks from 0.9 to 0.4 over time.
- Define three configurations to compare: Config A (swarm=20, fixed ω=0.9), Config B (swarm=30, fixed ω=0.7), Config C (APSO, swarm=50).

**Tips:**
- Import the fitness function from `problem/fitness.py`.
- Every run should accept a `seed` parameter so results are reproducible.
- Return `best_position`, `best_fitness`, and `convergence` (list of best fitness per iteration) from every run.
- Test that running with the same seed twice gives the exact same result.
- Test that fitness goes down over iterations, not up.

> **You are completely free to structure and implement this however you think is best. The above is just a starting point.**

---

### Member 3 — Genetic Algorithm
**File:** `algorithms/ga.py`

**Your job is to implement the genetic algorithm using the PyGAD library.**

**Tasks:**
- Implement GA Config 1: tournament selection, single-point crossover, Gaussian mutation, elitism.
- Implement GA Config 2: roulette wheel selection, uniform crossover, random reset mutation, generational replacement.
- Add fitness sharing to Config 2 to keep diversity in the population.
- Add a repair step so every individual always has valid allocations after each generation.

**Tips:**
- Install PyGAD: `pip install pygad`.
- PyGAD maximises by default — return the negative of your fitness score to turn it into minimisation.
- Fitness sharing means penalising individuals that are too similar to each other — this stops the whole population converging to the same spot.
- Return `best_position`, `best_fitness`, `convergence`, and `final_population` — the hybrid needs the final population.
- Test that allocations still sum to the budget after several generations.

> **You are completely free to structure and implement this however you think is best. The above is just a starting point.**

---

### Member 4 — Hybrid GA → PSO
**File:** `algorithms/hybrid.py`

**Your job is to combine GA and PSO into a hybrid system.  
Wait for Members 2 and 3 before final integration.**

---

## Option 1: Sequential Hybrid (Simpler)

### Idea:
- GA explores broadly
- PSO refines the best solution

### Steps:
1. Run GA for ~100 generations
2. Take the best GA solution
3. Use it to initialise PSO (as starting particle or part of swarm)
4. Run PSO for ~100 iterations
5. Return best result

This is called **GA → PSO seeding**

---

## Option 2: Island Model (Advanced)

### Idea:
- Run GA and PSO in parallel
- Occasionally exchange solutions

### Steps:
1. Run GA and PSO independently
2. Every N iterations (e.g. 10):
   - Send GA best → PSO
   - Send PSO best → GA
3. Continue until end
4. Return best overall result

---

**Tips:**
- Import `run_ga` from `algorithms/ga.py` and `run_pso` from `algorithms/pso.py`.
- Test that the hybrid performs better than either algorithm alone.

> **You are completely free to structure and implement this however you think is best. The above is just a starting point.**

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
