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
### Member 1 — Problem Definition
**Files:** `problem/scenario.py` and `problem/fitness.py`

**Your job is to build the foundation that everyone else imports from.
Please try to finish this first before the others start coding.**

**Tasks:**
- In `scenario.py`: define the disaster scenario. You have 8 regions, each with a need score, population, access difficulty, capacity, and resource weights. Also define the resource budgets (Food = 1000, Water = 800, Medicine = 600).
- In `fitness.py`: write the objective function that takes a 24-number solution and returns one score (lower = better). The score should reflect how much need went unmet, how much was wasted, and how hard delivery was.
- Write a repair function that fixes any solution so allocations always sum to the budget.
- Write two initialisation functions — one random, one that gives more to high-need regions from the start.
- Write a decode function that splits the 24 numbers into food[8], water[8], medicine[8].

**Tips:**
- The 24 numbers represent: first 8 = food per region, next 8 = water per region, last 8 = medicine per region.
- The repair function is simple — just rescale each group of 8 numbers so they sum to their budget.
- Test your fitness function with a random solution and make sure it returns a sensible number.
- Share your function names and what they return with the rest of the team early so everyone can import correctly.

> **You are completely free to structure and implement this however you think is best. The above is just a starting point.**

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
