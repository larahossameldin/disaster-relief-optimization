import sys  
import os  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'problem')))

from matplotlib.pylab import seed
import numpy as np
from problem.scenarioM import get_scenario
from problem.FitnessFinal import initialise_random, initialise_demand_proportional, initialise_urgency_biased
from problem.FitnessFinal import W1, W2, W3, BETA , F1_MODE, compute_fitness, decode
from problem.constraint import repair
sc = get_scenario()   # loads default 8-region scenario




class LinearInertia:
    def __init__(self,w_start=0.9,w_end=0.5):
        if(w_start>w_end):
            self.w_start=w_start
            self.w_end=w_end
        else:#swapping values in case caller swapped them accidentally
            self.w_start=w_end
            self.w_end=w_start
    def get(self,iteration,max_iterations):
        t= iteration/ max(max_iterations-1,1)
        return self.w_start-t*(self.w_start-self.w_end) #at the start of the iterations, inertia is w_start, and at the end it is w_end, and it changes linearly in between
    
class RandomInertia:
    #draws inertia randomly from a unifrom distribution [0.5,1)
    def __init__(self,rng):
        self.rng=rng
            
    def get(self,iteration,max_iterations):
        return float(self.rng.uniform(0.5,1))
    

class PSO:
    def __init__(self,scenario,num_particles=64,max_iterations=499,
                c1=1.5882518918201334,c2=0.6788100670905194,
                inertia='random',
                bare=False, bare_prob=0.5,
                ring=False, neighbors=7,
                seed=None, 
                w1=W1, w2=W2, w3=W3,beta=BETA, f1_mode="asymmetric",
                initialization_strategy='urgency_biased',
                stagnation_threshold=30, stagnation_tolerance=1e-1,
                near_zero_tolerance=1e-2):
        
        self.scenario=scenario
        self.num_particles=num_particles
        self.max_iterations=max_iterations
        self.c1=c1
        self.c2=c2
        self.bare=bare
        self.bare_prob=bare_prob
        self.ring=ring
        self.neighbors=min(neighbors,num_particles) #number of neighbors cant exceed swarm size
        self.w1=w1
        self.w2=w2
        self.w3=w3 
        self.beta= beta
        self.f1_mode=f1_mode
        self.dim = scenario['dim']
        self.initialization_strategy=initialization_strategy
        self.stagnation_threshold=stagnation_threshold
        self.stagnation_tolerance=stagnation_tolerance
        self.near_zero_tolerance=near_zero_tolerance
    
        #Random number generator,seeded for reproducibility
        self.rng=np.random.default_rng(seed)
        
        #Inertia strategy, default is linear inertia
        if inertia=='random':
            self.inertia=RandomInertia(self.rng)
        elif inertia=='linear':
            self.inertia=LinearInertia()
        else:
            self.inertia= inertia 
        
        #tracking best solution
        self.gbest_history=[] #history of global best fitness values
        self.gbest_position_history=[] #history of global best positions corresponding to the fitness values in gbest_history
        self.f1_history=[] 
        self.f2_history=[] 
        self.f3_history=[] 
        #el histories f1,f2,f3 are useful for PLOTTING the convergence of the algorithm,
        #and for analyzing the trade-offs between the objectives in a multi-objective optimization problem
        
        
        #INITIALIZE--------------------------------------------------------------
    def _initialize(self):
    #called once at the start of optimize()
        # Creates
            
        #     pos: (n_particles, dim)  — current particle positions
        #     vel: (n_particles, dim)  — current velocities (zeros at start)
        #     pbest_x: (n_particles, dim)  — each particle's personal best position
        #     pbest_f: (n_particles,)      — fitness at personal best
            
            #random initialization
        if self.initialization_strategy=='random':
            self.pos= initialise_random(self.num_particles,self.scenario,self.rng)
        elif self.initialization_strategy=='demand_proportional':
            self.pos= initialise_demand_proportional(self.num_particles,self.scenario,self.rng)
        elif self.initialization_strategy=='urgency_biased':
            self.pos= initialise_urgency_biased(self.num_particles,self.scenario,self.rng)
            
        #velocities initialized to zero
        self.vel=np.zeros((self.num_particles,self.dim)) #velocities of all particles(rows) in all dimensions(columns)
            
        #evaluating fitness of initial positions
        fits=self._evaluate_all(self.pos) 
        #shape(num_particles,) 
            
            
        #initializing personal bests to initial positions and fitnesses
        self.pbest_x=np.copy(self.pos) 
        self.pbest_f=np.copy(fits) 
            
        #swarm global best= best among all initial particles
        best_idx = int(np.argmin(fits)) #best fitness in our case is the smallest one(minimizing the objective function)
        self._gbest_x= self.pos[best_idx] #global best position
        self._gbest_f= fits[best_idx] #global best fitness
            
        #recording the fitness values of the initial positions in the history lists, for plotting and analysis later
        self._record() 
        
        
    #EVALUATION--------------------------------------------------------------
    def _evaluate_all(self,positions):
        return np.array([
            compute_fitness(positions[i],self.scenario,self.f1_mode,self.beta,self.w1,self.w2,self.w3)[0]
            for i in range(len(positions))
        ])
        
    #TOPOLOGY--------------------------------------------------------------
    #NEIGHBORHOOD BEST--------------------------
    def _ring_neighborhood(self,idx):
        #returns the indices of the neighbors of particle idx in a ring topology
        #num neighbors left and right:
        half= self.neighbors//2 #floor
        #range of neighbors:
        raw= np.arange(idx-half, idx+half+1) #inclusive of idx, so +1
        #mod ashan lama nwsl lel akher n wrap around 
        return raw% self.num_particles    
        ############example idx=0, raw= [-2,-1,0,1,2],num_particles=100, mod_raw= [98,99,0,1,2] so neighbors are 98,99,0,1,2 as expected in a ring topology
    
    def _neighborhood_best(self,idx):
        if not self.ring: # Global topology : return swarm's all-time best (gbest)
            return self._gbest_f, self._gbest_x
        else:# Ring topology   : scan only the ring neighbours' personal bests
            nbrs = self._ring_neighborhood(idx)
            best_f= np.inf #bc i want to minimize
            best_x= self.pbest_x[nbrs[0]] # Initialize with the first neighbor's personal best
            for nbr in nbrs:
                if self.pbest_f[nbr] < best_f:
                    best_f= self.pbest_f[nbr]
                    best_x= self.pbest_x[nbr]
            return best_f, best_x
        
        
    #CANONICAL PSO update (velocity and position updates)------------------------------------------------
    def _canonical_step(self,w):
        for i in range(self.num_particles):
            _, nbest_x= self._neighborhood_best(i) #get neighborhood best position for particle i
            # r1 and r2 introduce stochasticity in cognitive and social learning
            # ensuring diverse particle movement and preventing premature convergence
            r1= self.rng.uniform(0, 1, self.dim)   #cognitive randomness (for c1)
            r2= self.rng.uniform(0, 1, self.dim)   #social randomness (for c2)
            
            #velocity update: 3 terms - inertia, cognitive, social
            cognitive= self.c1 *r1* (self.pbest_x[i] - self.pos[i]) #moving towards personal best
            social = self.c2 *r2* (nbest_x - self.pos[i]) #moving towards neighborhood best
            self.vel[i] = w* self.vel[i] + cognitive + social
            
            #position update
            self.pos[i] = self.pos[i]+ self.vel[i]
            
            #repair new position to satisfy constraints
            self.pos[i] = repair(self.pos[i], self.scenario)
    
    #BARE_BONES PSO update-------------------------------------------------------
    def _bare_bones_step(self):
        new_pos= np.zeros_like(self.pos)
        for i in range(self.num_particles):
            _, nbest_x= self._neighborhood_best(i) #get neighborhood best position for particle i
            
            for j in range(self.dim):
                if self.rng.uniform(0, 1) < self.bare_prob:
                    #exploration : random value between personal best and neighborhood best
                    mu = 0.5 * (nbest_x[j] + self.pbest_x[i,j]) #i, j for the jth dimension of the ith particle
                    sigma = abs(nbest_x[j] - self.pbest_x[i,j]) #spread based on distance between personal best and neighborhood best
                    new_pos[i,j]= self.rng.normal(mu, sigma) #sample new position from normal distribution
                else:
                    #exploitation : move to personal best
                    new_pos[i,j]= self.pbest_x[i,j]
                    
            #repair new position to satisfy constraints
            new_pos[i]= repair(new_pos[i], self.scenario)
            
        self.pos=new_pos
        
    #RECORDING HISTORY-------------------------------------------------------
    def _record(self):
        #storing current best fitness and subobjective for plotting 
        self.gbest_history.append(self._gbest_f)
        _, details = compute_fitness(self._gbest_x, self.scenario, self.f1_mode,self.beta,self.w1,self.w2,self.w3)
        self.f1_history.append(details['f1'])
        self.f2_history.append(details['f2'])
        self.f3_history.append(details['f3'])
        
    #UPDATE MEMORY---------------------------------------------------------------
    def _update_memory(self,fits):
        # update:
        #Personal best (pbest) for each particle    
        #Global best (gbest) for the whole swarm
        #This is the SURVIVOR SELECTION step in EA terms
        for i in range(self.num_particles):
            if fits[i] < self.pbest_f[i]: 
                self.pbest_f[i]= fits[i]
                self.pbest_x[i]= self.pos[i].copy()
                
            if fits[i] < self._gbest_f:
                self._gbest_f= fits[i]
                self._gbest_x= self.pos[i].copy()
                
    #STAGNATION CHECK: STOPPING CRITERIA 
    
    def _is_stagnant(self):
        if len(self.gbest_history) < self.stagnation_threshold:
            return False #not enough history yet to judge
        window= self.gbest_history[-self.stagnation_threshold:] #[-k:] gets the last k elements of the list (so the last 'stagnation_threshold' fitness values from the history)
        improvement= window[0] - window[-1] #difference between the oldest and newest fitness in the window
        return improvement < self.stagnation_tolerance # if the improvement is less than the tolernace, it is stagnant, otherwise it is still improving
    
    
    #SWARM RADIUS NEAR ZERO : STOPPING CRITERIA
    # Returns True if the swarm has collapsed — all particles have crowded into
    # roughly the same point in search space and diversity is effectively gone.
    def _swarm_radius_near_zero(self):
        return np.std(self.pos) < self.near_zero_tolerance
                
    #MAIN OPTIMIZATION LOOP------------------------------------------------------
    def optimize(self):
        #returns :
            # best_fitness : float
            # best_solution: 2-D ndarray (n_regions, n_resources)
            # history      : dict of lists (for plotting convergence curves)
        self._initialize()
        
        for iter in range(self.max_iterations):
            #get inertia w 
            w= self.inertia.get(iter, self.max_iterations)
            
            #move particles according to the chosen PSO variant
            if self.bare:
                self._bare_bones_step()
            else:
                self._canonical_step(w)
                
            #evaluate fitness of new positions
            fits= self._evaluate_all(self.pos)
            
            #update memories (pbest and gbest) (survivor selection)
            self._update_memory(fits)
            
            #record state
            self._record()
            
            #stop optimization if stagnation detected
            if self._is_stagnant():
                break
            #stop optimization if swarm radius is near zero (swarm has collapsed)
            if self._swarm_radius_near_zero():
                break 
        #decode solution to matrix 
        best_solution =decode(self._gbest_x, self.scenario['n_regions']) #split the vector into n_regions parts, and reshape each part into (n_resources,) to get a (n_regions, n_resources) solution matrix
        #repair best solution to satisfy constraints
        best_solution= repair(best_solution, self.scenario)
            
        return self._gbest_f, best_solution, {
                "convergence" : self.gbest_history,
                "f1_history" : self.f1_history,
                "f2_history" : self.f2_history,
                "f3_history" : self.f3_history
            }
            



# PARAMETER STUDY----------------------------------------------------
# All combinations of the parameters 

def build_all_configs():
    # Returns a list of (label, kwargs) tuples — one per configuration.

    # Parameters swept in the study:
    # update_rule          : canonical  |  bare-bones (bare_prob 0.5 / 0.9)
    # topology             : global     |  ring (neighbors 2 / 4)
    # c1, c2               : balanced (1.5/1.5) | cognitive-heavy | social-heavy
    # inertia              : linear     |  random
    # initialization       : random | demand_proportional | urgency_biased
    # num_particles        : 10 | 30 | 50

    configs = []

    # ── helper so we never forget a label ────────────────────────────────────
    def add(label, **kw):
        configs.append((label, kw))

    # GROUP 1 — UPDATE RULE  (canonical vs bare-bones)
    # Baseline: global topology, balanced c1=c2=1.5, linear inertia, 30 particles
    add("Canonical-Global",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30)
    
    add("BareBones-p0.5-Global",
        bare=True, bare_prob=0.5, ring=False, c1=1.5, c2=1.5, num_particles=30)
    
    add("BareBones-p0.9-Global",
        bare=True, bare_prob=0.9, ring=False, c1=1.5, c2=1.5, num_particles=30)


    # GROUP 2 — TOPOLOGY  (global vs ring with different neighbour counts)

    add("Ring-k2",
        bare=False, ring=True, neighbors=2, c1=1.5, c2=1.5, num_particles=30)

    add("Ring-k4",
        bare=False, ring=True, neighbors=4, c1=1.5, c2=1.5, num_particles=30)

    add("global-k4",
        bare=False, bare_prob=0.5, ring=False, neighbors=4, c1=1.5, c2=1.5,
        num_particles=30)
    add("global-k2",
        bare=False, bare_prob=0.5, ring=False, neighbors=2, c1=1.5, c2=1.5,
        num_particles=30)


    # GROUP 3 — COGNITIVE vs SOCIAL BALANCE  (c1 / c2)

    add("Balanced-c1.5-c1.5",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30)

    add("Cognitive-c2.5-c0.5",
        bare=False, ring=False, c1=2.5, c2=0.5, num_particles=30)

    add("Social-c0.5-c2.5",
        bare=False, ring=False, c1=0.5, c2=2.5, num_particles=30)

    add("Equal-c1.49-c1.49",      # classic literature default
        bare=False, ring=False, c1=1.49, c2=1.49, num_particles=30)


    # GROUP 4 — INERTIA SCHEDULE  (linear vs random)
    add("Linear-Inertia",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        inertia='linear')

    add("Random-Inertia",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        inertia='random')


    # GROUP 5 — INITIALISATION STRATEGY

    add("Init-Random",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        initialization_strategy='random')

    add("Init-DemandProp",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        initialization_strategy='demand_proportional')

    add("Init-UrgencyBiased",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        initialization_strategy='urgency_biased')


    # GROUP 6 — SWARM SIZE  (num_particles)

    add("Swarm-10",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=10)

    add("Swarm-30",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30)

    add("Swarm-50",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=50)
    
    add("Swarm-100",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=100)           
    
    # BONUS — A few interesting cross-combinations
    add("Ring-k4-Cognitive-Large",
        bare=False, ring=True, neighbors=4,
        c1=2.5, c2=0.5, num_particles=50,
        initialization_strategy='urgency_biased')
    
    add("BareBones-Ring-k2-UrgencyInit",
        bare=True, bare_prob=0.5, ring=True, neighbors=2, num_particles=30,
        initialization_strategy='urgency_biased')

    add("Social-RandomInertia-DemandInit",
        bare=False, ring=False, c1=0.5, c2=2.5, num_particles=30,
        inertia='random',
        initialization_strategy='demand_proportional')
    
    
    # GROUP 7 — SCENARIO  (same baseline PSO, different disaster context)
    # Baseline: canonical, global, balanced c1=c2=1.5, linear inertia, 30 particles

    add("Scenario-Default",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario())

    add("Scenario-Epidemic",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario("Epidemic"))

    add("Scenario-Floods",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario("Floods"))

    add("Scenario-LargeDisaster",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario("Large Disaster"))

    add("Scenario-ResourceShortage",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario("Resource Shortage"))

    add("Scenario-WorstCase",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario("Worst Case"))

    # Cross: hardest scenario × best-performing init strategy
    add("WorstCase-UrgencyInit",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=30,
        scenario=get_scenario("Worst Case"),
        initialization_strategy='urgency_biased')

    add("WorstCase-LargeSwarm",
        bare=False, ring=False, c1=1.5, c2=1.5, num_particles=50,
        scenario=get_scenario("Worst Case"),
        initialization_strategy='urgency_biased')

    return configs

if __name__ == "__main__":
    import pandas as pd

    DEFAULT_SCENARIO = get_scenario()
    configs = build_all_configs()
    results = []

    for label, kwargs in configs:
        # use the scenario embedded in kwargs, or fall back to default
        scenario = kwargs.pop("scenario", DEFAULT_SCENARIO)

        print(f"\n{'='*55}") 
        print(f"  {label}")
        print(f"{'='*55}")

        pso = PSO(scenario=scenario, **kwargs)
        best_fitness, best_solution, history = pso.optimize()

        _, details = compute_fitness(best_solution, scenario)

        print(f"  Best fitness : {best_fitness:.4f}")
        print(f"  f1 (suffer)  : {details['f1']:.4f}")
        print(f"  f2 (waste)   : {details['f2']:.4f}")
        print(f"  f3 (delivery): {details['f3']:.4f}")
        print(f"  penalty      : {details['penalty']:.4f}")
        print(f"  feasible     : {details['penalty'] < 1e-6}")
        print(f"  convergence  : iter {len( history['convergence'])} (stagnation or swarm collapse)")

        results.append({
            "config"    : label,
            "fitness"   : round(best_fitness, 4),
            "f1"        : round(details["f1"], 4),
            "f2"        : round(details["f2"], 4),
            "f3"        : round(details["f3"], 4),
            "penalty"   : round(details["penalty"], 4),
            "feasible"  : details["penalty"] < 1e-6,
            "converge"  : int(len(history['convergence'])),
        })

    # ── Summary table ────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("fitness")
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY  (sorted by fitness)")
    print(f"{'='*55}")
    print(df.to_string(index=False))

    print(f"\n  Best config  : {df.iloc[0]['config']}")
    print(f"  Best fitness : {df.iloc[0]['fitness']}")
    print(f"  All feasible : {df['feasible'].all()}")