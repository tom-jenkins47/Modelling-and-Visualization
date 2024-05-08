"""
2023 Past Paper

In this exercise we will consider a cellular automaton model for the rock-paper-scissors
game. You will first implement a parallel deterministic update rule, and then a random
sequential one. In the model, each cell in a two-dimensional N ×N square lattice can be in
one of three states: ‘rock’ (R), ‘paper’ (P) or ‘scissors’ (S). Each cell has eight neighbours:
four along the principal axes and four along the diagonal (i.e., as in the Game of Life).
We consider periodic boundary conditions to determine the neighbours of each cell.

c. We now consider a variant of the algorithm, which is random and sequential, rather
than parallel and deterministic. As an initial state, you should now consider one
where each cell is randomly set to either R, S, or P. The update rules in this case
become:
    
    • A cell in the R state with at least one neighbour (1 out of 8) in the P state
      changes its state to P with probability p1.
    • A cell in the P state with at least one S neighbour becomes S with probability
      p2.
    • A cell in the S state with at least one R neighbour becomes R with probability
      p3.

Write another Python code to follow the evolution of this rock-paper-scissors random
sequential cellular automaton. While this new code may be a small variation of the
previous one, it is still best to keep two separate codes. 

"""

import random 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from scipy import stats
import os
import seaborn as sns
plt.style.use('ggplot')
from numba import njit 


class Simulation:
    
    def __init__(self, dim, p1, p2, p3):
        
        
        self.dim = dim
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.setLattice()
        self.activity = []
        self.Rstates = []
                               
    def setLattice(self):
        
        # Assume R = -1, P = 0, S = 1
        
        self.lattice = np.random.rand(self.dim, self.dim)
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if self.lattice[i, j] < 1/3:
                    self.lattice[i, j] = -1
                    
                elif self.lattice[i, j] < 2/3:
                    self.lattice[i, j] = 0
                    
                else:
                    self.lattice[i, j] = 1
        
    def countActivity(self):
        
        # Returns activity of lattice and also appends to a rolling list
        
        self.activity.append(np.sum(self.lattice))
        
        return np.sum(self.lattice)
       
        
    def update(self):
        
        # Updates the state of the lattice based on the conditions of the game
        
        d = self.dim
        
        for _ in range(self.dim**2):
        
            i = int(np.random.uniform()*self.dim)
            j = int(np.random.uniform()*self.dim)
            
            d = self.dim
        
            # 8 nearest neighbours labelled by polar direction
            
            N = self.lattice[i, (j+1)%d]
            E = self.lattice[(i+1)%d, j]
            S = self.lattice[i, (j-1)%d]    
            W = self.lattice[(i-1)%d, j]
           
            NE = self.lattice[(i+1)%d, (j+1)%d]
            SE = self.lattice[(i+1)%d, (j-1)%d]
            SW = self.lattice[(i-1)%d, (j-1)%d]
            NW = self.lattice[(i-1)%d, (j+1)%d]
            
            sampleState = self.lattice[i, j]
            
            neighbourStates = [N,E,S,W,NE,SE,SW,NW]
            
            if sampleState == -1:
                
                if 0 in set(neighbourStates):  
                    if random.random() < self.p1:
                        self.lattice[i, j] = 0   
                    else:
                        self.lattice[i, j] = -1
                        
            elif sampleState == 0:
                
                if 1 in set(neighbourStates):  
                    if random.random() < self.p2:
                        self.lattice[i, j] = 1   
                    else:
                        self.lattice[i, j] = 0
                        
            else:
                
                if -1 in set(neighbourStates):
                    if random.random() < self.p3:
                        self.lattice[i, j] = -1  
                    else:
                        self.lattice[i, j] = 1
                    
                
    def countR(self, i, j):
        
        state = self.lattice[i, j]
        
        if state == -1:   
            self.Rstates.append(1) 
        else:
            self.Rstates.append(0)
        
    def plotR(self):
        
        k = np.random.randint(0,self.dim)
        m = np.random.randint(0,self.dim)
        times = np.arange(0, 500)
        
        for i in range(500):
            
            self.countR(k, m)
            self.update()
            
        plt.plot(times, self.Rstates)
        plt.title('R States vs. Timestep')
        plt.xlabel('timestep')
        plt.ylabel('no. of R states')
        plt.show()
            
        
class Animation:
    
    def __init__(self, dim, p1, p2, p3):
        
        # Set up a simulation to be animated
        
        self.sim = Simulation(dim, p1, p2, p3)
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.sim.lattice, cmap='gray')
        self.ani = None
        
    def run(self):
        
        # Run the animation, updating every 25ms

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=25, blit=True)
        plt.show()

    def animate(self, frames):
        
        # Animation function
        
        self.sim.update()
        self.plot.set_data(self.sim.lattice)
    
        return (self.plot,)
    
    
def find_minority_phase_fraction(simulation, p3_values, steady_state_steps, measurement_steps):
    
    minority_fractions = []
    variances = []

    for p3 in p3_values:
        simulation.p3 = p3
        simulation.setLattice()  # Reset the lattice for the new simulation
        # Let the system evolve into steady state
        for _ in range(steady_state_steps):
            simulation.update()
        
        fractions = {'R': [], 'P': [], 'S': []}
        
        # Take measurements in the supposed steady state
        for _ in range(measurement_steps):
            simulation.update()
            counts = { 'R': np.sum(simulation.lattice == -1),
                       'P': np.sum(simulation.lattice == 0),
                       'S': np.sum(simulation.lattice == 1) }
            total_cells = simulation.dim**2
            for state in fractions:
                fractions[state].append(counts[state] / total_cells)
        
        # Calculate the average and variance for the minority phase
        for state in fractions:
            fractions[state] = np.array(fractions[state])
        
        # Determine the minority phase
        avg_fractions = {state: np.mean(fractions[state]) for state in fractions}
        minority_phase = min(avg_fractions, key=avg_fractions.get)
        minority_fractions.append(avg_fractions[minority_phase])
        
        # Calculate variance
        variances.append(np.var(fractions[minority_phase]))
        
        print(f'p3: {p3} complete')

    return p3_values, minority_fractions, variances


def plot_minority_fractions():
    
    # Define the resolution for p2 and p3
    p2_values = np.arange(0.05, 0.31, 0.05)
    p3_values = np.arange(0.05, 0.31, 0.05)
    
    measurement_steps = 200
    steady_state_steps = 200
    
    # Initialize matrix to hold the average fraction of the minority phase
    minority_fraction_matrix = np.zeros((len(p2_values), len(p3_values)))
    
    for i, p2 in enumerate(p2_values):
        for j, p3 in enumerate(p3_values):
            # Run the simulation for the current combination of p2 and p3
            simulation = Simulation(50, 0.5, p2, p3)
            p3_range, minority_fractions, variances = find_minority_phase_fraction(
                simulation, [p3], steady_state_steps, measurement_steps
            )
            # Store the average minority fraction in the matrix
            minority_fraction_matrix[i, j] = minority_fractions[0]
    
    # Plot the heatmap using seaborn for better visuals
    plt.figure(figsize=(10, 8))
    sns.heatmap(minority_fraction_matrix, xticklabels=np.round(p3_values, 2), yticklabels=np.round(p2_values, 2), annot=True, cmap='viridis')
    plt.title('Average Fraction of the Minority Phase')
    plt.xlabel('p3')
    plt.ylabel('p2')
    plt.show()


def find_minority():
    
    sim = Simulation(100, 0.5, 0.5, 0.5)
    
    p3_vals = np.arange(0, 0.1, 0.01)
    steady_steps = 200
    measurement_steps = 200
    
    _, minority_fractions, variances = find_minority_phase_fraction(sim, p3_vals, steady_steps, measurement_steps)
    
    plt.plot(p3_vals, minority_fractions)
    plt.xlabel('p3')
    plt.ylabel('minority fraction')
    plt.title('Minority Fraction vs. p3')
    plt.show()
    
    plt.plot(p3_vals, variances)
    plt.xlabel('p3')
    plt.ylabel('variances')
    plt.title('Variances vs. p3')
    plt.show()
    
    
#find_minority()
#plot_minority_fractions()  
anim = Animation(100, 0.2, 0.8, 0.5)
anim.run()





  


