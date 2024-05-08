"""
2023 Past Paper

In this exercise we will consider a cellular automaton model for the rock-paper-scissors
game. You will first implement a parallel deterministic update rule, and then a random
sequential one. In the model, each cell in a two-dimensional N ×N square lattice can be in
one of three states: ‘rock’ (R), ‘paper’ (P) or ‘scissors’ (S). Each cell has eight neighbours:
four along the principal axes and four along the diagonal (i.e., as in the Game of Life).
We consider periodic boundary conditions to determine the neighbours of each cell.

a. In the parallel deterministic version, all cells in the N × N lattice are updated at
the same time, according to the following three rules:
    
    • A cell in the R state with more than two (i.e., 3 or more out of 8) neighbours
      in the P state changes its state to P.
    • A cell in the P state with more than two S neighbours becomes S.
    • A cell in the S state with more than two R neighbours becomes R.
    
Write a Python code to follow the evolution of this rock-paper-scissors parallel
update cellular automaton. Your algorithm should allow you to change the size of
the lattice, N. The initial condition you should consider is one in which the lattice
is divided into three ‘pie wedges’, with a different state in each of the wedges.

For a.) parallel deterministic is as in the GOL.
    
Need to effectively combine the updates for GOL and SIRS

"""

import random 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from scipy import stats
import os
plt.style.use('ggplot')


class Simulation:
    
    def __init__(self, dim):
        
        
        self.dim = dim
        self.setLattice()
        self.activity = []
        self.Rstates = []
                               
    def setLattice(self):
        
        # Assume R = -1, P = 0, S = 1
        
        self.lattice = np.zeros((self.dim, self.dim))
        
        center = (self.dim // 2, self.dim // 2)  # Calculate the center of the grid
     
        for i in range(self.dim):
            for j in range(self.dim):
                # Calculate the angle from the center to the point (i, j)
                angle = np.arctan2((i - center[0]), (j - center[1])) * 180 / np.pi
                
                # Normalize the angle to be between 0 and 360 degrees
                if angle < 0:
                    angle += 360
                
                # Assign values based on the angle
                if 0 <= angle < 120:
                    self.lattice[i, j] = -1  # First wedge
                elif 120 <= angle < 240:
                    self.lattice[i, j] = 0   # Second wedge
                else:
                    self.lattice[i, j] = 1   # Third wedge
        
    def countActivity(self):
        
        # Returns activity of lattice and also appends to a rolling list
        
        self.activity.append(np.sum(self.lattice))
        
        return np.sum(self.lattice)
       
        
    def update(self):
        
        # Updates the state of the lattice based on the conditions of the game
        
        updatedLattice = np.zeros((self.dim, self.dim))
        d = self.dim
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                sampleState = self.lattice[i, j]
                
                # 8 nearest neighbours labelled by polar direction
                
                N = self.lattice[i, (j+1)%d]
                E = self.lattice[(i+1)%d, j]
                S = self.lattice[i, (j-1)%d]    
                W = self.lattice[(i-1)%d, j]
               
                NE = self.lattice[(i+1)%d, (j+1)%d]
                SE = self.lattice[(i+1)%d, (j-1)%d]
                SW = self.lattice[(i-1)%d, (j-1)%d]
                NW = self.lattice[(i-1)%d, (j+1)%d]
                
                neighbourStates = [N,E,S,W,NE,SE,SW,NW]
                
                if sampleState == -1:
                    
                    if neighbourStates.count(0) > 2:  
                        updatedLattice[i, j] = 0
                        
                    else:
                        updatedLattice[i, j] = -1
                        
                elif sampleState == 0:
                    
                    if neighbourStates.count(1) > 2:
                        updatedLattice[i, j] = 1
                        
                    else:
                        updatedLattice[i, j] = 0
                        
                else:
                    
                    if neighbourStates.count(-1) > 2:
                        updatedLattice[i, j] = -1
                        
                    else:
                        updatedLattice[i, j] = 1
                    
        self.lattice = updatedLattice
            
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
    
    def __init__(self, dim):
        
        # Set up a simulation to be animated
        
        self.sim = Simulation(dim)
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
    
    
#sim = Simulation(100)
#sim.plotR()
anim = Animation(100)
anim.run()

