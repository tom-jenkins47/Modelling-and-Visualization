"""
2024 EXAM
B151138

Parallel and deterministic code

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
rng = np.random.default_rng(42)
plt.style.use('ggplot')

class Simulation:
    
    def __init__(self, dim=100, p=0.01, n=2, cube=True):
        
        self.dim = dim
        self.p = p
        self.n = n
        
        if cube:
            self.setCube()
        else:
            self.setLattice()

        
    def setLattice(self):
        
        # initialize lattice based on the initial probability
        # label on as 1 and off as 0
        
        self.lattice = np.random.rand(self.dim, self.dim)
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if self.lattice[i, j] < self.p:
                    self.lattice[i, j] = 1
                    
                else:
                    self.lattice[i, j] = 0
                    
    def setCube(self):
    
        blockSize = 20
        startIdx = (self.dim - blockSize) // 2
        
        self.lattice = np.zeros((self.dim, self.dim))
        self.lattice[startIdx:startIdx+blockSize, startIdx:startIdx+blockSize] = 1


    def update(self):
        
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
                
                neighbourStatesSum = N+E+S+W+NE+SE+SW+NW
                
                if sampleState == 1:
                    updatedLattice[i,j] = 0
                        
                else:
                    if neighbourStatesSum == self.n:
                        updatedLattice[i, j] = 1
                    else:
                        updatedLattice[i, j] = 0
                        
        self.lattice = updatedLattice
        
        
    def plotOnStates(self):
        
        self.n = 2
        
        times = np.arange(1000)
        onStates = []
        
        for time in times:    
            self.update()
            onStates.append(np.sum(self.lattice))
            
        plt.plot(times, onStates)
        plt.xlabel('timestep')
        plt.ylabel('no. of on states')
        plt.title('Number of On States vs. Time')
        plt.show()
    
          
class Visualizer:
    
    def __init__(self, field):
        
        self.field = field
        self.step = 0
        self.fig, self.ax = plt.subplots()        
        self.plot = self.ax.imshow(self.field.lattice, cmap = 'gray')
        self.ani = None
        
    def run(self, save=False):
    
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=30, frames=2000, repeat=False)
        
        if save:
            self.ani.save(f"q1c_cube_n3.gif")
        else:
            plt.show()
        
    def animate(self, frame):
        
        self.field.update()
        self.step += 1
        self.plot.set_data(self.field.lattice)
        self.ax.set_title(f"Time: {self.step:.2f}")
         
        return (self.plot,)
    
def main():
    
    nVal = int(input('Value of n: '))
    dimVal = int(input('Dimension of the grid: '))
    useCube = str(input('Use cube [y/n]?: '))
    
    if useCube == 'y':
        field = Simulation(dim=dimVal, n=nVal)
        
    elif useCube == 'n':
        field = Simulation(dim=dimVal, n=nVal, cube=False)
        
    else:
        raise ValueError('Usage [y/n]')
        
    plotOn = str(input('Plot on states over time [y/n]?: '))
    
    if plotOn == 'y':
        field.plotOnStates()
        
    elif plotOn == 'n':
        anim = Visualizer(field)
        anim.run() # add argument save=True to save .gif
      
    else:
        raise ValueError('Usage [y/n]')
        
main()
    




