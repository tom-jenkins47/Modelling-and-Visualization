"""
2020 Past Paper

Each cell in the lattice represents an individual (or agent), which can be in one of two
states: “active” (or infected), and “inactive” (or healthy). The update rule defining the
contact process is the following. First, a cell (lattice site) is chosen randomly. If the
selected cell is inactive, nothing happens. If the cell is active, this becomes inactive with
probability 1 − p, whereas with probability p it infects one of its four nearest neighbours
(the neighbour to be infected is chosen randomly, and if the chosen neighbour is already
infected nothing happens).

Similar to the SIRS model 

Let 0 = inactive and 1 = active

"""

import random 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import os
import seaborn as sns
plt.style.use('ggplot')

class Simulation:
    
    def __init__(self, dim, p):
        
        self.dim = dim
        self.p = p
        self.setLattice()
        self.infectedFrac = []
        
    def setLattice(self):
        
        self.lattice = np.random.rand(self.dim, self.dim)
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if self.lattice[i, j] < 1/2:
                    self.lattice[i, j] = 0
                    
                else:
                    self.lattice[i, j] = 1
                    
    def setOneInfected(self):
        
        self.lattice = np.zeros((self.dim, self.dim))
        
        i = int(np.random.uniform()*self.dim)
        j = int(np.random.uniform()*self.dim)
        
        self.lattice[i, j] = 1
        
    def countInfectedFraction(self):
        
        self.infectedFrac.append(np.sum(self.lattice))
        
        return np.sum(self.lattice)
        
    def update(self):
        
        # Updates the state of the lattice based on the conditions of the game
        
        d = self.dim
        
        for _ in range(self.dim**2):
            
            # Choose point randomly
        
            i = int(np.random.uniform()*self.dim)
            j = int(np.random.uniform()*self.dim)
            
            N = self.lattice[i, (j+1)%d]
            E = self.lattice[(i+1)%d, j]
            S = self.lattice[i, (j-1)%d]    
            W = self.lattice[(i-1)%d, j]
            
            sampleState = self.lattice[i, j]
            
            changeProb = random.random()
            infectionProb = random.random()
            
            if sampleState == 0:
                pass
            else:
                if changeProb < 1 - self.p:
                    self.lattice[i, j] = 0
                else:
                    neighbours = [((i-1)%d, j),
                                  ((i+1)%d, j),
                                  (i, (j-1)%d),
                                  (i, (j+1)%d)]
                    neighbour = random.choice(neighbours)
                    self.lattice[neighbour] = 1
                        
        self.countInfectedFraction()
        
        
    def calcError(self, x, samples=1000):
            
        variances = []
        
        for _ in range(samples):
            
            resampledData = np.random.choice(x, size=len(x), replace=True)
            resampledVariance = np.var(resampledData, ddof=1)  
            variances.append(resampledVariance)
            
        return np.std(variances)  

        
    def plotInfected(self, plotIndividual=False):
        
        probs = np.arange(0.55, 0.705, 0.005)
        times = np.arange(0, 300)
        averageFractions = []
        variances = []
        errors = []
        
        for prob in probs:
            
            self.setLattice()
            self.p = prob
            
            for i in range(50): # equilibrates
                self.update()
            
            self.infectedFrac = []
            
            for time in times:
                self.update()  
                
            if plotIndividual:
          
                plt.plot(times, np.array(self.infectedFrac)/self.dim**2)
                plt.xlabel('iteration')
                plt.ylabel('infected fraction')
                plt.title(f'Fraction Infected vs. Iteration (p={self.p:.3f})')
                plt.show()
      
            averageFractions.append(np.average(self.infectedFrac)/self.dim**2)
            variance = np.var(self.infectedFrac, ddof=1)/self.dim**2
            variances.append(variance)
            errors.append(self.calcError(self.infectedFrac)/self.dim**2)
            
        plt.plot(probs, averageFractions)
        plt.xlabel('probability p')
        plt.ylabel('time-averaged infected fraction')
        plt.title('Time-Averaged Infected Fraction vs. Probability')
        plt.show()
        
        plt.errorbar(probs, variances, yerr=errors, ecolor='k')
        plt.xlabel('probability p')
        plt.ylabel('infected fraction variance')
        plt.title('Variance vs. Probability')
        plt.show()
        
    def survivalProbability(self, numSims=100):
        
        probs = [0.6, 0.625, 0.65]
        
        for prob in probs:
            
            self.p = prob
            survivalCount = np.zeros(300)
            times = np.arange(0, 300)
            
            for _ in range(numSims):
                self.setOneInfected()
                for t in times:
                    self.update()
                    if np.any(self.lattice == 1):  # Check if there are any infected cells
                        survivalCount[t] += 1
                    else:  # If no infected cells are left, the rest of the times will also be 0
                        break
            survivalProb = survivalCount / numSims
            
            plt.loglog(times, survivalProb)
            plt.xlabel('time')
            plt.ylabel('survival probability')
            plt.title(f'Survival Probability vs. Time (p={self.p:.3f})')
            plt.show()
        
class Animation:
    
    def __init__(self, dim, p):
        
        # Set up a simulation to be animated
        
        self.sim = Simulation(dim, p)
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.sim.lattice, cmap='Blues')
        self.ani = None
        
    def run(self):

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=25, blit=True)
        plt.show()

    def animate(self, frames):
        
        # Animation function
        
        self.sim.update()
        self.plot.set_data(self.sim.lattice)
    
        return (self.plot,)
    
#sim = Simulation(50, 0.6)
#sim.plotInfected(plotIndividual=True)
#sim.survivalProbability()

    
anim = Animation(50, 0.6)
anim.run()