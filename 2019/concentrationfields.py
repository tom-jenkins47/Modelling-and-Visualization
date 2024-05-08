"""
2019 Past Paper

Using an appropriate finite difference scheme, with periodic boundary conditions
in space, write a code to solve Eqs. (1) on an N × N grid, subject to the initial
conditions

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
plt.style.use('ggplot')
import random
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
rng = np.random.default_rng(42)

class Concentrations:
    
    def __init__(self, N=50, D1=0.2, D2=0.1, k=0.05, F=0.015, R=20, dt=0.1, dx=2):
        
        self.N = N
        self.D1 = D1
        self.D2 = D2
        self.k = k
        self.F = F
        self.R = R + rng.uniform(-0.01, 0.01, 1)[0]
        self.dt = dt
        self.dx = dx
        
        self.initializeLattice()
        self.initializeFields()
        
        
    def initializeLattice(self):
        
        centre = [int((self.N - 1)/2), int((self.N - 1)/2)]
        
        self.lattice = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                
                positionVector = np.sqrt((i-centre[0])**2 + (j-centre[1])**2)
                
                self.lattice[i, j] = positionVector
                
    def initializeFields(self):
        
        self.U = np.where(self.lattice < self.R, 0.5, 1)
        self.V = np.where(self.lattice < self.R, 0.25, 0.01)
        
    def laplacian(self, grid):
    
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx **2)
    
    def update(self):
        
        dUdt = self.D1 * self.laplacian(self.U) - self.U * self.V**2 + self.F * (1 - self.U)
        dVdt = self.D2 * self.laplacian(self.V) + self.U * self.V**2 - (self.F + self.k) * self.V
        
        self.U += self.dt * dUdt
        self.V += self.dt * dVdt
        
    def calcError(self, x, function, samples=50):
        
        variances = []
        
        for i in range(samples):
            
            resamples = rng.choice(x, size=(len(x), samples), replace=True)
            variances.append(function(resamples))
            
        return np.std(variances)
        
    def computeVariance(self, grid, axis=None):
        
        return np.var(grid, axis=axis)
    
    def plotVarianceAgainstTime(self):
        
        variance = []
        times = np.arange(0, 50000)*self.dt
        
        for time in times:
            self.update()
            variance.append(self.computeVariance(self.U))
            
        plt.plot(times, variance)
        plt.xlabel('time')
        plt.ylabel('variance')
        plt.title(f'Variance vs. Time for k={self.k:.2f}, F={self.F:.3f}')
        plt.show()
            
    def plotVarianceAgainstF(self):
        
        Fs = np.arange(0.020, 0.060, 0.005)
        
        averageVariances = []
        averageErrors = []
        
        for F in Fs:
            
            print(F)
            
            self.F = F
            self.initializeLattice()
            self.initializeFields()
          
            variances = []
            errors = []
            
            for i in range(500):
                
                self.update()
                variances.append(self.computeVariance(self.U))
                errors.append(self.calcError(self.U, self.computeVariance))
                
            averageVariances.append(np.average(variances))
            averageErrors.append(np.average(errors))
            
        plt.errorbar(Fs, averageVariances, yerr=averageErrors, ecolor='k')
        plt.xlabel('F')
        plt.ylabel('average variance')
        plt.title('Average Variance vs. F')
        plt.show()
            
        
class Visualizer:
    
    def __init__(self, field, nskip=80):
        
        self.field = field
        self.step = 0
        
        self.nskip = nskip
        
        self.fig, (self.axU, self.axV) = plt.subplots(1, 2)      
        self.plotU = self.axU.imshow(self.field.U)
        self.plotV = self.axV.imshow(self.field.V)
        self.title = self.fig.suptitle(f"Timestep: {self.step*self.nskip*self.field.dt:.1f}")
        
        self.axU.set_title('U')
        self.axV.set_title('V')
        
        self.ani = None
        
    def run(self, save=False):
        
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=30, frames=1000, repeat=False)
        
        if save:
            self.ani.save(f"F{self.field.F}.gif")
        else:
            plt.show()
        
    def animate(self, frame):
        
        self.step += 1
        
        for i in range(self.nskip):
            self.field.update()

        self.plotU.set_data(self.field.U)
        self.plotV.set_data(self.field.V)
        self.title.set_text(f"Time: {self.step*self.nskip*self.field.dt:.1f}")
         
        return (self.plotU, self.plotV,)
    
    
field = Concentrations()
#field.plotVarianceAgainstF()
#field.plotVarianceAgainstTime()
anim = Visualizer(field)
anim.run()
    
    
        
                        
        
        
        
        
        
        

