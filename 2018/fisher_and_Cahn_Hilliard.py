"""
2018 Past Paper

Using an appropriate finite difference scheme, with periodic boundary conditions in
space, write a Python code to solve the Fisher equation on an N × N grid, subject
to the initial condition that ϕ = 1 for |r| < R (where r is the position vector linking
any point of the grid to the grid centre, and R is the “radius” of the initial droplet),
and ϕ = 0 elsewhere. Your code should display the density field, ϕ, in real time as
it is running. It should also be possible to set the values of N and R when the code
is run. Here and in what follows you can set D = α = 1; you can further set the
spatial discretisation ∆x = 1 (you need to find a small enough time step ∆t for the
algorithm to remain stable).

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

class Fisher:
    
    def __init__(self, N=100, D=1, alpha=1, R=10, dt=0.1, dx=1, oneD=False):
        
        self.N = N
        self.D = D
        self.alpha = alpha
        self.R = R
        self.dt = dt
        self.dx = dx
        
        self.integral = []
        
        if oneD:
            self.initializeFields1D()
            self.k = 0.1
            
        else:
            self.initializeLattice()
            self.initializeFields()
        
        
    def initializeLattice(self):
        
        centre = [int(self.N/2), int(self.N/2)]
        
        self.lattice = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                
                positionVector = np.sqrt((i-centre[0])**2 + (j-centre[1])**2)
                self.lattice[i, j] = positionVector
                
    def initializeFields1D(self):
        
        self.phi = np.zeros(self.N)
        x0 = int(self.N/10)
        self.phi[:x0] = 1
        
    def initializeFields1Dexponential(self):
        
        self.phi = np.zeros(self.N)
        self.phi[0] = 1
        
        for i in range(1, len(self.phi)):
            
            self.phi[i] = np.exp(-self.k*i)
            
        
    def initializeFields(self):
        
        self.phi = np.where(self.lattice < self.R, 1.0, 0.0)
        
    def laplacian1D(self):
        
        laplacian = np.zeros(self.N)
        laplacian[1:-1] = (self.phi[:-2] + self.phi[2:] - 2 * self.phi[1:-1]) / (self.dx**2)
        #laplacian[0] = (self.phi[1] - self.phi[0]) / (self.dx**2)  
        #laplacian[-1] = (self.phi[-2] - self.phi[-1]) / (self.dx**2) 
        
        return laplacian

    def laplacian(self, grid):
    
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx **2)
    
    def update1D(self):
        
        dphidt = self.D * self.laplacian1D() + self.alpha * self.phi * (1 - self.phi)
        self.phi += self.dt * dphidt
        self.phi[0] = 1
        self.phi[-1] = self.phi[-2]


    def update(self):
        
        dphidt = self.D * self.laplacian(self.phi) + self.alpha * self.phi * (1 - self.phi)
        self.phi += self.dt * dphidt
        
    def calculateIntegral(self, plot=False, exponential=True):
        
        if exponential:
            self.initializeFields1Dexponential()
        
        times = np.arange(10000)*self.dt
        
        for time in times:
            
            self.update1D()
            self.integral.append(np.sum(self.phi) * self.dx)
        
        if plot:
            plt.plot(times, self.integral)
            plt.xlabel('time')
            plt.ylabel('integral')
            plt.title('Phi Integral vs. Time')
            plt.show()
            
        return times
        
    def fitIntegral(self):
        
        def objective(t, v, c):
            return v * t + c
        
        self.integral = []
        times = self.calculateIntegral()
        
        times = times[0:4000]
        self.integral = self.integral[0:4000]
        
        popt, _ = curve_fit(objective, times, self.integral)
        phiFit = objective(times, popt[0], popt[1])
        
        print(f'Velocity: {popt[0]:.3f} cells per unit time')
        
        plt.scatter(times[::400], self.integral[::400], label='data', color='k', marker='x')
        plt.plot(times, phiFit, label='fit')
        plt.xlabel('time')
        plt.ylabel('integral')
        plt.title('Phi Integral vs. Time')
        plt.legend()
        plt.show()
        
    def fitIntegralExponential(self):
        
        def objective(t, v, c):
            return v * t + c
        
        ks = np.arange(0.1, 1.1, 0.1)
        
        velocities = []
        
        for k in ks:
            
            self.k = k
            self.integral = []
            times = self.calculateIntegral()
            
            grad = np.gradient(self.integral)
            cutoff = np.where(grad==0)[0][0]
            
            times = times[:cutoff]
            self.integral = self.integral[:cutoff]
            
            popt, _ = curve_fit(objective, times, self.integral)
            phiFit = objective(times, popt[0], popt[1])
            
            plt.scatter(times[::400], self.integral[::400], color='k', marker='x')
            plt.plot(times, phiFit)
            plt.show()
            plt.title(f'k = {self.k:.2f}')
            
            print(f'Velocity = {popt[0]:.3f} cells per unit time for k = {self.k:.2f}')
            velocities.append(popt[0])
            
        plt.plot(ks, velocities)
        plt.xlabel('k')
        plt.ylabel('velocity')
        plt.title('Velocity vs. k')
        plt.show()
        
        
class CahnHilliard:
    
    def __init__(self, N=30, M=0.1, alpha=0.0003, k=0.1, a=0.1, dt=0.1, dx=1):
        
        self.N = N
        self.M = M
        self.alpha = alpha
        self.k = k
        self.a = a
        self.dt = dt
        self.dx = dx
        
        self.phi = rng.uniform(0.99, 1.01, size=(self.N, self.N))
        
    
    def laplacian(self, grid):
    
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx **2)
    
    def findMu(self):
        
        self.mu = self.a * self.phi * (self.phi - 1) * (self.phi - 2) - self.k * self.laplacian(self.phi)
    
    
    def update(self):
        
        self.findMu()
        
        dphidt = self.M * self.laplacian(self.mu) + self.alpha * self.phi * (1 - self.phi)
        self.phi += self.dt * dphidt
        change = np.abs(dphidt).max()
        
        return change
        
    def findSteadyState(self):
        
        for i in range(100):
            self.update()
        
        while True:
            change = self.update()
            if change < 1e-5:
                break
            
        self.plotContour()
        self.saveData()
        
    def plotContour(self):
        
        plt.figure()
        plt.contour(self.phi, cmap='Blues')
        plt.title('Contour Plot of Phi at Steady State')
        plt.colorbar()
        plt.show()
  
    def saveData(self):
        
        np.savetxt('steady_state_phi.csv', self.phi, delimiter=",")
        print('Data saved to steady_state_phi.csv.')
        
        
class Animation:
    
    def __init__(self, field, nskip=20):
        
        # Set up a simulation to be animated
        self.nskips = nskip
        self.field = field
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.field.phi, cmap='Blues')
        self.ani = None
        
    def run(self):
        

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=25, frames=1000)
        plt.show()

    def animate(self, frames):
        
        # Animation function
        for i in range(self.nskips):
            self.field.update()
        self.plot.set_data(self.field.phi)
    
        return (self.plot,)
      
  

    
    
#field = Fisher()
field = CahnHilliard()
#field.findSteadyState()
#field.calculateIntegral(plot=True)
#field.fitIntegralExponential()
#field.fitIntegral()
anim = Animation(field)
anim.run()


