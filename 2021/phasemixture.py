"""
2021 Past Paper

Using an appropriate finite difference scheme, with periodic boundary conditions in
space, write a Python code to solve Eqs. (1) on a 50 × 50 grid, subject to the initial
condition that: (i) φ equals a constant, φ0, with some noise – a random number
between −0.01 and 0.01 at each grid point – and (ii) m equals 0 with some noise
– again a random number between −0.01 and 0.01 at each grid point. Your code
should allow you to display both φ and m in real time as it is running. [Alternatively,
you could have an argument to decide which field is shown in a given simulation.]
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


class PhaseMixture:
    
    def __init__(self, N=50, M=0.1, D=1, c=0.1, a=0.1, chi=0.3, k=0.1, dx=1, dt=0.02, phi0=0.5):
        
        self.N = N
        self.M = M
        self.D = D
        self.c = c
        self.a = a
        self.k = k
        self.phi0 = phi0
        self.chi = chi
        self.dx = dx
        self.dt = dt
        
        self.phi = rng.uniform(phi0 - 0.01, phi0 + 0.01, size=(N, N))
        self.m = rng.uniform(-0.01, 0.01, size=(N, N))
        
        
    def laplacian(self, grid):
        
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx ** 2)
    
    def determineMu(self):
        
        self.mu = (-self.a * self.phi + self.a * self.phi**3
                   - (self.chi / 2) * self.m**2 - self.k * self.laplacian(self.phi))
    
    def update(self):
        
        self.determineMu()
        
        dphidt = self.M * self.laplacian(self.mu)
        dmdt = (self.D * self.laplacian(self.m) - ((self.c 
                - self.chi * self.phi) * self.m + self.c * self.m**3))
        
        self.phi += self.dt * dphidt
        self.m += self.dt * dmdt
        
    def modifiedUpdate(self):
        
        self.determineMu()
        
        dphidt = self.M * self.laplacian(self.mu) - self.alpha * (self.phi - self.phibar)
        dmdt = (self.D * self.laplacian(self.m) - ((self.c 
                - self.chi * self.phi) * self.m + self.c * self.m**3))
        
        self.phi += self.dt * dphidt
        self.m += self.dt * dmdt
        
    def spatialAverage(self, plot=False, modified=True):
        
        times = np.arange(0, 50000)*self.dt
        
        phiAv = []
        mAv = []
        mVar = []
        
        for time in times:
            
            if modified:
                self.modifiedUpdate()
            else:
                self.update()
                
            phiAv.append(np.average(self.phi))
            mAv.append(np.average(self.m))
            mVar.append(np.var(self.m))
            
        if plot:   
            plt.plot(times, phiAv)
            plt.xlabel('time')
            plt.ylabel('phi')
            plt.show()
            
            plt.plot(times, mAv)
            plt.xlabel('time')
            plt.ylabel('m')
            plt.show()
            
        return np.average(phiAv), np.average(mAv), np.average(mVar)
        
    def varyAlpha(self):
        
        alphas = np.arange(0.0005, 0.0055, 0.0005)
        
        mAvs = []
        mVars = []
        
        self.phibar = 0.5
        
        for alpha in alphas:
            
            self.alpha = alpha
            _, mAv, mVar = self.spatialAverage()
            mAvs.append(mAv)
            mVars.append(mVar)
            
        plt.plot(alphas, mAvs)
        plt.xlabel('alpha')
        plt.ylabel('m (time/spatial average)')
        plt.show()
        
        plt.plot(alphas, mVars)
        plt.xlabel('alpha')
        plt.ylabel('m variance (time average)')
        plt.show()
        
class Visualizer:
    
    def __init__(self, mixture):
        
        self.mixture = mixture
        self.step = 0
        self.nskip = 100
        self.fig, (self.axPhi, self.axMag) = plt.subplots(1, 2)      
        self.plotPhi = self.axPhi.imshow(self.mixture.phi)
        self.plotMag = self.axMag.imshow(self.mixture.m)
        self.title = self.fig.suptitle(f"Timestep: {self.step*self.nskip*self.mixture.dt:.1f}")
        
        self.axPhi.set_title('phi')
        self.axMag.set_title('m')
        
        self.ani = None
        
    def run(self, save=False):
        
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=30, frames=1000, repeat=False)
        
        if save:
            self.ani.save(f"phi0{self.mixture.phi0}_chi{self.mixture.chi}.gif")
        else:
            plt.show()
        
    def animate(self, frame):
        
        self.step += 1
        
        for i in range(self.nskip):
            self.mixture.modifiedUpdate()
               
        
        self.plotPhi.set_data(self.mixture.phi)
        self.plotMag.set_data(self.mixture.m)
        self.title.set_text(f"Time: {self.step*self.nskip*self.mixture.dt:.1f}")
         
        return (self.plotPhi, self.plotMag,)
    
    
mixture = PhaseMixture()
#mixture.spatialAverage()
mixture.varyAlpha()

#anim = Visualizer(mixture)
#anim.run()        
        
        
        
        
        
    
    
        
        
        
    
     
    

