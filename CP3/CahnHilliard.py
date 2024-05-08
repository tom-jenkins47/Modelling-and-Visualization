"""
Tom Jenkins MVP CP3

Cahn-Hilliard Visualization Classes

Note: The animation is very slow due to the excessive number of calculations.
To get around this, numba is used.
Note: In order to massively speed up computations, the numba package is used.
Details can be found here: https://numba.readthedocs.io/en/stable/user/jit.html
To install numba: $ conda install numba
This package is a compiler for Python that works best on code that uses NumPy 
arrays and functions, and loops. It effectively compiles code in C where possible.

"""

from Functions import *
from numba import njit 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
plt.style.use('ggplot')


class SystemCH:
    
    """
    Class for Cahn-Hilliard functions. This works when implemented into the 
    visualizer, however it is ~2 orders of magnitude slower than the numba functions.
    I tried to use the @jitclass decorator, but couldn't get it to work.
    I have left it in for the sake of it, but it is currently unused.
    """
    
    def __init__(self, order, cp, N, M, dx, dt):
        
        self.setParams(order, cp, N, M, dx, dt)
        
    def setParams(self, order, cp, N, M, dx, dt):
        
        self.order = order
        self.cp = cp
        self.N = N
        self.M = M
        self.dx = dx
        self.dt = dt
    
    def Order(self):
        
        update = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                east = self.cp[(i+1)%self.N, j]
                west = self.cp[(i-1)%self.N, j]
                north = self.cp[i, (j+1)%self.N]
                south = self.cp[i, (j-1)%self.N] 

                update[i, j] = self.order[i, j] + self.M * self.dt / (self.dx**2)\
                                * (east + west + north + south - 4 * self.cp[i, j])

        self.order = update
        
        return self.order
    
    def ChemicalPotential(self, a, k):
        
        cpUpdate = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                east = self.order[(i+1)%self.N, j]
                west = self.order[(i-1)%self.N, j]
                north = self.order[i, (j+1)%self.N]
                south = self.order[i, (j-1)%self.N]
                order_point = self.order[i, j]

                cpUpdate[i, j] = -(a * order_point) + a * (order_point**3) \
                                  - (k / self.dx**2) * (east + west + north + south - (4 * order_point))

        self.cp = cpUpdate
        
        return self.cp
    
    def FreeEnergyDensity(self, a, k):
        
        fUpdate = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                east = self.order[(i+1)%self.N, j]
                west = self.order[(i-1)%self.N, j]
                north = self.order[i, (j+1)%self.N]
                south = self.order[i, (j-1)%self.N]

                dphidx = (east - west) / (2 * self.dx)
                dphidy = (north - south) / (2 * self.dx)

                fUpdate[i, j] = -(a / 2) * (self.order[i, j]**2) \
                                + (a / 4) * (self.order[i, j]**4) \
                                + (k / 2) * ((dphidx**2) + (dphidy**2))

        fDensity = np.sum(fUpdate)
        
        return fDensity
    
    
class Update:
    
    """
    Updates the state of the system. This class is used to keep all update processes wrapped.
    """
    
    def __init__(self, phi, N, M, a, k, dx, dt):
       
        self.setParams(phi, N, M, a, k, dx, dt)
        
        self.boundLower = phi - 0.1
        self.boundUpper = phi + 0.1
        
        self.order = np.random.uniform(self.boundLower, self.boundUpper, size=(N, N))
        self.cp = np.zeros([N, N])
        self.f = np.zeros([N, N])
        
        #self.system = SystemCH(self.order, self.cp, N, M, dx, dt)
        
    def setParams(self, phi, N, M, a, k, dx, dt):
        
        self.phi = phi
        self.M = M
        self.dim = N
        self.a = a
        self.k = k
        self.dt = dt
        self.dx = dx
        
        
    def updateOrder(self):
        
        """
        Updates the order of the system.
        """
        
        #self.order = self.system.Order()
        self.order = Order(self.order, self.cp, self.dim, self.M, self.dx, self.dt)
        return self.order
    
    def updateChemicalPotential(self):
        
        """
        Updates the chemical potential.
        """
        
        #self.cp = self.system.ChemicalPotential(self.a, self.k)
        self.cp = ChemicalPotential(self.order, self.cp, self.dim, self.a, self.k, self.dx, self.dt)
        return self.cp 
    
    def updateFreeEnergyDensity(self):
        
        """
        Updates the free energy density.
        """
        
        #self.f = self.system.FreeEnergyDensity(self.a, self.k)
        self.f = FreeEnergyDensity(self.order, self.dim, self.a, self.k, self.dx)
        return self.f
    
    
class Visualizer:
    
    """
    Visualizes the system.
    """
    
    def __init__(self, phi, N, M, a, k, dx, dt):
        
        self.update = Update(phi, N, M, a, k, dx, dt)

        self.fig, self.ax = plt.subplots()        
        self.plot = self.ax.imshow(self.update.order, cmap = 'gray')
        self.ani = None
        
    def run(self):
        
        """
        Runs the animation, updating every 1ms.
        """

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=1, blit=True)
        plt.show()

        
    def animate(self, frame):
        
        """
        Animates the system for a particular duration.
        """
        
        for i in range(2500):
            
            self.update.updateOrder()
            self.update.updateChemicalPotential()
            
        self.plot.set_data(self.update.order)

        return (self.plot,)
    
class DataCollection:
    
    """
    Collects free energy density data and writes it to file.
    """
    
    def __init__(self, phi, N, M, a, k, dx, dt):
        
        self.phi = phi
        self.dim = N
        self.M = M
        self.a = a
        self.k = k
        self.dx = dx
        self.dt = dt
        
    def writeToFile(self, write=True):
        
        data = []
        iteration = []

        sim = Update(self.phi, self.dim, self.M, self.a, self.k, self.dx, self.dt)
        
        for i in range (0, 50000):
            
            sim.updateOrder()
            sim.updateChemicalPotential()
            fed = sim.updateFreeEnergyDensity()
            
            if write:
                if self.phi == 0:
                    fileExists = os.path.isfile('FED_phi_0p0.csv')
                    with open('FED_phi_0p0.csv', 'a+') as f:
                        if not fileExists:
                            f.write('Iteration, FED\n')
                        f.write(f'{i}, {str(fed)}\n')
                        
                elif self.phi == 0.5:
                    fileExists = os.path.isfile('FED_phi_0p5.csv')
                    with open('FED_phi_0p5.csv', 'a+') as f:
                        if not fileExists:
                            f.write('Iteration, FED\n')
                        f.write(f'{i}, {str(fed)}\n')
                
            iteration.append(i)
            data.append(fed)
            
        return iteration, data
    
    def plotter(self):
        
        iteration, data = self.writeToFile(write=True)
        
        plt.plot(iteration, data)
        plt.ylabel('free energy density')
        plt.xlabel('iteration')
        
        if self.phi == 0:
            plt.title('$\phi_0$=0 Free Energy Density vs. Time')
            
        elif self.phi == 0.5:
            plt.title('$\phi_0$=0.5 Free Energy Density vs. Time')
            
        plt.show()
        
        
def runVisualization(default=True):
    
    dx = 1
    dt = 1
    
    if default:
        dim = 100
        phi = 0.5
        M = 0.1
        a = 0.1
        k = 0.1
        
    if not default:
        dim = int(input("Dimension: "))
        phi = float(input("Initial phi value: "))
        M = float(input("M: "))
        a = float(input("a: "))
        k = float(input("k: "))
    
    visualization = Visualizer(phi, dim, M, a, k, dx, dt)
    visualization.run()
    
def main():
    
    """
    Prompts the user for relevant inputs.
    """
    
    data_type = str(input("Collect data? [y/n]: "))
  
    if data_type == 'y':
      
        phi = float(input('phi: '))
        data_coll = DataCollection(phi, 100, 0.1, 0.1, 0.1, 1, 1)
        data_coll.plotter()
        
    elif data_type == 'n':
        
        sim_type = str(input("Run default simulation [y/n]?: "))
        
        if sim_type == 'y':
            
            runVisualization()
            
        elif sim_type =='n':
            
            runVisualization(default=False)
            
        else:
            raise ValueError('Usage [y/n]')
            
    else:
        raise ValueError('Usage [y/n]')
        
if __name__ == '__main__':
    main()
