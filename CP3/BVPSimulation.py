"""
Tom Jenkins CP3

Boundary Value Problems


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

class Simulation:
    
    """
    Class to construct and analyze boundary value problems.
    Initializes system size, charge distribution (for electric), desired convergence accuracy, and algorithm of choice.
    """
    
    def __init__(self, N, initCharge, tolerance, algorithm):
        
        self.dim = N
        self.initCharge = initCharge
        self.tol = tolerance
        self.algorithm = algorithm
        
    def Jacobi(self, field_type):
        
        """
        Updates system using Jacobi algorithm until convergence achieved.
        """
        
        time = 0
        
        if field_type == 0:
            chargeDistribution = ChargeDistribution(self.dim, self.initCharge)
        elif field_type == 1:
            chargeDistribution = CurrentDistribution(self.dim)
            
        initPot = np.zeros((self.dim, self.dim, self.dim))
        pot = JacobiPotential(initPot, chargeDistribution, self.dim)
        error = abs(pot - initPot)
        error = np.sum(error)
        
        while error > self.tol:
            
            initPot = pot
            pot = JacobiPotential(pot, chargeDistribution, self.dim)        
            error = abs(pot - initPot)
            error = np.sum(error)
            time +=1
            
        return pot
        
    def Gauss(self, field_type):
        
        """
        Updates system using Gauss-Seidl algorithm until convergence achieved.
        """
        
        time = 0
        
        if field_type == 0:
            chargeDistribution = ChargeDistribution(self.dim, self.initCharge)
        elif field_type == 1:
            chargeDistribution = CurrentDistribution(self.dim)
     
        pot = np.zeros((self.dim, self.dim, self.dim))
        pot, initPot, finalPot = GaussPotential(pot, chargeDistribution, self.dim)
        error = abs(finalPot - initPot)
        
        while error > self.tol:
            pot, initPot, finalPot = GaussPotential(pot, chargeDistribution, self.dim)        
            error = abs(finalPot - initPot)
            time +=1
        
        return pot
    
    def SOR(self, write=False):
        
        """
        Optimizes successive over-relaxation, with respect to number of iterations.
        """
        
        timeData = []
        omegas = np.arange(1, 2, 0.01)
        
        for i in range(0, len(omegas)):
            omega = omegas[i]
            time = 0
            chargeDistribution = ChargeDistribution(self.dim, self.initCharge)
            pot = np.zeros((self.dim, self.dim, self.dim))
            pot, diffPot = SORPotential(pot, chargeDistribution, self.dim, omega)
            error = np.sum(np.abs(diffPot))
            
            while error > self.tol:
                time += 1
                pot, diffPot = SORPotential(pot, chargeDistribution, self.dim, omega)        
                error = np.sum(np.abs(diffPot))
                
            timeData.append(time)
            
            if write:
                fileExists = os.path.isfile('SOR_Optimization.csv')
                with open('SOR_Optimization.csv', 'a+') as f:
                    if not fileExists:
                        f.write('Omega, Time\n')
                    f.write(f'{omega}, {time}\n')
    
        plt.plot(omegas, timeData, marker='s')
        plt.title("Over-Relaxation vs. Convergence Time")
        plt.xlabel('time (no. of iterations)')
        plt.ylabel('omega')
        plt.show()
                
    def plotter(self, field_type, write=False):
        
        """
        Plots relevant graphs and writes data to outfile if desired.
        """
        
        if self.algorithm == 'jacobi':
            pot = self.Jacobi(field_type)
            
        elif self.algorithm == 'gauss':
            pot = self.Gauss(field_type)
            
        if field_type == 0:
            titleToken = 'Electric'
            Ex, Ey, Ez, Etotal = EField(pot, self.dim)  
            distance, potData, fieldData = collectData(pot, Etotal, self.dim)
            
        elif field_type == 1:
            titleToken = 'Magnetic'
            Ex, Ey, Ez, Etotal = MField(pot, self.dim)  # defining like this just makes next part easier
            distance, potData, fieldData = collectData(pot, Etotal, self.dim)
            
        if write:
            for i in range(1, self.dim-1):
                for j in range(1, self.dim-1):
                    
                    midplanePot = pot[i, j, int(self.dim/2)]
                    midplaneEx = Ex[i, j, int(self.dim/2)]
                    midplaneEy = Ey[i, j, int(self.dim/2)]
                    midplaneEz = Ez[i, j, int(self.dim/2)]
                    
                    fileExists = os.path.isfile(f'Midplane_{titleToken}_Potential_vs_field.csv')
                    with open(f'Midplane_{titleToken}_Potential_vs_field.csv', 'a+') as f:
                        if not fileExists:
                            f.write('x, y, Potential, x-field, y-field, z-field\n')
                        f.write(f' {i}, {j}, {midplanePot}, {midplaneEx}, {midplaneEy}, {midplaneEz}\n')
            
            for n in range(len(distance)):
                
                fileExists = os.path.isfile(f'Distance_vs_{titleToken}_Potential_vs_field.csv')
                with open(f'Distance_vs_{titleToken}_Potential_vs_field.csv', 'a+') as f:
                    if not fileExists:
                        f.write('Distance, Potential, Field Magnitude\n')
                    f.write(f'{distance[n]}, {potData[n]}, {fieldData[n]}\n')
        
        
        fig, ax = plt.subplots()
        image = ax.imshow(pot[:, :, int(self.dim/2)], extent = (0, self.dim, 0, self.dim), cmap='gray')
        plt.title(f'{titleToken} Potential Heatmap at Central Cut in z')
        cbar = ax.figure.colorbar(image, ax=ax)
        plt.show()
        
        plt.plot(distance, potData, marker='s')
        plt.title(f'{titleToken} Potential vs. Distance to Center')
        plt.xlabel('distance')
        plt.ylabel('potential')
        plt.show()
        
        plt.plot(distance, fieldData, marker='s')
        plt.title(f'{titleToken} Field vs. Distance to Center')
        plt.xlabel('distance')
        plt.ylabel('field')
        plt.show()
        
        x,y = np.meshgrid(np.arange(0,self.dim,1),np.arange(0,self.dim,1))
        U = Ex[:, :, int(self.dim/2)]
        V = Ey[:, :, int(self.dim/2)]
        plt.quiver(x,y,U,V)
        plt.title(f"{titleToken} Field at Central Cut in z")
        plt.show()
        
        
def main():
    
    """
    Prompts the user for relevant inputs.
    """
    
    use_SOR = str(input('Optimize successive over-relaxation (this will use electric field) [y/n]?: '))
    field_type = int(input('Electric or magnetic field [0/1]: '))
    algorithm = str(input('Jacobi or Gauss [jacobi/gauss]?: '))
    charge_type = int(input('Random or point charge distribution (0 if magnetic field) [0/1]?: '))
    dim = int(input('Dimension: '))
    tolerance = float(input('Desired accuracy: '))
    write_data = str(input('Write data to file [y/n]?: '))
    
    sim = Simulation(dim, charge_type, tolerance, algorithm)
    
    if use_SOR == 'y':
        
        if write_data == 'y':
            sim.SOR(write=True)
            
        if write_data == 'n':
            sim.SOR()
            
    elif use_SOR == 'n':
        
        if write_data == 'y':
            sim.plotter(field_type, write=True)
            
        if write_data == 'n':
            sim.plotter(field_type)
            
    else:
        raise ValueError('Usage [y/n]: ')
        
if __name__ == '__main__':
    main()



