"""
2024 EXAM
B151138

Random and sequential code

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
plt.style.use('ggplot')

class Simulation:
    
    def __init__(self, dim=50, p1=0.1, p2=0.1):
        
        self.n = 2
        self.dim = dim
        self.p1 = p1
        self.p2 = p2
        
        self.setLattice()
      
        
    def setLattice(self):
        
        # initialize lattice randomly
        # label on as 1 and off as 0
        
        self.lattice = np.random.rand(self.dim, self.dim)
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if self.lattice[i, j] < 1/2:
                    self.lattice[i, j] = 1
                    
                else:
                    self.lattice[i, j] = 0
                
    def update(self):
        
        d = self.dim
        
        for _ in range(self.dim**2):
        
            i = int(np.random.uniform()*self.dim)
            j = int(np.random.uniform()*self.dim) # pick a random point
            
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
            
            if sampleState == 1:
                if random.random() < self.p1:
                    self.lattice[i, j] = 0
                else:
                    pass
                    
            else:
                if np.sum(neighbourStates) == self.n:
                    if random.random() < self.p2:
                        self.lattice[i, j] = 1
                    else:
                        pass
                else:
                    pass
                    
    def phaseDiagram(self):
        
        p1s = np.arange(0.1, 1.1, 0.1)
        p2s = np.arange(0.1, 1.1, 0.1)
        
        averageOnFraction = []
        p2sMaximal = []
        
        for p1 in p1s:
            print(f'{p1:.2f}')
            variance = []
            
            for p2 in p2s:
                self.setLattice()
                self.p1 = p1
                self.p2 = p2
                
                onFractions = []
                
                # ideally I would like to increase the number of interations
                # due to time constraints not really possible
                
                for i in range(25): # equilibrate
                    self.update()
    
                for n in range(75):  
                    self.update()
                    onFractions.append(np.sum(self.lattice)/self.dim**2)
                    
                variance.append(np.var(onFractions))
                averageOnFraction.append(np.average(onFractions))
                
            maxIdx = np.argmax(variance)
            p2sMaximal.append(p2s[maxIdx])

        averageOnFraction = np.array(averageOnFraction)
        averageOnFraction = averageOnFraction.reshape(len(p1s), len(p2s))
        
        def objective(x, m ,c):
            return m * x + c
        
        popt, _ = curve_fit(objective, p1s, p2sMaximal)
        fit = objective(p1s, popt[0], popt[1])
        
        print(f'Equation of fit: y = {popt[0]:.3f}x + {popt[1]:.3f}')
        
        fig, ax = plt.subplots()
        image = ax.imshow(averageOnFraction, extent=(p1s.min(), p1s.max(), p2s.max(), p2s.min()))
        bar = ax.figure.colorbar(image)
        ax.plot(p1s, p2sMaximal, 'r-', label='Max variance line')
        ax.plot(p1s, fit, 'k--', label='Max variance fit')
        bar.ax.set_ylabel('On fraction', rotation=90)
        ax.set_xlabel('p2')
        ax.set_ylabel('p1')
        ax.set_title('Phase Diagram for Varying p1 and p2')
        plt.legend()
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
    
    simType = int(input('Collect data or visualize [0/1]: '))
    dimVal = int(input('Dimension of the grid: '))
    
    if simType == 1:
        
        p1Val = float(input('Value of p1: '))
        p2Val = float(input('Value of p2: '))
        
        field = Simulation(dim=dimVal, p1=p1Val, p2=p2Val)
        anim = Visualizer(field)
        anim.run() # add argument save=True to save a .gif
        
    elif simType == 0:
        
        field = Simulation(dim=dimVal)
        field.phaseDiagram()

    else:
        raise ValueError('Usage [0/1]')
        
main()
