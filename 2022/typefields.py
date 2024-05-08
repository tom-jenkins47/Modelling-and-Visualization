"""
2022 Past Paper

Using an appropriate finite difference scheme, with periodic boundary conditions in
space, write a Python code to solve Eqs. (1) on a 50 × 50 grid, subject to the initial
condition that a, b and c are all equal to a different random number, between 0 and
1/3, at each grid point. For the visualisation, define a type field τ for each point in
the grid, which is equal to 1 (e.g., shown in red), 2 (e.g., shown in green), 3 (e.g.,
shown in blue) or 0 (e.g., shown in gray), if the maximum field at that point is a,
b, c, or (1 − a − b − c) respectively. Your code should allow you to display the type
field τ in real time as it is running.


"""

from numba import njit 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
plt.style.use('ggplot')
from tqdm import tqdm
import random
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
rng = np.random.default_rng(42)


class TypeFields:
    
    def __init__(self, N=50, D=1.0, q=1.0, p=0.5, dx=1, dt=0.02):
        
        self.N = N
        self.D = D
        self.q = q
        self.p = p
        self.dx = dx
        self.dt = dt
        
        self.a = rng.random(size=(self.N, self.N)) / 3
        self.b = rng.random(size=(self.N, self.N)) / 3
        self.c = rng.random(size=(self.N, self.N)) / 3
        
        self.abcMinus = 1 - self.a - self.b - self.c
        
        self.typeField = np.empty((N, N))
        
        self.t = -1 
        
        
    def calculateTypeField(self):
        
        maxField = np.maximum(np.maximum(np.maximum(self.a, self.b), self.c), self.abcMinus)
        self.typeField[maxField == self.abcMinus] = 0
        self.typeField[maxField == self.a] = 1
        self.typeField[maxField == self.b] = 2
        self.typeField[maxField == self.c] = 3
  
        
    def laplacian(self, grid):
        
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx **2)
    
    def update(self):
        
        da = self.D * self.laplacian(self.a) + self.q*self.a*self.abcMinus - self.p*self.a*self.c
        db = self.D * self.laplacian(self.b) + self.q*self.b*self.abcMinus - self.p*self.a*self.b
        dc = self.D * self.laplacian(self.c) + self.q*self.c*self.abcMinus - self.p*self.b*self.c
        
        self.a += self.dt * da
        self.b += self.dt * db
        self.c += self.dt * dc
        
        self.abcMinus = 1 - self.a - self.b - self.c
        
        self.calculateTypeField()
        
        return self.typeField
    
    
    def fieldFractions(self):
        
        times = np.arange(0, 20000)
        
        aFrac = []
        bFrac = []
        cFrac = []
        
        for _ in range(len(times)):

            self.update()
            aFrac.append(len(self.typeField[self.typeField == 1]) / self.N**2)
            bFrac.append(len(self.typeField[self.typeField == 2]) / self.N**2)
            cFrac.append(len(self.typeField[self.typeField == 3]) / self.N**2)
            
        plt.plot(times*self.dt, aFrac)
        plt.xlabel('time')
        plt.ylabel('a fraction')
        plt.title('a Fraction vs. Time')
        plt.show()
        
        plt.plot(times*self.dt, bFrac)
        plt.xlabel('time')
        plt.ylabel('b fraction')
        plt.title('b Fraction vs. Time')
        plt.show()
        
        plt.plot(times*self.dt, cFrac)
        plt.xlabel('time')
        plt.ylabel('b fraction')
        plt.title('b Fraction vs. Time')
        plt.show()
        
        plt.plot(times*self.dt, aFrac, label = 'a')
        plt.plot(times*self.dt, bFrac, label = 'b')
        plt.plot(times*self.dt, cFrac, label = 'c')
        plt.xlabel('time')
        plt.ylabel('fraction')
        plt.title('Fraction vs. Time')
        plt.legend()
        plt.show()
        
    def adsorption(self):
        
        times = np.arange(0, 50000)

        
        executionTimes = []
        
        for run in range(10):
            
            self.a = np.random.random(size=(self.N, self.N)) / 3
            self.b = np.random.random(size=(self.N, self.N)) / 3
            self.c = np.random.random(size=(self.N, self.N)) / 3
        
            for time in times:
                
                self.update()
                aFrac = len(self.typeField[self.typeField == 1]) / self.N**2
                bFrac = len(self.typeField[self.typeField == 2]) / self.N**2
                cFrac = len(self.typeField[self.typeField == 3]) / self.N**2
                
                if aFrac == 1.0 or bFrac == 1.0 or cFrac == 1.0:
                    executionTimes.append(time*self.dt)
                    print(time*self.dt)
                    break
                
        mean = np.average(executionTimes)
        error = np.std(executionTimes)/np.sqrt(10)
        
        print(f'Mean: {mean}')
        print(f'Error: {error}')
        
    def aFieldAnalysis(self):
        
        i = random.randint(0, 50)
        j = random.randint(0, 50)
        
        k = random.randint(0, 50)
        m = random.randint(0, 50)
        
        point1 = []
        point2 = []
        
        timeRange = np.arange(0, 20000)*self.dt
           
        for time in timeRange:
            
            self.update()
            point1.append(self.a[i, j])
            point2.append(self.a[k, m])
            
        plt.plot(timeRange, point1, label = 'point 1', linestyle='--', color='r')
        plt.plot(timeRange, point2, label = 'point 2', linestyle='-.', color='k')
        plt.xlabel('time')
        plt.ylabel('a value')
        plt.title('a Value vs. Time')
        plt.legend()
        plt.show()
        
        def objective(t, amplitude, frequency, phase, offset):
            
            return amplitude * np.sin(frequency * t + phase) + offset
        
        guess = [0.5, 0.2, 0, 0.3]
        
        popt1, _ = curve_fit(objective, timeRange, point1, p0=guess)
        popt2, _ = curve_fit(objective, timeRange, point2, p0=guess)
        
        period1 = 2 * np.pi / popt1[1]
        period2 = 2 * np.pi / popt2[1]
        
        print(f'point 1 period: {period1}')
        print(f'point 2 period: {period2}')
        
        point1fit = objective(timeRange, popt1[0], popt1[1], popt1[2], popt1[3])
        
        plt.scatter(timeRange, point1, marker='.', color='k')
        plt.plot(timeRange, point1fit, color='r')
        plt.title('Fit Test')
        plt.show()
        
    def radialProbability(self):
        
        timeSteps = 1000
        snapshots = 10
        
        plt.figure(figsize=(10, 7))  # Adjust the size as needed

        for D in [0.3, 0.4, 0.5]:
            
            self.D = D
            self.a = rng.random(size=(self.N, self.N)) / 3
            self.b = rng.random(size=(self.N, self.N)) / 3
            self.c = rng.random(size=(self.N, self.N)) / 3

            # Collect probabilities at different time steps
            allTypeProbabilities = {r: [] for r in range(self.N // 2 + 1)}

            # Run the simulation and take 'snapshots' at multiple time steps
            for step in range(1, timeSteps + 1):
                self.update()

                # Calculate the probabilities at specific time steps ('snapshots')
                if step % (timeSteps // snapshots) == 0 or step == timeSteps:
                    self.calculateTypeField()
    
                    typeProbabilities = {r: [] for r in range(self.N // 2 + 1)}
    
                    for i in range(self.N):
                        for j in range(self.N):
                            for r in range(self.N // 2 + 1):  
                                if j + r < self.N:  
                                    sameType = (self.typeField[i, j] == self.typeField[i, j + r])
                                    typeProbabilities[r].append(int(sameType))

                    # Average the probabilities for this snapshot
                    for r in range(self.N // 2 + 1):
                        allTypeProbabilities[r].append(np.mean(typeProbabilities[r]))
            
            # Calculate average probability for each distance across all snapshots
            averageProbabilities = {r: np.mean(probs) for r, probs in allTypeProbabilities.items()}
            
            plt.plot(list(averageProbabilities.keys()), list(averageProbabilities.values()), marker='.', label=f'D={D}')

        plt.xlabel('r')
        plt.ylabel('same type probability')
        plt.title('Cell Type Correlation')
        plt.legend()
        plt.show()
        
        return averageProbabilities
        
class Visualizer:
    
    def __init__(self, field):
        
        self.field = field
        self.step = 0
        self.fig, self.ax = plt.subplots()        
        self.plot = self.ax.imshow(self.field.typeField, cmap = 'PiYG', vmin=0, vmax=3)
        plt.colorbar(self.plot)
        self.ani = None
        
    def run(self):
        
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=30, frames=1000, repeat=False)
        plt.show()
        
    def animate(self, frame):
               
        self.field.update()
        self.step += 1
        self.plot.set_data(self.field.typeField)
        self.ax.set_title(f"Step: {self.step}")
         
        return (self.plot,)
        
field = TypeFields(D=0.5, p=2.5)
field = TypeFields()
#field.adsorption()
#field.aFieldAnalysis()
#field.radialProbability()
anim = Visualizer(field)
anim.run()