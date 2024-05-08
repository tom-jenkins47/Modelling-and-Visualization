import random 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from scipy import stats
import os
plt.style.use('ggplot')


class Simulation(object):
    
    def __init__(self, dim, pi, pr, ps):
        
        # Initializes system size and relevant probabilities
    
        self.dim = dim
        self.pi = pi
        self.pr = pr
        self.ps = ps
        
        self.useRandom()
        
    def useRandom(self):
        
        # Initializes a random state where each cell can either be infected, recovered or susceptible
        
        self.lattice = np.random.rand(self.dim, self.dim)
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if self.lattice[i, j] < 1/3:
                    self.lattice[i, j] = -1
                    
                elif self.lattice[i, j] < 2/3:
                    self.lattice[i, j] = 0
                    
                else:
                    self.lattice[i, j] = 1
                    
    def update(self):
        
        # Updates the state of the lattice, selecting points to update at random
        
        for _ in range(self.dim**2):
        
            i = int(np.random.uniform()*self.dim)
            j = int(np.random.uniform()*self.dim)
            
            d = self.dim
            
            if self.lattice[i, j] == -1:
              if random.random() < self.ps:
                  self.lattice[i, j] = 0
            
            elif self.lattice[i, j] == 0:
                
                N = self.lattice[i, (j+1)%d]
                E = self.lattice[(i+1)%d, j]
                S = self.lattice[i, (j-1)%d]    
                W = self.lattice[(i-1)%d, j]
                
                if 1 in set([N, E, S, W]):
                    if random.random() < self.pi:
                        self.lattice[i, j] = 1         
        
            elif self.lattice[i, j] == 1:
                if random.random() < self.pr:
                  self.lattice[i, j] = -1
            
            else:
                
                # This line is required for when immunity is implemented
                
                self.lattice[i, j] == self.lattice[i, j]
                
            
    def countInfected(self):
        
        # Counts the number of infected cells in the lattice
    
        coords = np.argwhere(self.lattice==1)
        return len(coords)

    
    def initializeImmune(self, pimm):
        
        # Initializes certain cells to be immune to infection
        
        self.useRandom()
        
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if random.random() < pimm:
                    self.lattice[i, j] = -2
                    
class Animation(object):
    
    def __init__(self, dim, pi, pr, ps, immune=False):
        
        # Sets up simulation to be animated
        
        self.sim = Simulation(dim, pi, pr, ps)
        
        if immune:
            pimm = float(input('Immune probability: '))
            self.sim.initializeImmune(pimm)
            
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.sim.lattice)
        self.ani = None
        
    def run(self):
        
        # Runs animation, updating every 5ms

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=5, blit=True)
        plt.show()
        
    def animate(self, frame):
        
        # Animation function 
        
        self.sim.update()
        self.plot.set_data(self.sim.lattice)
        
        return (self.plot,)
    
    
class DataCollection(object):
    
    def __init__(self, dim):
        
        # Initializer for data collection
    
        self.dim = dim
        self.pr = 0.5
        
    def calcError(self, x):
        
        # Computes error using bootstrap resampling method
        
        errors = []
        
        for i in range(500):
            resamples = []
        
            for j in range(1000): 
                y = [random.choice(x) for item in x]                                            
                resamples.append(y)
                
            errors.append(np.var(resamples)/self.dim**2)
            
        return np.std(errors)
      
    def phaseDiagram(self):
        
        # Determines how the average infection number varies with both infection and suceptibility probabilities
        
        self.pi = np.arange(0, 1.05, 0.05)
        self.ps = np.arange(0, 1.05, 0.05)
        
        averageInfections = []
        
        for i in range(len(self.pi)):
            for j in range(len(self.ps)):
                
                infectedData = []
                self.sim = Simulation(self.dim, self.pi[i], self.pr, self.ps[j])
    
                for n in range(1000):  
                    self.sim.update()
                    
                    if n > 100:
                        infectedData.append(self.sim.countInfected())
                        
                mean = np.mean(np.array(infectedData))
                averageInfections.append(mean)
                
                # Write data to outfile
        
                fileExists = os.path.isfile('infections_data.csv')
                with open('infections_data.csv', 'a+') as f:
                    if not fileExists:
                        f.write('Average Infections, Infection Prob., Susceptibility Prob.\n')
                    f.write(f'{mean}, {self.pi[i]}, {self.ps[j]}\n')
        
        return np.array(averageInfections), self.pi, self.ps

    
    def waveAnalysis(self):
        
        # Determines how the variance of the number of infected changes for a varying infection probability
        
        self.pi = np.arange(0.2, 0.5, 0.01)
        self.ps = 0.5
        
        infectedVar = []
        errors = []
        
        for i in range(len(self.pi)):
            
            infectedData = []
            
            self.sim = Simulation(self.dim, self.pi[i], self.pr, self.ps)

            for n in range(10000):
                self.sim.update()
                
                if n > 100:
                    infectedData.append(self.sim.countInfected())
                    
                if n%1000 == 0:
                    print(f'Cycle {n/100}% complete.')
                
            var = np.var(infectedData)/self.dim**2
            infectedVar.append(var)
            
            error = self.calcError(infectedData)
            errors.append(error)
            
            # Write data to outfile
        
            fileExists = os.path.isfile('wave_data.csv')
            with open('wave_data.csv', 'a+') as f:
                if not fileExists:
                    f.write('Infection Variance, Infection Prob., Error\n')
                f.write(f'{var}, {self.pi[i]}, {error}\n')
                
        return infectedVar, errors, self.pi
                
                
    def immunityAnalysis(self):
        
        # Determines how the number of infections responds to various immune fractions in the population
        
        fracImmune = np.linspace(0, 1, 50)
        averageInfections = []
        
        for frac in fracImmune:
            
            infectedData = []  
            self.sim = Simulation(self.dim, 0.5, 0.5, 0.5)
            self.sim.initializeImmune(frac)
            
            for n in range(2500):
                self.sim.update()
                
            if n > 100:
                infectedData.append(self.sim.countInfected())
                
            mean = np.mean(np.array(infectedData))
            averageInfections.append(mean)
            
            # Write data to outfile 
            
            fileExists = os.path.isfile('immunity_data.csv')
            with open('immunity_data.csv', 'a+') as f:
                if not fileExists:
                    f.write('Immune Fraction, Average Infections\n')
                f.write(f'{np.round(frac, 2)}, {mean}\n')
                
        return fracImmune, averageInfections
    
    def plotPhaseDiagram(self):
        
        averageInfections, pi, ps = self.phaseDiagram()
        averageInfections = averageInfections.reshape(len(pi), len(ps))
        
        fig, ax = plt.subplots()
        image = ax.imshow(averageInfections, extent=(pi.min(), pi.max(), ps.max(), ps.min()))
        bar = ax.figure.colorbar(image)
        bar.ax.set_ylabel('average number infected', rotation=90)
        ax.set_xlabel('susceptibility probability')
        ax.set_ylabel('infection probability')
        ax.set_title('Phase Contour Plot (p$_{2}$ = 0.5)')
        plt.show()
        
    def plotWaves(self):
        
        infectedVar, errors, pi = self.waveAnalysis()
        
        plt.plot(pi, infectedVar)
        plt.errorbar(pi, infectedVar, yerr=errors, fmt=".", color='k')
        plt.title('Infection Variance Plot (p$_{2}$ = p$_{3}$ = 0.5)')
        plt.xlabel('infection probability')
        plt.ylabel('infection variance')
        plt.show()
        
    def plotImmunity(self):
        
        fracImmune, averageInfections = self.immunityAnalysis()
        
        plt.plot(fracImmune, averageInfections, color='k')
        plt.title('Average Number Infected vs. Immune Fraction')
        plt.xlabel('immune fraction')
        plt.ylabel('average number infected')
        plt.show()
           
    
def runExperiment():

    sim_dim = int(input('System size: '))
    sim_type = int(input('Choose probabilities / use set state / collect data [0/1/2]: '))

    if sim_type == 0:
        
        pi = float(input('Infection probability: '))
        pr = float(input('Recovery probability: '))
        ps = float(input('Susceptibility probability: '))
        
        anim = Animation(sim_dim, pi, pr, ps)
        anim.run()
        
    elif sim_type == 1:
        
        state = str(input('Choose state [absorbing/dynamic/cyclic/immune]: '))
        
        if state == 'absorbing':
            anim = Animation(sim_dim, 0.25, 0.7, 0.1)
            
        elif state == 'dynamic':
            anim = Animation(sim_dim, 0.5, 0.5, 0.5)
            
        elif state == 'cyclic':
            anim = Animation(sim_dim, 0.8, 0.08, 0.011)
        
        elif state == 'immune':
            anim = Animation(sim_dim, 0.5, 0.5, 0.5, immune=True)
            
        else:
            raise ValueError('Usage [absorbing/dynamic/cyclic/immune]')
            
        anim.run()
        
    elif sim_type == 2:
        
        data_coll = DataCollection(sim_dim)
        data = str(input('Choose data to collect [phase/waves/immunity]: '))
        
        if data == 'phase':
            data_coll.plotPhaseDiagram()
            
        elif data == 'waves':
            data_coll.plotWaves()
            
        elif data == 'immunity':
            data_coll.plotImmunity()
            
        else:
            raise ValueError('Usage [phase/waves/immunity]')
            
    else:
        raise ValueError('Usage [0/1/2]')
            
            
runExperiment()
        
       
        
        
        
                
                
            
        
         
 
        
        
        
        

