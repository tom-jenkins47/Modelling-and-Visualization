import random 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from scipy import stats
import os
plt.style.use('ggplot')



class Simulation(object):
    
    def __init__(self, dim, initializer='random'):
        
        # Initializer - if no state initializer selected, default to random
        
        self.dim = dim
        self.init = initializer
        self.lattice = None
        
        if self.init == 'random':
            self.useRandom()
            
        elif self.init == 'absorbing':
            self.useAbsorbing()
        
        elif self.init == 'glider':
            self.useGlider()
            
        elif self.init == 'blinker':
            self.useBlinker()
            
        elif self.init == 'beehive':
            self.useBeehive()
            
        else:
            raise ValueError('Initializer usage [random/absorbing/glider/blinker/beehive]')
            
        self.activity = []
                    
    def useRandom(self):
        
        # Initializes a random lattice where each point is assigned a value of 0 (dead) or 1 (alive)
        
        self.lattice = np.random.rand(self.dim, self.dim)
       
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                
                if self.lattice[i, j] < 0.5:
                    self.lattice[i ,j] = 0 
                else:
                    self.lattice[i, j] = 1
                    
    def useAbsorbing(self):
        
        # Absorbing initializer, pretty useless if I'm honest
        
        self.lattice = np.zeros((self.dim, self.dim))
        
        i = int(np.random.uniform()*self.dim)
        j = int(np.random.uniform()*self.dim)

        self.lattice[i, j] = 1
    
    def useGlider(self):
        
        # Glider initializer, will display in top left corner of animation and move diagonally
        
        self.lattice = np.zeros((self.dim, self.dim))
        
        self.lattice[5, 6] = 1
        self.lattice[7, 5] = 1
        self.lattice[6, 7] = 1
        self.lattice[7, 6] = 1
        self.lattice[7, 7] = 1
        
    def useBlinker(self):
        
        # Initializes a blinker at a random point in the lattice
        
        self.lattice = np.zeros((self.dim, self.dim))
        
        selectionRange = np.arange(5, self.dim-5)
        i = random.choice(selectionRange)
        
        self.lattice[i-1, i] = 1
        self.lattice[i, i] = 1
        self.lattice[i+1, i] = 1

    def useBeehive(self):
        
        # Initializes a beehive at a random point in the lattice
        
        self.lattice = np.zeros((self.dim, self.dim))
        
        selectionRange = np.arange(5, self.dim-5)
        i = random.choice(selectionRange)
        
        self.lattice[i-1, i] = 1 
        self.lattice[i, i+1] = 1 
        self.lattice[i+1, i+1] = 1
        self.lattice[i, i-1] = 1
        self.lattice[i+1, i-1] = 1
        self.lattice[i+2, i] = 1
        
        
    def countActivity(self):
        
        # Returns activity of lattice and also appends to a rolling list
        
        self.activity.append(np.sum(self.lattice))
        
        return np.sum(self.lattice)
       
        
    def update(self):
        
        # Updates the state of the lattice based on the conditions of the game
        
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
                    if neighbourStatesSum == 2 or neighbourStatesSum == 3:  
                        updatedLattice[i, j] = 1
                        
                    else:
                        updatedLattice[i, j] = 0
                        
                else:
                    if neighbourStatesSum == 3:
                        updatedLattice[i, j] = 1
                    
        self.lattice = updatedLattice
        
        
    def COM(self):
        
        # Finds the centre of mass of the alive cells in the lattice
        
        coords = np.argwhere(self.lattice==1)
        
        xCOM = int(np.average(coords[0]))
        yCOM = int(np.average(coords[1]))
        
        return [xCOM, yCOM]
    
    @staticmethod
    def gliderVelocity(time, position):
        
        # Static method to compute a velocity from time and position lists
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(time, position)
        fit = slope * np.array(time) + intercept
        
        return slope, fit
        
class Animation(object):
    
    def __init__(self, dim, init):
        
        # Set up a simulation to be animated
        
        self.sim = Simulation(dim, initializer=init)
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.sim.lattice, cmap='gray')
        self.ani = None
        
    def run(self):
        
        # Run the animation, updating every 25ms

        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=25, blit=True)
        plt.show()

    def animate(self, frames):
        
        # Animation function
        
        self.sim.update()
        self.plot.set_data(self.sim.lattice)
    
        return (self.plot,)
    
    
class DataCollection(object):
    
    def __init__(self, dim):
        
        # Initializer for data collection
        
        self.dim = dim
        self.sim = Simulation(self.dim)
        
    def equilibrationTime(self):
        
        # Determines equilibration time
        
        time = 0  
        unique = set()
        prevUnique = set()  
        prevAverage = 0  
        
        average = self.sim.countActivity()
        
        # Loops until activity has flatlined
        
        while prevUnique != unique or prevAverage != average:
            
            prevUnique = unique
            prevAverage = average
        
            for _ in range(10):
                
                time += 1
                self.sim.update()
                self.sim.countActivity()
        
            prevIter = self.sim.activity[-10:]
            unique = set(prevIter)  
            average = sum(unique)/len(unique)
        
        return time
    
    def calcGliderVelocity(self):
        
        # Calculates glider velocity
        
        self.sim = Simulation(self.dim, initializer='glider')
        
        tList = []
        positionList = []
        
        t = 0
        vectorDistance = lambda x,y: np.hypot(x, y)
        
        while t<200:
            
            self.sim.update()
            COM = self.sim.COM()
            
            # Only append to time list if glider is not at boundary
            
            if 3<COM[0]<(self.sim.dim-3) and 3<COM[1]<(self.sim.dim-3):
                
                t+=1
                tList.append(t)
                positionList.append(vectorDistance(COM[0], COM[1]))
                
                fileExists = os.path.isfile('glider_data.csv')
                with open('glider_data.csv', 'a+') as f:
                    if not fileExists:
                        f.write('Time, Position\n')
                    f.write(f'{t}, {vectorDistance(COM[0], COM[1])}\n')
        
        cutoff = np.argmax(positionList)
        gliderVel, fit = self.sim.gliderVelocity(tList[:cutoff], positionList[:cutoff])
        
        return gliderVel, tList[:cutoff], positionList[:cutoff], fit[:cutoff] # units indices/timestep for velocity
        
    
    def equilibrationTimeExperiment(self):
        
        # Conducts 750 simulations and returns the list of equilibration times
        
        tList = []
        
        for i in range(750):
            
            self.sim = Simulation(self.dim)
            time = self.equilibrationTime()
            tList.append(time)
            
        return tList
    
    def plotGliderVelocity(self):
        
        gliderVel, gliderTimeList, gliderPositionList, fit = self.calcGliderVelocity()
       
        plt.scatter(gliderTimeList, gliderPositionList, marker='.')
        plt.plot(gliderTimeList, fit, label='linear fit', linestyle='--', color='k', alpha=0.5)
        plt.xlabel('timestep')
        plt.ylabel('index')
        plt.title('Glider Position vs. Timestep')
        plt.text(60, 15, f'glider velocity = {np.round(gliderVel, 3)} indices/timestep')
        plt.show()
        
    def plotEquilibriumTimes(self):
        
        equiTimeList = self.equilibrationTimeExperiment()
        
        plt.hist(equiTimeList, bins=50)
        plt.xlabel('timestep')
        plt.ylabel('counts')
        plt.title('Histogram of System Equilibration Time')
        plt.show()
        

def runExperiment():
    
    sim_dim = int(input('System size: '))
    sim_or_data = int(input('Visualize / collect data [0/1]: '))
    
    if sim_or_data == 0:
        
        sim_type = str(input('What simulation type? [random/absorbing/glider/blinker/beehive]: '))
        anim = Animation(sim_dim, sim_type)
        anim.run()
        
    elif sim_or_data == 1:
        
        data_coll = DataCollection(sim_dim)
        data_type = int(input('Glider velocity / equilibration times [0/1]: '))
        
        if data_type == 0:
            data_coll.plotGliderVelocity()
            
        elif data_type == 1:
            data_coll.plotEquilibriumTimes()
            
        else:
            raise ValueError('Usage [0/1]')
            
    else:
        raise ValueError('Usage [0/1]')
        
        
runExperiment()

    


        
        
    
        
        
        
             

    
    
    
    
                    
                    
        
        
        
        
