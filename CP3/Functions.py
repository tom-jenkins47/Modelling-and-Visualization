"""
Tom Jenkins CP3

Field Functions

Note: In order to massively speed up computations, the numba package is used.
Details can be found here: https://numba.readthedocs.io/en/stable/user/jit.html
To install numba: $ conda install numba
This package is a compiler for Python that works best on code that uses NumPy 
arrays and functions, and loops. It effectively compiles code in C where possible.
"""

from numba import njit 
import numpy as np

@njit
def Order(order, cp, N, M, dx, dt):
    
    """
    Updates the state of the system using the Euler algorithm.
    Params order and cp represent the order and chemical potential lattices.
    """
    
    update = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            
            east = cp[(i+1)%N, j] 
            west = cp[(i-1)%N, j]
            north = cp[i, (j+1)%N]
            south = cp[i, (j-1)%N] 
            
            update[i, j] = order[i, j] + M*dt/(dx**2) * (east+west+north+south-4*cp[i, j])
            
    order = update
    
    return order
    
@njit         
def ChemicalPotential(order, cp, N, a, k, dx, dt):
    
    """
    Updates the chemical potential, given the order.
    """
    
    update_cp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            
            east = order[(i+1)%N, j] 
            west = order[(i-1)%N, j]
            north = order[i, (j+1)%N]
            south = order[i, (j-1)%N] 
            
            update_cp[i, j] = -(a * order[i, j]) + a * (order[i, j]**3)\
                              - (k/dx**2)*(east+west+north+south-(4*order[i, j]))
    
    cp = update_cp
    
    return cp

@njit
def FreeEnergyDensity(order, N, a, k, dx):
    
    """
    Updates the free energy density, given the order.
    """
  
    update_f = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            
            east = order[(i+1)%N, j]
            west = order[(i-1)%N, j]
            north = order[i, (j+1)%N]
            south = order[i, (j-1)%N]

            dphi_dx = (east-west)/(2*dx)
            dphi_dy = (north-south)/(2*dx)
            
            update_f[i, j] = -(a/2) * (order[i, j]**2)\
                              + (a/4) * (order[i, j]**4)\
                              + (k/2)*((dphi_dx**2)+(dphi_dy**2))
            
    f = update_f
    density_f = np.sum(f) 
    
    return density_f


# =============================================================================


@njit
def EField(phi, N):
    
    """
    Computes the electric field, given the potential.
    """

    Ex = np.zeros((N,N,N))
    Ey = np.zeros((N,N,N))
    Ez = np.zeros((N,N,N))
    Etotal = np.zeros((N,N,N))
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                
                x = -(phi[i+1, j, k] - phi[i-1, j, k]) / 2
                y = -(phi[i, j+1, k] - phi[i, j-1, k]) / 2
                z = -(phi[i, j, k+1] - phi[i, j, k-1]) / 2
                 
                Eval = np.sqrt((x**2)+(y**2)+(z**2))
                
                normx = x/Eval
                normy = y/Eval
                normz = z/Eval
                
                Ex[i, j, k] = normx
                Ey[i, j, k] = normy
                Ez[i, j, k] = normz
                Etotal[i, j, k] = Eval   
        
    return Ex, Ey, Ez, Etotal  

@njit
def MField(Az, N):
    
    """
    Computes the magnetic field, given the vector potential.
    """

    cut = int(N/2)
    
    Mx = np.zeros((N,N,N))
    My = np.zeros((N,N,N))
    Mz = np.zeros((N,N,N))
    Mtotal = np.zeros((N,N,N))
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            
            x = -(Az[(i+1), j, cut] - Az[(i-1), j, cut]) / 2
            y = -(Az[i, (j+1), cut] - Az[i, (j-1), cut]) / 2
            z = - (Az[i, j, (cut+1)] - Az[i, j, (cut-1)]) / 2
            
            Mval = np.sqrt((x**2) + (y**2) + (z**2))  #Magnetic vector magnitude
        
            normx = x/Mval
            normy = y/Mval
            normz = z/Mval
            
            Mx[i, j, cut] = normx
            My[i, j, cut] = normy
            Mz[i, j, cut] = normz
            Mtotal[i, j, cut] = Mval   
        
    return Mx, My, Mz, Mtotal
            
@njit
def JacobiPotential(phi, rho, N):
    
    """
    Uses the Jacobi algorithm to update the potential.
    This works for both electric and magnetic fields.
    """
    
    updatePhi = np.zeros((N, N, N))
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                
                east = phi[(i+1), j, k] 
                west = phi[(i-1), j, k]
                north = phi[i, (j+1), k]
                south = phi[i, (j-1), k]
                front = phi[i, j, (k+1)]
                behind = phi[i, j, (k-1)]  
                
                sumPhiPoints = east + west + north + south + front + behind
    
                updatePhi[i, j, k] = (1/6) * (sumPhiPoints + rho[i, j, k])

    phi = updatePhi    
    
    return phi

@njit
def GaussPotential(phi, rho, N):
    
    """
    Uses the Gauss-Seidl algorithm to update the potential.
    This works for both electric and magnetic fields.
    """
    
    initPhi = np.sum(phi)
    updatePhi = JacobiPotential(phi, rho, N)
    finalPhi = np.sum(updatePhi)

    return updatePhi, initPhi, finalPhi

@njit
def collectData(phi, e, N):
    
    """
    Collects data on the vector distance magnitude, potential and field.
    The field is denoted as e, however this works for both electric and magnetic fields.
    """
    
    fieldData = np.empty((N-2)*(N-2), dtype=np.float64)
    potData = np.empty((N-2)*(N-2), dtype=np.float64)
    distance = np.empty((N-2)*(N-2), dtype=np.float64)
    center = N/2
    
    idx = 0
    centerDist = np.sqrt((np.arange(1, N-1) - center)**2 + (np.arange(1, N-1)[:, np.newaxis] - center)**2)
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            
            potData[idx] = phi[i, j, int(center)]
            fieldData[idx] = e[i, j, int(center)]
            distance[idx] = centerDist[i-1, j-1]
            idx += 1
                
    return distance, potData, fieldData       


def ChargeDistribution(N, system):
    
    """
    Initializes a charge distribution.
    The binary parameter 'system' dictates whether random or point charge distribution is initialized.
    """
    
    if system == 0:
        rho = np.random.uniform(size=(N, N, N))
        
    elif system == 1:
        rho = np.zeros((N, N, N))
        center = int(N/2)
        rho[center, center, center] = 1
        
    return rho

def CurrentDistribution(N):
    
    """
    Initializes a current distribution.
    This corresponds to a wire passing through the origin and running parallel to the z-axis.
    """
    
    current = np.zeros((N, N, N))
    center = int(N/2)
    
    for i in range(1, N-1):
        current[center, center, i] = 1
    
    return current

@njit
def SORPotential(phi, rho, N, w):
    
    """
    Uses successive over-relaxation to update the potential.
    The value of the parameter w can be between 1 and 2.
    """
    
    diffPhi = np.zeros((N,N,N))
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                  
                east = phi[(i+1), j, k] 
                west = phi[(i-1), j, k]
                north = phi[i, (j+1), k]
                south = phi[i, (j-1), k]
                front = phi[i, j, (k+1)]
                behind = phi[i, j, (k-1)]
                
                sumPhiPoints = east + west + north + south + front + behind
            
                rhoState = rho[i, j, k]  
                phiState = phi[i, j, k]
    
                phi[i, j, k] = ((1-w)*phiState) + (w*((1/6) * (sumPhiPoints+rhoState)))
                diffPhi[i, j, k] = phi[i, j, k]-phiState
   
    return phi, diffPhi

