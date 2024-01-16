import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
import copy
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import quad
from scipy.misc import derivative
import random

def initialstate(N):   
    '''
    Generates a random spin configuration for initial condition for 2D lattice
    N: number
    '''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def cold_start(N):
    lattice = random.choice([-1,1]) * np.ones((N,N))
    return lattice

def mcmove(config, beta):
    '''
    Moetropolis algorithm for 2D Ising model: nearest neighbours are able to interact with periodic boundary
    config: input configuration
    beta: inverse temperature
    '''
    N = len(config)

    for i in range(N):
        for j in range(N):
                #a=i
                #b=j
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb #Enegy difference between the new and old configurations (E1-E0)
                
                if cost < 0: #or rand() < np.exp(-cost*beta): #flip if energeticallaly more optimal
                    s *= -1
                elif rand() < np.exp(-cost*beta): #flip if energeticallaly non-optimal with the exp. prob.
                    s *= -1
                config[a, b] = s
    return config

def calcEnergy(config):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    
    N = len(config)

    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
            #print(-nb*S)
    return energy/2.  # double counting is eliminated

def calcMag(config):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(config)
    return mag


def generating_samples(N = 40, eqit = 2^50, Temp_lower= .25, Temp_upper = 4, Temp_number = 16, num_samp = 100):
    """
    N: sample size
    eqit: iteration until equilibrium
    Temp_lower: lower limit of temp
    Temp_upper: upper limit of temp
    Temp_number: temp numbers
    """
    
    All_Temp = {}
    for Temp in np.linspace(Temp_lower,Temp_upper,Temp_number):
        CO= []
        print(f"-------Temp {Temp}-------")
        #print(int(4*np.round(Temp,4)*100+1))
        #np.random.seed(int(4*np.round(Temp,4)*100+1))
        for i in range(num_samp):            
            np.random.seed(int((i+15)*np.round(Temp,4)*100))
            #config = initialstate(N)         # initial
            config = cold_start(N)
            for j in range(eqit):       # equilibrate
                np.random.seed(int((i+16)*(j+17)*np.round(Temp,4)*100+1))
                mcmove(config, 1/Temp)
            #print("magnetization:", sum(sum(config))/(N*N))
            CO.append(config)
        with open(f'Destinationfolder\ising_{Temp}.npy', 'wb') as f:
            np.save(f, CO)
        All_Temp[f"Temp {Temp}"] = np.asanyarray(CO)

    return All_Temp

if __name__ == "__main__":

    Ising_lattices = generating_samples()
    

    """
    print(Ising_lattices.keys())

    from mpl_toolkits.axes_grid1 import ImageGrid
    
    for i in Ising_lattices.keys():
        fig = plt.figure(figsize=(15., 15.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(2, 10),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        L = Ising_lattices[i]
        for ax, im in zip(grid, L.astype(int)):
            # Iterating over the grid returns the Axes.
            ax.imshow(im+.000000001)
    
    plt.show()
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print(Ising_lattices["Temp 0.25"][0:8])
    """
   
