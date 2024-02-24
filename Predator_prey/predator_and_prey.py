#import packages 

import numpy as np
import matplotlib.pyplot as plt

# Define parameters of the predator-prey dynamical system
a = 0.07  # prey growth rate
b = 0.08 # predator death rate
c = 0.05 # predator growth rate
d = .45 # prey death rate

# Create time vector
t = np.arange(0, 200, 0.1)

# Define the initial conditions
prey_0 = 13  # initial number of prey
pred_0 = 5  # initial number of predators

# Define the functions that govern the dynamical system
def prey_growth(prey, pred):
    return a*prey - b*prey*pred

def pred_growth(prey, pred):
    return c*prey*pred - d*pred

# Define the arrays to store the prey and predator values over time
prey_vals = []
pred_vals = []

# Set the initial values
prey_vals.append(prey_0)
pred_vals.append(pred_0)

# Iterate over the time vector to calculate the values of prey and predator
for i in range(1,len(t)):
    prey_curr = prey_vals[i-1]
    pred_curr = pred_vals[i-1]
    prey_next = prey_curr + prey_growth(prey_curr, pred_curr)*0.05
    pred_next = pred_curr + pred_growth(prey_curr, pred_curr)*np.random.rand()
    prey_vals.append(prey_next)
    pred_vals.append(pred_next)

# Plot the time series of prey and predator populations
plt.plot(t, prey_vals, 'b', label='Prey')
plt.plot(t, pred_vals, 'r', label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Dynamics')
plt.legend()
plt.show()

# Plot the phase space of the predator-prey dynamical system
plt.plot(prey_vals, pred_vals)
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Space')
plt.show()

# Plot the critical points of the system
crit_prey = (d/c)*np.asarray(pred_vals)
crit_pred = (b/a)*np.asarray(prey_vals)

plt.plot(crit_prey, crit_pred)
plt.xlabel('Critical Prey')
plt.ylabel('Critical Predator')
plt.title('Critical Points')
plt.show()