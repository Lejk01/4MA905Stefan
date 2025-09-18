import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import (construct_convection_matrix,
                       construct_mass_matrix,
                       construct_stiffness_matrix,
                       analytical_m, 
                       analytical_s,
                       lambdeq,
                       phi_j,
                       phi_j_vectorized,
                       finite_diff_xi,
                       finite_diff
                       )
from scipy.optimize import fsolve
from functools import partial

# Water–ice constants.
k_L = 0.5664         # W/(mK)
rho_L = 1000        # kg/m^3
c_L = 4189.9        # J/(kgK)
m_L = 298.15
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 334e3             # temperature difference driving phase change [K]
N_nodes = 300
L = 0.1
lambda_guess = 0.5

lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]
nodes = np.linspace(0, 1, N_nodes)
h_nodes = nodes[1] - nodes[0]

# Time horizon (seconds).
HORIZON = 600     # 5 minutes
time = np.linspace(1e-6, HORIZON, 1000)
dt = time[1] - time[0]
space = np.linspace(0, L, 1000) 
h_space = space[1] - space[0]

### Analytical solution - Start ###
s = analytical_s(alpha_L, lambd, time)

T, X = np.meshgrid(time, space, indexing="ij")
S = analytical_s(alpha_L, lambd, T)
M = analytical_m(m_L, alpha_L, lambd, X, T)

M[X > S] = np.nan

### Analytical solution - End ###

## Finite element method Uniform mesh - Start ##

mass_matrix = construct_mass_matrix(N_nodes, h_nodes)
stiffnes_matrix = construct_stiffness_matrix(N_nodes, h_nodes)
convection_matrix = construct_convection_matrix(N_nodes, h_nodes, nodes)

heat_distribution_x = np.zeros((len(time), len(space))) # m(x,t)
heat_distribution_x[:, 0] = m_L # m_L at the intial point for all times. 

initial_heat_distribution_xi = np.zeros((len(time), N_nodes)) # mtilde(xi,t)
initial_heat_distribution_xi[:, 0] = m_L # m_L at the intial point for all times. 
interface = np.zeros(len(time)) # this is s(t), so the position of the interface at each time point.
interface[0] = h_nodes # The position of the interface is close to 0 at t=0. This is s(0) = s_0 >= 0

# a coefficients
a = np.zeros(N_nodes)
a[0] = m_L

# initial condition:
initial_condition = np.zeros(N_nodes)
initial_condition[0] = m_L

# Solve problem for each time point:
for n in range(len(time)-1):
  dmdxi = finite_diff(heat_distribution_x, interface, n, h_space, space)
  s_next = interface[n] - dmdxi * dt * k_L/(rho_L*l*interface[n])
  interface[n+1] = s_next
  a = np.linalg.solve(mass_matrix/h_nodes - (1/s_next)*(s_next - interface[n] )/h_nodes * convection_matrix + alpha_L / s_next**2 * stiffnes_matrix, np.dot(a, mass_matrix/h_nodes) -  initial_condition)
  
  phis_evaluated = np.zeros((N_nodes, len(space)))

  for i in range(N_nodes): # For each function phi.
    for j, x in enumerate(space): # For each point in space.
      phis_evaluated[i, j] = phi_j(i, h_nodes, x, nodes) # evaluate every phi on every point in space. 


  for idx in range(len(space)):
    heat_distribution_x[n, idx] = np.dot(a, phis_evaluated[:, idx])
  
  if n % 100 == 0:
    print("At n=", n, " Interface at x=", interface[interface > 0][-1])
    print("Len interface=", len(interface[interface > 0]))
    print("heat ", heat_distribution_x[n, 5])

## Finite element method Uniform mesh - End ##

### Plots - Start ###

# Plot interface s(t)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.lineplot(x=time, y=s)
plt.xlabel("Time [s]")
plt.ylabel("Interface position s(t) [m]")
plt.title("Stefan problem (water–ice): interface evolution")
plt.show()

# Plot heat distribution
plt.figure(figsize=(10, 6))

# Swap axes: X=time, Y=space, transpose M
pcm = plt.pcolormesh(time, space, M.T, shading="auto", cmap="magma")

plt.colorbar(pcm, label="m(x,t) [J/kg]")
plt.xlabel("Time t [s]")
plt.ylabel("Space x [m]")
plt.title("Heat distribution m(x,t) (time on x-axis)")
plt.show()

### Plots - End ###