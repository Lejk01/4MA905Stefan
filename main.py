import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import (construct_convection_matrix,
                       construct_mass_matrix,
                       construct_stiffness_matrix,
                       analytical_m, 
                       analytical_s,
                       lambdeq
                       )
from scipy.optimize import fsolve

# Water–ice constants.
k_L = 0.6         # W/(mK)
rho_L = 1000      # kg/m^3
c_L = 4200        # J/(kgK)
m_L = 334e3       # J/kg
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 10            # temperature difference driving phase change [K]

lambda_guess = 0.5 # Shit guess probably?

lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

# Time horizon (seconds).
HORIZON = 600     # 10 minutes
time = np.linspace(1e-6, HORIZON, 300)  # 300 time steps

space = np.linspace(0, 0.05, 300) 

s = analytical_s(alpha_L, lambd, time)

T, X = np.meshgrid(time, space, indexing="ij")
S = analytical_s(alpha_L, lambd, T)
M = analytical_m(m_L, alpha_L, lambd, X, T)

M[X > S] = np.nan

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