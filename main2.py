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
                       finite_diff,
                       phi_j_dt
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
s_analytic = analytical_s(alpha_L, lambd, time)

T, X = np.meshgrid(time, space, indexing="ij")
S = analytical_s(alpha_L, lambd, T)
M = analytical_m(m_L, alpha_L, lambd, X, T)

M[X > S] = np.nan

### Analytical solution - End ###

## Finite element method Uniform mesh - Start ##

mass_matrix = construct_mass_matrix(N_nodes, h_nodes)
stiffness_matrix = construct_stiffness_matrix(N_nodes, h_nodes)
convection_matrix = construct_convection_matrix(N_nodes, h_nodes, nodes)

## Interior matrices (no boundaries) ##
interior_idx = np.arange(1, N_nodes-1)
mass_int = mass_matrix[np.ix_(interior_idx, interior_idx)]
stiffness_int = stiffness_matrix[np.ix_(interior_idx, interior_idx)]
convection_int = convection_matrix[np.ix_(interior_idx, interior_idx)]

## u(0) = m_L, u(1) = 0
u_vals = m_L * (1.0 - nodes)
u_int = u_vals[interior_idx]

## F(xi, 0) = 0 -> v(xi, 0) = -u
a_v = - u_int.copy()

## F on the nodes, F = u + v
F = np.zeros((len(time), N_nodes))
F[0, :] = u_vals.copy()
F[0, 1:-1] += a_v

## interface s(t)
s = np.zeros(len(time))
s[0] = h_nodes

# Solve for each time point:
for n in range(len(time)-1):
  # F at time n
  F_n = u_vals.copy()
  F_n[1:-1] += a_v

  # dF/dxi at 1, for Stefan condition
  dFdxi = 0
  for j in range(N_nodes):
    dFdxi += F_n[j] * phi_j_dt(j, h_nodes, 1, nodes)
  
  # Stefan condition
  dsdt = -(k_L/(rho_L*l)) * (1/s[n]) * dFdxi

  # Update s(t)
  s_next = s[n] + dt * dsdt
  s[n+1] = s_next


  # Finding the coefficients
  LHS = mass_int + dt * (alpha_L / s_next**2) * stiffness_int
  RHS = np.dot(mass_int, a_v) + dt * (dsdt/s[n]) * np.dot(convection_int, u_int + a_v)

  a_v = np.linalg.solve(LHS, RHS)

  F[n+1, :] = u_vals.copy()
  F[n+1, 1:-1] += a_v

# Plots
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8,4))
plt.plot(time, s, label="s (FEM, numeric)", linestyle="--")
plt.plot(time, s_analytic, label="s (analytical)")
plt.xlabel("Time [s]")
plt.ylabel("Interface s(t) [units of xi]")
plt.legend()
plt.title("Interface evolution (xi-domain FEM using your matrices)")

plt.figure(figsize=(10,6))
pcm = plt.pcolormesh(time, nodes, F.T, shading='auto')
plt.xlabel("Time [s]")
plt.ylabel("xi")
plt.title("F(ξ,t) - FEM solution")
plt.colorbar(pcm, label="F")
plt.show()