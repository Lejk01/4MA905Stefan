import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import (construct_convection_matrix,
                       construct_mass_matrix,
                       construct_stiffness_matrix,
                       analytical_m, 
                       analytical_s,
                       lambdeq,
                       phi_j_dt
                       )
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

# Water–ice constants.
k_L = 1           # W/(mK)
rho_L = 1         # kg/m^3
c_L = 1           # J/(kgK)
m_L = 1
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 1             # temperature difference driving phase change [K]
L = 2
'''
k_L = 0.5664         # W/(mK)
rho_L = 1000         # kg/m^3
c_L = 4189.9         # J/(kgK)
m_L = 298.15
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 334e3             # temperature difference driving phase change [K]
L = 0.1
'''
lambda_guess = 0.5
lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

# Time horizon (seconds).
HORIZON = 1     # 5 minutes
TIME_POINTS = 4000
time = np.linspace(1e-6, HORIZON, TIME_POINTS)
dt = time[1] - time[0]

## Analytical solution ##
s_analytic = analytical_s(alpha_L, lambd, time, 0.02)

plt.figure(figsize=(8,4))
N_nodes_list = [25, 51]

# For convergence calculations
h_list = []
mse_s_list = []
mse_F_list = []

for N_nodes in N_nodes_list:
  nodes = np.linspace(0, 1, N_nodes)
  h_nodes = nodes[1] - nodes[0]

  ## Finite element method Uniform mesh - Start ##

  # Finest granulariy
  mass_matrix = construct_mass_matrix(N_nodes, h_nodes)
  stiffness_matrix = construct_stiffness_matrix(N_nodes, h_nodes)
  convection_matrix = construct_convection_matrix(N_nodes, h_nodes, nodes)

  ## Interior matrices (no boundaries) ##
  interior_idx = np.arange(1, N_nodes-1)
  mass_int = mass_matrix[np.ix_(interior_idx, interior_idx)]
  stiffness_int = stiffness_matrix[np.ix_(interior_idx, interior_idx)]
  convection_int = convection_matrix[np.ix_(interior_idx, interior_idx)]

  ## u(0) = m_L, u(1) = 0, in between does not matter. 
  u_vals = m_L * (1.0 - nodes)
  u_int = u_vals[interior_idx]

  # load vector / forcing function.
  load_vector = -m_L * h_nodes * nodes[interior_idx]

  ## F(xi, 0) = 0 -> v(xi, 0) = -u
  a_vh = - u_int.copy()

  ## F on the nodes, F = u + v
  F_h = np.zeros((len(time), N_nodes))
  F_h[0, :] = u_vals.copy()
  F_h[0, 1:-1] += a_vh

  ## interface s(t).
  s = np.zeros(len(time))
  s[0] = s_analytic[0]

  # Solve for each time point:
  for n in range(len(time)-1):
    # temp F at time n.
    F_nh = u_vals.copy()
    F_nh[1:-1] += a_vh

    # dF/dxi at 1, for Stefan condition.
    dFhdxi = 0
    for j in range(N_nodes):
      dFhdxi += F_nh[j] * phi_j_dt(j, h_nodes, 1, nodes)
    
    # Stefan condition.
    dshdt = -(k_L/(rho_L*l)) * (1/s[n]) * dFhdxi

    # Update s(t).
    s_next = s[n] + dt * dshdt
    s[n+1] = s_next

    # Finding the coefficients.
    LHSh = mass_int + dt * (alpha_L / s_next**2) * stiffness_int - dt * (dshdt/s_next) * convection_int
    RHSh = np.dot(mass_int, a_vh) + dt * (dshdt/s_next) * load_vector

    a_vh = np.linalg.solve(LHSh, RHSh)

    # Apply u everywhere then overwrite interior nodes with our approximation.
    F_h[n+1, :] = u_vals.copy()
    F_h[n+1, 1:-1] += a_vh

  # Compute MSE
  mse_s = mean_squared_error(s_analytic, s)
  print(f'Mean squared error for s(t) with {N_nodes} nodes: {mse_s:.4}')

  plt.plot(time, s, label=f"s (FEM, numeric, nodes={N_nodes})", linestyle="--")

  h_list.append(h_nodes)
  mse_s_list.append(mse_s)

plt.plot(time, s_analytic, label="s (analytical)")
plt.xlabel("Time [s]")
plt.ylabel("Interface s(t) [units of ξ]")
plt.legend()
plt.title("Interface evolution (ξ-domain FEM)")
plt.tight_layout()
plt.show()

# e(h) ~ C*h^p --- p = log(e(h1)/e(h2)) / log(h1/h2)
p = np.log(mse_s_list[0]/mse_s_list[1])/np.log(h_list[0]/h_list[1])
print(f'Order of convergence: {p:.4}')

# FEM in physical domain.
space = np.linspace(0, L, 1000)

m_xt = np.full((len(time), len(space)), np.nan)

T, X = np.meshgrid(time, space, indexing='ij')
S = analytical_s(alpha_L, lambd, T, s0=h_nodes)
M = analytical_m(m_L, alpha_L, lambd, X, T)
M[X > S] = np.nan

Mh  = np.full_like(M, np.nan)

for t in range(len(time)):
  Mh[t, space <= s[t]] = analytical_m(m_L, alpha_L, lambd, space[space <= s[t]], time[t])
    
  xi_h  = space / s[t]

  interp_h  = interp1d(nodes,  F_h[t, :],  kind='linear', bounds_error=False, fill_value=np.nan)

  # evaluate at the xi coordinates
  vals_h  = interp_h(xi_h)

  # assign only where xi <= 1 (physical domain of that FEM)
  mask1 = (xi_h  <= 1.0)

  m_xt[t,  mask1] = vals_h[mask1]

Mh_flat = Mh.flatten()
m_xt_h_flat = m_xt.flatten()

mask = ~(np.isnan(Mh_flat) | np.isnan(m_xt_h_flat))

mse_F  = mean_squared_error(Mh_flat[mask], m_xt_h_flat[mask])
print(f'Mean squared error for F(x,t) with {N_nodes} nodes: {mse_F:.4}')

# Plot FEM solution (h)
plt.figure(figsize=(10,6))
pcm = plt.pcolormesh(time, space, m_xt.T, shading='auto')
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.title("F(x,t) - FEM solution")
plt.colorbar(pcm, label="F")
plt.tight_layout()
plt.show()