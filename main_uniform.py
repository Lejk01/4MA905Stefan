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
k_L = 0.5664         # W/(mK)
rho_L = 1000         # kg/m^3
c_L = 4189.9         # J/(kgK)
m_L = 298.15
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 334e3             # temperature difference driving phase change [K]
N_nodes_h = 750
N_nodes_h2 = int(N_nodes_h/2)
L = 0.1
lambda_guess = 0.5

lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

nodes_h = np.linspace(0, 1, N_nodes_h)
h_nodes = nodes_h[1] - nodes_h[0]

nodes_h2 = np.linspace(0, 1, N_nodes_h2)
h2_nodes = nodes_h2[1] - nodes_h2[0]

# Time horizon (seconds).
HORIZON = 600     # 5 minutes
TIME_POINTS = 1000
time = np.linspace(1e-6, HORIZON, TIME_POINTS)
dt = time[1] - time[0]

space = np.linspace(0, L, 1000)
h_space = space[1] - space[0]

### Analytical solution - Start ###
s_analytic = analytical_s(alpha_L, lambd, time)
s_analytic[0] = h_nodes

### Analytical solution - End ###

## Finite element method Uniform mesh - Start ##

# Finest granulariy -> h
mass_matrix_h = construct_mass_matrix(N_nodes_h, h_nodes)
stiffness_matrix_h = construct_stiffness_matrix(N_nodes_h, h_nodes)
convection_matrix_h = construct_convection_matrix(N_nodes_h, h_nodes, nodes_h)

# h/2
mass_matrix_h2 = construct_mass_matrix(N_nodes_h2, h2_nodes)
stiffness_matrix_h2 = construct_stiffness_matrix(N_nodes_h2, h2_nodes)
convection_matrix_h2 = construct_convection_matrix(N_nodes_h2, h2_nodes, nodes_h2)

## Interior matrices (no boundaries) (h) ##
interior_idx_h = np.arange(1, N_nodes_h-1)
mass_int_h = mass_matrix_h[np.ix_(interior_idx_h, interior_idx_h)]
stiffness_int_h = stiffness_matrix_h[np.ix_(interior_idx_h, interior_idx_h)]
convection_int_h = convection_matrix_h[np.ix_(interior_idx_h, interior_idx_h)]

## Interior matrices (no boundaries) (h/2) ##
interior_idx_h2 = np.arange(1, N_nodes_h2-1)
mass_int_h2 = mass_matrix_h2[np.ix_(interior_idx_h2, interior_idx_h2)]
stiffness_int_h2 = stiffness_matrix_h2[np.ix_(interior_idx_h2, interior_idx_h2)]
convection_int_h2 = convection_matrix_h2[np.ix_(interior_idx_h2, interior_idx_h2)]

## u(0) = m_L, u(1) = 0, in between does not matter. 
u_vals_h = m_L * (1.0 - nodes_h)
u_int_h = u_vals_h[interior_idx_h]

u_vals_h2 = m_L * (1.0 - nodes_h2)
u_int_h2 = u_vals_h2[interior_idx_h2]

# load vector / forcing function.
load_vector_h = -m_L * h_nodes * nodes_h[interior_idx_h]
load_vector_h2 = -m_L * h2_nodes * nodes_h2[interior_idx_h2]

## F(xi, 0) = 0 -> v(xi, 0) = -u
a_vh = - u_int_h.copy()
a_vh2 = - u_int_h2.copy()


## F on the nodes, F = u + v
F_h = np.zeros((len(time), N_nodes_h))
F_h[0, :] = u_vals_h.copy()
F_h[0, 1:-1] += a_vh

F_h2 = np.zeros((len(time), N_nodes_h2))
F_h2[0, :] = u_vals_h2.copy()
F_h2[0, 1:-1] += a_vh2

## interface s(t).
s_h = np.zeros(len(time))
s_h[0] = h_nodes

s_h2 = np.zeros(len(time))
s_h2[0] = h2_nodes

# Solve for each time point:
for n in range(len(time)-1):
  # temp F at time n.
  F_nh = u_vals_h.copy()
  F_nh[1:-1] += a_vh

  F_nh2 = u_vals_h2.copy()
  F_nh2[1:-1] += a_vh2

  # dF/dxi at 1, for Stefan condition.
  dFhdxi = 0
  for j in range(N_nodes_h):
    dFhdxi += F_nh[j] * phi_j_dt(j, h_nodes, 1, nodes_h)
  
  dFh2dxi = 0
  for j in range(N_nodes_h2):
    dFh2dxi += F_nh2[j] * phi_j_dt(j, h2_nodes, 1, nodes_h2)

  # Stefan condition.
  dshdt = -(k_L/(rho_L*l)) * (1/s_h[n]) * dFhdxi
  dsh2dt = -(k_L/(rho_L*l)) * (1/s_h2[n]) * dFh2dxi

  # Update s(t).
  s_next_h = s_h[n] + dt * dshdt
  s_h[n+1] = s_next_h

  s_next_h2 = s_h2[n] + dt * dsh2dt
  s_h2[n+1] = s_next_h2

  # Finding the coefficients.
  LHSh = mass_int_h + dt * (alpha_L / s_next_h**2) * stiffness_int_h - dt * (dshdt/s_next_h) * convection_int_h
  RHSh = np.dot(mass_int_h, a_vh) + dt * (dshdt/s_next_h) * load_vector_h
  a_vh = np.linalg.solve(LHSh, RHSh)

  LHSh2 = mass_int_h2 + dt * (alpha_L / s_next_h2**2) * stiffness_int_h2 - dt * (dsh2dt/s_next_h2) * convection_int_h2
  RHSh2 = np.dot(mass_int_h2, a_vh2) + dt * (dsh2dt/s_next_h2) * load_vector_h2

  a_vh = np.linalg.solve(LHSh, RHSh)
  a_vh2 = np.linalg.solve(LHSh2, RHSh2)

  # Apply u everywhere then overwrite interior nodes with our approximation.
  F_h[n+1, :] = u_vals_h.copy()
  F_h[n+1, 1:-1] += a_vh

  F_h2[n+1, :] = u_vals_h2.copy()
  F_h2[n+1, 1:-1] += a_vh2

# Compute MSE
mse_h = mean_squared_error(s_analytic, s_h)
mse_h2 = mean_squared_error(s_analytic, s_h2)

order_of_convergence_s = mse_h2 / mse_h # h2 has larger distance between nodes than h since fewer points. 

plt.figure(figsize=(8,4))
plt.plot(time, s_h, label="s (FEM, numeric, grid h)", linestyle="--")
plt.plot(time, s_h2, label="s (FEM, numeric, grid h/2)", linestyle="--")
plt.plot(time, s_analytic, label="s (analytical)")

# Annotate MSE at the end of the curve
plt.text(time[-1], s_h[91],
         f"MSE for h = {mse_h:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")

plt.text(time[-1], s_h[50],
         f"MSE for 2*h = {mse_h2:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")

plt.text(time[-1], s_h[1],
         f"Order of convergence = {order_of_convergence_s:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")

plt.xlabel("Time [s]")
plt.ylabel("Interface s(t) [units of ξ]")
plt.legend()
plt.title("Interface evolution (ξ-domain FEM)")
plt.tight_layout()
plt.show()

# FEM in physical domain.
m_xt_h = np.full((len(time), len(space)), np.nan)
m_xt_h2 = np.full((len(time), len(space)), np.nan)

for t in range(len(time)):
  xi_vals_h = space / s_h[t]
  interp_h = np.interp(xi_vals_h, nodes_h, F_h[t, :])
  m_xt_h[t, :] = np.where(xi_vals_h <= 1.0, interp_h, np.nan)

  xi_vals_h2 = space / s_h2[t]
  interp_h2 = np.interp(xi_vals_h2, nodes_h2, F_h2[t, :])
  m_xt_h2[t, :] = np.where(xi_vals_h2 <= 1.0, interp_h2, np.nan)

# Analytical solutions
Th, Xh = np.meshgrid(time, xi_vals_h, indexing="ij")
Sh = analytical_s(alpha_L, lambd, Th)
Mh = analytical_m(m_L, alpha_L, lambd, Xh, Th)
Mh[Xh > Sh] = np.nan

Th2, Xh2 = np.meshgrid(time, xi_vals_h, indexing="ij")
Mh2 = analytical_m(m_L, alpha_L, lambd, Xh2, Th2)
Mh2[Xh2 > Sh] = np.nan

Mh_flat = Mh.flatten()
Mh2_flat = Mh2.flatten()
m_xt_h_flat = m_xt_h.flatten()
m_xt_h2_flat = m_xt_h2.flatten()

print("Before drop, ", len(Mh_flat))
maskh = ~np.isnan(Mh_flat) & ~np.isnan(m_xt_h_flat) 
maskh2 = ~np.isnan(Mh2_flat) & ~np.isnan(m_xt_h2_flat) 

Mh_flat = Mh_flat[maskh]
Mh2_flat = Mh2_flat[maskh2]
m_xt_h_flat = m_xt_h_flat[maskh]
m_xt_h2_flat = m_xt_h2_flat[maskh2]

print("After drop, ", len(Mh_flat))
mse_h = mean_squared_error(Mh_flat, m_xt_h_flat)
mse_h2 = mean_squared_error(Mh2_flat, m_xt_h2_flat)

order_of_convergence_m = mse_h2 / mse_h

# Plot FEM solution (h)
plt.figure(figsize=(10,6))
pcm = plt.pcolormesh(time, space, m_xt_h.T, shading='auto')
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.title("F(x,t) - FEM solution (grid h)")
plt.colorbar(pcm, label="F")

plt.text(time[-1], space[-50],
         f"MSE for h = {mse_h:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")

plt.text(time[-1], space[-1],
         f"MSE for h/2 = {mse_h2:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")

plt.text(time[-1], space[-90],
         f"Order of convergence = {order_of_convergence_m:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")

plt.tight_layout()
plt.show()