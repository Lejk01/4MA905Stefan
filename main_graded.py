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

# ---------------------------
# Physical / numeric params
# ---------------------------
# Water–ice constants.
k_L = 0.5664         # W/(mK)
rho_L = 1000         # kg/m^3
c_L = 4189.9         # J/(kgK)
m_L = 323.15 * 5 # 150 C
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 334e3             # latent heat [J/kg]
L = 0.1               # physical domain length [m]


#k_L = 1         # W/(mK)
#rho_L = 1         # kg/m^3
#c_L = 1         # J/(kgK)
#m_L = 1
#alpha_L = 1
#l = 1

# Spatial discretizations (same as your uniform-file)
N_nodes_h = 75
N_nodes_h2 = int(N_nodes_h/2)

nodes_h = np.linspace(0, 1, N_nodes_h)
h_nodes = nodes_h[1] - nodes_h[0]

nodes_h2 = np.linspace(0, 1, N_nodes_h2)
h2_nodes = nodes_h2[1] - nodes_h2[0]

# Time horizon + points
HORIZON = 600.0
TIME_POINTS = 4000

# Solver helper (Stefan parameter lambda)
lambda_guess = 0.5
lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

# ---------------------------
# Graded time mesh
# ---------------------------
# Choose grading exponent gamma > 1 to cluster near t=0. (gamma = 1 -> uniform)
gamma = 2.0

tau = np.linspace(0.0, 1.0, TIME_POINTS)
# avoid exact zero at start (some functions expect >0). keep small epsilon.
eps = 1e-9
time_graded = HORIZON * (tau**gamma)
time_graded[0] = eps  # avoid exact zero
dt_graded = np.diff(time_graded)  # length TIME_POINTS-1

# ---------------------------
# Analytical solution on graded times
# ---------------------------
# Note: analytical_s signature assumed as in your uniform file:
# analytical_s(alpha_L, lambd, time_array, maybe_s0) -- keep same call pattern.
# In the uniform script you used analytical_s(alpha_L, lambd, time, h_nodes).
# We'll mimic that: pass h_nodes as s0 if that's what your function expects.
# If your analytical_s signature is different, adapt the call accordingly.
s_analytic = analytical_s(alpha_L, lambd, time_graded, h_nodes)

# Physical x grid for later plotting / M comparison
space = np.linspace(0, L, 1000)
h_space = space[1] - space[0]

# ---------------------------
# FEM matrices (same as uniform code)
# ---------------------------
mass_matrix_h = construct_mass_matrix(N_nodes_h, h_nodes)
stiffness_matrix_h = construct_stiffness_matrix(N_nodes_h, h_nodes)
convection_matrix_h = construct_convection_matrix(N_nodes_h, h_nodes, nodes_h)

mass_matrix_h2 = construct_mass_matrix(N_nodes_h2, h2_nodes)
stiffness_matrix_h2 = construct_stiffness_matrix(N_nodes_h2, h2_nodes)
convection_matrix_h2 = construct_convection_matrix(N_nodes_h2, h2_nodes, nodes_h2)

interior_idx_h = np.arange(1, N_nodes_h - 1)
mass_int_h = mass_matrix_h[np.ix_(interior_idx_h, interior_idx_h)]
stiffness_int_h = stiffness_matrix_h[np.ix_(interior_idx_h, interior_idx_h)]
convection_int_h = convection_matrix_h[np.ix_(interior_idx_h, interior_idx_h)]

interior_idx_h2 = np.arange(1, N_nodes_h2 - 1)
mass_int_h2 = mass_matrix_h2[np.ix_(interior_idx_h2, interior_idx_h2)]
stiffness_int_h2 = stiffness_matrix_h2[np.ix_(interior_idx_h2, interior_idx_h2)]
convection_int_h2 = convection_matrix_h2[np.ix_(interior_idx_h2, interior_idx_h2)]

# ---------------------------
# Dirichlet u(0)=m_L, u(1)=0 and initial fields
# ---------------------------
u_vals_h = m_L * (1.0 - nodes_h)
u_int_h = u_vals_h[interior_idx_h]
load_vector_h = -m_L * h_nodes * nodes_h[interior_idx_h]

u_vals_h2 = m_L * (1.0 - nodes_h2)
u_int_h2 = u_vals_h2[interior_idx_h2]
load_vector_h2 = -m_L * h2_nodes * nodes_h2[interior_idx_h2]

# initial correction v so F = u + v and F(xi,0)=0 -> v = -u (interior)
a_vh = - u_int_h.copy()
a_vh2 = - u_int_h2.copy()

F_h = np.zeros((len(time_graded), N_nodes_h))
F_h[0, :] = u_vals_h.copy()
F_h[0, 1:-1] += a_vh

F_h2 = np.zeros((len(time_graded), N_nodes_h2))
F_h2[0, :] = u_vals_h2.copy()
F_h2[0, 1:-1] += a_vh2

# interface s(t) arrays (in xi-domain)
s_h = np.zeros(len(time_graded))
s_h[0] = s_analytic[0]  # initialize from analytic or small value

s_h2 = np.zeros(len(time_graded))
s_h2[0] = s_analytic[0]

# ---------------------------
# Time-stepping loop on graded mesh
# ---------------------------
for n in range(len(time_graded) - 1):
  # use current F (u + v)
  F_nh = u_vals_h.copy()
  F_nh[1:-1] += a_vh

  F_nh2 = u_vals_h2.copy()
  F_nh2[1:-1] += a_vh2

  # compute dF/dxi at xi=1 via basis derivatives
  dFhdxi = 0.0
  for j in range(N_nodes_h):
    dFhdxi += F_nh[j] * phi_j_dt(j, h_nodes, 1.0, nodes_h)

  dFh2dxi = 0.0
  for j in range(N_nodes_h2):
    dFh2dxi += F_nh2[j] * phi_j_dt(j, h2_nodes, 1.0, nodes_h2)

  # Stefan ODE: ds/dt = -(k_L/(rho_L*l)) * (1/s) * dF/dxi
  # use s_h[n] (current) to compute dsdt
  dsdt_h = -(k_L / (rho_L * l)) * (1.0 / s_h[n]) * dFhdxi
  dsdt_h2 = -(k_L / (rho_L * l)) * (1.0 / s_h2[n]) * dFh2dxi

  # advance with graded timestep dt_graded[n]
  dt_n = dt_graded[n]

  s_next_h = s_h[n] + dt_n * dsdt_h
  s_h[n + 1] = s_next_h

  s_next_h2 = s_h2[n] + dt_n * dsdt_h2
  s_h2[n + 1] = s_next_h2

  # Assemble LHS / RHS with graded dt
  LHSh = mass_int_h + dt_n * (alpha_L / s_next_h**2) * stiffness_int_h - dt_n * (dsdt_h / s_next_h) * convection_int_h
  RHSh = np.dot(mass_int_h, a_vh) + dt_n * (dsdt_h / s_next_h) * load_vector_h

  LHSh2 = mass_int_h2 + dt_n * (alpha_L / s_next_h2**2) * stiffness_int_h2 - dt_n * (dsdt_h2 / s_next_h2) * convection_int_h2
  RHSh2 = np.dot(mass_int_h2, a_vh2) + dt_n * (dsdt_h2 / s_next_h2) * load_vector_h2

  # Solve for updated a_v
  a_vh = np.linalg.solve(LHSh, RHSh)
  a_vh2 = np.linalg.solve(LHSh2, RHSh2)

  # update global F arrays
  F_h[n + 1, :] = u_vals_h.copy()
  F_h[n + 1, 1:-1] += a_vh

  F_h2[n + 1, :] = u_vals_h2.copy()
  F_h2[n + 1, 1:-1] += a_vh2

# ---------------------------
# Evaluate analytic m(x,t) and compare to FEM (map xi->x)
# ---------------------------
# Analytical S and M on the graded time mesh
T_grid, X_grid = np.meshgrid(time_graded, space, indexing='ij')
S_grid = analytical_s(alpha_L, lambd, T_grid, h_nodes)
M_grid = analytical_m(m_L, alpha_L, lambd, X_grid, T_grid)
M_grid[X_grid > S_grid] = np.nan

# FEM m(x,t) from F( xi , t ) using xi = x / s(t)
m_xt_h = np.full((len(time_graded), len(space)), np.nan)
m_xt_h2 = np.full((len(time_graded), len(space)), np.nan)

for tidx in range(len(time_graded)):
  # analytical temperature inside physical domain for reference
  # fill by calling analytical_m where x <= S(t)
  # Mh[t, x<=s] used only for mse later
  # interpolation in xi to physical x/s
  xi_h = space / s_h[tidx]
  xi_h2 = space / s_h2[tidx]

  interp_h = interp1d(nodes_h, F_h[tidx, :], kind='linear', bounds_error=False, fill_value=np.nan)
  interp_h2 = interp1d(nodes_h2, F_h2[tidx, :], kind='linear', bounds_error=False, fill_value=np.nan)

  vals_h = interp_h(xi_h)
  vals_h2 = interp_h2(xi_h2)

  mask1 = (xi_h <= 1.0)
  mask2 = (xi_h2 <= 1.0)

  m_xt_h[tidx, mask1] = vals_h[mask1]
  m_xt_h2[tidx, mask2] = vals_h2[mask2]

# Build analytic M restricted to domain (for mse)
Mh = np.full_like(M_grid, np.nan)
Mh2 = np.full_like(M_grid, np.nan)

for t in range(len(time_graded)):
  # evaluate analytic m only on x <= S(t)
  s_val = S_grid[t, 0] if S_grid.ndim == 2 else S_grid[t]  # defensive
  mask_space = space <= s_val
  if np.any(mask_space):
    Mh[t, mask_space] = analytical_m(m_L, alpha_L, lambd, space[mask_space], time_graded[t])
    Mh2[t, mask_space] = analytical_m(m_L, alpha_L, lambd, space[mask_space], time_graded[t])

# Flatten and mask for MSE
Mh_flat = Mh.flatten()
Mh2_flat = Mh2.flatten()
m_xt_h_flat = m_xt_h.flatten()
m_xt_h2_flat = m_xt_h2.flatten()

maskh = ~(np.isnan(Mh_flat) | np.isnan(m_xt_h_flat))
maskh2 = ~(np.isnan(Mh2_flat) | np.isnan(m_xt_h2_flat))

mse_h = mean_squared_error(Mh_flat[maskh], m_xt_h_flat[maskh])
mse_h2 = mean_squared_error(Mh2_flat[maskh2], m_xt_h2_flat[maskh2])

# MSE for s(t)
mse_s_h = mean_squared_error(s_analytic, s_h)
mse_s_h2 = mean_squared_error(s_analytic, s_h2)

# order of convergence (same metric you used before)
order_of_convergence_s = mse_s_h2 / mse_s_h
order_of_convergence_m = mse_h2 / mse_h

# ---------------------------
# Plots
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(time_graded, s_h, label="s (FEM, numeric, grid h)", linestyle="--")
plt.plot(time_graded, s_h2, label="s (FEM, numeric, grid 2*h)", linestyle="--")
plt.plot(time_graded, s_analytic, label="s (analytical)", linewidth=1)
plt.xlabel("Time [s]")
plt.ylabel("Interface s(t) [ξ-units]")
plt.legend()
plt.title("Interface evolution (ξ-domain FEM) — graded time mesh")

# annotate MSEs
plt.text(time_graded[-1], s_h[-1],
         f"MSE s (h) = {mse_s_h:.3e}",
         fontsize=9, va="bottom", ha="right", color="red")

plt.text(time_graded[-1], s_h2[-1] * 0.9,
         f"MSE s (2h) = {mse_s_h2:.3e}",
         fontsize=9, va="bottom", ha="right", color="red")

plt.text(time_graded[-1], s_h2[-1] * 0.8,
         f"Order conv (s) = {order_of_convergence_s:.3e}",
         fontsize=9, va="bottom", ha="right", color="red")

plt.tight_layout()
plt.show()

# F(ξ,t) plot (h)
plt.figure(figsize=(10, 6))
pcm = plt.pcolormesh(time_graded, nodes_h, F_h.T, shading='auto')
plt.xlabel("Time [s]")
plt.ylabel("ξ")
plt.title("F(ξ,t) - FEM solution (grid h) — graded time mesh")
plt.colorbar(pcm, label="F")
plt.tight_layout()
plt.show()

# m(x,t) FEM plot (h)
plt.figure(figsize=(10, 6))
pcm = plt.pcolormesh(time_graded, space, m_xt_h.T, shading='auto')
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.title("m(x,t) - FEM (grid h) — graded time mesh")
plt.colorbar(pcm, label="m")
plt.text(time_graded[-10], space[-50],
         f"MSE m (h) = {mse_h:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")
plt.text(time_graded[-10], space[-90],
         f"MSE m (2h) = {mse_h2:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")
plt.text(time_graded[-10], space[-130],
         f"Order conv (m) = {order_of_convergence_m:.3e}",
         fontsize=10, va="bottom", ha="right", color="red")
plt.tight_layout()
plt.show()
