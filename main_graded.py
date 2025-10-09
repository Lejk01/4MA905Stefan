import numpy as np
import matplotlib.pyplot as plt
from functions import (construct_convection_matrix,
                      construct_mass_matrix,
                      construct_stiffness_matrix,
                      analytical_m,
                      analytical_s,
                      lambdeq,
                      s_l2_err
                      )
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
import seaborn as sns

# ============================================================
# Solver function (graded time mesh) - same structure as uniform run_solver
# ============================================================
def run_solver(max_nodes, N_nodes, L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd, gamma=2.0, TIME_POINTS=5000, CFL_const=0.5):
  # Mesh
  nodes = np.linspace(0, 1, N_nodes)
  nodes_max = np.linspace(0, 1, max_nodes)
  h = nodes[1] - nodes[0]
  h_max = nodes_max[1] - nodes_max[0]
  dt = CFL_const * (h_max**2) / alpha_L

  tau = np.linspace(0.0, 1.0, TIME_POINTS)
  eps = 1e-9
  time = HORIZON * (tau**gamma)
  time[0] = eps
  dt_arr1 = np.diff(time)
  dt_subset = np.where(dt_arr1 <= [dt]*len(dt_arr1))
  time_subset = time[:len(dt_subset)]

  if time_subset[-1] < HORIZON:
    extended_time = np.arange(time_subset[-1], HORIZON, dt)
    time = np.sort(np.concatenate([time_subset, extended_time]))
  dt_arr = np.diff(time)
  print("len time: ", len(time), " time", time)
  print("cfl dt: ", dt, " last dt: ", dt_arr[-1])

  print(f"Running solver (graded) with N={N_nodes}, h={h:.3e}, gamma={gamma:.2f}, Nt={len(time)}")

  # Analytical s on the graded time points
  s_analytic = analytical_s(alpha_L, lambd, time, h)

  # FEM matrices (use same constructors as uniform code)
  M = construct_mass_matrix(N_nodes, h)
  K = construct_stiffness_matrix(N_nodes, h)
  C = construct_convection_matrix(N_nodes, h, nodes)

  # Interior (Dirichlet nodes removed)
  interior_idx = np.arange(1, N_nodes - 1)
  M_int = M[np.ix_(interior_idx, interior_idx)]
  K_int = K[np.ix_(interior_idx, interior_idx)]
  C_int = C[np.ix_(interior_idx, interior_idx)]

  # Initial conditions & load vector
  u_vals = m_L * (1.0 - nodes)
  u_int = u_vals[interior_idx]

  load_vector = -m_L * h * nodes[interior_idx]
  a_v = - u_int.copy()

  F = np.zeros((len(time), N_nodes))
  F[0, :] = u_vals.copy()
  F[0, 1:-1] += a_v

  s_vals = np.zeros(len(time))
  s_vals[0] = s_analytic[0]

  # Time stepping (graded dt).
  for n in range(len(time) - 1):
    dt = dt_arr[n]

    # Current F (u + v)
    F_n = u_vals.copy()
    F_n[1:-1] += a_v

    # dF/dxi at xi = 1 using the same 3-point backward finite difference.
    dFdxi = (3.0 * F[n, -1] - 4.0 * F[n, -2] + F[n, -3]) / (2.0 * h)

    # Stefan ODE.
    dsdt = -(k_L / (rho_L * l)) * (1.0 / s_vals[n]) * dFdxi
    s_next = s_vals[n] + dt * dsdt
    s_vals[n + 1] = s_next

    LHS = M_int + dt * (alpha_L / s_next**2) * K_int - dt * (dsdt / s_next) * C_int
    RHS = np.dot(M_int, a_v) + dt * (dsdt / s_next) * load_vector

    a_v = np.linalg.solve(LHS, RHS)

    # Update global F.
    F[n + 1, :] = u_vals.copy()
    F[n + 1, 1:-1] += a_v

  return time, s_vals, s_analytic, nodes, F

# ============================================================
# Main script - uses run_solver twice (h and 2h) and performs full comparison
# ============================================================
if __name__ == "__main__":
  k_L = 1.0
  rho_L = 1.0
  c_L = 1.0
  m_L = 1.0
  alpha_L = k_L / (rho_L * c_L)
  l = 1.0
  L = 2.0
  HORIZON = 1.0

  lambda_guess = 0.5
  lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

  MAX_NODES = 200
  N_nodes_h  = 200
  N_nodes_h2 = N_nodes_h // 2 + 1   # coarser grid (2h)

  gamma = 2.0
  time_h,  s_h,  s_analytic_h,  nodes_h,  F_h  = run_solver(MAX_NODES, N_nodes_h,  L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd, gamma=gamma)
  time_h2, s_h2, s_analytic_h2, nodes_h2, F_h2 = run_solver(MAX_NODES, N_nodes_h2, L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd, gamma=gamma)

  # Map xi-domain -> physical x domain.
  space_h  = np.tile(nodes_h,  (len(time_h), 1)) * s_h[:, None]
  space_h2 = np.tile(nodes_h2, (len(time_h2), 1)) * s_h2[:, None]

  h = nodes_h[1] - nodes_h[0]
  h2 = nodes_h2[1] - nodes_h2[0]

  # --------------------------------------------------------
  # Compare s(t)
  # --------------------------------------------------------
  # Interpolate coarse solution to fine time grid for fair comparison
  dt_h = np.diff(time_h)
  dt_h2 = np.diff(time_h2)
  dt_h = np.concatenate([dt_h, [dt_h[-1]]])
  dt_h2 = np.concatenate([dt_h2, [dt_h2[-1]]])

  mse_h = s_l2_err(s_analytic_h, s_h, len(time_h), dt_h)
  mse_h2 = s_l2_err(s_analytic_h2, s_h2, len(time_h2), dt_h2)

  print("mse_h=", mse_h, " mse_h2=", mse_h2, " h=", h, " h2=", h2)
  order_of_convergence_s = np.log(mse_h/mse_h2)/np.log(h/h2)

  plt.figure(figsize=(8,4))
  plt.plot(time_h, s_h,  label="s (FEM, grid h)", linestyle="--")
  plt.plot(time_h2, s_h2, label="s (FEM, grid 2h)", linestyle="--")
  plt.plot(time_h, s_analytic_h, label="s (analytical)")
  plt.xlabel("Time [s]")
  plt.ylabel("Interface s(t)")
  plt.legend()
  plt.title("Interface evolution (graded time mesh)")
  plt.text(time_h[-1], 0.8,
            f"MSE h = {mse_h:.3e}\nMSE 2h = {mse_h2:.3e}\nOrder ≈ {order_of_convergence_s:.5f}",
            fontsize=10, va="top", ha="right", color="red")
  plt.tight_layout()
  plt.show()

  # --------------------------------------------------------
  # Physical domain solution comparison (map xi->x and compute L2/MSE)
  # --------------------------------------------------------
  real_space    = np.linspace(0, L, MAX_NODES*2) # Define the subset of the real axis that is our "ice-rod".
  # Include nodes from h discretization and also the regular real_space; create a consistent fine-space per time-step
  fine_space_h  = np.unique(np.sort(np.concatenate([space_h,  np.tile(real_space, (len(time_h), 1))], axis=1), axis=1), axis=1)
  fine_space_h2 = np.unique(np.sort(np.concatenate([space_h2, np.tile(real_space, (len(time_h2), 1))], axis=1), axis=1), axis=1)

  fine_h  = np.mean(np.diff(fine_space_h))
  fine_h2 = np.mean(np.diff(fine_space_h2))

  print("Fine space h: ", fine_h)
  print("Fine space h2: ", fine_h2)

  print("Solving analytically")
  Th  = time_h[:, None]   # shape (Nt, 1)
  Th2 = time_h2[:, None]
  sh  = s_h[:, None]
  sh2 = s_h2[:, None]

  # Analytical field evaluated on fine-space arrays: ensure analytical_m supports vectorized inputs
  Mh  = analytical_m(m_L, alpha_L, lambd, fine_space_h, Th)
  Mh2 = analytical_m(m_L, alpha_L, lambd, fine_space_h2, Th2)

  # Mask outside moving interface
  Mh[fine_space_h > sh]    = np.nan
  Mh2[fine_space_h2 > sh2] = np.nan

  print("Interpolating FEM -> physical x grid")
  # Interpolate FEM solutions (xi -> x) onto fine_space arrays and build m_xt arrays
  m_xt_h   = np.full((len(time_h), fine_space_h.shape[1]), np.nan)
  m_xt_h2  = np.full((len(time_h2), fine_space_h2.shape[1]), np.nan)

  for t_idx, t in enumerate(time_h):
    xi_h  = fine_space_h[t_idx] / s_h[t_idx]
    interp_h = interp1d(nodes_h,  F_h[t_idx, :],  kind='linear', bounds_error=False, fill_value=np.nan)
    vals_h = interp_h(xi_h)
    mask1 = xi_h <= 1
    m_xt_h[t_idx, mask1] = vals_h[mask1]

    if t_idx < len(time_h2):
      xi_h2 = fine_space_h2[t_idx] / s_h2[t_idx]
      interp_h2 = interp1d(nodes_h2, F_h2[t_idx, :], kind='linear', bounds_error=False, fill_value=np.nan)
      vals_h2 = interp_h2(xi_h2)
      mask2 = xi_h2 <= 1
      m_xt_h2[t_idx, mask2] = vals_h2[mask2]

  nan_mask_h  = ~np.isnan(m_xt_h)
  nan_mask_h2 = ~np.isnan(m_xt_h2)

  nan_mask_Mh  = ~np.isnan(Mh)
  nan_mask_Mh2 = ~np.isnan(Mh2)

  # Drop nan values to prepare for mse calc.
  Mh_fem  = m_xt_h[nan_mask_h]
  Mh2_fem = m_xt_h2[nan_mask_h2]

  Mh_analytic  = Mh[nan_mask_h]
  Mh2_analytic = Mh2[nan_mask_h2]

  print("Mh analytic: ", Mh.shape, "   Mh fem", m_xt_h.shape)
  print("Mh2 analytic: ", Mh2.shape, "   Mh2 fem", m_xt_h2.shape)
  l2_h  = root_mean_squared_error(Mh_analytic,  Mh_fem) * np.sqrt(h)
  l2_h2 = root_mean_squared_error(Mh2_analytic, Mh2_fem) * np.sqrt(h2)

  order_of_convergence_m = np.log(l2_h/l2_h2)/np.log(h/h2)
  print(f'Order of convergence (m): {order_of_convergence_m:.4}')

  plt.figure(figsize=(10,6))
  pcm = plt.pcolormesh(time_h, np.linspace(0, L, m_xt_h.shape[1]), m_xt_h.T, shading="auto")
  plt.xlabel("Time [s]")
  plt.ylabel("x [m]")
  plt.title("F(x,t) FEM solution (grid h) - graded time mesh")
  plt.colorbar(pcm, label="F")
  plt.text(time_h[-50], 0.9,
            f"MSE h = {l2_h:.3e}\nMSE 2h = {l2_h2:.3e}\nOrder ≈ {order_of_convergence_m:.3f}",
            fontsize=10, va="top", ha="right", color="red")
  plt.tight_layout()
  plt.show()
