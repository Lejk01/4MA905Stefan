import numpy as np
import matplotlib.pyplot as plt
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

# ============================================================
# Solver function
# ============================================================
def run_solver(N_nodes, L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd, CFL_const=0.25):
  # Mesh
  nodes = np.linspace(0, 1, N_nodes)
  h = nodes[1] - nodes[0]
  
  # Stable time step scaling with h^2
  dt = CFL_const * h**2 / alpha_L
  time = np.arange(1e-6, HORIZON, dt)
  print(f"Running solver with N={N_nodes}, h={h:.3e}, dt={dt:.3e}, Nt={len(time)}")
  
  # Analytical solution for interface
  s_analytic = analytical_s(alpha_L, lambd, time, h)
  
  # Matrices
  M = construct_mass_matrix(N_nodes, h)
  K = construct_stiffness_matrix(N_nodes, h)
  C = construct_convection_matrix(N_nodes, h, nodes)

  # Interior
  interior_idx = np.arange(1, N_nodes-1)
  M_int = M[np.ix_(interior_idx, interior_idx)]
  K_int = K[np.ix_(interior_idx, interior_idx)]
  C_int = C[np.ix_(interior_idx, interior_idx)]

  # Initial conditions
  u_vals = m_L * (1.0 - nodes)
  u_int  = u_vals[interior_idx]

  load_vector = -m_L * h * nodes[interior_idx]
  a_v = - u_int.copy()

  F = np.zeros((len(time), N_nodes))
  F[0, :] = u_vals.copy()
  F[0, 1:-1] += a_v

  s_vals = np.zeros(len(time))
  s_vals[0] = s_analytic[0]

  # Time stepping
  for n in range(len(time)-1):
      # Current field
      F_n = u_vals.copy()
      F_n[1:-1] += a_v

      # dF/dxi at xi=1
      dFdxi = 0
      for j in range(N_nodes):
          dFdxi += F_n[j] * phi_j_dt(j, h, 1, nodes)

      # Stefan condition
      dshdt = -(k_L/(rho_L*l)) * (1/s_vals[n]) * dFdxi
      s_next = s_vals[n] + dt * dshdt
      s_vals[n+1] = s_next

      # Solve system
      LHS = M_int + dt * (alpha_L / s_next**2) * K_int - dt * (dshdt/s_next) * C_int
      RHS = np.dot(M_int, a_v) + dt * (dshdt/s_next) * load_vector
      a_v = np.linalg.solve(LHS, RHS)

      # Update field
      F[n+1, :] = u_vals.copy()
      F[n+1, 1:-1] += a_v

  return time, s_vals, s_analytic, nodes, F


# ============================================================
# Main script
# ============================================================
if __name__ == "__main__":
  # Physical constants (nondimensionalised for simplicity)
  k_L = 1.0
  rho_L = 1.0
  c_L = 1.0
  m_L = 1.0
  alpha_L = k_L / (rho_L * c_L)
  l = 1.0
  L = 1.0
  HORIZON = 1.0
  
  lambda_guess = 0.5
  lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

  # Grids
  N_nodes_h  = 49
  N_nodes_h2 = N_nodes_h // 2   # coarser grid (2h)

  # Run solvers
  time_h,  s_h,  s_analytic_h,  nodes_h,  F_h  = run_solver(N_nodes_h,  L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd)
  time_h2, s_h2, s_analytic_h2, nodes_h2, F_h2 = run_solver(N_nodes_h2, L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd)

  h = nodes_h[1] - nodes_h[0]
  h2 = nodes_h2[1] - nodes_h2[0]

  # --------------------------------------------------------
  # Compare s(t)
  # --------------------------------------------------------
  # Interpolate coarse solution to fine time grid for fair comparison
  from scipy.interpolate import interp1d
  interp_s_h2 = interp1d(time_h2, s_h2, bounds_error=False, fill_value="extrapolate")
  s_h2_interp = interp_s_h2(time_h)

  mse_h  = mean_squared_error(s_analytic_h, s_h)
  mse_h2 = mean_squared_error(s_analytic_h, s_h2_interp)
  order_of_convergence_s = np.log(mse_h/mse_h2)/np.log(h/h2)


  plt.figure(figsize=(8,4))
  plt.plot(time_h, s_h,  label="s (FEM, grid h)", linestyle="--")
  plt.plot(time_h2, s_h2, label="s (FEM, grid 2h)", linestyle="--")
  plt.plot(time_h, s_analytic_h, label="s (analytical)")
  plt.xlabel("Time [s]")
  plt.ylabel("Interface s(t)")
  plt.legend()
  plt.title("Interface evolution")
  plt.text(time_h[-1], 0.8,
            f"MSE h = {mse_h:.3e}\nMSE 2h = {mse_h2:.3e}\nOrder ≈ {order_of_convergence_s:.3f}",
            fontsize=10, va="top", ha="right", color="red")
  plt.tight_layout()
  plt.show()

  # --------------------------------------------------------
  # Physical domain solution comparison
  # --------------------------------------------------------
  space = np.linspace(0, L, 500)
  T, X = np.meshgrid(time_h, space, indexing="ij")
  S = analytical_s(alpha_L, lambd, T, s0=0.0)
  M = analytical_m(m_L, alpha_L, lambd, X, T)
  M[X > S] = np.nan

  m_xt_h  = np.full_like(M, np.nan)
  m_xt_h2 = np.full_like(M, np.nan)

  for t_idx, t in enumerate(time_h):
    xi_h  = space / s_h[t_idx]
    xi_h2 = space / interp_s_h2(t)

    interp_h  = interp1d(nodes_h,  F_h[t_idx,:],  kind='linear', bounds_error=False, fill_value=np.nan)
    interp_h2 = interp1d(nodes_h2, F_h2[min(t_idx, len(time_h2)-1),:], kind='linear', bounds_error=False, fill_value=np.nan)

    vals_h  = interp_h(xi_h)
    vals_h2 = interp_h2(xi_h2)

    mask1 = xi_h  <= 1.0
    mask2 = xi_h2 <= 1.0
    m_xt_h[t_idx, mask1]  = vals_h[mask1]
    m_xt_h2[t_idx, mask2] = vals_h2[mask2]

  # Flatten + mask
  Mh_flat   = M.flatten()
  Mh_fem    = m_xt_h.flatten()
  Mh2_fem   = m_xt_h2.flatten()

  mask = ~(np.isnan(Mh_flat) | np.isnan(Mh_fem) | np.isnan(Mh2_fem))

  mse_h  = mean_squared_error(Mh_flat[mask],  Mh_fem[mask])
  mse_h2 = mean_squared_error(Mh_flat[mask], Mh2_fem[mask])

  order_of_convergence_m = np.log(mse_h/mse_h2)/np.log(h/h2)
  print(f'Order of convergence: {order_of_convergence_m:.4}')

  plt.figure(figsize=(10,6))
  pcm = plt.pcolormesh(time_h, space, m_xt_h.T, shading="auto")
  plt.xlabel("Time [s]")
  plt.ylabel("x [m]")
  plt.title("F(x,t) FEM solution (grid h)")
  plt.colorbar(pcm, label="F")
  plt.text(time_h[-50], 0.9,
            f"MSE h = {mse_h:.3e}\nMSE 2h = {mse_h2:.3e}\nOrder ≈ {order_of_convergence_m:.3f}",
            fontsize=10, va="top", ha="right", color="red")
  plt.tight_layout()
  plt.show()
