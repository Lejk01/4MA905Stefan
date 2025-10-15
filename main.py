import numpy as np
import matplotlib.pyplot as plt
from functions import (
    construct_convection_matrix,
    construct_mass_matrix,
    construct_stiffness_matrix,
    analytical_m,
    analytical_s,
    lambdeq,
)
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error


# ============================================================
# Solver function
# ============================================================
def run_solver(
    N_nodes, L, HORIZON, alpha_L, k_L, rho_L, l, m_L, lambd, gamma, M_points
):
    # Mesh
    nodes = np.linspace(0, 1, N_nodes)
    h = nodes[1] - nodes[0]

    n = np.arange(0, M_points)
    time = HORIZON * (n / M_points) ** gamma

    time[0] = time[1] / 2

    dt_arr = time[1:] - time[:-1]

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
    a_v = -u_int.copy()

    F = np.zeros((len(time), N_nodes))
    F[0, :] = u_vals.copy()
    F[0, 1:-1] += a_v

    s_vals = np.zeros(len(time))
    s_vals[0] = s_analytic[0]

    # Time stepping.
    for n in range(len(dt_arr)):
        dt = dt_arr[n]

        # Current F (u + v)
        F_n = F[n, :].copy()

        # dF/dxi at xi = 1 using the same 3-point backward finite difference.
        dFdxi = (3.0 * F_n[-1] - 4.0 * F_n[-2] + F_n[-3]) / (2.0 * h)

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
# Main script
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

    N_nodes_h = 320
    N_nodes_h2 = 160

    gamma = 1
    time_points = 100_000
    time_h, s_h, s_analytic_h, nodes_h, F_h = run_solver(
        N_nodes_h,
        L,
        HORIZON,
        alpha_L,
        k_L,
        rho_L,
        l,
        m_L,
        lambd,
        gamma=gamma,
        M_points=time_points,
    )
    time_h2, s_h2, s_analytic_h2, nodes_h2, F_h2 = run_solver(
        N_nodes_h2,
        L,
        HORIZON,
        alpha_L,
        k_L,
        rho_L,
        l,
        m_L,
        lambd,
        gamma=gamma,
        M_points=time_points,
    )

    space_h = (
        np.tile(nodes_h, (len(time_h), 1)) * s_h[:, None]
    )  # Transform to space x = xi * s(t)
    space_h2 = np.tile(nodes_h2, (len(time_h2), 1)) * s_h2[:, None]

    h = nodes_h[1] - nodes_h[0]
    h2 = nodes_h2[1] - nodes_h2[0]

    # --------------------------------------------------------
    # Compare s(t)
    # --------------------------------------------------------
    # Interpolate coarse solution to fine time grid for fair comparison

    mse_h = np.sqrt(np.sum(np.abs(s_analytic_h - s_h) ** 2 * np.max(np.diff(time_h))))
    mse_h2 = np.sqrt(
        np.sum(np.abs(s_analytic_h2 - s_h2) ** 2 * np.max(np.diff(time_h2)))
    )

    order_of_convergence_s = np.log(mse_h / mse_h2) / np.log(h / h2)

    plt.figure(figsize=(8, 4))
    plt.plot(time_h, s_h, label=f"s (FEM, grid {N_nodes_h} nodes)", linestyle="--")
    plt.plot(time_h2, s_h2, label=f"s (FEM, grid {N_nodes_h2} nodes)", linestyle="--")
    plt.plot(time_h, s_analytic_h, label="s (analytical)")
    plt.xlabel("Time [s]")
    plt.ylabel("Interface s(t)")
    plt.legend()
    plt.title("Interface evolution")
    plt.text(
        time_h[-1],
        0.8,
        f"MSE {N_nodes_h2} nodes = {mse_h2:.3e}\nMSE {N_nodes_h} nodes = {mse_h:.3e}\nOrder of convergence ≈ {order_of_convergence_s:.5f}",
        fontsize=10,
        va="top",
        ha="right",
        color="red",
    )
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Physical domain solution comparison
    # --------------------------------------------------------
    real_space = np.linspace(
        0, L, 400
    )  # Define the subset of the real axis that is our "ice-rod".
    fine_space_h = np.unique(
        np.sort(
            np.concatenate([space_h, np.tile(real_space, (len(time_h), 1))], axis=1),
            axis=1,
        ),
        axis=1,
    )  # Include nodes from h discretization.
    fine_space_h2 = np.unique(
        np.sort(
            np.concatenate([space_h2, np.tile(real_space, (len(time_h2), 1))], axis=1),
            axis=1,
        ),
        axis=1,
    )  # Include nodes from h2 discretization.

    fine_h = np.mean(np.diff(fine_space_h))
    fine_h2 = np.mean(np.diff(fine_space_h2))

    print("Solving analytically")
    Th = time_h[:, None]  # shape (Nt, 1) — analytical_m should broadcast
    Th2 = time_h2[:, None]  # shape (Nt, 1) — analytical_m should broadcast
    sh = s_h[:, None]
    sh2 = s_h2[:, None]
    Mh = analytical_m(m_L, alpha_L, lambd, fine_space_h, Th)
    Mh2 = analytical_m(m_L, alpha_L, lambd, fine_space_h2, Th2)

    Mh[fine_space_h > sh] = np.nan
    Mh2[fine_space_h2 > sh2] = np.nan
    print("Interpolating")
    # Interpolate
    m_xt_h = np.full((len(time_h), fine_space_h.shape[1]), np.nan)
    m_xt_h2 = np.full((len(time_h2), fine_space_h2.shape[1]), np.nan)

    for t_idx, t in enumerate(time_h):
        xi_h = fine_space_h[t_idx] / s_h[t_idx]
        interp_h = interp1d(
            nodes_h, F_h[t_idx, :], kind="linear", bounds_error=False, fill_value=np.nan
        )
        vals_h = interp_h(xi_h)
        mask1 = xi_h <= 1
        m_xt_h[t_idx, mask1] = vals_h[mask1]

        if t_idx < len(time_h2):
            xi_h2 = fine_space_h2[t_idx] / s_h2[t_idx]
            interp_h2 = interp1d(
                nodes_h2,
                F_h2[t_idx, :],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            vals_h2 = interp_h2(xi_h2)
            mask2 = xi_h2 <= 1
            m_xt_h2[t_idx, mask2] = vals_h2[mask2]

    nan_mask_h = ~np.isnan(m_xt_h)
    nan_mask_h2 = ~np.isnan(m_xt_h2)

    nan_mask_Mh = ~np.isnan(Mh)
    nan_mask_Mh2 = ~np.isnan(Mh2)

    # Drop nan values to prepare for mse calc.
    Mh_fem = m_xt_h[nan_mask_h]
    Mh2_fem = m_xt_h2[nan_mask_h2]

    Mh_analytic = Mh[nan_mask_h]
    Mh2_analytic = Mh2[nan_mask_h2]

    l2_h = np.sqrt(
        np.sum(
            np.abs(Mh_analytic[-1] - Mh_fem[-1]) ** 2 * np.max(np.diff(fine_space_h))
        )
    )
    l2_h2 = np.sqrt(
        np.sum(
            np.abs(Mh2_analytic[-1] - Mh2_fem[-1]) ** 2 * np.max(np.diff(fine_space_h2))
        )
    )

    order_of_convergence_m = np.log(l2_h / l2_h2) / np.log(h / h2)
    print(f"Order of convergence: {order_of_convergence_m:.4}")

    # For the plot
    x_uniform = np.linspace(0, np.max(s_analytic_h), 300)
    m_phys_xt = np.zeros((len(time_h), len(x_uniform)))

    for i, (s_i, t_i) in enumerate(zip(s_h, time_h)):
        x_phys = nodes_h * s_i
        interp_func = interp1d(x_phys, F_h[i, :], bounds_error=False, fill_value=np.nan)
        m_phys_xt[i, :] = interp_func(x_uniform)

    plt.figure(figsize=(10, 6))
    pcm = plt.pcolormesh(time_h, x_uniform, m_phys_xt.T, shading="auto")
    plt.xlabel("Time [s]")
    plt.ylabel("x [m]")
    plt.title("F(x,t) FEM solution (grid h)")
    plt.colorbar(pcm, label="Temperature")
    plt.plot(time_h, s_analytic_h, color="red", linewidth=2, label="s(t) analytic")
    plt.text(
        time_h[-50],
        1.75,
        f"L2 error {N_nodes_h2} nodes = {l2_h2:.3e}\n L2 error {N_nodes_h} nodes = {l2_h:.3e}\nOrder of convergence ≈ {order_of_convergence_m:.3f}",
        fontsize=10,
        va="top",
        ha="right",
        color="red",
    )
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.show()
