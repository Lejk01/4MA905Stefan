import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve

# Water–ice constants.
k_L = 0.5664          # W/(mK)
rho_L = 1000          # kg/m^3
c_L = 4189.9          # J/(kgK)
m_L = 298.15
alpha_L = k_L / (rho_L * c_L)  # m^2/s
l = 334e3              # temperature difference driving phase change [K]
N_nodes = 300
lambda_guess = 0.5

# Time horizon (seconds).
HORIZON = 600
time = np.linspace(1e-6, HORIZON, 1000)
dt = time[1] - time[0]

# Discretize the transformed domain xi in [0,1]
nodes = np.linspace(0, 1, N_nodes)
h_nodes = nodes[1] - nodes[0]

# --- Helper functions ---
def lambdeq(lambd, c_L, m_L, l):
    from scipy.special import erf
    return lambd * np.exp(lambd**2) * erf(lambd) - (m_L * c_L) / (l*np.sqrt(np.pi))

def construct_mass_matrix(N, h):
    M = np.zeros((N, N))
    np.fill_diagonal(M, 2*h / 3)
    M[0, 0] = h / 3
    M[N-1, N-1] = h / 3
    val = h / 6
    for i in range(N-1):
        M[i, i+1] = val
        M[i+1, i] = val
    return M

def construct_stiffness_matrix(N, h):
    A = np.zeros((N, N))
    np.fill_diagonal(A, 2 / h)
    A[0, 0] = 1 / h
    A[N-1, N-1] = 1 / h
    val = -1 / h
    for i in range(N-1):
        A[i, i+1] = val
        A[i+1, i] = val
    return A

def construct_convection_matrix(N, h, nodes):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Integral int_0^1 xi * phi_i * phi_j' dxi
            # This is hardcoded for linear elements
            if i == j:
                if i > 0 and i < N - 1:
                    K[i,j] = 0.5
                elif i == 0:
                    K[i,j] = 1/6
                else: # i == N-1
                    K[i,j] = 5/6
            elif i == j+1:
                K[i,j] = -1/6
            elif i == j-1:
                K[i,j] = 1/3
    return K

# --- Main simulation loop ---

# Solve for lambda
lambd = fsolve(lambdeq, lambda_guess, args=(c_L, m_L, l))[0]

# Pre-compute matrices (they are constant for a uniform mesh)
M = construct_mass_matrix(N_nodes, h_nodes)
A = construct_stiffness_matrix(N_nodes, h_nodes)
K = construct_convection_matrix(N_nodes, h_nodes, nodes)

# Initial conditions
a = np.zeros(N_nodes)
a[0] = m_L # Temperature at the boundary x=0
s = np.zeros(len(time))
s[0] = 1e-6 # A small non-zero initial interface position

# Store results
heat_distribution_xi = np.zeros((len(time), N_nodes))
heat_distribution_xi[0,:] = a

# Time loop
for n in range(len(time) - 1):
    # Predict s_next from the current state
    # Use simple finite difference for the derivative at xi=1
    dmdxi_at_1 = (a[N_nodes-1] - a[N_nodes-2]) / h_nodes
    s_next = s[n] - dt * (k_L / (rho_L * l * s[n])) * dmdxi_at_1
    s[n+1] = s_next

    # Construct the linear system for the next time step
    # LHS matrix for the FEM system
    LHS = (1.0 / (alpha_L * dt)) * M + (1.0 / s_next**2) * A + ((s_next - s[n]) / (s_next**2 * dt)) * K

    # RHS vector
    RHS = (1.0 / (alpha_L * dt)) * M @ a

    # Solve the system
    a_next = np.linalg.solve(LHS, RHS)

    # Apply boundary conditions
    a_next[0] = m_L
    a_next[N_nodes-1] = 0.0

    # Store result and update
    a = a_next
    heat_distribution_xi[n+1, :] = a

    if n % 100 == 0:
        print(f"Time step {n}: Interface position s(t) = {s[n]:.6f} m")

# Plotting (optional)
# You can now transform the xi-domain solution to the x-domain for visualization
# This part is for plotting the results and not part of the core FEM algorithm
X_plot = np.zeros((len(time), N_nodes))
T_plot = np.zeros((len(time), N_nodes))
for i in range(len(time)):
    X_plot[i,:] = nodes * s[i]
    T_plot[i,:] = heat_distribution_xi[i,:]

# Create a 2D time grid to match the shape of X_plot and T_plot
time_2d = np.broadcast_to(time[:, None], (len(time), N_nodes))

# The corrected call to pcolormesh
plt.figure(figsize=(12, 8))
plt.pcolormesh(X_plot, time_2d, T_plot, shading='auto', cmap='viridis')

plt.plot(s, time, 'r--', label='Numerical Interface Position')
plt.xlabel('Position (x) [m]')
plt.ylabel('Time (s)')
plt.title('Temperature Distribution M(x,t) - Numerical Solution')
cbar = plt.colorbar(label='Temperature (m)')
plt.legend()
plt.show()