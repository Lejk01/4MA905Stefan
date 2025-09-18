import numpy as np
from scipy.special import erf

def phi_j(j, h, x, nodes):
  x_lb = nodes[j-1]
  x_j = nodes[j]
  if j+1 < len(nodes): # Nåt fel här.
    x_ub = nodes[j+1]
  else:
    return 1

  if x >= x_lb  and x <= x_j:
    return (x - x_lb) / h
  elif x >= x_j and x <= x_ub:
    return (x_ub - x) / h
  else:
    return 0
  
def phi_j_vectorized(j, h, x, nodes):
  x_lb = nodes[j - 1] if j - 1 >= 0 else nodes[j]
  x_j = nodes[j]
  x_ub = nodes[j + 1] if j + 1 < len(nodes) else nodes[j]

  res = np.zeros_like(x, dtype=float)

  mask_left = (x >= x_lb) & (x <= x_j)
  res[mask_left] = (x[mask_left] - x_lb) / h

  mask_right = (x >= x_j) & (x <= x_ub)
  res[mask_right] = (x_ub - x[mask_right]) / h
  
  return res

def phi_j_dt(j, h, x, nodes):
  x_lb = nodes[j-1]
  x_j = nodes[j]
  x_ub = nodes[j+1]
  
  if x >= x_lb  and x <= x_j:
    return 1 / h
  elif x >= x_j and x <= x_ub:
    return - 1 / h
  else:
    return 0
  
def construct_mass_matrix(N, h):
  M = np.zeros((N, N))
  np.fill_diagonal(M, 2*h / 3)
  M[0, 0] = M[-1, -1] = h / 3

  val = h / 6
  for i in range(N-1):
    M[i, i+1] = val
    M[i+1, i] = val
  return M

def construct_stiffness_matrix(N, h):
  A = np.zeros((N, N))
  np.fill_diagonal(A, 2 / h)
  A[0, 0] = A[N-1, N-1] = 1 / h

  val = - 1 / h
  for i in range(N-1):
    A[i, i+1] = val
    A[i+1, i] = val
  return A

def construct_convection_matrix(N, h, nodes):
  K = np.zeros((N, N))
  K[0, 0] = -(nodes[1]**3 - nodes[0]**2 * (3*nodes[1] - 2*nodes[0])) / (6*h**2)
  K[-1, -1] = (nodes[-2]**3 - nodes[-1]**2 * (3*nodes[-2] - 2*nodes[-1])) / (6*h**2)

  for i in range(N-1):
    if i > 0 and i < N-1:
      K[i, i] = ((nodes[i+1]**3 - nodes[i-1]**3)/6 - nodes[i]**2 * h) / h**2
    K[i, i+1] =  (nodes[i+1]**3 - nodes[i]**2 * (3*nodes[i+1] - 2*nodes[i])) / (6*h**2)
    K[i+1, i] = -(nodes[i]**3 - nodes[i+1]**2 * (3*nodes[i] - 2*nodes[i+1])) / (6*h**2)
  
  return K

def analytical_m(m_L, alpha_L, lambd, x, t):
  return m_L * (1 - erf(x / (2*np.sqrt(alpha_L*t))) / erf(lambd) )

def analytical_s(alpha_L, lambd, t):
  return 2*lambd*np.sqrt(alpha_L*t)

def lambdeq(lambd, c_L, m_L, l):
  return lambd * np.exp(lambd**2) * erf(lambd) - (m_L * c_L) / (l*np.sqrt(np.pi))

def finite_diff(heat_distribution, interface, n, h, space):
  x_domain_idx = np.where(space < interface[n])[0]  # enforce s 0 < x < s(t).
  m = heat_distribution[n, x_domain_idx] # grab m with support on x.
  dmdx = np.zeros_like(m) 

  # interior points.
  dmdx[1:-1] = (m[2:] - m[:-2]) / (2 * h)

  # one-sided for boundaries.
  dmdx[0] = (m[1] - m[0]) / h
  dmdx[-1] = (m[-1] - m[-2]) / h

  # scale to xi-space and return dmdxi at s(t) <=> xi = 1.
  dmdxi = interface[n] * dmdx
  return dmdxi[dmdxi < 0][-1]

def finite_diff_xi(heat_distribution, n, h):
  m_tilde = heat_distribution[n, :] # m_tilde()
  # last two points: one-sided difference
  dmdxi_interior = np.diff(m_tilde[1:-1]) / (2*h)
  print("dmdxi_interior=", dmdxi_interior)

  dmdxi_start = (m_tilde[1] - m_tilde[0]) / h
  print("dmdxi_start=", dmdxi_start)

  dmdxi_end = (m_tilde[-1] - m_tilde[-2]) / h
  print("dmdxi_end=", dmdxi_end)

  dmdxi = np.sum(dmdxi_interior) + dmdxi_start + dmdxi_end
  print("dmdxi=", dmdxi)

  return dmdxi
  