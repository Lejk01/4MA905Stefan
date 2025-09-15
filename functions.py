import numpy as np
from scipy.special import erf

def phi_j(j, h, x, nodes):
  x_lb = nodes[j-1]
  x_j = nodes[j]
  x_ub = nodes[j+1]
  
  if x >= x_lb  and x <= x_j:
    return (x - x_lb) / h
  elif x >= x_j and x <= x_ub:
    return (x_ub - x) / h
  else:
    return 0
  
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
  M[0, 0] = M[N-1, N-1] = h / 3

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

def construct_convection_matrix(N):
  K = np.zeros((N, N))
  val = .5
  for i in range(N-1):
    K[i, i+1] = val
    K[i+1, i] = -val

def analytical_m(m_L, alpha_L, lambd, x, t):
  return m_L * (1 - erf(x / (2*np.sqrt(alpha_L*t))) / erf(lambd) )

def analytical_s(alpha_L, lambd, t):
  return 2*lambd*np.sqrt(alpha_L*t)

def f_lambd(lambd, c_L, m_L, l):
  return lambd * np.exp(lambd**2) * erf(lambd) - (m_L * c_L) / (l*np.sqrt(np.pi))