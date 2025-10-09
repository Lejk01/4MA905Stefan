import numpy as np
from scipy.special import erf

def phi_j(j, h, x, nodes):
  x_j = nodes[j]
  x_lb = nodes[j-1] if j-1 >= 0 else x_j
  x_ub = nodes[j+1] if j+1 < len(nodes) else x_j

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
  x_j = nodes[j]
  x_lb = nodes[j - 1] if j - 1 >= 0 else x_j
  x_ub = nodes[j + 1] if j + 1 < len(nodes) else x_j
  
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

  for i in range(N-1):
    K[i,i] = -h/3
    if i > 0:
      K[i, i-1] = -(2*nodes[i+1]**3 - 2*nodes[i]**3 - 3*nodes[i+1]**2*nodes[i] + 3*nodes[i]**3) / (6*h**2)
      K[i-1, i] = -(2*nodes[i+1]**3 - 2*nodes[i]**3 - 3*nodes[i+1]**3 + 3*nodes[i]**2*nodes[i+1]) / (6*h**2)

  return K

def construct_convection2_matrix(N, h, nodes):
  K = np.zeros((N, N))
  for e in range(N-1):
      xL, xR = e*h, (e+1)*h
      # ∫ x*(xR-x) dx and ∫ x*(x-xL) dx (antiderivatives)
      base1 = (xR*(xR**2)/2 - (xR**3)/3) - (xR*(xL**2)/2 - (xL**3)/3)   # ∫ x*(xR - x) dx
      base2 = ((xR**3)/3 - xL*(xR**2)/2) - ((xL**3)/3 - xL*(xL**2)/2)   # ∫ x*(x - xL) dx

      # Local matrix with a = trial(derivative) index, b = test index
      I = np.zeros((2,2))
      I[0,0] = (-1.0/(h*h))*base1   # a=1 (left),  b=1 (left)
      I[0,1] = (-1.0/(h*h))*base2   # a=1,          b=2
      I[1,0] = ( 1.0/(h*h))*base1   # a=2 (right),  b=1
      I[1,1] = ( 1.0/(h*h))*base2   # a=2,          b=2

      nodes = [e, e+1]
      for aL, aG in enumerate(nodes):    # trial/derivative basis index -> column
          if aG==0 or aG==N-1: continue
          col = aG-1
          for bL, bG in enumerate(nodes):  # test basis index -> row
              if bG==0 or bG==N-1: continue
              row = bG-1
              K[row, col] += I[aL, bL]
  return K

def analytical_m(m_L, alpha_L, lambd, x, t):
  return m_L * (1 - erf(x / (2*np.sqrt(alpha_L*t))) / erf(lambd) )

def analytical_s(alpha_L, lambd, t, s0):
  return np.sqrt(s0**2 + 4*alpha_L*lambd**2*t)

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
  