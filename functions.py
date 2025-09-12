import numpy as np

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
  M[0, 0] = M[N-1, N-1] = h/3

  val = h / 6
  # Set off-diagonal entries
  val = h / 6
  for i in range(N-1):  # loop to N-1 so i+1 is valid
      M[i, i+1] = val
      M[i+1, i] = val   # assign symmetric entry

  return M