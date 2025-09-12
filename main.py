import numpy as np
from functions import construct_mass_matrix

N_nodes = 100
space = np.linspace(start=0, stop=1, num=10_000, endpoint=True)
h = 0.1
nodes = np.arange(0, N_nodes, step=h)

print(construct_mass_matrix(N_nodes, h))