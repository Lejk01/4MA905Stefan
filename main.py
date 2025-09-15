import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import (construct_mass_matrix, 
                       construct_convection_matrix, 
                       construct_stiffness_matrix,
                       analytical_m,
                       analytical_s
                       )

N_nodes = 100
HORIZON = 10000 # [0, T] = [0, HORIZON]
l = 2
k_L = 0.5
p_L = 0.5
m_L = 10
c_L = 0.5
space = np.linspace(start=0, stop=1, num=10_000, endpoint=True)
time = np.linspace(start=0, stop=HORIZON, num=10_000, endpoint=True)
h = 0.1
alpha_L = k_L / (p_L * c_L)
lambd = 0.5
nodes = np.arange(0, N_nodes, step=h)

s = np.array([analytical_s(alpha_L, lambd, t) for t in time])

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.lineplot(x=time, y=s)
plt.xlabel("Time")
plt.ylabel("s(t)")
plt.title("Interface position in x dimension")
plt.show()
