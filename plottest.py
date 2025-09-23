import numpy as np
import matplotlib.pyplot as plt

HORIZON = 600
TIME_POINTS = 2000
time_p = np.linspace(0, 1, TIME_POINTS)
time_h = np.linspace(0, HORIZON, TIME_POINTS)
time_graded = time_p**2 * time_h

plt.plot(range(1, TIME_POINTS + 1), time_graded, linestyle=':')
plt.show()