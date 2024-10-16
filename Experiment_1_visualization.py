"""
Simulation studies Experiment 1:

        Resolution matrix calculation and comparisons

- Activate source activity in one source at a time
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the results
with open('results/Experiment1_Osc_results.pickle', 'rb') as openfile:
    all_x_t_n_Ar1, all_P_t_n_Ar1, em_iters_Ar1 = pickle.load(openfile)

nsources = len(all_x_t_n_Ar1)
res_mat = np.zeros((nsources, nsources), dtype=np.float64)

for vidx in range(nsources):
    x_t_n = all_x_t_n_Ar1[vidx]
    res_mat[:, vidx] = np.sqrt(np.mean(x_t_n ** 2, axis=1))[0:-1:2]

plt.figure()
plt.imshow(res_mat, vmin=0, vmax=0.05, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('AR1 Matrix', fontsize=20)
plt.colorbar()