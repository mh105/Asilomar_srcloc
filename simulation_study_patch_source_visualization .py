"""
Simulation studies Experiment 1:

        Resolution matrix calculation and comparisons

- Activate source activity in one source at a time
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import mne
from plot_utils import plot_resmat_metrics

# Simulation parameters
fwd = mne.read_forward_solution('eeganes02-neeg64-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']

# Load the precomputed resolution matrix using the old simulation_study_patch_source.py code
with open('results/simulate_study_patch_source_res_mat_patch_source.pickle', 'rb') as openfile:
    res_mat, em_iters = pickle.load(openfile)

plt.figure()
plt.imshow(res_mat, cmap='hot', vmin=0, vmax=0.01, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('Osc ResMat', fontsize=20)
plt.colorbar()
plt.show()

# %% Load the results for the oscillator dynamic source localization
with open('results/simulate_study_patch_source_Osc_results.pickle', 'rb') as openfile:
    all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc = pickle.load(openfile)

nsources = len(all_x_t_n_Osc)
res_mat_Osc = np.zeros((nsources, nsources), dtype=np.float64)

for vidx in range(nsources):
    x_t_n = all_x_t_n_Osc[vidx]
    res_mat_Osc[:, vidx] = np.sqrt(np.mean(x_t_n ** 2, axis=1))[0:-1:2]

plt.figure()
plt.imshow(res_mat_Osc, cmap='hot', vmin=0, vmax=0.01, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('Osc ResMat', fontsize=20)
plt.colorbar()
plt.show()

# %% Visualize the metrics for the resolution matrices
fig, ax = plot_resmat_metrics(res_mat_Osc, fwd['src'], 'Osc')
fig.show()
