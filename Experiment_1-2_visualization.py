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
R = 1
SNR_power = 9

# %% Load the results for the oscillator dynamic source localization
with open('results/Experiment_1-2_Osc_results.pickle', 'rb') as openfile:
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

# %% Load the results for the AR1 dynamic source localization
with open('results/Experiment_1-2_Arn_results.pickle', 'rb') as openfile:
    all_x_t_n_Ar1, all_P_t_n_Ar1, em_iters_Ar1 = pickle.load(openfile)

nsources = len(all_x_t_n_Ar1)
res_mat_Ar1 = np.zeros((nsources, nsources), dtype=np.float64)

for vidx in range(nsources):
    x_t_n = all_x_t_n_Ar1[vidx]
    res_mat_Ar1[:, vidx] = np.sqrt(np.mean(x_t_n ** 2, axis=1))

plt.figure()
plt.imshow(res_mat_Ar1, cmap='hot', vmin=0, vmax=0.01, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('Ar1 ResMat', fontsize=20)
plt.colorbar()
plt.show()

# %% Compute the MNE resolution matrix
MNE_R = R * np.eye(G.shape[0], dtype=np.float64)
MNE_G = G
MNE_Q = np.trace(MNE_R) / np.trace(MNE_G @ MNE_G.T) * SNR_power * np.eye(MNE_G.shape[1], dtype=np.float64)
M = MNE_Q @ MNE_G.T @ np.linalg.inv(MNE_G @ MNE_Q @ MNE_G.T + MNE_R)
res_mat_MNE = np.abs(M @ G)

plt.figure()
plt.imshow(res_mat_MNE, cmap='hot', vmin=0, vmax=0.01, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('MNE ResMat', fontsize=20)
plt.colorbar()
plt.show()

# %% Visualize the metrics for the resolution matrices
fig, ax = plot_resmat_metrics(res_mat_Osc, fwd['src'], 'Osc')
fig.show()

fig, ax = plot_resmat_metrics(res_mat_Ar1, fwd['src'], 'Ar1')
fig.show()

fig, ax = plot_resmat_metrics(res_mat_MNE, fwd['src'], 'MNE')
fig.show()
