"""
Simulation study
- new resolution matrix, activate a patch of sources at a time
"""

import mne
import pickle
import numpy as np
from codetiming import Timer
from somata import OscillatorModel as Osc
from somata.source_loc import SourceLocModel as Src
from simulation_utils import simulate_oscillation  # resolution_matrix_metrics
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter

# Load forward model G
fwd = mne.read_forward_solution('eeganes02-neeg64-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']

# Oscillator parameters for simulating source activity
simulation_mode = 'sinusoid'  # (oscillator or sinusoid) used for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 10  # (s) total duration of simulated activity
a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
f = 10  # (Hz) center frequency of oscillation in Hertz
Q = 0  # (Am^2) state noise covariance for the active oscillator only
mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
Q0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

# Set the random seed
np.random.seed(226)

# Under the assumption of white observation noise, we can control the average SNR on the scalp
SNR_power = 3
SNR_amplitude = np.sqrt(SNR_power)
scalp_amplitude = np.sqrt(R * SNR_power * 4)  # sine wave two-sided power: A^2/4
src_scale = scalp_amplitude / np.mean(abs(G), axis=(0, 1))

# Assume a noiseless background of source activity for resolution matrix calculation
neeg, nsources = G.shape  # (64,1162)
ntime = T * Fs + 1
x_blank = np.zeros((G.shape[1], ntime))

res_mat = np.zeros((nsources, nsources), dtype=np.float64)
em_iters = np.zeros(nsources, dtype=np.float64)
max_iter = 10

all_x_t_n_Osc = []
all_P_t_n_Osc = []
em_iters_Osc = np.zeros(nsources, dtype=np.float64)

# Create dictionaries to identify neighbors
src = fwd['src']
vert_to_source, source_to_vert = Src._vertex_source_mapping(src)
neighbors = Src._define_neighbors(src)

# Simulate source activity in one patch of sources at a time
for hemi in range(len(src)):
    for vidx in source_to_vert[hemi]:
        with Timer():
            print('vertex' + str(vidx))

            # Simulate the source activity in a single source point
            simulated_src = simulate_oscillation(f, a, Q, mu0, Q0, Fs, T, oscillation_type=simulation_mode)

            # Place simulated_src in the correct row of x that correspond to the activated source/vertex index
            x = np.copy(x_blank)
            x[vidx, :] += src_scale * simulated_src  # scale to the right average scalp SNR

            # Find the indices of first/second-order neighbors
            vert = source_to_vert[hemi][vidx]  # vertex indexing
            for order, neighbor_scale in zip(['first', 'second'], (0.1, 0.05)):
                vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                            for x in neighbors[hemi][vert][order]])

                # Filter out neighbor vertices that are not sources
                valid_idx = np.invert(np.isnan(vert_neighbor))
                vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)

                # Add the activity to the neighbor sources
                x[vert_neighbor, :] += neighbor_scale * src_scale * simulated_src

            # Multiply by fwd model to get EEG scalp activity and add observation noise
            y = G @ x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

            # Dynamic source localization
            components = Osc(a=0.99, freq=f, Fs=Fs)
            src1 = Src(components=components, fwd=fwd, d1=0.1, d2=0.05, m1=0.9, m2=0.1)
            x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, keep_param='R')

            # Store the hidden state estimates in resolution matrix
            res_mat[:, vidx] = x_t_n[:, 100:-100].max(axis=1)[0:-1:2]  # cutoff beginning and end

            # Record how many EM iterations successfully completed
            em_iters[vidx] = src1.em_log['em_iter']

            components = Osc(a=0.99, freq=f, Fs=Fs)
            src1 = Src(components=components, fwd=fwd)
            x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, keep_param='R')
            all_x_t_n_Osc.append(x_t_n)
            all_P_t_n_Osc.append(P_t_n)
            em_iters_Osc[vidx] = src1.em_log['em_iter']

            # plt.figure()
            # plt.plot(res_mat[:, vidx])
            # plt.title('all source activity')
            #
            # print(src1.components[0])

            """
                plt.figure(); plt.plot(x[vidx, :]); plt.title('True source')
                plt.figure(); plt.plot(x_t_n[vidx*2, :]); plt.title('estimated activity')
                plt.figure(); plt.plot(x_t_n[:, 4]); plt.title('all source activity')
            """

# Save the results
if res_mat[0, 0] != 0:
    with open('results/simulate_study_patch_source_res_mat_patch_source.pickle', 'wb') as openfile:
        pickle.dump((res_mat, em_iters), openfile)

with open('results/simulate_study_patch_source_Osc_results.pickle', 'wb') as openfile:
    pickle.dump((all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc), openfile)

# plt.figure()
# plt.imshow(res_mat, vmin=0, vmax=0.08, interpolation='none')
# plt.xlabel('# Source Vertex', fontsize=16)
# plt.ylabel('# Source Vertex', fontsize=16)
# plt.title('Dynamic Resolution Matrix', fontsize=20)
# plt.colorbar()

# #
# #
# #
# #
# #
# # Compare with MNE resolution matrix
# MNE_R = R * np.eye(G.shape[0], dtype=np.float64)
# MNE_G = G
# MNE_Q = np.trace(MNE_R) / np.trace(MNE_G @ MNE_G.T) * SNR_power * np.eye(MNE_G.shape[1], dtype=np.float64)
# M = MNE_Q @ MNE_G.T @ np.linalg.inv(MNE_G @ MNE_Q @ MNE_G.T + MNE_R)
# MNE_res_mat = np.abs(M @ G)

# plt.figure()
# plt.imshow(MNE_res_mat, vmin=0, vmax=0.05, interpolation='none')
# plt.xlabel('# Source Vertex', fontsize=16)
# plt.ylabel('# Source Vertex', fontsize=16)
# plt.title('MNE Resolution Matrix', fontsize=20)
# plt.colorbar()

# #
# #
# #
# #
# #
# # Load the saved results and compute resolution matrix metrics
# with open('pickle_files/res_mat_patch_source.pickle', 'rb') as openfile:
#     res_mat, em_iters = pickle.load(openfile)

# SD, DLE, RI = resolution_matrix_metrics(res_mat, fwd['src'])
# MNE_SD, MNE_DLE, MNE_RI = resolution_matrix_metrics(MNE_res_mat, fwd['src'])

# bins = np.arange(-0.25, 13.75, 0.5)
# bin_centers = 0.5 * (bins[1:] + bins[:-1])

# plt.figure()
# plt.hist(MNE_SD, bins=bins, alpha=0.5, density=True, edgecolor='black', label='MNE')
# plt.hist(SD, bins=bins, alpha=0.5, density=True, edgecolor='black',  label='Dynamic')
# plt.legend(prop={'size': 10})
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# plt.xlabel('Spatial Dispersion (cm)', fontsize=16)
# plt.ylabel('pmf', fontsize=16)
# plt.title('Resolution Matrix Metric: SD', fontsize=16)
# plt.savefig('SD_compare_MNE_Dynamic.svg')

# plt.figure()
# plt.hist(MNE_DLE, bins=bins, alpha=0.5, density=True, edgecolor='black',  label='MNE')
# plt.hist(DLE, bins=bins, alpha=0.5, density=True, edgecolor='black', label='Dynamic')
# plt.legend(prop={'size': 10})
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# plt.xlabel('Dipole Localization Error (cm)', fontsize=16)
# plt.ylabel('pmf', fontsize=16)
# plt.title('Resolution Matrix Metric: DLE', fontsize=16)
# plt.savefig('DLE_compare_MNE_Dynamic.svg')
