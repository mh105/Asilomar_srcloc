"""
Simulation studies Experiment 1:

        Resolution matrix calculation and comparisons

- Activate source activity in one patch of sources at a time
"""

import mne
import pickle
import numpy as np
from codetiming import Timer
from somata import OscillatorModel as Osc
from somata import AutoRegModel as Arn
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import simulate_oscillation

# Load forward model G
fwd = mne.read_forward_solution('eeganes02-neeg64-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']

# Oscillator parameters for simulating source activity
simulation_mode = 'oscillator'  # (oscillator or sinusoid) used for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 10  # (s) total duration of simulated activity
a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
f = 10  # (Hz) center frequency of oscillation in Hertz
Q = 1  # (Am^2) state noise covariance for the active oscillator only
mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
Q0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

# Set the random seed
rng = np.random.default_rng(1015)

# Under the assumption of white observation noise, we can control the average SNR on the scalp
SNR_power = 9
SNR_amplitude = np.sqrt(SNR_power)
scalp_amplitude = np.sqrt(R * SNR_power * 2)  # two-sided power of wave: A_rms^2/2
src_scale = scalp_amplitude / np.mean(abs(G), axis=(0, 1))

# Assume a noiseless background of source activity for resolution matrix calculation
neeg, nsources = G.shape  # (64,1162)
ntime = T * Fs + 1
x_blank = np.zeros((G.shape[1], ntime))

all_x_t_n_Osc = []
all_P_t_n_Osc = []
em_iters_Osc = np.zeros(nsources, dtype=np.float64)

all_x_t_n_Ar1 = []
all_P_t_n_Ar1 = []
em_iters_Ar1 = np.zeros(nsources, dtype=np.float64)

max_iter = 10

# Create dictionaries to identify neighbors
src = fwd['src']
vert_to_source, source_to_vert = Src._vertex_source_mapping(src)
neighbors = Src._define_neighbors(src)

# Simulate source activity in one patch of sources at a time
with Timer():
    for hemi in range(len(src)):
        for vidx in source_to_vert[hemi]:
            print('vertex' + str(vidx))

            # Simulate the source activity in a single source point
            simulated_src = simulate_oscillation(f, a, Q, mu0, Q0, Fs, T, oscillation_type=simulation_mode)
            rms_amplitude = np.sqrt(np.mean(simulated_src ** 2))

            # Place simulated_src in the correct row of x that corresponds to the activated source/vertex index
            x = np.copy(x_blank)
            x[vidx, :] += src_scale / rms_amplitude * simulated_src  # scale to the right average scalp SNR

            # Find the indices of first/second-order neighbors
            vert = source_to_vert[hemi][vidx]  # vertex indexing
            for order, neighbor_scale in zip([1, 2], (0.1, 0.05)):
                vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                            for x in neighbors[hemi][vert][order]])

                # Filter out neighbor vertices that are not sources
                valid_idx = np.invert(np.isnan(vert_neighbor))
                vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)

                # Add the same activity to the neighbor sources
                x[vert_neighbor, :] += neighbor_scale * x[vidx, :]

            # Multiply by fwd model to get EEG scalp activity and add observation noise
            y = G @ x + rng.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

            # Dynamic source localization
            components = Osc(a=0.95, freq=f, Fs=Fs)
            src1 = Src(components=components, fwd=fwd)
            x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, update_param='Q')
            all_x_t_n_Osc.append(x_t_n)
            all_P_t_n_Osc.append(P_t_n)
            em_iters_Osc[vidx] = src1.em_log['em_iter']

            components = Arn(coeff=0.95)
            src1 = Src(components=components, fwd=fwd)
            x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, update_param='Q')
            all_x_t_n_Ar1.append(x_t_n)
            all_P_t_n_Ar1.append(P_t_n)
            em_iters_Ar1[vidx] = src1.em_log['em_iter']

# Save the results
with open('results/Experiment_1-9-1_Osc_results.pickle', 'wb') as openfile:
    pickle.dump((all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc), openfile)

with open('results/Experiment_1-9-1_Arn_results.pickle', 'wb') as openfile:
    pickle.dump((all_x_t_n_Ar1, all_P_t_n_Ar1, em_iters_Ar1), openfile)