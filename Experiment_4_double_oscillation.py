"""
Simulation studies Experiment 4:

        Activate a patch of sources with both slow and alpha oscillations

- Activate a patch of sources within an ROI from atlas
- Add both slow and alpha oscillations to the patch
- Localize and examine the recovered source activity
"""

import mne
import pickle
import numpy as np
from codetiming import Timer
from somata import OscillatorModel as Osc
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import get_atlas_source_indices

# Set the random seed
np.random.seed(1023)

# %% Figure out which sources to activate

# Load the atlas labels
subject = 'm2m_recon'
subjects_dir = 'data/JCHU_F_92_young/mri/simnibs_pipe/bem_surfaces/final_surface'
labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)

# Load forward model G
fwd = mne.read_forward_solution('data/JCHU_F_92_young/fif/four_layer-ico4-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']  # (129, 5124)
src = fwd['src']

# Get the indices of the source space corresponding to atlas ROIs
atlas_info = get_atlas_source_indices(labels, src)
vert_to_source, source_to_vert = Src._vertex_source_mapping(src)
patch_order = 3
scaling = (1, 1, 1)  # 0.5, 0.25, 0.125
neighbors = Src._define_neighbors(src, order=patch_order)

# Pick one large ROI in the left medial frontal cortex
active_idx = atlas_info['rostralmiddlefrontal-lh']
hemi = 0

# Pick some sources from the ROI
num_src_loop = 10
center_seeds = np.random.choice(active_idx, num_src_loop, replace=False)

# (Optional) Visualize the selected sources
if __name__ != '__main__':
    data = np.zeros((G.shape[1], 1000))
    data[center_seeds, :] = 1
    stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                             tmin=0, tstep=1, subject='fs_FS6')
    brain = stc.plot(hemi='both',
                     title='Testing',
                     subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                     clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                     views='lateral', initial_time=100,  smoothing_steps=10)

# %% Now we simulate activity in the active sources and perform localization

# Oscillator parameters for simulating source activity
# simulation_mode = 'oscillator'  # (oscillator or sinusoid) used for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 10  # (s) total duration of simulated activity
# a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
# f = 10  # (Hz) center frequency of oscillation in Hertz
Q = 1  # (Am^2) state noise covariance for the active oscillator only
# mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
# Q0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

# Under the assumption of white observation noise, we can control the average SNR on the scalp
SNR_power = 9
SNR_amplitude = np.sqrt(SNR_power)
scalp_amplitude = np.sqrt(R * SNR_power * 2)  # two-sided power of wave: A_rms^2/2
src_scale = scalp_amplitude / np.mean(abs(G), axis=(0, 1))

# Assume a noiseless background of source activity for resolution matrix calculation
neeg, nsources = G.shape  # (129, 5124)
ntime = T * Fs  # + 1 removed since using o1.simulate()
x_blank = np.zeros((G.shape[1], ntime))

all_x_t_n_Osc = []
all_P_t_n_Osc = []
em_iters_Osc = np.zeros(nsources, dtype=np.float64)

max_iter = 10

# Simulate the same source activity that will be re-used across ROIs
o1 = Osc(a=0.98, freq=1, sigma2=Q * 3, Fs=Fs, R=0)
_, slow_activity = o1.simulate(duration=T)
o2 = Osc(a=0.98, freq=10, sigma2=Q, Fs=Fs, R=0)
_, alpha_activity = o2.simulate(duration=T)
simulated_src = slow_activity + alpha_activity
rms_amplitude = np.sqrt(np.mean(simulated_src ** 2))

# scale to produce the right average scalp SNR
slow_true = src_scale / rms_amplitude * slow_activity
alpha_true = src_scale / rms_amplitude * alpha_activity

if __name__ != '__main__':
    import matplotlib.pyplot as plt
    plt.plot(np.squeeze(slow_true + alpha_true))
    plt.plot(slow_true.T)
    plt.plot(alpha_true.T)

# Simulate the same observation noise that will be re-used across ROIs
observation_noise = np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

vidx = center_seeds[0]

# %% Simulate activity in the selected patch of sources and localize
with Timer():
    # Place simulated_src in the correct row of x that corresponds to the center source
    x = np.copy(x_blank)
    x[vidx, :] += np.squeeze(slow_true + alpha_true)

    # Activate the neighboring sources around the center source
    vert = source_to_vert[hemi][vidx]  # vertex indexing
    for order, neighbor_scale in zip(list(range(1, patch_order + 1)), scaling):
        vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                    for x in neighbors[hemi][vert][order]])
        # Filter out neighbor vertices that are not sources
        valid_idx = np.invert(np.isnan(vert_neighbor))
        vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)

        # Add the same activity to the neighbor sources
        x[vert_neighbor, :] += neighbor_scale * x[vidx, :]

    # Multiply by fwd model to get EEG scalp activity and add observation noise
    y = G @ x + observation_noise

    # Dynamic source localization
    components = [Osc(a=0.98, freq=1, Fs=Fs), Osc(a=0.96, freq=10, Fs=Fs)]
    src1 = Src(components=components, fwd=fwd, d1=0.5, d2=0.25, m1=0.5, m2=0.5)
    x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, update_param='Q')
    all_x_t_n_Osc.append(x_t_n)
    all_P_t_n_Osc.append(P_t_n)
    em_iters_Osc = src1.em_log['em_iter']

# Save the results
with open('results/Experiment_4_Osc_results.pickle', 'wb') as openfile:
    pickle.dump((all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc, slow_true, alpha_true, center_seeds), openfile)

# %% Visualize the results
if __name__ != '__main__':
    # Load the results
    with open('results/Experiment_4_Osc_results.pickle', 'rb') as openfile:
        all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc, \
            slow_true_save, alpha_true_save, center_seeds_save = pickle.load(openfile)

    x_t_n = all_x_t_n_Osc[0]
    slow_x_t_n = x_t_n[:nsources * 2, :]
    alpha_x_t_n = x_t_n[nsources * 2:, :]

    # Visualize the localized source activity for slow
    data = np.sqrt(np.mean(slow_x_t_n ** 2, axis=1))[0:-1:2]
    stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                             tmin=0, tstep=1, subject='fs_FS6')
    brain = stc.plot(hemi='both',
                     title='Testing',
                     subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                     clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                     views='lateral', initial_time=100,  smoothing_steps=10)

    # Visualize the localized source activity for alpha
    data = np.sqrt(np.mean(alpha_x_t_n ** 2, axis=1))[0:-1:2]
    stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                             tmin=0, tstep=1, subject='fs_FS6')
    brain = stc.plot(hemi='both',
                     title='Testing',
                     subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                     clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                     views='lateral', initial_time=100,  smoothing_steps=10)

    # Visualize the true activated patch of sources
    data = np.zeros((G.shape[1], 1000))
    data[vidx, :] = 1
    # Activate the neighboring sources around the center source
    vert = source_to_vert[hemi][vidx]  # vertex indexing
    for order, neighbor_scale in zip(list(range(1, patch_order + 1)), scaling):
        vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                    for x in neighbors[hemi][vert][order]])
        # Filter out neighbor vertices that are not sources
        valid_idx = np.invert(np.isnan(vert_neighbor))
        vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)

        # Add the same activity to the neighbor sources
        data[vert_neighbor, :] += neighbor_scale * data[vidx, :]

    stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                             tmin=0, tstep=1, subject='fs_FS6')
    brain = stc.plot(hemi='both',
                     title='Testing',
                     subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                     clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                     views='lateral', initial_time=100,  smoothing_steps=10)

    import matplotlib.pyplot as plt

    plt.plot(slow_x_t_n[0:-1:2, :][vidx, :])
    plt.plot(alpha_x_t_n[0:-1:2, :][vidx, :])

    plt.plot(slow_x_t_n[0:-1:2, :][np.argmax(np.sqrt(np.mean(slow_x_t_n ** 2, axis=1))[0:-1:2]), :])
    plt.plot(alpha_x_t_n[0:-1:2, :][np.argmax(np.sqrt(np.mean(alpha_x_t_n ** 2, axis=1))[0:-1:2]), :])

    # Inspect the true source activity in comparison to the localized source activity
    plt.plot(np.squeeze(slow_true_save + alpha_true_save))
    plt.plot(slow_true_save.T)
    plt.plot(alpha_true_save.T)

    plt.plot(slow_true_save.T)
    plt.plot(slow_x_t_n[0:-1:2, :][np.argmax(np.sqrt(np.mean(slow_x_t_n ** 2, axis=1))[0:-1:2]), :])

    plt.plot(alpha_true_save.T)
    plt.plot(alpha_x_t_n[0:-1:2, :][np.argmax(np.sqrt(np.mean(alpha_x_t_n ** 2, axis=1))[0:-1:2]), :])
