"""
Simulation studies Experiment 3:

        Activate a region on the cortex and compare the recovered result

- Activate a patch of sources within an ROI from the Destrieux Atlas
- Rotate through all atlas ROIs on one hemisphere
- Compute localized energy ratio as well as ROC curve
"""

import mne
import pickle
import numpy as np
from codetiming import Timer
from somata import OscillatorModel as Osc
from somata import AutoRegModel as Arn
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import simulate_oscillation, get_atlas_source_indices

# Set the random seed
rng = np.random.default_rng(1015)

# %% Figure out which sources to activate

# Load the atlas labels
subject = 'm2m_recon'
subjects_dir = 'data/JCHU_F_92_young/mri/simnibs_pipe/bem_surfaces/final_surface'
labels = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)

# Load forward model G
fwd = mne.read_forward_solution('data/JCHU_F_92_young/fif/four_layer-ico4-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']

# Load additional variables needed for visualizing the cortex
raw = mne.io.read_raw_fif('data/JCHU_F_92_young/fif/JCHU_F_92_night2_Resting_ds500_Z3-raw.fif')
trans = mne.read_trans('data/JCHU_F_92_young/fif/JCHU_F_92_night2-trans.fif')
src = fwd['src']

# Get the indices of the source space corresponding to atlas ROIs
atlas_info = get_atlas_source_indices(labels, src)

# Filter down to only the left hemisphere
atlas_info = {k: v for k, v in atlas_info.items() if 'lh' in k and 'Unknown' not in k}

# %% Now we simulate activity in the active sources and perform localization

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

# Under the assumption of white observation noise, we can control the average SNR on the scalp
SNR_power = 9
SNR_amplitude = np.sqrt(SNR_power)
scalp_amplitude = np.sqrt(R * SNR_power * 2)  # two-sided power of wave: A_rms^2/2
src_scale = scalp_amplitude / np.mean(abs(G), axis=(0, 1))

# Assume a noiseless background of source activity for resolution matrix calculation
neeg, nsources = G.shape  # (129, 5124)
ntime = T * Fs + 1
x_blank = np.zeros((G.shape[1], ntime))

all_x_t_n_Osc = []
all_P_t_n_Osc = []
em_iters_Osc = np.zeros(nsources, dtype=np.float64)

all_x_t_n_Ar1 = []
all_P_t_n_Ar1 = []
em_iters_Ar1 = np.zeros(nsources, dtype=np.float64)

max_iter = 10

# Simulate the same source activity that will be re-used across ROIs
simulated_src = simulate_oscillation(f, a, Q, mu0, Q0, Fs, T, oscillation_type=simulation_mode)
rms_amplitude = np.sqrt(np.mean(simulated_src ** 2))

# Simulate the same observation noise that will be re-used across ROIs
observation_noise = rng.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

for (ROI_name, active_idx), vidx in zip(atlas_info.items(), range(len(atlas_info))):
    with Timer():
        print(f'Processing the {vidx}th ROI: {ROI_name}...')

        # Place simulated_src in the correct row of x that corresponds to the activated source/vertex index
        x = np.copy(x_blank)
        x[active_idx, :] += src_scale / rms_amplitude * simulated_src  # scale to the right average scalp SNR

        # Multiply by fwd model to get EEG scalp activity and add observation noise
        y = G @ x + observation_noise

        # Dynamic source localization
        components = Osc(a=0.95, freq=f, Fs=Fs)
        src1 = Src(components=components, fwd=fwd, d1=0.5, d2=0.25, m1=0.5, m2=0.5)
        x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, update_param='Q')
        all_x_t_n_Osc.append(x_t_n)
        all_P_t_n_Osc.append(P_t_n)
        em_iters_Osc[vidx] = src1.em_log['em_iter']

        components = Arn(coeff=0.95)
        src1 = Src(components=components, fwd=fwd, d1=0.5, d2=0.25, m1=0.5, m2=0.5)
        x_t_n, P_t_n = src1.learn(y=y, R=R, SNR=SNR_amplitude, max_iter=max_iter, update_param='Q')
        all_x_t_n_Ar1.append(x_t_n)
        all_P_t_n_Ar1.append(P_t_n)
        em_iters_Ar1[vidx] = src1.em_log['em_iter']

# Save the results
with open('results/Experiment_3_Osc_results.pickle', 'wb') as openfile:
    pickle.dump((all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc), openfile)

with open('results/Experiment_3_Arn_results.pickle', 'wb') as openfile:
    pickle.dump((all_x_t_n_Ar1, all_P_t_n_Ar1, em_iters_Ar1), openfile)
