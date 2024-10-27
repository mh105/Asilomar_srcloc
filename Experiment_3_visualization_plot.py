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
from somata.source_loc.source_loc_utils import get_atlas_source_indices
import matplotlib.pyplot as plt

# Load the atlas labels
subject = 'm2m_recon'
subjects_dir = 'data/JCHU_F_92_young/mri/simnibs_pipe/bem_surfaces/final_surface'
labels = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)

# Load forward model G and source space
fwd = mne.read_forward_solution('data/JCHU_F_92_young/fif/four_layer-ico4-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']  # (129, 5124)
src = fwd['src']

# Get the indices of the source space corresponding to atlas ROIs
atlas_info = get_atlas_source_indices(labels, src)

# Filter down to only the left hemisphere
atlas_info = {k: v for k, v in atlas_info.items() if 'lh' in k and 'Unknown' not in k}

# Load the computed results
with open('results/Experiment_3_visualization_results.pickle', 'rb') as openfile:
    ROI_names, rms_Osc, rms_Ar1, e_ratios_Osc, e_ratios_Ar1, percentiles, \
        roc_hit_Osc, roc_fa_Osc, roc_hit_Ar1, roc_fa_Ar1 = pickle.load(openfile)

# Plot scatter plot of the localized energy ratios
fig, ax = plt.subplots(1, 1)
for idx in range(len(ROI_names)):
    Ar1_x = 1 + (np.random.random() - 0.5) * 0.1
    ax.scatter(Ar1_x, e_ratios_Ar1[idx], color='blue')
    Osc_x = 2 + (np.random.random() - 0.5) * 0.1
    ax.scatter(Osc_x, e_ratios_Osc[idx], color='magenta')
    if e_ratios_Ar1[idx] >= e_ratios_Osc[idx]:
        ax.plot([Ar1_x, Osc_x], [e_ratios_Ar1[idx], e_ratios_Osc[idx]], color='red', alpha=0.25)
    else:
        ax.plot([Ar1_x, Osc_x], [e_ratios_Ar1[idx], e_ratios_Osc[idx]], color='green', alpha=0.25)
ax.set_xlim([0.5, 3])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Ar1', 'Oscillator'])
ax.set_ylabel('Localized Energy Ratio')
ax.plot([0, 0], [0, 0], color='red', label='Ar1 > Oscillator')
ax.plot([0, 0], [0, 0], color='green', label='Oscillator > Ar1')
ax.legend(loc='best')

# %% Plot the ROC curves
fig, ax = plt.subplots(1, 1)

ax.plot(roc_fa_Ar1.mean(axis=0), roc_hit_Ar1.mean(axis=0),
        color='blue', label='Ar1')
hit_std = np.std(roc_hit_Ar1, axis=0)
ax.fill_between(roc_fa_Ar1.mean(axis=0),
                roc_hit_Ar1.mean(axis=0)-hit_std,
                np.clip(roc_hit_Ar1.mean(axis=0)+hit_std, None, 1.0),
                color='blue',
                alpha=0.1)

ax.plot(roc_fa_Osc.mean(axis=0), roc_hit_Osc.mean(axis=0),
        color='magenta', label='Oscillator')
hit_std = np.std(roc_hit_Osc, axis=0)
ax.fill_between(roc_fa_Osc.mean(axis=0),
                roc_hit_Osc.mean(axis=0)-hit_std,
                np.clip(roc_hit_Osc.mean(axis=0)+hit_std, None, 1.0),
                color='magenta',
                alpha=0.1)

ax.set_xlabel('False Alarm Rate')
ax.set_ylabel('Hit Rate')

ax.legend(loc='best')
fig.show()

# Zoom in on the ROC curve
fig, ax = plt.subplots(1, 1)

ax.plot(roc_fa_Ar1.mean(axis=0), roc_hit_Ar1.mean(axis=0),
        color='blue', label='Ar1')
hit_std = np.std(roc_hit_Ar1, axis=0)
ax.fill_between(roc_fa_Ar1.mean(axis=0),
                roc_hit_Ar1.mean(axis=0)-hit_std,
                np.clip(roc_hit_Ar1.mean(axis=0)+hit_std, None, 1.0),
                color='blue',
                alpha=0.1)

ax.plot(roc_fa_Osc.mean(axis=0), roc_hit_Osc.mean(axis=0),
        color='magenta', label='Oscillator')
hit_std = np.std(roc_hit_Osc, axis=0)
ax.fill_between(roc_fa_Osc.mean(axis=0),
                roc_hit_Osc.mean(axis=0)-hit_std,
                np.clip(roc_hit_Osc.mean(axis=0)+hit_std, None, 1.0),
                color='magenta',
                alpha=0.1)

ax.set_xlim([0, 0.2])
ax.set_ylim([0.6, 1])

# remove ticks
ax.set_xticks([])
ax.set_yticks([])

# remove frame
for spine in ax.spines.values():
    spine.set_visible(False)

fig.show()

# %% Inspect the ROIs where the localized energy ratio is very low
with open('results/Experiment_3_low_e_ratio_data.pickle', 'rb') as openfile:
    low_e_ratio_data = pickle.load(openfile)

low_e_ratio_ROIs = list(low_e_ratio_data.keys())

idx = 8

print(low_e_ratio_ROIs[idx])
active_idx = atlas_info[low_e_ratio_ROIs[idx]]
data = np.zeros((G.shape[1], 1000))
data[active_idx, :] = 1
stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                         tmin=0, tstep=1, subject='fs_FS6')
brain = stc.plot(hemi='both',
                 title='Testing',
                 subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                 clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                 views='lateral', initial_time=100,  smoothing_steps=10)

data = low_e_ratio_data[low_e_ratio_ROIs[idx]]
stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                         tmin=0, tstep=1, subject='fs_FS6')
brain = stc.plot(hemi='both',
                 title='Testing',
                 subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                 clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                 views='lateral', initial_time=100,  smoothing_steps=10)
