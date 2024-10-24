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

# %% Load saved results
with open('results/Experiment_3_Osc_results.pickle', 'rb') as openfile:
    all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc = pickle.load(openfile)
    assert len(all_x_t_n_Osc) == len(atlas_info), 'Mismatch in the number of ROIs'

with open('results/Experiment_3_Arn_results.pickle', 'rb') as openfile:
    all_x_t_n_Ar1, all_P_t_n_Ar1, em_iters_Ar1 = pickle.load(openfile)
    assert len(all_x_t_n_Ar1) == len(atlas_info), 'Mismatch in the number of ROIs'

# %% Compute localized energy ratio and ROC curve for all ROIs
ROI_names = []
rms_Osc = []
rms_Ar1 = []
e_ratios_Osc = np.zeros(len(atlas_info))
e_ratios_Ar1 = np.zeros(len(atlas_info))
percentiles = np.hstack((np.linspace(1, 90, 1000), np.linspace(90, 100, 1000)))
roc_hit_Osc = np.zeros((len(atlas_info), len(percentiles)))
roc_fa_Osc = np.zeros((len(atlas_info), len(percentiles)))
roc_hit_Ar1 = np.zeros((len(atlas_info), len(percentiles)))
roc_fa_Ar1 = np.zeros((len(atlas_info), len(percentiles)))

for (ROI_name, active_idx), ii in zip(atlas_info.items(), range(len(atlas_info))):
    print(f'Processing the {ii}th ROI: {ROI_name}...')
    # Save the ROI_names in order to be reused later
    ROI_names.append(ROI_name)
    # Compute the RMS of the localization distributions
    Osc_activity = np.sqrt(np.mean(all_x_t_n_Osc[ii] ** 2, axis=1))[0:-1:2]
    Ar1_activity = np.sqrt(np.mean(all_x_t_n_Ar1[ii] ** 2, axis=1))
    # Save the localization distributions across sources
    rms_Osc.append(Osc_activity)
    rms_Ar1.append(Ar1_activity)
    # Localized energy ratios
    squared_x = np.sum(all_x_t_n_Osc[ii] ** 2, axis=1)[0:-1:2]
    e_ratios_Osc[ii] = squared_x[active_idx].sum() / squared_x.sum()
    squared_x = np.sum(all_x_t_n_Ar1[ii] ** 2, axis=1)
    e_ratios_Ar1[ii] = squared_x[active_idx].sum() / squared_x.sum()
    # Compute ROC curves
    inactive_idx = np.setdiff1d(np.arange(G.shape[1]), active_idx)
    # Osc
    roc_hit = np.zeros(len(percentiles))
    roc_fa = np.zeros(len(percentiles))
    for ii in range(len(percentiles)):
        threshold = np.percentile(Osc_activity, percentiles[ii])
        roc_hit[ii] = np.mean(Osc_activity[active_idx] > threshold)
        roc_fa[ii] = np.mean(Osc_activity[inactive_idx] > threshold)
    roc_hit_Osc[ii, :] = roc_hit
    roc_fa_Osc[ii, :] = roc_fa
    # Ar1
    roc_hit = np.zeros(len(percentiles))
    roc_fa = np.zeros(len(percentiles))
    for ii in range(len(percentiles)):
        threshold = np.percentile(Ar1_activity, percentiles[ii])
        roc_hit[ii] = np.mean(Ar1_activity[active_idx] > threshold)
        roc_fa[ii] = np.mean(Ar1_activity[inactive_idx] > threshold)
    roc_hit_Ar1[ii, :] = roc_hit
    roc_fa_Ar1[ii, :] = roc_fa

# Save the computed results
with open('results/Experiment_3_visualization_results.pickle', 'wb') as openfile:
    pickle.dump((ROI_names, rms_Osc, rms_Ar1, e_ratios_Osc, e_ratios_Ar1, percentiles,
                 roc_hit_Osc, roc_fa_Osc, roc_hit_Ar1, roc_fa_Ar1), openfile)

# %% Plot the ROC curves
# fig, ax = plt.subplots(1,1)
# ax.plot(roc_fa_Osc.mean(axis=0), roc_hit_Osc.mean(axis=0))
# hit_std = np.std(roc_hit_Osc, axis=0)
# ax.fill_between(roc_fa_Osc.mean(axis=0),
#                 roc_hit_Osc.mean(axis=0)-hit_std,
#                 np.clip(roc_hit_Osc.mean(axis=0)+hit_std, None, 1.0),
#                 alpha=0.1)
# ax.plot(roc_fa_Ar1.mean(axis=0), roc_hit_Ar1.mean(axis=0))
# hit_std = np.std(roc_hit_Ar1, axis=0)
# ax.fill_between(roc_fa_Ar1.mean(axis=0),
#                 roc_hit_Ar1.mean(axis=0)-hit_std,
#                 np.clip(roc_hit_Ar1.mean(axis=0)+hit_std, None, 1.0),
#                 alpha=0.1)
# fig.show()

# %% Inspect the ROIs where the localized energy ratio is very low
# low_e_ratio_data = {}
# low_e_ratio_idx = np.where(e_ratios_Osc < 0.1)[0]

# for idx in low_e_ratio_idx:
#     ROI_name = ROI_names[idx]
#     x_t_n = all_x_t_n_Osc[idx]
#     low_e_ratio_data[ROI_name] = np.sqrt(np.mean(x_t_n ** 2, axis=1))[0:-1:2]

# with open('results/Experiment_3_low_e_ratio_data.pickle', 'wb') as openfile:
#     pickle.dump(low_e_ratio_data, openfile)

# if __name__ != '__main__':
#     with open('results/Experiment_3_low_e_ratio_data.pickle', 'rb') as openfile:
#         low_e_ratio_data = pickle.load(openfile)

#     low_e_ratio_ROIs = list(low_e_ratio_data.keys())

#     idx = 8

#     print(low_e_ratio_ROIs[idx])
#     active_idx = atlas_info[low_e_ratio_ROIs[idx]]
#     data = np.zeros((G.shape[1], 1000))
#     data[active_idx, :] = 1
#     stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
#                              tmin=0, tstep=1, subject='fs_FS6')
#     brain = stc.plot(hemi='both',
#                      title='Testing',
#                      subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
#                      clim=dict(kind='value', lims=[0, 0.04, 0.15]),
#                      views='lateral', initial_time=100,  smoothing_steps=10)

#     data = low_e_ratio_data[low_e_ratio_ROIs[idx]]
#     stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
#                              tmin=0, tstep=1, subject='fs_FS6')
#     brain = stc.plot(hemi='both',
#                      title='Testing',
#                      subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
#                      clim=dict(kind='value', lims=[0, 0.04, 0.15]),
#                      views='lateral', initial_time=100,  smoothing_steps=10)
