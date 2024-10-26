"""
Simulation studies Experiment 5:

        Activate a larger patch of slow, which contains a smaller patch of alpha

- Activate an ROI from atlas with slow oscillation
- Activate a patch of sources within an ROI from atlas
- Add alpha oscillation to the patch
- Localize and examine the recovered source activity
"""

import mne
import pickle
import numpy as np
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import get_atlas_source_indices
import matplotlib.pyplot as plt

# Set the random seed
np.random.seed(1024)

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

# (Optional) Visualize the ROI
if __name__ != '__main__':
    data = np.zeros((G.shape[1], 1000))
    data[active_idx, :] = 1
    stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                             tmin=0, tstep=1, subject='fs_FS6')
    brain = stc.plot(hemi='both',
                     title='Testing',
                     subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                     clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                     views='lateral', initial_time=100,  smoothing_steps=10)

# %% Load the results
with open('results/Experiment_5_Osc_results.pickle', 'rb') as openfile:
    all_x_t_n_Osc, all_P_t_n_Osc, em_iters_Osc, \
        slow_true_save, alpha_true_save, center_seeds_save = pickle.load(openfile)

# %% Manually visualize the localization results for one patch at a time
ii = 5
vidx = center_seeds_save[ii]

# Extract the estimated hidden states and separate by slow and alpha oscillations
x_t_n = all_x_t_n_Osc[ii]
slow_x_t_n = x_t_n[:G.shape[1] * 2, :]
alpha_x_t_n = x_t_n[G.shape[1] * 2:, :]

# Visualize the localized source activity for slow
data = np.sqrt(np.mean(slow_x_t_n ** 2, axis=1))[0:-1:2]
stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                         tmin=0, tstep=1, subject='fs_FS6')
brain = stc.plot(hemi='both',
                 title='Testing',
                 subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                 clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                 views='lateral', initial_time=100,  smoothing_steps=10)

# Visualize the true activated patch of sources for slow
data = np.zeros((G.shape[1], 1000))
data[active_idx, :] = 1
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

# Visualize the true activated patch of sources for alpha
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

# %% Manually visualize the localized source activity time traces
plt.plot(slow_x_t_n[0:-1:2, :][vidx, :])
plt.plot(alpha_x_t_n[0:-1:2, :][vidx, :])

# Inspect the true source activity in comparison to the localized source activity
plt.plot(np.squeeze(slow_true_save + alpha_true_save))
plt.plot(slow_true_save.T)
plt.plot(alpha_true_save.T)

plt.plot(slow_x_t_n[0:-1:2, :][vidx, :])
plt.plot(slow_true_save.T)

plt.plot(alpha_x_t_n[0:-1:2, :][vidx, :])
plt.plot(alpha_true_save.T)

# Sources with the highest RMS
plt.plot(slow_true_save.T)
plt.plot(slow_x_t_n[0:-1:2, :][np.argmax(np.sqrt(np.mean(slow_x_t_n ** 2, axis=1))[0:-1:2]), :])

plt.plot(alpha_true_save.T)
plt.plot(alpha_x_t_n[0:-1:2, :][np.argmax(np.sqrt(np.mean(alpha_x_t_n ** 2, axis=1))[0:-1:2]), :])

# %% Compute localized energy ratio in each patch separately for slow and alpha
e_ratios_slow = np.zeros(len(all_x_t_n_Osc))
e_ratios_alpha = np.zeros(len(all_x_t_n_Osc))

for ii in range(len(all_x_t_n_Osc)):
    vidx = center_seeds_save[ii]
    x_t_n = all_x_t_n_Osc[ii]
    slow_x_t_n = x_t_n[:G.shape[1] * 2, :]
    alpha_x_t_n = x_t_n[G.shape[1] * 2:, :]

    # Figure out the active sources in the patch
    active_idx_patch = np.array([vidx])
    # Activate the neighboring sources around the center source
    vert = source_to_vert[hemi][vidx]  # vertex indexing
    for order, neighbor_scale in zip(list(range(1, patch_order + 1)), scaling):
        vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                    for x in neighbors[hemi][vert][order]])
        # Filter out neighbor vertices that are not sources
        valid_idx = np.invert(np.isnan(vert_neighbor))
        vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)
        # Append to the active source idx array
        active_idx_patch = np.append(active_idx_patch, vert_neighbor)
    assert len(active_idx_patch) == len(np.unique(active_idx_patch)), 'Duplicated source indices in the patch'

    squared_x_slow = np.sum(slow_x_t_n ** 2, axis=1)[0:-1:2]
    e_ratios_slow[ii] = squared_x_slow[active_idx].sum() / squared_x_slow.sum()

    squared_x_alpha = np.sum(alpha_x_t_n ** 2, axis=1)[0:-1:2]
    e_ratios_alpha[ii] = squared_x_alpha[active_idx_patch].sum() / squared_x_alpha.sum()
