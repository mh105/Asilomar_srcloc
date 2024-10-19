"""
Simulation studies Experiment 2:

        Activate a region on the cortex and compare the recovered result

- Activate a patch of source within an ROI from atlas
"""
import mne
import numpy as np
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import get_atlas_source_indices

# Load the atlas labels
subject = 'm2m_recon'
subjects_dir = 'data/JCHU_F_92_young/mri/simnibs_pipe/bem_surfaces/final_surface'
labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)

# Load forward model G
fwd = mne.read_forward_solution('data/JCHU_F_92_young/fif/four_layer-ico3-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']

# Load additional variables needed for visualizing the cortex
raw = mne.io.read_raw_fif('data/JCHU_F_92_young/fif/JCHU_F_92_night2_Resting_ds500_Z3-raw.fif')
trans = mne.read_trans('data/JCHU_F_92_young/fif/JCHU_F_92_night2-trans.fif')
src = fwd['src']

# # Visualize the sensors and brain surfaces
# mne.viz.plot_alignment(raw.info, trans=trans, eeg='original', subject=subject,
#                        subjects_dir=subjects_dir, dig=True,
#                        surfaces={'head': 0.2}, coord_frame='head', interaction='terrain')

# mne.viz.plot_alignment(raw.info, trans=trans, subject=subject,
#                        src=src, subjects_dir=subjects_dir, dig=True,
#                        surfaces={'head': 0.2, 'white': 1}, coord_frame='head',
#                        interaction='terrain')

# Get the indices of the source space corresponding to atlas ROIs
atlas_info = get_atlas_source_indices(labels, src)
vert_to_source, source_to_vert = Src._vertex_source_mapping(src)
neighbors = Src._define_neighbors(src)

# Pick one ROI
active_idx = atlas_info['rostralmiddlefrontal-lh']

# Pick one of the sources in the ROI
center_seed = np.random.choice(active_idx)

data = np.zeros((G.shape[1], 1000))
data[center_seed, :] = 1

hemi = 0
vert = source_to_vert[hemi][center_seed]  # vertex indexing
for order, neighbor_scale in zip(['first', 'second'], (0.5, 0.25)):
    vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                for x in neighbors[hemi][vert][order]])
    # Filter out neighbor vertices that are not sources
    valid_idx = np.invert(np.isnan(vert_neighbor))
    vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)

    # Add the same activity to the neighbor sources
    data[vert_neighbor, :] += neighbor_scale * data[center_seed, :]

# brain_kwargs = dict(alpha=1, background="white", cortex="low_contrast")
# brain = mne.viz.Brain(subject, subjects_dir=subjects_dir, **brain_kwargs)

stc = mne.SourceEstimate(data=data, vertices=[src[0]['vertno'], src[1]['vertno']],
                         tmin=0, tstep=1, subject='fs_FS6')
brain = stc.plot(hemi='both',
                 title='Testing',
                 subjects_dir='data/JCHU_F_92_young/mri/simnibs_pipe/mri2mesh',
                 clim=dict(kind='value', lims=[0, 0.04, 0.15]),
                 views='lateral', initial_time=100,  smoothing_steps=10)
