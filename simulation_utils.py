import numpy as np
from somata import OscillatorModel as Osc
from somata.source_loc import SourceLocModel as Src
import mne


def simulate_oscillation(center_frequency_Hz, damping_factor, state_noise_variance, initial_state_mean,
                         initial_state_variance, sampling_frequency, duration, oscillation_type='oscillator'):
    """Simulate oscillation activity time series"""
    time_step = 1 / sampling_frequency
    time_axis = np.arange(0, duration + time_step, time_step)
    n_time_steps = time_axis.shape[0]

    if oscillation_type == 'oscillator':
        transition_matrix = Osc.get_rot_mat(damping_factor, Osc.hz_to_rad(center_frequency_Hz, sampling_frequency))
        simulated = np.zeros((2, n_time_steps))
        simulated[:, 0] += np.matmul(transition_matrix,
                                     np.random.multivariate_normal(initial_state_mean,
                                                                   initial_state_variance * np.eye(2, 2))
                                     ) + np.random.multivariate_normal([0, 0], state_noise_variance * np.eye(2, 2))
        for ii in range(n_time_steps - 1):
            simulated[:, ii + 1] = np.matmul(transition_matrix, simulated[:, ii]) + \
                                   np.random.multivariate_normal([0, 0], state_noise_variance * np.eye(2, 2))
        simulated = simulated[1, :]  # keep only the real component

    elif oscillation_type == 'sinusoid':
        # amplitude = 2 * np.pi * damping_factor * np.sqrt(state_noise_variance)  # this might not be correct
        # simulated = amplitude * np.sin(2 * np.pi * center_frequency_Hz * time_axis) + \
        #             np.random.normal(0, np.sqrt(state_noise_variance), n_time_steps)
        simulated = np.sin(2 * np.pi * center_frequency_Hz * time_axis)

    else:
        raise ValueError('Unrecognized oscillation_type value')

    return simulated


def resolution_matrix_metrics(K, src):
    """
    Compute the resolution matrix metrics:
        - Spatial dispersion (SD)
        - Dipole localization error (DLE)
        - Resolution index (RI)
    """
    nsources = K.shape[0]

    # Compute the pairwise distance between all triangulation vertices
    if src[0]['dist'] is None:
        mne.add_source_space_distances(src)

    # Create a distance matrix between sources
    D = np.zeros((nsources, nsources), dtype=np.float64)
    _, source_to_vert = Src._vertex_source_mapping(src)

    for hemi in range(len(src)):
        for vidx in source_to_vert[hemi]:
            vert = source_to_vert[hemi][vidx]  # vertex indexing

            for vidx2 in source_to_vert[hemi]:
                vert2 = source_to_vert[hemi][vidx2]  # vertex indexing

                D[vidx, vidx2] = src[hemi]['dist'][vert, vert2] * 100  # get into cm

    # Calculate SD, DLE, RI
    SD = np.zeros(nsources, dtype=np.float64)
    DLE = np.zeros(nsources, dtype=np.float64)
    RI = np.zeros(nsources, dtype=np.float64)
    max_dist = np.max(D)

    for hemi in range(len(src)):
        for vidx in source_to_vert[hemi]:

            vidx2_list = np.asarray(list(source_to_vert[hemi].keys()))

            numerator = ((D[vidx, vidx2_list] * K[vidx2_list, vidx]) ** 2).sum()
            denominator = (K[vidx2_list, vidx] ** 2).sum()
            SD[vidx] = np.sqrt(numerator / denominator)

            max_r_idx = np.argmax(abs(K[vidx2_list, vidx]))
            DLE[vidx] = D[vidx2_list[max_r_idx], vidx]

            max_c_idx = np.argmax(abs(K[vidx, vidx2_list]))
            RI[vidx] = ((max_dist - D[vidx, vidx2_list[max_c_idx]]) * abs(K[vidx, vidx])
                        ) / (max_dist * abs(K[vidx, vidx2_list[max_c_idx]]))

    return SD, DLE, RI
