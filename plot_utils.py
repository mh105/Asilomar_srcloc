import numpy as np
from somata.source_loc.source_loc_utils import resolution_matrix_metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# %% Compute the metrics for the resolution matrices
def plot_resmat_metrics(res_mat, src, legend_str):
    SD, DLE, RI = resolution_matrix_metrics(res_mat, src)

    bins = np.arange(-0.25, 13.75, 0.5)
    # bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.rcParams["svg.fonttype"] = "none"
    font_size = 20

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].hist(SD, bins=bins, alpha=0.5, density=True, edgecolor='black', label=legend_str)
    ax[0].legend(prop={'size': 10})
    ax[0].gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].xlabel('SD (cm)', fontsize=font_size)
    ax[0].ylabel('pmf', fontsize=font_size)
    ax[0].xticks(fontsize=font_size)
    ax[0].yticks(fontsize=font_size)
    ax[0].title('Spatial Dispersion (SD)', fontsize=font_size)

    ax[1].hist(DLE, bins=bins, alpha=0.5, density=True, edgecolor='black',  label=legend_str)
    ax[1].legend(prop={'size': 10})
    ax[1].gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].xlabel('DLE (cm)', fontsize=font_size)
    ax[1].ylabel('pmf', fontsize=font_size)
    ax[1].xticks(fontsize=font_size)
    ax[1].yticks(fontsize=font_size)
    ax[1].title('Dipole Localization Error (DLE)', fontsize=font_size)

    return fig, ax
