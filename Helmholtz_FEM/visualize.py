import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

cmap_source = LinearSegmentedColormap.from_list('source', ['green', 'white', 'purple'])

def plot_solution(domain, frequency, u_array, source_array, title_suffix=""):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'True FEM Solution - Frequency: {frequency:.1f} Hz {title_suffix}', fontsize=16)
    rect_params = dict(
        xy=(domain.main_extension[0], domain.main_extension[2]),
        width=domain.main_extension[1]-domain.main_extension[0],
        height=domain.main_extension[3]-domain.main_extension[2],
        linewidth=2, edgecolor='r', facecolor='none')

    vmax = np.max(np.abs(source_array))
    im0 = ax0.imshow(np.real(source_array).T, extent=domain.extension,
                     origin='lower', cmap=cmap_source, vmin=-vmax, vmax=vmax)
    fig.colorbar(im0, ax=ax0, shrink=0.5)
    ax0.add_patch(patches.Rectangle(**rect_params))
    ax0.set_title('Source (Real part)')
    ax0.set_xlabel('x'); ax0.set_ylabel('y')

    vmax = np.max(np.abs(np.real(u_array)))
    im1 = ax1.imshow(np.real(u_array).T, extent=domain.extension,
                     origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.add_patch(patches.Rectangle(**rect_params))
    ax1.set_title('Field (Real part)')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    im2 = ax2.imshow(np.angle(u_array).T, extent=domain.extension,
                     origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.add_patch(patches.Rectangle(**rect_params))
    ax2.set_title('Phase')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')

    fig.tight_layout()
    plt.show()
