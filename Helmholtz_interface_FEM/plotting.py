import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from source import cmap_source


def plot_solution(domain, frequency, u_array, source_array, title_suffix="", savepath=None):
    """
    Tres paneles: fuente (parte real), campo (parte real) y fase.
    Se grafica sobre el rectángulo completo (incluye PML).
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'True FEM Solution - Frequency: {frequency:.3f} Hz {title_suffix}', fontsize=16)

    # Marco rojo para el dominio físico central (sin PML)
    rect_width  = 0.5 * (domain.main_extension[1] - domain.main_extension[0])
    rect_height = 0.5 * (domain.main_extension[3] - domain.main_extension[2])
    rect_x = domain.main_extension[0] + 0.25 * (domain.main_extension[1] - domain.main_extension[0])
    rect_y = domain.main_extension[2] + 0.25 * (domain.main_extension[3] - domain.main_extension[2])
    rect_params = dict(xy=(rect_x, rect_y), width=rect_width, height=rect_height,
                       linewidth=2, edgecolor='r', facecolor='none')

    # Fuente
    vmax_src = np.max(np.abs(source_array))
    im0 = ax0.imshow(np.real(source_array).T, extent=domain.extension,
                     origin='lower', cmap=cmap_source, vmin=-vmax_src, vmax=vmax_src)
    fig.colorbar(im0, ax=ax0, shrink=0.6)
    ax0.add_patch(patches.Rectangle(**rect_params))
    ax0.set_title('Source (Real part)'); ax0.set_xlabel('x'); ax0.set_ylabel('y')

    # Campo
    vmax = np.max(np.abs(np.real(u_array)))
    im1 = ax1.imshow(np.real(u_array).T, extent=domain.extension,
                     origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=ax1, shrink=0.6)
    ax1.add_patch(patches.Rectangle(**rect_params))
    ax1.set_title('Field (Real part)'); ax1.set_xlabel('x'); ax1.set_ylabel('y')

    # Fase
    im2 = ax2.imshow(np.angle(u_array).T, extent=domain.extension,
                     origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im2, ax=ax2, shrink=0.6)
    ax2.add_patch(patches.Rectangle(**rect_params))
    ax2.set_title('Phase'); ax2.set_xlabel('x'); ax2.set_ylabel('y')

    fig.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300)
        print(f"✅ Figura de solución guardada en {savepath}")

    plt.show()


def plot_velocity(domain, velocity_array, title="Velocity Model", savepath=None):
    """
    Grafica el modelo de velocidades en el dominio.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(velocity_array.T,
                   extent=domain.extension,
                   origin="lower",
                   cmap="viridis")
    fig.colorbar(im, ax=ax, label="Velocity")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Marco rojo para el dominio físico (sin PML)
    rect_width  = 0.5 * (domain.main_extension[1] - domain.main_extension[0])
    rect_height = 0.5 * (domain.main_extension[3] - domain.main_extension[2])
    rect_x = domain.main_extension[0] + 0.25 * (domain.main_extension[1] - domain.main_extension[0])
    rect_y = domain.main_extension[2] + 0.25 * (domain.main_extension[3] - domain.main_extension[2])
    rect_params = dict(xy=(rect_x, rect_y), width=rect_width, height=rect_height,
                       linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(patches.Rectangle(**rect_params))

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300)
        print(f"✅ Figura de velocidad guardada en {savepath}")

    plt.show()


def plot_mesh(domain, title="Discretized Domain", savepath=None):
    """
    Grafica la malla del dominio completo (incluyendo PML).
    Se resaltará el dominio físico central con un marco rojo.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Nodos
    X, Y = np.meshgrid(domain.x_array, domain.y_array, indexing="ij")
    ax.plot(X.flatten(), Y.flatten(), 'k.', markersize=2, label="Nodes")

    # Líneas de la malla (grilla)
    for x in domain.x_array:
        ax.plot([x, x], [domain.y_array[0], domain.y_array[-1]], color='lightgray', linewidth=0.5)
    for y in domain.y_array:
        ax.plot([domain.x_array[0], domain.x_array[-1]], [y, y], color='lightgray', linewidth=0.5)

    # Marco rojo para el dominio físico
    rect_width  = 0.5 * (domain.main_extension[1] - domain.main_extension[0])
    rect_height = 0.5 * (domain.main_extension[3] - domain.main_extension[2])
    rect_x = domain.main_extension[0] + 0.25 * (domain.main_extension[1] - domain.main_extension[0])
    rect_y = domain.main_extension[2] + 0.25 * (domain.main_extension[3] - domain.main_extension[2])
    rect_params = dict(xy=(rect_x, rect_y), width=rect_width, height=rect_height,
                       linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(patches.Rectangle(**rect_params))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300)
        print(f"✅ Malla guardada en {savepath}")

    plt.show()
