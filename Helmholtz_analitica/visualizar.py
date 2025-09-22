import numpy as np
import matplotlib.pyplot as plt

def plot_solution(X, Y, psi, source, config, title="Solución de Helmholtz"):
    """
    Visualiza la solución de Helmholtz 2D - VERSIÓN CORREGIDA
    """
    # Crear figura con más espacio para los títulos
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Definir los subplots con más espacio
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.4)
    
    # 1. Fuente
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(source, extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap=config['CMAP_SOURCE'], origin='lower')
    ax0.set_title('Fuente Gaussiana', fontsize=12, pad=10)
    ax0.set_xlabel('X [m]')
    ax0.set_ylabel('Y [m]')
    cbar0 = plt.colorbar(im0, ax=ax0, shrink=0.8)
    cbar0.set_label('Amplitud')

    # 2. Parte real
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(np.real(psi), extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap=config['CMAP_FIELD'], origin='lower')
    ax1.set_title('Parte Real (Hankel)', fontsize=12, pad=10)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Amplitud')

    # 3. Parte imaginaria
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(np.imag(psi), extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap=config['CMAP_FIELD'], origin='lower')
    ax2.set_title('Parte Imaginaria (Hankel)', fontsize=12, pad=10)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Amplitud')

    # 4. Magnitud
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(np.abs(psi), extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap=config['CMAP_SOURCE'], origin='lower')
    ax3.set_title('Magnitud', fontsize=12, pad=10)
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Amplitud')

    # 5. Fase
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.angle(psi), extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap=config['CMAP_PHASE'], vmin=-np.pi, vmax=np.pi, origin='lower')
    ax4.set_title('Fase', fontsize=12, pad=10)
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Radianes')

    # 6. Perfil radial
    ax5 = fig.add_subplot(gs[1, 2])
    r = np.linspace(0.01, 1.0, 200)
    from config import analytical_helmholtz_2d_solution
    psi_radial = analytical_helmholtz_2d_solution(r, np.zeros_like(r), config['k'])
    
    ax5.plot(r, np.real(psi_radial), 'b-', label='Real', linewidth=2)
    ax5.plot(r, np.imag(psi_radial), 'r-', label='Imag', linewidth=2)
    ax5.plot(r, np.abs(psi_radial), 'g-', label='Magnitud', linewidth=2)
    ax5.set_title('Perfil Radial (Hankel)', fontsize=12, pad=10)
    ax5.set_xlabel('Distancia radial [m]')
    ax5.set_ylabel('Amplitud')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()