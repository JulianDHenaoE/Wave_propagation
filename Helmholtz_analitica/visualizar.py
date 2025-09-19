import numpy as np
import matplotlib.pyplot as plt
from helmholtz_solver import analytical_helmholtz_solution

def plot_solution(X, Y, psi, source, config):
    """
    Visualiza la solución de Helmholtz y la fuente en varias representaciones:
        1. Fuente gaussiana
        2. Parte real del campo
        3. Parte imaginaria
        4. Magnitud
        5. Fase
        6. Perfil radial 1D
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # --- 1. Fuente ---
    im = axes[0,0].imshow(np.real(source),
                          extent=[config['X_MIN'], config['X_MAX'], 
                                  config['Y_MIN'], config['Y_MAX']],
                          cmap=config['CMAP_SOURCE'], origin='lower')
    axes[0,0].set_title('Fuente')
    plt.colorbar(im, ax=axes[0,0])  # <- Colorbar añadida

    # --- 2. Parte real ---
    im = axes[0,1].imshow(np.real(psi),
                          extent=[config['X_MIN'], config['X_MAX'], 
                                  config['Y_MIN'], config['Y_MAX']],
                          cmap=config['CMAP_FIELD'], origin='lower')
    axes[0,1].set_title('Parte Real')
    plt.colorbar(im, ax=axes[0,1])  # <- Colorbar añadida

    # --- 3. Parte imaginaria ---
    im = axes[0,2].imshow(np.imag(psi),
                          extent=[config['X_MIN'], config['X_MAX'], 
                                  config['Y_MIN'], config['Y_MAX']],
                          cmap=config['CMAP_FIELD'], origin='lower')
    axes[0,2].set_title('Parte Imaginaria')
    plt.colorbar(im, ax=axes[0,2])  # <- Colorbar añadida

    # --- 4. Magnitud ---
    im = axes[1,0].imshow(np.abs(psi),
                          extent=[config['X_MIN'], config['X_MAX'], 
                                  config['Y_MIN'], config['Y_MAX']],
                          cmap=config['CMAP_SOURCE'], origin='lower')
    axes[1,0].set_title('Magnitud')
    plt.colorbar(im, ax=axes[1,0])  # <- Colorbar añadida

    # --- 5. Fase ---
    im = axes[1,1].imshow(np.angle(psi),
                          extent=[config['X_MIN'], config['X_MAX'], 
                                  config['Y_MIN'], config['Y_MAX']],
                          cmap=config['CMAP_PHASE'], 
                          vmin=-np.pi, vmax=np.pi, origin='lower')
    axes[1,1].set_title('Fase')
    plt.colorbar(im, ax=axes[1,1])  # <- Colorbar añadida

    # --- 6. Perfil radial ---
    r = np.linspace(0.01, 1.0, 200)
    psi_radial = analytical_helmholtz_solution(r, 0, config['k'])
    axes[1,2].plot(r, np.real(psi_radial), 'b-', label='Real')
    axes[1,2].plot(r, np.imag(psi_radial), 'r-', label='Imag')
    axes[1,2].plot(r, np.abs(psi_radial), 'g-', label='Magnitud')
    axes[1,2].set_title('Perfil Radial')
    axes[1,2].legend()
    axes[1,2].grid(True)

    plt.tight_layout()
    plt.show()