import numpy as np
import matplotlib.pyplot as plt
from helmholtz_solver import analytical_helmholtz_solution  # ¡IMPORTANTE!

def plot_solution(X, Y, psi, source, config):
    """Grafica los resultados"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Fuente
    axes[0,0].imshow(np.real(source), extent=[config['X_MIN'], config['X_MAX'], 
                     config['Y_MIN'], config['Y_MAX']], cmap='viridis', origin='lower')
    axes[0,0].set_title('Fuente')
    
    # 2. Parte real
    axes[0,1].imshow(np.real(psi), extent=[config['X_MIN'], config['X_MAX'], 
                     config['Y_MIN'], config['Y_MAX']], cmap='RdBu_r', origin='lower')
    axes[0,1].set_title('Parte Real')
    
    # 3. Parte imaginaria
    axes[0,2].imshow(np.imag(psi), extent=[config['X_MIN'], config['X_MAX'], 
                     config['Y_MIN'], config['Y_MAX']], cmap='RdBu_r', origin='lower')
    axes[0,2].set_title('Parte Imaginaria')
    
    # 4. Magnitud
    axes[1,0].imshow(np.abs(psi), extent=[config['X_MIN'], config['X_MAX'], 
                     config['Y_MIN'], config['Y_MAX']], cmap='viridis', origin='lower')
    axes[1,0].set_title('Magnitud')
    
    # 5. Fase
    axes[1,1].imshow(np.angle(psi), extent=[config['X_MIN'], config['X_MAX'], 
                     config['Y_MIN'], config['Y_MAX']], cmap='twilight', 
                     vmin=-np.pi, vmax=np.pi, origin='lower')
    axes[1,1].set_title('Fase')
    
    # 6. Perfil radial
    r = np.linspace(0.01, 1.0, 100)
    psi_radial = analytical_helmholtz_solution(r, 0, config['k'])
    axes[1,2].plot(r, np.real(psi_radial), 'b-', label='Real')
    axes[1,2].plot(r, np.imag(psi_radial), 'r-', label='Imag')
    axes[1,2].plot(r, np.abs(psi_radial), 'g-', label='Magnitud')
    axes[1,2].set_title('Perfil Radial')
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()