# ==============================================================
# VISUALIZACIÓN - Gráficas de resultados
# ==============================================================

import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """Clase para visualizar los resultados de la simulación"""
    
    def __init__(self, config, domain):
        self.config = config
        self.domain = domain
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10
        
    def plot_results(self, Ei, Ez_total, Es):
        """Genera todas las gráficas de resultados"""
        # Crear figura con ajustes de espacio
        fig = plt.figure(figsize=(20, 14))
        
        # Definir la cuadrícula de subplots
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
        
        # Configuración común para plots de imágenes
        plot_params = {
            'extent': [-self.config.Lwin, self.config.Lwin, 
                      -self.config.Lwin, self.config.Lwin],
            'origin': 'lower',
            'aspect': 'equal'
        }
        
        # 1. |E_total|
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(np.abs(Ez_total), **plot_params, cmap='turbo')
        ax1.set_title('|E_total| - Solución Mie', pad=20)
        ax1.set_xlabel('x/λ')
        ax1.set_ylabel('y/λ')
        plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05)
        
        # 2. |E_scattered|
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(np.abs(Es), **plot_params, cmap='turbo')
        ax2.set_title('|E_scattered| - Solución Mie', pad=20)
        ax2.set_xlabel('x/λ')
        ax2.set_ylabel('y/λ')
        plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.05)
        
        # 3. Re{E_incident}
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(np.real(Ei), **plot_params, cmap='seismic', vmin=-1, vmax=1)
        ax3.set_title('Re{E_incident} (Onda Plana)', pad=20)
        ax3.set_xlabel('x/λ')
        ax3.set_ylabel('y/λ')
        plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.05)
        
        # 4. Re{E_total}
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(np.real(Ez_total), **plot_params, cmap='seismic')
        ax4.set_title('Re{E_total} - Solución Mie', pad=20)
        ax4.set_xlabel('x/λ')
        ax4.set_ylabel('y/λ')
        plt.colorbar(im4, ax=ax4, shrink=0.8, pad=0.05)
        
        # 5. Im{E_total}
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(np.imag(Ez_total), **plot_params, cmap='seismic')
        ax5.set_title('Im{E_total} - Solución Mie', pad=20)
        ax5.set_xlabel('x/λ')
        ax5.set_ylabel('y/λ')
        plt.colorbar(im5, ax=ax5, shrink=0.8, pad=0.05)
        
        # 6. Corte transversal (y=0)
        ax6 = fig.add_subplot(gs[1, 2])
        ny = self.config.ny
        xx = self.domain.X[0, :]
        ax6.plot(xx, np.real(Ei[ny//2, :]), 'g-', linewidth=2, label='Re{E_incident}')
        ax6.plot(xx, np.real(Ez_total[ny//2, :]), 'b-', linewidth=2, label='Re{E_total}')
        ax6.plot(xx, np.real(Es[ny//2, :]), 'r--', linewidth=2, label='Re{E_scattered}')
        ax6.set_xlabel('x/λ')
        ax6.set_ylabel('Re{E}')
        ax6.set_title('Corte transversal en y=0', pad=20)
        ax6.legend(fontsize=9, loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim([-self.config.Lwin, self.config.Lwin])
        
        # Dibujar contorno del cilindro en todas las imágenes
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = self.config.a * np.cos(theta)
        y_circle = self.config.a * np.sin(theta)
        
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.plot(x_circle, y_circle, 'w--', linewidth=1.5, alpha=0.8)
        
        # Ajustar diseño final
        plt.tight_layout()
        
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]
    
    def plot_individual(self, data, title, cmap='turbo', vmin=None, vmax=None):
        """Genera una gráfica individual con mejor formato"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        plot_params = {
            'extent': [-self.config.Lwin, self.config.Lwin, 
                      -self.config.Lwin, self.config.Lwin],
            'origin': 'lower',
            'aspect': 'equal',
            'cmap': cmap
        }
        
        if vmin is not None and vmax is not None:
            plot_params['vmin'] = vmin
            plot_params['vmax'] = vmax
        
        im = ax.imshow(data, **plot_params)
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('x/λ', fontsize=12)
        ax.set_ylabel('y/λ', fontsize=12)
        
        # Dibujar contorno del cilindro
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = self.config.a * np.cos(theta)
        y_circle = self.config.a * np.sin(theta)
        ax.plot(x_circle, y_circle, 'w--', linewidth=2, alpha=0.8)
        
        # Colorbar con buen espaciado
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Magnitud', fontsize=10)
        
        plt.tight_layout()
        
        return fig, ax