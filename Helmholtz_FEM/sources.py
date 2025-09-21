import numpy as np
import matplotlib.pyplot as plt


class GaussianSource:
    """Gaussian source function"""
    def __init__(self, amplitude=0.02, x_pos=0, y_pos=0, sigma=0.05, phase=0):
        self.amplitude = amplitude   # Amplitud
        self.x_pos = x_pos           # x0
        self.y_pos = y_pos           # y0
        self.sigma = sigma           # σ (ancho)
        self.phase = phase           # φ (fase)
    
    def __call__(self, x, y): # Evaluar la fuente en (x, y)
        r_squared = (x - self.x_pos)**2 + (y - self.y_pos)**2 # r^2= (x - x0)² + (y - y0)²
        return self.amplitude * np.exp(-r_squared / (2 * self.sigma**2)) * np.exp(1j * self.phase) # A * exp(-r²/(2σ²)) * exp(iφ)


if __name__ == "__main__": # Prueba de la fuente Gaussiana
    # Crear la fuente
    source = GaussianSource(amplitude=0.02, x_pos=0, y_pos=0, sigma=0.1, phase=0) # Parámetros de la fuente

    # Crear una malla de puntos donde evaluarla
    x = np.linspace(-1, 1, 500) # 500 puntos entre -1 y 1
    y = np.linspace(-1, 1, 500) # 500 puntos entre -1 y 1
    X, Y = np.meshgrid(x, y) # Crear malla 2D

    # Evaluar la fuente en toda la malla
    Z = source(X, Y) # Evaluar la fuente en la malla

    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 1 fila, 2 columnas

    # Parte real
    im1 = ax1.imshow(np.real(Z), extent=[-1, 1, -1, 1], origin='lower', cmap="viridis") # Mostrar parte real
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Re{Fuente}") # Barra de color
    ax1.set_title("Fuente Gaussiana (parte real)") # Título
    ax1.set_xlabel("x") # Etiqueta eje x
    ax1.set_ylabel("y") # Etiqueta eje y

    # Magnitud
    im2 = ax2.imshow(np.abs(Z), extent=[-1, 1, -1, 1], origin='lower', cmap="inferno") # Mostrar magnitud
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="|Fuente|") # Barra de color
    ax2.set_title("Fuente Gaussiana (magnitud)") # Título
    ax2.set_xlabel("x") # Etiqueta eje x
    ax2.set_ylabel("y")    # Etiqueta eje y

    # Ajustar y mostrar
    plt.tight_layout() # Ajustar diseño
    plt.show() # Mostrar figura

