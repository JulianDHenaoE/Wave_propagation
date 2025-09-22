# ===========================
# Parámetros de configuración - Problema de Helmholtz 2D
# ===========================

import numpy as np
from scipy import special

# Parámetros físicos de la onda
FREQUENCY = 5.0        # Frecuencia [Hz]
VELOCITY = 1.5         # Velocidad de propagación [m/s]
SOURCE_AMPLITUDE = 0.02 # Amplitud de la fuente gaussiana
SOURCE_SIGMA = 0.05     # Desviación estándar de la fuente gaussiana

# Dominio espacial de simulación
X_MIN, X_MAX = -1.0, 1.0   # Extensión en X [m]
Y_MIN, Y_MAX = -1.0, 1.0   # Extensión en Y [m]
GRID_RESOLUTION = 200      # Número de puntos por eje de la malla

# Colormaps para visualización
CMAP_SOURCE = 'viridis'   # Fuente
CMAP_FIELD  = 'RdBu_r'    # Campos (parte real e imaginaria)
CMAP_PHASE  = 'twilight'  # Fase

def calculate_wave_number(frequency, velocity):
    """
    Calcula el número de onda k = 2πf / v
    """
    return 2 * np.pi * frequency / velocity

def analytical_helmholtz_2d_solution(x, y, k, x0=0, y0=0):
    """
    Solución analítica CORRECTA de la ecuación de Helmholtz en 2D.
    Usa la función de Hankel de primera especie H₀⁽¹⁾(kr)
    ψ(r) = (i/4) * H₀⁽¹⁾(k·r)
    
    Parámetros:
        x, y : coordenadas donde evaluar
        k    : número de onda [rad/m]
        x0,y0: posición de la fuente
    Retorna:
        ψ(x,y): campo complejo en cada punto
    """
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_safe = np.where(r < 1e-10, 1e-10, r)  # evita división por cero
    
    # Función de Hankel de primera especie de orden 0
    return (1j/4) * special.hankel1(0, k * r_safe)

def gaussian_source(x, y, amplitude=1, x0=0, y0=0, sigma=0.05):
    """
    Define una fuente gaussiana localizada en (x0,y0).
    """
    r_squared = (x - x0)**2 + (y - y0)**2
    return amplitude * np.exp(-r_squared / (2 * sigma**2))

def create_grid(x_min, x_max, y_min, y_max, resolution):
    """
    Crea la malla cartesiana para la simulación.
    """
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    return np.meshgrid(x, y)