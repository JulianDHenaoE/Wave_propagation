import numpy as np
import matplotlib.pyplot as plt
from config import *
from visualizar import plot_solution

print("=== SOLUCIÓN DE HELMHOLTZ 2D CON FUNCIONES DE HANKEL ===")

# Configuración
config = {
    'FREQUENCY': FREQUENCY,
    'VELOCITY': VELOCITY,
    'X_MIN': X_MIN,
    'X_MAX': X_MAX,
    'Y_MIN': Y_MIN,
    'Y_MAX': Y_MAX,
    'GRID_RESOLUTION': GRID_RESOLUTION,
    'CMAP_SOURCE': CMAP_SOURCE,
    'CMAP_FIELD': CMAP_FIELD,
    'CMAP_PHASE': CMAP_PHASE
}

# Calcular número de onda
config['k'] = calculate_wave_number(FREQUENCY, VELOCITY)
print(f"Frecuencia: {FREQUENCY} Hz")
print(f"Velocidad: {VELOCITY} m/s") 
print(f"Número de onda k: {config['k']:.3f} rad/m")

# Crear malla
X, Y = create_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RESOLUTION)
print(f"Malla creada: {GRID_RESOLUTION}x{GRID_RESOLUTION} puntos")

# Calcular solución analítica CORRECTA (2D)
psi_analytical = analytical_helmholtz_2d_solution(X, Y, config['k'])
source = gaussian_source(X, Y, SOURCE_AMPLITUDE, sigma=SOURCE_SIGMA)

# Graficar
print("Generando gráficos...")
plot_solution(X, Y, psi_analytical, source, config, title="Solución Analítica 2D")

print("=== COMPLETADO ===")