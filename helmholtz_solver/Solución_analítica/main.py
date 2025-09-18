import numpy as np
from config import *
from helmholtz_solver import calculate_wave_number, create_grid, analytical_helmholtz_solution, gaussian_source
from visualizar import plot_solution

print("=== SOLUCIÓN ANALÍTICA DE HELMHOLTZ ===")

# Empaquetar la configuración en un diccionario
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

# Crear malla de puntos
X, Y = create_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RESOLUTION)
print(f"Malla creada: {GRID_RESOLUTION}x{GRID_RESOLUTION} puntos")

# Calcular campo de Helmholtz y fuente
psi = analytical_helmholtz_solution(X, Y, config['k'])
source = gaussian_source(X, Y, SOURCE_AMPLITUDE, sigma=SOURCE_SIGMA)

# Graficar resultados
print("Generando gráficos...")
plot_solution(X, Y, psi, source, config)

print("=== COMPLETADO ===")
