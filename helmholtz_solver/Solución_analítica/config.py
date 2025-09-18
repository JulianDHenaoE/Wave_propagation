# ===========================
# Parámetros de configuración..Problema de Helmholtz 
# ===========================

# Parámetros físicos de la onda
FREQUENCY = 5.0        # Frecuencia [Hz]
VELOCITY = 1.5         # Velocidad de propagación [m/s]
SOURCE_AMPLITUDE = 0.02 # Amplitud de la fuente gaussiana
SOURCE_SIGMA = 0.05     # Desviación estándar de la fuente gaussiana

# Dominio espacial de simulación
X_MIN, X_MAX = -1.0, 1.0   # Extensión en X [m]
Y_MIN, Y_MAX = -1.0, 1.0   # Extensión en Y [m]
GRID_RESOLUTION = 500      # Número de puntos por eje de la malla

# Colormaps para visualización
CMAP_SOURCE = 'viridis'   # Fuente
CMAP_FIELD  = 'RdBu_r'    # Campos (parte real e imaginaria)
CMAP_PHASE  = 'twilight'  # Fase
