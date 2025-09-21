from sources import GaussianSource

# Dominio principal
MAIN_DOMAIN_SHAPE = (101, 101)               # resolución de la malla
MAIN_DOMAIN_EXTENSION = (-0.5, 0.5, -0.5, 0.5)  # dominio físico

# PML
NBL =40 #número de nodos en la PML
ALPHA = 1.5 # coeficiente de atenuación en la PML

# Física
FREQUENCY = 5.0   # Hz #
VELOCITY = 1.5    # m/s

# Fuente
SOURCE = GaussianSource(amplitude=0.02, x_pos=0, y_pos=0, sigma=0.05, phase=0) # fuente gaussiana
