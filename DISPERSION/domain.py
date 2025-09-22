# ==============================================================
# DOMINIO - Definición de la grilla computacional
# ==============================================================

import numpy as np

class Domain:
    """Clase para definir el dominio computacional"""
    
    def __init__(self, config): # Inicializa con la configuración dada
        self.config = config # Guarda la configuración
        self.X = None # Grilla X
        self.Y = None # Grilla Y
        self.R = None # Radio
        self.PHI = None # Ángulo polar
        
    def create_grid(self): 
        """Crea la grilla computacional"""
        xx = np.linspace(-self.config.Lwin, self.config.Lwin, self.config.nx) # Eje x
        yy = np.linspace(-self.config.Lwin, self.config.Lwin, self.config.ny) # Eje y
        self.X, self.Y = np.meshgrid(xx, yy) # Grilla 2D
        self.R = np.sqrt(self.X**2 + self.Y**2) # Radio
        self.PHI = np.arctan2(self.Y, self.X) # Ángulo polar
        return self.X, self.Y, self.R, self.PHI # Retorna las grillas creadas
    
    def get_cylinder_mask(self): 
        """Retorna máscara para el interior del cilindro"""
        return self.R < self.config.a # Máscara booleana