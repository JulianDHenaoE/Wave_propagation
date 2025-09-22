# ==============================================================
# FUENTES - Definición del campo incidente
# ==============================================================

import numpy as np

class Source:
    """Clase para definir la fuente (onda plana incidente)"""
    
    def __init__(self, config, domain):
        self.config = config
        self.domain = domain
        
    def plane_wave(self):
        """Genera una onda plana incidente propagándose en dirección x"""
        return self.config.E0 * np.exp(-1j * self.config.k0 * self.domain.X)