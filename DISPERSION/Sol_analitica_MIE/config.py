# ==============================================================
# CONFIGURACIÓN - Parámetros de simulación
# ==============================================================

import numpy as np  

class SimulationConfig:
    """Configuración de la simulación de Mie para cilindro PEC"""
    
    def __init__(self, regime="lambda_mayor_radio"):  
        # Parámetros físicos
        self.lam = 1.0                    # Longitud de onda
        self.k0 = 2 * np.pi / self.lam    # Número de onda
        self.E0 = 1.0                     # Amplitud del campo incidente
        
        # ¡ESTA ES LA LÍNEA QUE DEBES CAMBIAR!
        if regime == "lambda_menor_radio":
            self.a = 2.0 * self.lam       # λ < r (r = 2λ)
        elif regime == "lambda_igual_radio":
            self.a = 1.0 * self.lam       # λ = r (r = λ)
        else:  # lambda_mayor_radio
            self.a = 0.25 * self.lam      # λ > r (r = λ/4)
        
        # Parámetros de la grilla (ajustar ventana según el tamaño)
        self.Lwin = 2.5 * max(self.a, self.lam)  # Ventana adaptativa
        self.nx = 500                     # Resolución en x
        self.ny = 500                     # Resolución en y
        
        # Parámetros de la solución
        self.n_max = None                 # Se calculará automáticamente
        
    def calculate_n_max(self):
        """Calcula el número máximo de términos para la serie"""
        self.n_max = int(np.round(10 + 10 * self.k0 * self.a))
        return self.n_max