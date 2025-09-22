# ==============================================================
# SOLVER - Solución analítica de Mie
# ==============================================================

import numpy as np
from scipy.special import jv, hankel2

class MieSolver:
    """Solver para la solución analítica de Mie para cilindro PEC"""
    
    def __init__(self, config, domain, source):
        self.config = config
        self.domain = domain
        self.source = source
        
    def solve(self):
        """Calcula la solución de Mie para el campo total"""
        print("Calculando solución de Mie...")
        
        # Calcular número máximo de términos
        n_max = self.config.calculate_n_max()
        print(f"Truncando serie a {2*n_max + 1} términos (n_max = {n_max})")
        
        Ez_total = np.zeros_like(self.domain.R, dtype=complex)
        
        # Sumatoria sobre todos los modos
        for n in range(-n_max, n_max + 1):
            Jn_k0a = jv(n, self.config.k0 * self.config.a)
            H2n_k0a = hankel2(n, self.config.k0 * self.config.a)
            H2n_k0r = hankel2(n, self.config.k0 * self.domain.R)
            
            # Coeficiente de Mie para cilindro PEC
            coeff = -self.config.E0 * (1j**(-n)) * (Jn_k0a / H2n_k0a)
            
            # Contribución del modo n
            Ez_total += coeff * H2n_k0r * np.exp(1j * n * self.domain.PHI)
        
        return Ez_total
    
    def apply_pec_conditions(self, Ez_total, Ei):
        """Aplica condiciones de contorno PEC"""
        mask = self.domain.get_cylinder_mask()
        Ez_total[mask] = 0 + 0j  # Campo total CERO dentro del PEC
        Es = Ez_total - Ei        # Campo dispersado
        Es[mask] = 0 + 0j         # Campo dispersado CERO dentro del PEC
        
        return Ez_total, Es