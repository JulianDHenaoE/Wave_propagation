# -*- coding: utf-8 -*-
import numpy as np

def kc_rect(m: int, n: int, W: float, H: float) -> float: 
    """Número de corte analítico para el modo (m,n) en rectángulo WxH."""
    return np.sqrt((m*np.pi/W)**2 + (n*np.pi/H)**2) # Se resuelve la frecuencia de corte
 
def Hz_TE_nodes(m, n, x, y, W, H):
    """
    Forma analítica del campo longitudinal para TE_mn: H_z = cos(mπx/W)cos(nπy/H).
    Válido para m,n>=0 (ojo: (0,0) se descarta en el problema).
    """
    return np.cos(m*np.pi*x/W)*np.cos(n*np.pi*y/H) # Se resuelve la función de campo longitudinal

def Ez_TM_nodes(m, n, x, y, W, H): # m,n>=1
    """
    Forma analítica del campo longitudinal para TM_mn: E_z = sin(mπx/W)sin(nπy/H).
    Válido para m,n>=1.
    """
    return np.sin(m*np.pi*x/W)*np.sin(n*np.pi*y/H) # Se resuelve la función de campo longitudinal (Modo transverso eléctrico)
