# -*- coding: utf-8 -*-
"""
convergence_demo.py
Muestra cómo disminuye el tamaño de malla h al aumentar Nx, Ny,
y cómo el error del FEM en kc decae como O(h^2).
"""

import numpy as np
import matplotlib.pyplot as plt
from eigen import solve_mode  # tu función ya hecha
from analytic import kc_rect  # fórmula analítica


# --- Parámetros físicos ---
W, H = 1.0, 0.5
modo = (1, 0)   # probemos con TE10

# --- Refinamientos de malla ---
N_list = [8, 12, 16, 20, 24, 32, 40, 50]

h_list = []
err_list = []

for N in N_list:
    Nx, Ny = N, int(N*H/W)  # mantener aspecto de la caja
    h = max(W/Nx, H/Ny)     # tamaño característico de la malla
    
    kc_fem, kc_ana, err, vec, Xy, Tri = solve_mode("TE", *modo, Nx, Ny, W, H)
    
    h_list.append(h)
    err_list.append(err/100.0)  # convertir % a valor decimal

    print(f"Nx={Nx:3d}, Ny={Ny:3d}, h={h:.4f}, kc_FEM={kc_fem:.4f}, kc_Ana={kc_ana:.4f}, err={err:.2f}%")

h_list = np.array(h_list)
err_list = np.array(err_list)


# --- Curva de referencia O(h^2) ---
Cref = err_list[-1] / (h_list[-1]**2)
err_ref = Cref * h_list**2


# --- Graficar ---
plt.figure(figsize=(6,5))
plt.loglog(h_list, err_list, 'o-b', label="Error relativo FEM")
plt.loglog(h_list, err_ref, 'r--', label="O(h^2)")

plt.xlabel("Tamaño de malla h")
plt.ylabel("Error relativo en kc")
plt.title("Convergencia de kc con refinamiento de malla")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.show()
