# -*- coding: utf-8 -*-
"""
Visualización de refinamiento de malla rectangular
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ----------------------------
# Función para generar malla rectangular
# ----------------------------
def rect_mesh(Nx, Ny, W=1.0, H=0.5):
    """
    Construye malla rectangular estructurada y la triangula.

    Parámetros
    ----------
    Nx, Ny : int
        Número de subdivisiones en x e y
    W, H : float
        Ancho y alto del rectángulo

    Retorna
    -------
    Xy : ndarray
        Coordenadas de nodos (N,2)
    Tri : ndarray
        Conectividad triangular (M,3)
    h   : float
        Tamaño característico de la malla
    """
    x = np.linspace(0, W, Nx + 1)
    y = np.linspace(0, H, Ny + 1)
    xx, yy = np.meshgrid(x, y)
    Xy = np.column_stack([xx.ravel(), yy.ravel()])

    # Conectividad de triángulos
    Tri = []
    for j in range(Ny):
        for i in range(Nx):
            n0 = j * (Nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (Nx + 1)
            n3 = n2 + 1
            # dos triángulos por celda
            Tri.append([n0, n1, n3])
            Tri.append([n0, n3, n2])
    Tri = np.array(Tri)

    # tamaño de malla
    hx, hy = W / Nx, H / Ny
    h = max(hx, hy)
    return Xy, Tri, h


# ----------------------------
# Visualizar varias mallas
# ----------------------------
W, H = 1.0, 0.5
subdivisiones = [(4, 2), (8, 4), (16, 8), (32, 16)]  # refinamientos

fig, axs = plt.subplots(1, len(subdivisiones), figsize=(14, 3))
for ax, (Nx, Ny) in zip(axs, subdivisiones):
    Xy, Tri, h = rect_mesh(Nx, Ny, W, H)
    triang = Triangulation(Xy[:, 0], Xy[:, 1], Tri)
    ax.triplot(triang, lw=0.7, color="k")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Nx={Nx}, Ny={Ny}\nh={h:.3f}")

plt.suptitle("Refinamiento de malla FEM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
