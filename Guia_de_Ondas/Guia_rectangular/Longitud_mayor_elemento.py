# -*- coding: utf-8 -*-
"""
Visualización de un elemento de la malla con hx, hy y el diámetro h.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del rectángulo
hx = 0.25   # ancho
hy = 0.5    # alto

# Coordenadas de los nodos del rectángulo
n0 = np.array([0, 0])
n1 = np.array([hx, 0])
n2 = np.array([0, hy])
n3 = np.array([hx, hy])

# Triángulos (n0,n1,n3) y (n0,n3,n2)
tri1 = [n0, n1, n3]
tri2 = [n0, n3, n2]

# Calcular longitudes de lados del triángulo (n0,n1,n3)
def lado(p1, p2):
    return np.linalg.norm(p1 - p2)

l1 = lado(n0, n1)  # = hx
l2 = lado(n1, n3)  # = hy
l3 = lado(n0, n3)  # diagonal

h_exacto = max(l1, l2, l3)

# --- Dibujar
plt.figure(figsize=(6, 4))
# Rectángulo completo
plt.plot([n0[0], n1[0], n3[0], n2[0], n0[0]],
         [n0[1], n1[1], n3[1], n2[1], n0[1]], 'k--', lw=1)

# Triángulo resaltado
plt.fill([p[0] for p in tri1], [p[1] for p in tri1],
         color="lightblue", alpha=0.6, label="Elemento triangular")

# Lados
plt.plot([n0[0], n1[0]], [n0[1], n1[1]], 'r-', lw=2, label=f"$h_x={l1:.2f}$")
plt.plot([n1[0], n3[0]], [n1[1], n3[1]], 'g-', lw=2, label=f"$h_y={l2:.2f}$")
plt.plot([n0[0], n3[0]], [n0[1], n3[1]], 'b-', lw=2, label=f"$h=\\sqrt{{h_x^2+h_y^2}}={h_exacto:.2f}$")

# Nodos
for i, (x, y) in enumerate([n0, n1, n2, n3]):
    plt.plot(x, y, 'ko')
    plt.text(x, y+0.02, f"P{i}", ha="center", fontsize=9, color="black")

plt.gca().set_aspect("equal")
plt.title("Definición de $h_x$, $h_y$ y diámetro $h$")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.grid(True, ls=":", alpha=0.6)
plt.tight_layout()
plt.show()
