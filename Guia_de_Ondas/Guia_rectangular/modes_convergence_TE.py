# -*- coding: utf-8 -*-
"""
modes_convergence_TE.py
Estudia la convergencia FEM para un modo TE_mn en guía rectangular.
- Muestra el campo del modo en TODOS los refinamientos (una columna por nivel).
- (Opcional) Muestra las mallas de TODOS los refinamientos.
- Grafica convergencia de kc y del error relativo; el eje-x se invierte para
  que el error disminuya de izquierda -> derecha.

FUNDAMENTO MATEMÁTICO: Este script demuestra la convergencia del Método de 
Elementos Finitos P1 (lineales) para la ecuación de Helmholtz. La teoría 
predice convergencia O(h²) para el error en autovalores cuando h → 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from eigen import solve_mode
from mesh import rect_mesh

# ==============================================================================
# PARÁMETROS DE CONFIGURACIÓN DEL ESTUDIO DE CONVERGENCIA
# ==============================================================================

# Tipo de modo y modo específico a estudiar
KIND     = "TE"     # Tipo de modo: "TE" (Transverso Eléctrico)
MODE_MN  = (1, 0)   # Modo específico: TE₁₀ (modo fundamental)
W, H     = 1.0, 0.5 # Dimensiones de la guía [metros]

# ==============================================================================
# NIVELES DE REFINAMIENTO DE MALLA
# ==============================================================================
"""
Secuencia de refinamientos h-adaptivos:
Cada tupla (Nx, Ny) representa un nivel de refinamiento con:
- Nx: divisiones en dirección x
- Ny: divisiones en dirección y
- h: tamaño característico de elemento = max(W/(Nx-1), H/(Ny-1))

La secuencia va de malla GRUESA a FINA para estudiar convergencia.
"""
REF_LEVELS = [(4,3),(8, 6), (12, 9), (16, 12), (22, 16), (30, 20), (40, 30), (44,33)]

# Control de visualización adicional
SHOW_ALL_MESHES = True  # Mostrar también las mallas de todos los refinamientos

# ==============================================================================
# FUNCIÓN PRINCIPAL: ESTUDIO DE CONVERGENCIA
# ==============================================================================

# ------------------------
# CÁLCULO EN TODOS LOS NIVELES DE REFINAMIENTO
# ------------------------
records = []  # Almacenará resultados de cada refinamiento
m, n = MODE_MN  # Desempaquetar índices del modo

print("=" * 70)
print(f"ESTUDIO DE CONVERGENCIA FEM - Modo {KIND}{m}{n}")
print("=" * 70)

for (Nx, Ny) in REF_LEVELS:
    print(f"Calculando: {Nx}×{Ny} divisiones...")
    
    # Generar malla rectangular
    Xy, Tri, h = rect_mesh(Nx, Ny, W, H)

    # Resolver el modo específico usando FEM
    kc_fem, kc_ana, err_pct, vec, Xy, Tri = solve_mode(KIND, m, n, Nx, Ny, W, H)

    # Almacenar resultados
    records.append({
        "Nx": Nx, "Ny": Ny, "h": h,                    # Información de malla
        "kc_fem": kc_fem, "kc_ana": kc_ana, "err_pct": err_pct,  # Resultados numéricos
        "Xy": Xy, "Tri": Tri, "vec": vec               # Datos para visualización
    })

# Ordenar registros por tamaño de malla (de GRUESO a FINO)
# Orden descendente por h: mallas más gruesas primero
records.sort(key=lambda r: r["h"], reverse=True)

print("✓ Cálculos completados para todos los refinamientos")

# ==============================================================================
# VISUALIZACIÓN 1: MALLA EN TODOS LOS REFINAMIENTOS (OPCIONAL)
# ==============================================================================
if SHOW_ALL_MESHES:
    ncols = len(records)
    fig_m, axs_m = plt.subplots(1, ncols, figsize=(2.8 * ncols, 2.6))
    
    # Manejar caso de un solo refinamiento
    if ncols == 1:
        axs_m = [axs_m]
    
    fig_m.suptitle("Evolución de la Malla - Refinamiento h-Adaptivo", 
                   fontsize=14, fontweight="bold")

    for ax, rec in zip(axs_m, records):
        Xy, Tri = rec["Xy"], rec["Tri"]
        
        # Crear triangulación para visualización
        triang = Triangulation(Xy[:, 0], Xy[:, 1], Tri)
        
        # Dibujar malla
        ax.triplot(triang, lw=0.6, color="k")
        ax.set_aspect("equal")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Título informativo
        ax.set_title(f"{rec['Nx']}×{rec['Ny']}\n"
                    f"h = {rec['h']:.3f}\n"
                    f"Elementos: {Tri.shape[0]}", 
                    fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    print("✓ Visualización de mallas generada")

# ==============================================================================
# VISUALIZACIÓN 2: CAMPO MODAL EN TODOS LOS REFINAMIENTOS
# ==============================================================================
ncols = len(records)
fig_f, axs_f = plt.subplots(1, ncols, figsize=(2.9 * ncols, 2.8))

# Manejar caso de un solo refinamiento
if ncols == 1:
    axs_f = [axs_f]

fig_f.suptitle(f"Convergencia del Campo {KIND}{m}{n} - Refinamiento de Malla", 
               fontsize=16, fontweight="bold")

# Escala de color común para todos los subplots (permite comparación directa)
vmin, vmax = 0.0, 1.0

for k, (ax, rec) in enumerate(zip(axs_f, records)):
    Xy, Tri, vec = rec["Xy"], rec["Tri"], rec["vec"]
    
    # Crear triangulación
    triang = Triangulation(Xy[:, 0], Xy[:, 1], Tri)
    
    # Procesar campo para visualización
    v = np.abs(vec)  # Valor absoluto del campo
    vmax_local = np.max(np.abs(v))
    if vmax_local > 0:
        v = v / vmax_local  # Normalizar al rango [0,1]

    # Visualizar campo con tripcolor
    tpc = ax.tripcolor(triang, v, shading="gouraud", cmap="viridis", 
                       vmin=vmin, vmax=vmax)
    
    # Configurar subplot
    ax.set_aspect("equal")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Título con información numérica del modo
    ax.set_title(
        f"{rec['Nx']}×{rec['Ny']}\n"
        f"kc_FEM = {rec['kc_fem']:.4f}\n"
        f"kc_Ana = {rec['kc_ana']:.4f}\n"
        f"Error = {rec['err_pct']:.2f}%",
        fontsize=9, pad=3
    )

# Añadir barra de color común (solo una para toda la figura)
cbar = fig_f.colorbar(tpc, ax=axs_f[-1], fraction=0.046, pad=0.04)
if KIND == "TE":
    cbar.set_label(r"$|H_z|$ (normalizado)", rotation=90)
else:
    cbar.set_label(r"$|E_z|$ (normalizado)", rotation=90)

plt.tight_layout(rect=[0, 0, 1, 0.92])
print("✓ Visualización de campos modales generada")

# ==============================================================================
# VISUALIZACIÓN 3: ANÁLISIS DE CONVERGENCIA NUMÉRICA
# ==============================================================================
# Extraer datos para análisis de convergencia
hs = np.array([r["h"] for r in records])               # Tamaños de malla
kc_fem_values = np.array([r["kc_fem"] for r in records]) # Valores FEM de kc
kc_analytical = records[0]["kc_ana"]                   # Valor analítico (constante)
errs_relative = np.array([r["err_pct"] / 100.0 for r in records]) # Errores en fracción

# Crear figura para análisis de convergencia
fig_c, (ax_k, ax_e) = plt.subplots(1, 2, figsize=(11, 4.5))
fig_c.suptitle(f"Análisis de Convergencia FEM – Modo {KIND}{m}{n}", 
               fontsize=16, fontweight="bold")

# ==============================================================================
# SUBPLOT 1: CONVERGENCIA DE kc vs h
# ==============================================================================
ax_k.plot(hs, kc_fem_values, "-o", lw=2, ms=5, label="kc_FEM", color='blue')
ax_k.axhline(kc_analytical, color="red", ls="--", lw=2, 
            label=f"kc_Analítico = {kc_analytical:.4f}")
ax_k.set_xlabel("Tamaño de malla h [m]")
ax_k.set_ylabel("Constante de corte kc [m⁻¹]")
ax_k.grid(True, ls=":", alpha=0.6)
ax_k.legend()

# INVERTIR EJE X: h decrece de izquierda a derecha (convergencia → derecha)
ax_k.invert_xaxis()

# ==============================================================================
# SUBPLOT 2: ANÁLISIS DE ERROR (LOG-LOG)
# ==============================================================================
ax_e.loglog(hs, errs_relative, "-s", lw=2, ms=5, label="Error relativo", color='green')

# Referencia teórica: O(h²) para elementos P1
# Calcular constante de referencia del primer punto
C_ref = errs_relative[0] / (hs[0]**2 + 1e-30)  # Evitar división por cero
ax_e.loglog(hs, C_ref * hs**2, "r--", lw=2, label="O(h²)")

ax_e.set_xlabel("Tamaño de malla h [m]")
ax_e.set_ylabel("Error relativo")
ax_e.grid(True, which="both", ls=":", alpha=0.6)
ax_e.legend()

# INVERTIR EJE X también en escala logarítmica
ax_e.invert_xaxis()

plt.tight_layout(rect=[0, 0, 1, 0.92])
print("✓ Análisis de convergencia generado")

# ==============================================================================
# MOSTRAR TODAS LAS FIGURAS
# ==============================================================================
plt.show()

# ==============================================================================
# ANÁLISIS NUMÉRICO ADICIONAL (CONSOLA)
# ==============================================================================
print("\n" + "=" * 70)
print("RESUMEN DE CONVERGENCIA NUMÉRICA")
print("=" * 70)

# Calcular tasas de convergencia empíricas
if len(hs) >= 2:
    rates = []
    for i in range(1, len(hs)):
        rate = np.log(errs_relative[i] / errs_relative[i-1]) / np.log(hs[i] / hs[i-1])
        rates.append(rate)
    
    print(f"Tasas de convergencia empíricas: {[f'{r:.2f}' for r in rates]}")
    print(f"Tasa promedio: {np.mean(rates):.2f} (teórica: 2.00)")

print(f"\nEvolución del error:")
for i, rec in enumerate(records):
    print(f"  h={rec['h']:.4f} → Error={rec['err_pct']:.2f}% → kc_FEM={rec['kc_fem']:.6f}")

print(f"\nValor analítico de referencia: kc = {kc_analytical:.6f}")
print("=" * 70)