# -*- coding: utf-8 -*-
"""
modes_TM.py
Visualiza los primeros modos TM (Transverso Magnético) de una guía rectangular con FEM.
Compara soluciones numéricas FEM con soluciones analíticas y muestra patrones de campo.

FUNDAMENTO FÍSICO: Este script calcula y visualiza modos Transverso Magnético (TM)
en una guía de onda rectangular. Los modos TM se caracterizan por:
- Componente magnética transversal (H_z = 0)
- Componente eléctrica longitudinal (E_z ≠ 0)
- Campo: E_z(x,y) = sin(mπx/W)sin(nπy/H)
- Condiciones de contorno: E_z = 0 en las paredes (Dirichlet)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from eigen import solve_mode

# ==============================================================================
# PARÁMETROS DE CONFIGURACIÓN
# ==============================================================================

# Dimensiones físicas de la guía de onda [metros]
W, H = 4.0, 3.0  # Relación de aspecto 4:3

# Resolución de la malla FEM - suficiente para capturar variaciones modales
Nx, Ny = 40, 20  # 40×20 = 800 nodos, ~1482 triángulos

# Modo de visualización del campo electromagnético
mode_view = "aligned"  # Opciones: "raw" (directo) o "aligned" (normalizado)

# ==============================================================================
# LISTA DE MODOS TM A CALCULAR
# ==============================================================================
"""
Los modos TM_mn están definidos por dos números enteros:
- m: número de semi-variaciones en dirección x (ancho W)
- n: número de semi-variaciones en dirección y (alto H)

Restricciones físicas para modos TM:
- m ≥ 1, n ≥ 1 (ambos deben ser positivos)
- TM00, TM01, TM10 no existen físicamente
- El modo fundamental TM es el TM11
"""
modes_TM = [
    (1, 1), (1, 2), (1, 3),  # Modos con m=1
    (2, 1), (2, 2), (2, 3),  # Modos con m=2  
    (3, 1), (3, 2), (3, 3)   # Modos con m=3
]

# Nota: TM11 es el modo TM fundamental (menor frecuencia de corte entre modos TM)

# ==============================================================================
# FUNCIÓN DE PROCESAMIENTO DEL CAMPO ELECTROMAGNÉTICO
# ==============================================================================
def process_field(vec, mode_view="aligned"):
    """
    Procesa el campo vectorial FEM del modo TM para visualización óptima.

    Parámetros
    ----------
    vec : ndarray
        Vector del campo FEM calculado (autovector para E_z).
    mode_view : str
        Modo de procesamiento: "raw" o "aligned".

    Retorna
    -------
    processed_field : ndarray
        Campo procesado listo para visualización.

    Notas
    -----
    Para modos TM, el campo vectorial representa E_z (componente eléctrica longitudinal).
    El procesamiento mejora la consistencia visual entre diferentes modos y ejecuciones.
    """
    # Crear copia del vector para no modificar el original
    v = np.array(vec, dtype=float)
    
    if mode_view == "raw":
        # ======================================================================
        # MODO RAW: CAMPO FEM DIRECTO
        # ======================================================================
        """
        Ventajas:
        - Muestra la solución numérica real sin modificaciones
        - Preserva la escala física real del campo
        
        Desventajas:
        - Diferentes escalas entre modos dificultan la comparación visual
        - Signo aleatorio en autovectores puede causar inconsistencia
        """
        return np.abs(v)  # Valor absoluto para visualización de intensidad

    elif mode_view == "aligned":
        # ======================================================================
        # MODO ALIGNED: CAMPO NORMALIZADO Y ALINEADO
        # ======================================================================
        """
        Procesamiento para visualización consistente:
        1. Normalización: Escala al rango [0,1] para comparación uniforme
        2. Corrección de signo: Evita inversiones aleatorias entre ejecuciones
        3. Valor absoluto: Muestra intensidad del campo (|E_z|)
        """
        
        # Paso 1: Normalizar por el valor máximo absoluto
        vmax = np.max(np.abs(v))
        if vmax > 0: 
            v = v / vmax  # Escalar al rango [-1, 1]
        
        # Paso 2: Corregir signo para consistencia entre ejecuciones
        # Los solvers de autovalores pueden devolver ±φ para el mismo modo físico
        # Forzamos que la suma sea positiva para consistencia visual
        if np.sum(v) < 0: 
            v = -v  # Invertir si la suma es negativa
        
        # Paso 3: Tomar valor absoluto para visualización de intensidad
        # |E_z| muestra la distribución de intensidad del campo eléctrico
        return np.abs(v)
    
    else:
        raise ValueError(f"Modo de visualización no reconocido: {mode_view}")

# ==============================================================================
# CONFIGURACIÓN DE LA FIGURA PRINCIPAL
# ==============================================================================

# Calcular disposición de subplots (3 columnas)
ncols = 3
nrows = int(np.ceil(len(modes_TM) / ncols))  # Filas necesarias

# Crear figura con tamaño adaptativo
# figsize: (ancho, alto) en pulgadas - altura proporcional al número de filas
fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))

# Título principal de la figura
fig.suptitle(f"Modos TMmn - Guía Rectangular {W}×{H}", 
             fontsize=16, y=0.98, fontweight="bold")

# ==============================================================================
# BUCLE PRINCIPAL: CÁLCULO Y VISUALIZACIÓN DE CADA MODO TM
# ==============================================================================
for idx, (m, n) in enumerate(modes_TM):
    """
    Para cada modo TM(m,n):
    1. Resolver problema de autovalores FEM
    2. Crear triangulación para visualización
    3. Procesar campo para visualización óptima
    4. Graficar y configurar subplot
    5. Añadir información numérica del modo
    """
    
    # Calcular posición en la grilla de subplots
    row = idx // ncols  # Fila actual
    col = idx % ncols   # Columna actual
    
    # Obtener referencia al subplot actual
    ax = axs[row, col]
    
    # ==========================================================================
    # PASO 1: RESOLVER MODO TM CON FEM
    # ==========================================================================
    """
    solve_mode para modos TM:
    - Aplica condiciones de Dirichlet (E_z=0) en el contorno automáticamente
    - Retorna solución FEM y datos de referencia analítica
    """
    kc_fem, kc_ana, err, vec, Xy, Tri = solve_mode("TM", m, n, Nx, Ny, W, H)
    
    # ==========================================================================
    # PASO 2: CREAR TRIANGULACIÓN PARA VISUALIZACIÓN
    # ==========================================================================
    triang = Triangulation(Xy[:, 0], Xy[:, 1], Tri)
    
    # ==========================================================================
    # PASO 3: PROCESAR CAMPO PARA VISUALIZACIÓN
    # ==========================================================================
    v_proc = process_field(vec, mode_view=mode_view)
    
    # ==========================================================================
    # PASO 4: VISUALIZACIÓN CON TRIPCOLOR
    # ==========================================================================
    """
    tripcolor con shading="gouraud" crea una visualización suavizada
    cmap="plasma": mapa de colores que resalta bien los patrones modales TM
    """
    tpc = ax.tripcolor(triang, v_proc, shading="gouraud", cmap="plasma")
    
    # ==========================================================================
    # PASO 5: CONFIGURACIÓN DEL SUBPLOT
    # ==========================================================================
    
    # Mantener relación de aspecto 1:1 (sin distorsión geométrica)
    ax.set_aspect("equal")
    
    # Remover marcas de ejes para claridad visual
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ==========================================================================
    # PASO 6: TÍTULO INFORMATIVO DEL MODO
    # ==========================================================================
    """
    Información mostrada por subplot:
    - Identificación del modo: TMmn
    - Constante de corte FEM: kc_fem
    - Constante de corte analítica: kc_ana  
    - Error relativo porcentual: err
    """
    ax.set_title(f"TM{m}{n}\n"
                 f"FEM: {kc_fem:.3f}\n"
                 f"Ana: {kc_ana:.3f}\n"
                 f"Err: {err:.2f}%", 
                 fontsize=7, pad=2)

# ==============================================================================
# LIMPIEZA DE SUBPLOTS VACÍOS
# ==============================================================================
"""
Desactivar los subplots sobrantes si el número de modos 
no llena completamente la grilla nrows×ncols.
"""
for idx in range(len(modes_TM), nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    axs[row, col].axis('off')  # Ocultar ejes y contenido

# ==============================================================================
# AJUSTES FINALES DE DISPOSICIÓN
# ==============================================================================
"""
Ajustar espaciamiento para óptima legibilidad:
- tight_layout: ajuste automático de márgenes
- rect: área usable [left, bottom, right, top] en coordenadas normalizadas (0-1)
- subplots_adjust: control manual fino del espaciado
"""
plt.tight_layout(rect=[0, 0, 1, 0.99])  # 99% de altura para subplots
plt.subplots_adjust(top=0.90, hspace=0.5, wspace=0.25)  # Espaciado entre subplots

# ==============================================================================
# MOSTRAR LA FIGURA
# ==============================================================================
plt.show()

# ==============================================================================
# INFORMACIÓN ADICIONAL EN CONSOLA
# ==============================================================================
print("=" * 70)
print("SIMULACIÓN DE MODOS TM EN GUÍA RECTANGULAR")
print("=" * 70)
print(f"Dimensiones de la guía: {W} × {H} m")
print(f"Resolución de malla: {Nx} × {Ny}")
print(f"Nodos: {Nx * Ny}, Elementos triangulares: ~{2 * (Nx-1) * (Ny-1)}")
print(f"Modos calculados: {len(modes_TM)} modos TM")
print(f"Visualización: {mode_view}")
print(f"Modo TM fundamental: TM11")
print(f"Frecuencia de corte TM11: {(3e8 * np.sqrt((np.pi/W)**2 + (np.pi/H)**2))/(2*np.pi)/1e6:.2f} MHz")
print("=" * 70)
print("CARACTERÍSTICAS DE MODOS TM:")
print("- Campo eléctrico longitudinal E_z ≠ 0")
print("- Campo magnético transversal (H_z = 0)")
print("- E_z = 0 en todas las paredes (condición Dirichlet)")
print("- m ≥ 1, n ≥ 1 (ambos índices deben ser positivos)")
print("=" * 70)