# -*- coding: utf-8 -*-
"""
modes_TE.py
Visualiza los primeros modos TE de una guía rectangular con FEM.
Permite elegir entre graficar los modos crudos ("raw") o alineados con la solución analítica ("aligned").

FUNDAMENTO FÍSICO: Este script calcula y visualiza los modos Transverso Eléctrico (TE)
en una guía de onda rectangular. Los modos TE se caracterizan por:
- Componente eléctrica transversal (E_z = 0)
- Componente magnética longitudinal (H_z ≠ 0)
- Campo: H_z(x,y) = cos(mπx/W)cos(nπy/H)
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

# Resolución de la malla FEM
Nx, Ny = 40, 20  # 40×20 = 800 nodos, ~1482 triángulos

# Modo de visualización del campo electromagnético
# "raw"     → Muestra el campo FEM directo (puede tener signos inconsistentes)
# "aligned" → Alinea y normaliza el campo FEM con la solución analítica
mode_view = "aligned"

# ==============================================================================
# LISTA DE MODOS TE A CALCULAR
# ==============================================================================
"""
Los modos TE_mn están definidos por dos números enteros:
- m: número de semi-variaciones en dirección x (ancho W)
- n: número de semi-variaciones en dirección y (alto H)

Restricciones físicas:
- m ≥ 0, n ≥ 0 (enteros no negativos)
- m y n no pueden ser ambos cero (TE00 no existe)
- La frecuencia de corte aumenta con m y n
"""
modes_TE = [
    (0, 1), (0, 2), (0, 3),      # Modos con m=0 (variación solo en y)
    (1, 0), (1, 1), (1, 2), (1, 3),  # Modos con m=1
    (2, 0), (2, 1), (2, 2), (2, 3),  # Modos con m=2  
    (3, 0), (3, 1), (3, 2), (3, 3)   # Modos con m=3
]

# Nota: TE10 es el modo fundamental (menor frecuencia de corte)


# ==============================================================================
# FUNCIÓN DE PROCESAMIENTO DEL CAMPO
# ==============================================================================
def process_field(vec, Xy, Tri, mode_view="aligned"):
    """
    Procesa el campo vectorial FEM para mejorar la visualización.

    Parámetros
    ----------
    vec : ndarray
        Vector del campo FEM calculado (autovector).
    Xy : ndarray
        Coordenadas de los nodos (no usado aquí, pero mantiene interfaz).
    Tri : ndarray  
        Conectividad triangular (no usado aquí, pero mantiene interfaz).
    mode_view : str
        Modo de procesamiento: "raw" o "aligned".

    Retorna
    -------
    processed_field : ndarray
        Campo procesado listo para visualización.
    """
    # Crear copia del vector para no modificar el original
    v = np.array(vec, dtype=float)

    if mode_view == "raw":
        # ======================================================================
        # MODO RAW: CAMPO FEM DIRECTO
        # ======================================================================
        """
        Muestra el campo tal como sale del solver FEM.
        Ventaja: Muestra la solución numérica real.
        Desventaja: Puede tener signos inconsistentes entre ejecuciones
                    y diferentes escalas entre modos.
        """
        return np.abs(v)  # Valor absoluto para visualización

    elif mode_view == "aligned":
        # ======================================================================
        # MODO ALIGNED: CAMPO NORMALIZADO Y ALINEADO
        # ======================================================================
        """
        Procesamiento para mejor visualización:
        1. Normalización: Escala todos los modos al mismo rango [0,1]
        2. Alineación de signo: Evita inversiones aleatorias entre ejecuciones
        3. Valor absoluto: Mejor contraste visual
        """
        
        # Paso 1: Normalizar por el valor máximo absoluto
        vmax = np.max(np.abs(v))
        if vmax > 0:
            v = v / vmax  # Escalar al rango [-1, 1]

        # Paso 2: Corregir signo para consistencia
        # Si la suma es negativa, invertir todo el campo
        # Esto asegura que el "lóbulo principal" sea siempre positivo
        if np.sum(v) < 0:
            v = -v

        # Paso 3: Tomar valor absoluto para visualización
        # En modos TE, |H_z| muestra la intensidad del campo magnético
        return np.abs(v)

    else:
        raise ValueError(f"Modo de vista no reconocido: {mode_view}")


# ==============================================================================
# CONFIGURACIÓN DE LA FIGURA PRINCIPAL
# ==============================================================================

# Calcular disposición de subplots
ncols = 4  # 4 columnas de modos
nrows = int(np.ceil(len(modes_TE) / ncols))  # Filas necesarias

# Crear figura con tamaño proporcional al número de filas
# figsize: (ancho, alto) en pulgadas
fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))

# Título principal de la figura
fig.suptitle(f"Modos TEmn - Guía Rectangular {W}×{H} - Visualización: {mode_view}",
             fontsize=16, fontweight="bold", y=0.98)


# ==============================================================================
# BUCLE PRINCIPAL: CÁLCULO Y VISUALIZACIÓN DE CADA MODO
# ==============================================================================
for idx, (m, n) in enumerate(modes_TE):
    """
    Para cada modo TE(m,n):
    1. Resolver el problema de autovalores FEM
    2. Procesar el campo para visualización  
    3. Graficar en subplot correspondiente
    4. Añadir información del modo
    """
    
    # Calcular posición en la grilla de subplots
    row = idx // ncols  # Fila actual (división entera)
    col = idx % ncols   # Columna actual (módulo)

    # Obtener referencia al subplot actual
    ax = axs[row, col]
    
    # ==========================================================================
    # PASO 1: RESOLVER MODO CON FEM
    # ==========================================================================
    """
    solve_mode retorna:
    - kc_fem: constante de corte FEM [m⁻¹]
    - kc_ana: constante de corte analítica [m⁻¹]  
    - err: error relativo porcentual
    - vec: autovector (campo H_z en los nodos)
    - Xy: coordenadas de nodos
    - Tri: conectividad triangular
    """
    kc_fem, kc_ana, err, vec, Xy, Tri = solve_mode("TE", m, n, Nx, Ny, W, H)
    
    # Crear objeto triangulación para matplotlib
    triang = Triangulation(Xy[:, 0], Xy[:, 1], Tri)

    # ==========================================================================
    # PASO 2: PROCESAR CAMPO PARA VISUALIZACIÓN
    # ==========================================================================
    v_proc = process_field(vec, Xy, Tri, mode_view=mode_view)

    # ==========================================================================
    # PASO 3: VISUALIZACIÓN CON TRIPCOLOR
    # ==========================================================================
    """
    tripcolor crea un plot de contorno coloreado sobre la malla triangular.
    shading="gouraud": suavizado para mejor apariencia
    cmap="viridis": mapa de colores perceptualmente uniforme
    """
    tpc = ax.tripcolor(triang, v_proc, shading="gouraud", cmap="viridis")
    
    # ==========================================================================
    # PASO 4: CONFIGURACIÓN DEL SUBPLOT
    # ==========================================================================
    
    # Mantener relación de aspecto correcta (sin distorsión)
    ax.set_aspect("equal")
    
    # Remover ticks numéricos para claridad visual
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ==========================================================================
    # PASO 5: TÍTULO INFORMATIVO DEL MODO
    # ==========================================================================
    """
    Información mostrada:
    - Modo: TEmn
    - kc_FEM: constante de corte numérica
    - kc_Ana: constante de corte analítica  
    - Error: diferencia relativa porcentual
    """
    ax.set_title(f"TE{m}{n}\n"
                 f"FEM: {kc_fem:.3f}\n"
                 f"Ana: {kc_ana:.3f}\n"
                 f"Err: {err:.2f}%", 
                 fontsize=7, pad=2)


# ==============================================================================
# LIMPIEZA DE SUBPLOTS VACÍOS
# ==============================================================================
"""
Si el número total de modos no llena completamente la grilla,
desactivar los subplots sobrantes.
"""
for idx in range(len(modes_TE), nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    axs[row, col].axis("off")


# ==============================================================================
# AJUSTES FINALES DE DISPOSICIÓN
# ==============================================================================
"""
Ajustar espaciado y disposición para mejor legibilidad:
- tight_layout: ajusta automáticamente los márgenes
- rect: define el área usable [left, bottom, right, top] en coordenadas normalizadas
- subplots_adjust: control fino del espaciado
"""
plt.tight_layout(rect=[0, 0, 1, 0.93])  # 93% de la altura para los subplots
plt.subplots_adjust(top=0.90, hspace=0.5, wspace=0.25)  # Espaciado entre subplots

# ==============================================================================
# MOSTRAR RESULTADOS
# ==============================================================================
plt.show()

# ==============================================================================
# INFORMACIÓN ADICIONAL (impresa en consola)
# ==============================================================================
print("=" * 70)
print("SIMULACIÓN DE MODOS TE EN GUÍA RECTANGULAR")
print("=" * 70)
print(f"Dimensiones de la guía: {W} × {H} m")
print(f"Resolución de malla: {Nx} × {Ny}")
print(f"Nodos: {Nx * Ny}, Elementos: ~{2 * (Nx-1) * (Ny-1)}")
print(f"Modos calculados: {len(modes_TE)} modos TE")
print(f"Visualización: {mode_view}")
print(f"Modo fundamental: TE10")
print(f"Frecuencia de corte TE10: {(3e8 * np.pi/W)/(2*np.pi)/1e6:.2f} MHz")
print("=" * 70)