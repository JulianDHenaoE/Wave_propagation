# -*- coding: utf-8 -*-
"""
mesh.py
Genera mallas triangulares estructuradas para la guía rectangular.
Devuelve coordenadas, conectividad y tamaño característico de malla.

USO PRINCIPAL: Discretización espacial para el Método de Elementos Finitos (FEM)
en la solución de ecuaciones electromagnéticas en guías de onda rectangulares.
"""

import numpy as np  # Librería numérica para operaciones matemáticas eficientes

def rect_mesh(Nx: int, Ny: int, W: float, H: float):
    """
    Genera una malla estructurada de triángulos en el rectángulo [0,W]x[0,H].
    
    La malla se utiliza como base para discretizar ecuaciones diferenciales
    en simulaciones electromagnéticas usando el Método de Elementos Finitos.

    Parámetros
    ----------
    Nx, Ny : int
        Número de divisiones en x y en y. Deben ser enteros porque representan
        conteos discretos de elementos en la discretización espacial.
    W, H : float
        Dimensiones físicas de la guía de onda en metros. Son decimales porque
        representan medidas físicas continuas del dominio de simulación.

    Retorna
    -------
    Xy : ndarray (N,2)
        Matriz de coordenadas de los nodos. Cada fila contiene las coordenadas (x,y)
        de un punto de la malla. Formato: [[x0,y0], [x1,y1], ..., [xN-1,yN-1]]
    Tri : ndarray (M,3)
        Matriz de conectividad triangular. Cada fila contiene los ÍNDICES de los 3 nodos
        que forman un triángulo. Formato: [[nodo0, nodo1, nodo2], ...]
    h : float
        Tamaño característico de la malla (máximo entre hx y hy). Representa la
        resolución espacial máxima y determina la precisión de la simulación FEM.
    """
    
    # ==========================================================================
    # ETAPA 1: GENERACIÓN DE COORDENADAS DE NODOS (PUNTOS DE LA MALLA)
    # ==========================================================================
    
    # Crear arrays de coordenadas 1D equiespaciadas en cada dirección
    xs = np.linspace(0, W, Nx)  # Puntos desde 0 hasta W, con Nx divisiones
    ys = np.linspace(0, H, Ny)  # Puntos desde 0 hasta H, con Ny divisiones
    
    # Crear malla 2D: genera matrices de coordenadas X e Y para todos los puntos
    # indexing="xy" significa: primera dimensión = filas (Y), segunda = columnas (X)
    XS, YS = np.meshgrid(xs, ys, indexing="xy")
    
    # Aplanar las matrices 2D y combinar en una sola matriz de coordenadas Nx2
    # np.c_[] concatena columnas, .ravel() aplana las matrices por filas
    Xy = np.c_[XS.ravel(), YS.ravel()]  # Forma final: (Nx*Ny, 2)
    

    # ==========================================================================
    # ETAPA 2: FUNCIÓN AUXILIAR PARA NUMERACIÓN DE NODOS
    # ==========================================================================
    
    def nidx(i: int, j: int) -> int:
        """
        Calcula el índice global de un nodo dada su posición (i,j) en la malla.
        
        Sistema de numeración por FILAS (row-major order):
        - i: índice en dirección x (columna)
        - j: índice en dirección y (fila)
        
        Fórmula: índice_global = fila_actual × ancho_grid + columna_actual
                 índice = j × Nx + i
                 
        Ejemplo para Nx=3:
            (i=0,j=0)->0  (i=1,j=0)->1  (i=2,j=0)->2
            (i=0,j=1)->3  (i=1,j=1)->4  (i=2,j=1)->5
        """
        return j * Nx + i
    

    # ==========================================================================
    # ETAPA 3: GENERACIÓN DE CONECTIVIDAD TRIANGULAR
    # ==========================================================================
    
    tris = []  # Lista temporal para almacenar los triángulos
    
    # Recorrer todas las celdas rectangulares de la malla
    # Hay (Ny-1) filas de celdas y (Nx-1) columnas de celdas
    for j in range(Ny - 1):      # j recorre filas de celdas (dirección Y)
        for i in range(Nx - 1):  # i recorre columnas de celdas (dirección X)
            
            # Obtener los ÍNDICES de los 4 vértices de la celda rectangular actual
            # Convención de nomenclatura: n[filas][columnas]
            n00 = nidx(i, j)      # Esquina inferior-izquierda (0,0)
            n10 = nidx(i + 1, j)  # Esquina inferior-derecha (1,0)
            n01 = nidx(i, j + 1)  # Esquina superior-izquierda (0,1)
            n11 = nidx(i + 1, j + 1)  # Esquina superior-derecha (1,1)
            
            # Dividir la celda rectangular en DOS triángulos mediante la diagonal
            # Triángulo 1: n00 → n10 → n11 (diagonal inferior-derecha)
            tris.append([n00, n10, n11])
            # Triángulo 2: n00 → n11 → n01 (diagonal superior-izquierda)
            tris.append([n00, n11, n01])
            
            # Visualización de la división:
            # n01 ───── n11
            # │       ╱ │
            # │     ╱   │  
            # │   ╱     │
            # │ ╱       │
            # n00 ───── n10
    
    # Convertir la lista de triángulos a array numpy para eficiencia computacional
    Tri = np.array(tris, dtype=int)  # dtype=int porque son índices de nodos
    

    # ==========================================================================
    # ETAPA 4: CÁLCULO DEL TAMAÑO CARACTERÍSTICO DE MALLA
    # ==========================================================================
    
    # Calcular espaciamiento entre nodos en cada dirección
    hx = W / (Nx - 1)  # Distancia entre nodos consecutivos en dirección X
    hy = H / (Ny - 1)  # Distancia entre nodos consecutivos en dirección Y
    
    # El tamaño característico h es el MÁXIMO espaciamiento
    # Esto garantiza que en el análisis de error del FEM se considere el peor caso
    h = max(hx, hy)
    

    # ==========================================================================
    # ETAPA 5: RETORNO DE LOS RESULTADOS
    # ==========================================================================
    
    return Xy, Tri, h


# ==============================================================================
# BLOQUE DE PRUEBA Y DEMOSTRACIÓN
# ==============================================================================
if __name__ == "__main__":
    """
    Bloque que solo se ejecuta cuando el archivo se corre directamente,
    no cuando se importa como módulo.
    
    Propósito: Demostrar el funcionamiento de la función rect_mesh
    y visualizar la malla generada.
    """
    
    # Importar librerías de visualización (solo para este bloque de prueba)
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation  # Para visualización de mallas triangulares

    # ==========================================================================
    # PARÁMETROS DE EJEMPLO
    # ==========================================================================
    
    # Dimensiones físicas de la guía de onda de ejemplo
    W, H = 2.0, 1.0  # Ancho = 2.0 unidades, Alto = 1.0 unidad
    
    # Resolución de la malla: número de divisiones en cada dirección
    Nx, Ny = 10, 4   # 10 divisiones en X, 4 divisiones en Y
                     # Total de nodos = Nx × Ny = 10 × 4 = 40 nodos
                     # Total de triángulos = 2 × (Nx-1) × (Ny-1) = 2 × 9 × 3 = 54 triángulos

    # ==========================================================================
    # GENERACIÓN DE LA MALLA
    # ==========================================================================
    
    # Llamar a la función principal para generar la malla
    # Xy: coordenadas de los 40 nodos
    # Tri: conectividad de los 54 triángulos  
    # h: tamaño característico de la malla
    Xy, Tri, h = rect_mesh(Nx, Ny, W, H)
    
    # Mostrar estadísticas de la malla generada
    print(f"Malla generada: {len(Xy)} nodos, {len(Tri)} triángulos, h={h:.3f}")
    # Ejemplo de salida: "Malla generada: 40 nodos, 54 triángulos, h=0.222"

    # ==========================================================================
    # VISUALIZACIÓN DE LA MALLA
    # ==========================================================================
    
    # Crear objeto Triangulation para matplotlib
    triang = Triangulation(Xy[:, 0],  # Coordenadas X de todos los nodos
                          Xy[:, 1],  # Coordenadas Y de todos los nodos  
                          Tri)       # Conectividad triangular
    
    # Configurar la figura para visualización
    plt.figure(figsize=(5, 3))  # Tamaño de figura: 5 pulgadas ancho × 3 alto
    
    # Dibujar la malla triangular
    plt.triplot(triang, color="k")  # 'k' = color negro, líneas continuas
    
    # Configurar aspecto de la visualización
    plt.gca().set_aspect("equal")  # Misma escala en ejes X e Y (sin distorsión)
    
    # Añadir título descriptivo
    plt.title("Malla estructurada para guía rectangular")
    
    # Mostrar la figura en pantalla
    plt.show()
    
    # ==========================================================================
    # INFORMACIÓN ADICIONAL PARA EL USUARIO
    # ==========================================================================
    print("\n" + "="*50)
    print("INFORMACIÓN SOBRE LA MALLA GENERADA:")
    print("="*50)
    print(f"• Dimensiones físicas: {W} × {H}")
    print(f"• Resolución: {Nx} × {Ny} divisiones")
    print(f"• Nodos totales: {len(Xy)}")
    print(f"• Elementos triangulares: {len(Tri)}")
    print(f"• Tamaño característico (h): {h:.3f}")
    print(f"• Espaciamiento en X (hx): {W/(Nx-1):.3f}")
    print(f"• Espaciamiento en Y (hy): {H/(Ny-1):.3f}")
    print("\nEsta malla está lista para usar en simulaciones FEM!")