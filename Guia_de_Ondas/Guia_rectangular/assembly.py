# -*- coding: utf-8 -*-
"""
assembly.py
Ensambla matrices globales K (rigidez) y M (masa) para FEM P1 (triángulos)
en una guía rectangular. No depende de boundary_flags.

FUNDAMENTO MATEMÁTICO: Este módulo implementa el ensamblaje de las matrices
del Método de Elementos Finitos para la ecuación de Helmholtz:
    -∇²φ = kc²φ
donde K es la matriz de rigidez (discretización de -∇²) y M es la matriz 
de masa (discretización de la identidad).
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def assemble_K_M(coords: np.ndarray, elems: np.ndarray):
    """
    Ensambla matrices globales K y M (elementos P1 sobre triángulos).

    Parámetros
    ----------
    coords : (N,2) ndarray
        Coordenadas de los N nodos. Cada fila es [x_i, y_i].
    elems : (M,3) ndarray
        Conectividad triangular (índices de nodos por triángulo).
        Cada fila es [nodo0, nodo1, nodo2] definiendo un triángulo.

    Retorna
    -------
    K, M : scipy.sparse.csr_matrix
        Matrices globales de rigidez y masa en formato CSR sparse.
        K representa el operador Laplaciano discreto (-∇²)
        M representa el operador Identidad discreta

    Notas
    -----
    Para elementos P1 (lineales), las funciones de forma en un triángulo son:
        N_i(x,y) = a_i + b_i*x + c_i*y, i=0,1,2
    donde los coeficientes se calculan usando coordenadas de área.
    """
    
    # Número total de nodos en la malla
    N = coords.shape[0]
    
    # Inicializar matrices globales en formato LIL (List of Lists)
    # Este formato es eficiente para construcción incremental
    K = lil_matrix((N, N), dtype=float)  # Matriz de rigidez
    M = lil_matrix((N, N), dtype=float)  # Matriz de masa

    # ==========================================================================
    # MATRIZ DE MASA LOCAL (ELEMENTAL) - CONSTANTE PARA TODOS LOS TRIÁNGULOS
    # ==========================================================================
    """
    Para elementos P1 en triángulos, la matriz de masa local es:
        M_ij = ∫∫_T N_i(x,y) N_j(x,y) dA
    
    Para un triángulo de área A, la matriz es:
        M_local = (A/12) * [[2, 1, 1],
                            [1, 2, 1], 
                            [1, 1, 2]]
    
    Esta matriz es la misma para todos los triángulos, solo se escala por el área.
    """
    Mloc = (1.0 / 12.0) * np.array([[2, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 2]], dtype=float)

    # ==========================================================================
    # BUCLE PRINCIPAL: ENSAMBLAJE ELEMENTO POR ELEMENTO
    # ==========================================================================
    for tri in elems:
        # Obtener índices globales de los 3 nodos del triángulo actual
        ids = tri  # ids = [i0, i1, i2] - índices globales de los nodos
        
        # Obtener coordenadas (x,y) de los 3 vértices del triángulo
        x = coords[ids, 0]  # Coordenadas x de los 3 nodos: [x0, x1, x2]
        y = coords[ids, 1]  # Coordenadas y de los 3 nodos: [y0, y1, y2]

        # ======================================================================
        # CÁLCULO DE LA MATRIZ DE RIGIDEZ LOCAL (Ke)
        # ======================================================================
        
        # Matriz A para calcular gradientes de las funciones de forma
        # A = [[1, x0, y0],
        #      [1, x1, y1], 
        #      [1, x2, y2]]
        A = np.array([[1.0, x[0], y[0]],
                      [1.0, x[1], y[1]],
                      [1.0, x[2], y[2]]], dtype=float)
        
        # Determinante de A: relacionado con el área del triángulo
        # |det(A)| = 2 * área_del_triángulo
        detA = np.linalg.det(A)
        
        # Calcular área del triángulo
        area = 0.5 * abs(detA)
        
        # Verificar que el área no sea demasiado pequeña (elemento degenerado)
        if area <= 1e-18:
            continue  # Saltar este elemento si es numéricamente degenerado

        # Calcular inversa de A para obtener coeficientes b_i, c_i
        # Las funciones de forma lineales son: N_i(x,y) = a_i + b_i*x + c_i*y
        # donde [a_i, b_i, c_i]^T es la i-ésima columna de inv(A)
        invA = np.linalg.inv(A)
        
        # b = [b0, b1, b2] - coeficientes para derivadas en x
        b = invA[1, :]  # Segunda fila de inv(A) da los b_i
        
        # c = [c0, c1, c2] - coeficientes para derivadas en y  
        c = invA[2, :]  # Tercera fila de inv(A) da los c_i

        # ======================================================================
        # CONSTRUCCIÓN EXPLÍCITA DE LAS MATRICES LOCALES
        # ======================================================================
        
        # MATRIZ DE RIGIDEZ LOCAL (Ke):
        # Ke_ij = ∫∫_T ∇N_i · ∇N_j dA = área * (b_i*b_j + c_i*c_j)
        # Esto viene de: ∇N_i = (b_i, c_i) y ∇N_j = (b_j, c_j)
        Ke = area * (np.outer(b, b) + np.outer(c, c))
        
        # MATRIZ DE MASA LOCAL (Me):
        # Me_ij = ∫∫_T N_i N_j dA = área * Mloc[i,j]
        Me = area * Mloc

        # ======================================================================
        # ENSAMBLAJE EN MATRICES GLOBALES
        # ======================================================================
        """
        Estrategia de ensamblaje: para cada par de nodos (i,j) en el elemento,
        sumar la contribución elemental Ke[i,j] a la posición global K[I,J]
        donde I = ids[i], J = ids[j] son los índices globales.
        
        Este proceso se llama "scatter" o dispersión de contribuciones locales
        a las posiciones globales correspondientes.
        """
        for i in range(3):      # i: índice local del nodo (0,1,2)
            I = ids[i]          # I: índice global del nodo i
            for j in range(3):  # j: índice local del nodo (0,1,2)
                J = ids[j]      # J: índice global del nodo j
                
                # Sumar contribuciones elementales a matrices globales
                K[I, J] += Ke[i, j]  # K_IJ += Ke_ij
                M[I, J] += Me[i, j]  # M_IJ += Me_ij

    # ==========================================================================
    # CONVERSIÓN A FORMATO CSR (COMPRESSED SPARSE ROW)
    # ==========================================================================
    """
    CSR es más eficiente para operaciones algebraicas (multiplicaciones, 
    resoluciones de sistemas) que el formato LIL usado para construcción.
    """
    return csr_matrix(K), csr_matrix(M)


# ==============================================================================
# BLOQUE DE PRUEBA Y VALIDACIÓN
# ==============================================================================
if __name__ == "__main__":
    """
    Bloque de prueba que se ejecuta solo cuando el script corre directamente.
    Propósito: Validar el ensamblaje con una malla pequeña y visualizar.
    """
    
    # Importar librerías de visualización (solo para pruebas)
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    # Importar función de generación de malla desde módulo local
    from mesh import rect_mesh

    print("=" * 60)
    print("PRUEBA DE ENSAMBLAJE DE MATRICES K y M")
    print("=" * 60)

    # ==========================================================================
    # PARÁMETROS DE LA MALLA DE PRUEBA
    # ==========================================================================
    
    # Dimensiones de la guía de onda de prueba
    W, H = 1.0, 0.5  # Rectángulo 1.0 × 0.5
    
    # Resolución de malla (suficiente para prueba pero no muy grande)
    Nx, Ny = 10, 6   # 10×6 = 60 nodos, ~90 triángulos

    # Generar malla rectangular estructurada
    Xy, Tri, h = rect_mesh(Nx, Ny, W, H)
    
    print(f"Geometría: {W} × {H}")
    print(f"Malla: {Nx} × {Ny} divisiones")
    print(f"Nodos: {len(Xy)}, Triángulos: {len(Tri)}")
    print(f"Tamaño característico h: {h:.4f}")

    # ==========================================================================
    # ENSAMBLAJE DE MATRICES
    # ==========================================================================
    
    print("\nEnsamblando matrices K y M...")
    K, M = assemble_K_M(Xy, Tri)
    
    # ==========================================================================
    # INFORMACIÓN SOBRE LAS MATRICES ENSAMBLADAS
    # ==========================================================================
    
    print("\n" + "-" * 40)
    print("PROPIEDADES DE LAS MATRICES:")
    print("-" * 40)
    print(f"Matriz K (Rigidez):")
    print(f"  Dimensiones: {K.shape}")
    print(f"  Elementos no cero: {K.nnz}")
    print(f"  Densidad: {K.nnz / (K.shape[0]*K.shape[1]) * 100:.2f}%")
    print(f"  Tipo: {type(K).__name__}")
    
    print(f"\nMatriz M (Masa):")
    print(f"  Dimensiones: {M.shape}") 
    print(f"  Elementos no cero: {M.nnz}")
    print(f"  Densidad: {M.nnz / (M.shape[0]*M.shape[1]) * 100:.2f}%")
    print(f"  Tipo: {type(M).__name__}")

    # ==========================================================================
    # VERIFICACIONES NUMÉRICAS BÁSICAS
    # ==========================================================================
    
    print("\n" + "-" * 40)
    print("VERIFICACIONES NUMÉRICAS:")
    print("-" * 40)
    
    # Verificar simetría (K y M deben ser simétricas)
    K_sym_err = np.max(np.abs(K - K.T))
    M_sym_err = np.max(np.abs(M - M.T))
    print(f"Error simetría K: {K_sym_err:.2e}")
    print(f"Error simetría M: {M_sym_err:.2e}")
    
    # Verificar que K es semidefinida positiva (autovalor mínimo ≥ 0)
    # Nota: Para ahorrar tiempo, solo verificamos con unos pocos autovalores
    from scipy.sparse.linalg import eigsh
    lambda_min_K = eigsh(K, k=1, which='SA', return_eigenvectors=False)[0]
    print(f"Autovalor mínimo de K: {lambda_min_K:.2e}")

    # ==========================================================================
    # VISUALIZACIÓN DE LA MALLA
    # ==========================================================================
    
    print("\n" + "-" * 40)
    print("VISUALIZACIÓN DE LA MALLA:")
    print("-" * 40)
    
    # Crear objeto triangulación para matplotlib
    triang = Triangulation(Xy[:, 0], Xy[:, 1], Tri)
    
    # Configurar figura
    plt.figure(figsize=(8, 4))
    
    # Dibujar malla triangular
    plt.triplot(triang, lw=0.7, color="k")
    
    # Marcar nodos
    plt.plot(Xy[:, 0], Xy[:, 1], 'o', markersize=3, color='red', alpha=0.6)
    
    # Configurar gráfico
    plt.gca().set_aspect("equal")
    plt.title(f"Malla de Prueba: {Nx}×{Ny} divisiones\n"
              f"{len(Xy)} nodos, {len(Tri)} triángulos")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Mostrar gráfico
    plt.show()

    # ==========================================================================
    # INFORMACIÓN ADICIONAL SOBRE EL ENSAMBLAJE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("INFORMACIÓN TÉCNICA:")
    print("=" * 60)
    print("• K representa el operador Laplaciano: -∇²")
    print("• M representa el operador Identidad")
    print("• Formato LIL: Eficiente para construcción")
    print("• Formato CSR: Eficiente para operaciones algebraicas")
    print("• Elementos P1: Funciones de forma lineales por triángulo")
    print("• Ensamblaje: Estrategia 'scatter' elemento por elemento")
    print("=" * 60)