import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse

def solve_helmholtz_numerically(X, Y, source, k, boundary_condition='sommerfeld'):
    """
    Resuelve la ecuación de Helmholtz 2D numéricamente usando diferencias finitas.
    ∇²ψ + k²ψ = f(x,y)
    
    Parámetros:
        X, Y: Mallas de coordenadas
        source: Campo de fuente f(x,y)
        k: Número de onda
        boundary_condition: Condición de frontera
    """
    nx, ny = X.shape
    dx = (X[0,1] - X[0,0])
    dy = (Y[1,0] - Y[0,0])
    
    # Número total de puntos
    N = nx * ny
    
    # Constantes de la discretización
    hx2 = dx * dx
    hy2 = dy * dy
    k2 = k * k
    
    # Constantes para la matriz
    diag_const = -2/hx2 - 2/hy2 + k2
    x_const = 1/hx2
    y_const = 1/hy2
    
    # Construir matriz dispersa
    diagonals = []
    offsets = []
    
    # Diagonal principal
    diagonals.append(np.ones(N) * diag_const)
    offsets.append(0)
    
    # Sub/super diagonales para derivadas en x
    diagonals.append(np.ones(N-1) * x_const)
    offsets.append(1)
    diagonals.append(np.ones(N-1) * x_const)
    offsets.append(-1)
    
    # Sub/super diagonales para derivadas en y
    diagonals.append(np.ones(N-nx) * y_const)
    offsets.append(nx)
    diagonals.append(np.ones(N-nx) * y_const)
    offsets.append(-nx)
    
    # Crear matriz
    A = diags(diagonals, offsets, shape=(N, N), format='csr')
    
    # Aplicar condiciones de frontera (simplificado)
    if boundary_condition == 'dirichlet':
        # Para fronteras, establecer 1 en la diagonal y 0 en otros lugares
        pass  # Implementación simplificada
    
    # Vector fuente
    b = source.ravel()
    
    # Resolver sistema lineal
    psi_numeric = spsolve(A, b)
    
    return psi_numeric.reshape((nx, ny))