import numpy as np

class Q1:
    """Bilinear quadrilateral element (Q1) on [-1,1]^2."""
    
    @staticmethod
    def shape_functions(xi, eta):
        """
        Funciones de forma N1...N4 evaluadas en coordenadas naturales (xi, eta).
        xi, eta ∈ [-1,1]
        """
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        return np.array([N1, N2, N3, N4], dtype=float)

    @staticmethod
    def shape_derivatives(xi, eta):
        """
        Derivadas de las funciones de forma respecto a xi y eta.
        Devuelve (dN_dxi, dN_deta) como arrays de longitud 4.
        """
        dN_dxi  = np.array([
            -0.25 * (1 - eta),  # ∂N1/∂xi
             0.25 * (1 - eta),  # ∂N2/∂xi
             0.25 * (1 + eta),  # ∂N3/∂xi
            -0.25 * (1 + eta)   # ∂N4/∂xi
        ], dtype=float)

        dN_deta = np.array([
            -0.25 * (1 - xi),   # ∂N1/∂eta
            -0.25 * (1 + xi),   # ∂N2/∂eta
             0.25 * (1 + xi),   # ∂N3/∂eta
             0.25 * (1 - xi)    # ∂N4/∂eta
        ], dtype=float)

        return dN_dxi, dN_deta
