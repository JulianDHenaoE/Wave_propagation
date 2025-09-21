import numpy as np

class BilinearElement:
    """Bilinear quadrilateral element for FEM"""

    @staticmethod
    def shape_functions(xi, eta):
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def shape_derivatives(xi, eta):
        dN_dxi = np.array([
            -0.25 * (1 - eta),
             0.25 * (1 - eta),
             0.25 * (1 + eta),
            -0.25 * (1 + eta)
        ])
        dN_deta = np.array([
            -0.25 * (1 - xi),
            -0.25 * (1 + xi),
             0.25 * (1 + xi),
             0.25 * (1 - xi)
        ])
        return dN_dxi, dN_deta
