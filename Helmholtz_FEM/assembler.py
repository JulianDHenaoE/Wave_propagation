import numpy as np
from scipy.sparse import lil_matrix

from elements import BilinearElement
from pml import pml_functions

def assemble_fem_system(domain, frequency, velocity_array, source, alpha):
    """
    Ensambla el sistema FEM (A, F) para:
      ∫ ( (s_y/s_x) u_x v_x + (s_x/s_y) u_y v_y - k^2 s_x s_y u v ) dΩ = ∫ f v dΩ
    con elementos Q1 y cuadratura de Gauss 2×2.
    """
    nx, ny = domain.nx, domain.ny # nodos en x, y
    nk = nx * ny # total nodos

    K = lil_matrix((nk, nk), dtype=complex)  # rigidez
    M = lil_matrix((nk, nk), dtype=complex)  # masa-PML
    F = np.zeros(nk, dtype=complex)

    omega = 2 * np.pi * frequency
    pml_tr = pml_functions(domain, alpha, frequency)

    # 2x2 puntos de Gauss
    g = 1.0 / np.sqrt(3.0)
    gauss_points  = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    gauss_weights = np.ones(4)

    # bucle de elementos (rectángulos entre nodos)
    for j in range(ny - 1):
        for i in range(nx - 1):
            # nodos globales del elemento
            nodes = np.array([j*nx + i, j*nx + (i+1), (j+1)*nx + (i+1), (j+1)*nx + i])

            # coords de los 4 vértices
            x_elem = np.array([domain.x_array[i], domain.x_array[i+1],
                               domain.x_array[i+1], domain.x_array[i]])
            y_elem = np.array([domain.y_array[j], domain.y_array[j],
                               domain.y_array[j+1], domain.y_array[j+1]])

            Ke = np.zeros((4, 4), dtype=complex)
            Me = np.zeros((4, 4), dtype=complex)
            Fe = np.zeros(4, dtype=complex)

            for gp in range(4):
                xi, eta = gauss_points[gp]
                w = gauss_weights[gp]

                N = BilinearElement.shape_functions(xi, eta)
                dN_dxi, dN_deta = BilinearElement.shape_derivatives(xi, eta)

                # Jacobiano
                dx_dxi  = np.dot(dN_dxi,  x_elem)
                dx_deta = np.dot(dN_deta, x_elem)
                dy_dxi  = np.dot(dN_dxi,  y_elem)
                dy_deta = np.dot(dN_deta, y_elem)

                J = np.array([[dx_dxi, dx_deta],
                              [dy_dxi, dy_deta]])
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)

                # derivadas en coordenadas físicas
                dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
                dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta

                # coordenadas del GP (por si quieres evaluar campos allí)
                x_gp = np.dot(N, x_elem)
                y_gp = np.dot(N, y_elem)

                # índice cercano (para muestrear velocidad)
                i_gp = min(max(int((x_gp - domain.x_array[0]) / domain.dx), 0), nx - 1)
                j_gp = min(max(int((y_gp - domain.y_array[0]) / domain.dy), 0), ny - 1)

                v_gp = velocity_array[i_gp, j_gp]
                k_gp = omega / v_gp

                sx, sy = pml_tr(i_gp, j_gp)

                # rigidez con PML (término ∇u·∇v anisótropo)
                for a in range(4):
                    for b in range(4):
                        Ke[a, b] += w * detJ * (
                            (sy / sx) * dN_dx[a] * dN_dx[b] +
                            (sx / sy) * dN_dy[a] * dN_dy[b]
                        )

                # masa con PML (término -k^2 s_x s_y u v)
                for a in range(4):
                    for b in range(4):
                        Me[a, b] += w * detJ * (k_gp ** 2 * sx * sy * N[a] * N[b])

                # fuerza (fuente volumétrica)
                f_gp = source(x_gp, y_gp)
                for a in range(4):
                    Fe[a] += w * detJ * f_gp * N[a]

            # ensamblaje global
            for a in range(4):
                A = nodes[a]
                F[A] += Fe[a]
                for b in range(4):
                    B = nodes[b]
                    K[A, B] += Ke[a, b]
                    M[A, B] += Me[a, b]

    A = K - M  # sistema (K - M) U = F
    return A.tocsr(), F
