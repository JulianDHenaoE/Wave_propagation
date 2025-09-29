import numpy as np
from scipy.sparse import lil_matrix
from elements import Q1
from pml import make_pml_transform


def assemble_fem_system(domain, frequency, velocity_array, source, alpha):
    """
    Ensambla el sistema FEM (K - M)U = F para la ecuación de Helmholtz con PML.
    """
    nx, ny = domain.nx, domain.ny
    nk = nx * ny

    K = lil_matrix((nk, nk), dtype=complex)
    M = lil_matrix((nk, nk), dtype=complex)
    F = np.zeros(nk, dtype=complex)

    omega = 2 * np.pi * frequency
    pml_transform = make_pml_transform(domain, alpha, frequency)

    # Cuadratura de Gauss 2x2
    gp = 1.0/np.sqrt(3.0)
    gauss_points = np.array([[-gp,-gp],[gp,-gp],[gp,gp],[-gp,gp]])
    weights = np.ones(4)

    for j in range(ny-1):
        for i in range(nx-1):
            # Nodos del elemento
            nodes = np.array([
                j*nx + i,
                j*nx + (i+1),
                (j+1)*nx + (i+1),
                (j+1)*nx + i
            ], dtype=int)

            x_elem = np.array([domain.x_array[i], domain.x_array[i+1],
                               domain.x_array[i+1], domain.x_array[i]])
            y_elem = np.array([domain.y_array[j], domain.y_array[j],
                               domain.y_array[j+1], domain.y_array[j+1]])

            Ke = np.zeros((4,4), dtype=complex)
            Me = np.zeros((4,4), dtype=complex)
            Fe = np.zeros(4, dtype=complex)

            # Integración numérica
            for (xi, eta), w in zip(gauss_points, weights):
                N = Q1.shape_functions(xi, eta)
                dN_dxi, dN_deta = Q1.shape_derivatives(xi, eta)

                dx_dxi = np.dot(dN_dxi, x_elem)
                dx_deta = np.dot(dN_deta, x_elem)
                dy_dxi = np.dot(dN_dxi, y_elem)
                dy_deta = np.dot(dN_deta, y_elem)

                J = np.array([[dx_dxi, dx_deta],[dy_dxi, dy_deta]], dtype=float)
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)

                dN_dx = Jinv[0,0]*dN_dxi + Jinv[0,1]*dN_deta
                dN_dy = Jinv[1,0]*dN_dxi + Jinv[1,1]*dN_deta

                x_gp = np.dot(N, x_elem)
                y_gp = np.dot(N, y_elem)

                ix = min(max(int((x_gp - domain.x_array[0])/domain.dx), 0), nx-1)
                iy = min(max(int((y_gp - domain.y_array[0])/domain.dy), 0), ny-1)
                v_gp = velocity_array[ix, iy]
                k_gp = omega / v_gp

                s_tx, s_ty = pml_transform(ix, iy)

                # Rigidez
                for a in range(4):
                    for b in range(4):
                        Ke[a,b] += w*detJ*((s_ty/s_tx)*dN_dx[a]*dN_dx[b] +
                                           (s_tx/s_ty)*dN_dy[a]*dN_dy[b])
                # Masa
                for a in range(4):
                    for b in range(4):
                        Me[a,b] += w*detJ*(k_gp**2 * s_tx * s_ty * N[a]*N[b])

                # Fuente
                f_gp = source(x_gp, y_gp)
                for a in range(4):
                    Fe[a] += w*detJ*f_gp*N[a]

            # Ensamblaje global
            for a in range(4):
                Arow = nodes[a]
                for b in range(4):
                    Acol = nodes[b]
                    K[Arow, Acol] += Ke[a,b]
                    M[Arow, Acol] += Me[a,b]
                F[Arow] += Fe[a]

    A = K - M
    return A.tocsr(), F
