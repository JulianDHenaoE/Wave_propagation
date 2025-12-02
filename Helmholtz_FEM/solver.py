import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from elements import BilinearElement
from pml import pml_functions

def assemble_fem_system(domain, frequency, velocity_array, source, alpha):
    nx, ny = domain.nx, domain.ny
    nk = nx * ny
    
    K = lil_matrix((nk, nk), dtype=complex)
    M = lil_matrix((nk, nk), dtype=complex)
    F = np.zeros(nk, dtype=complex)
    
    omega = 2*np.pi*frequency
    pml_transform = pml_functions(domain, alpha, frequency)
    
    gauss_points = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                             [ 1/np.sqrt(3), -1/np.sqrt(3)],
                             [ 1/np.sqrt(3),  1/np.sqrt(3)],
                             [-1/np.sqrt(3),  1/np.sqrt(3)]])
    gauss_weights = np.array([1, 1, 1, 1])
    
    for j in range(ny-1):
        for i in range(nx-1):
            nodes = np.array([
                j*nx + i,
                j*nx + (i+1),
                (j+1)*nx + (i+1),
                (j+1)*nx + i
            ])
            x_elem = np.array([domain.x_array[i], domain.x_array[i+1],
                               domain.x_array[i+1], domain.x_array[i]])
            y_elem = np.array([domain.y_array[j], domain.y_array[j],
                               domain.y_array[j+1], domain.y_array[j+1]])
            K_elem = np.zeros((4, 4), dtype=complex)
            M_elem = np.zeros((4, 4), dtype=complex)
            F_elem = np.zeros(4, dtype=complex)
            
            for gp in range(4):
                xi, eta = gauss_points[gp]
                w = gauss_weights[gp]
                N = BilinearElement.shape_functions(xi, eta)
                dN_dxi, dN_deta = BilinearElement.shape_derivatives(xi, eta)
                dx_dxi = np.dot(dN_dxi, x_elem)
                dx_deta = np.dot(dN_deta, x_elem)
                dy_dxi = np.dot(dN_dxi, y_elem)
                dy_deta = np.dot(dN_deta, y_elem)
                J = np.array([[dx_dxi, dx_deta],[dy_dxi, dy_deta]])
                det_J = np.linalg.det(J)
                J_inv = np.linalg.inv(J)
                x_gp = np.dot(N, x_elem)
                y_gp = np.dot(N, y_elem)
                dN_dx = J_inv[0,0]*dN_dxi + J_inv[0,1]*dN_deta
                dN_dy = J_inv[1,0]*dN_dxi + J_inv[1,1]*dN_deta
                i_closest = min(max(int((x_gp - domain.x_array[0])/domain.dx), 0), nx-1)
                j_closest = min(max(int((y_gp - domain.y_array[0])/domain.dy), 0), ny-1)
                v_gp = velocity_array[i_closest, j_closest]
                k_gp = omega / v_gp
                s_tilde_x, s_tilde_y = pml_transform(i_closest, j_closest)
                
                for a in range(4):
                    for b in range(4):
                        K_elem[a,b] += w * det_J * (
                            (s_tilde_y/s_tilde_x) * dN_dx[a] * dN_dx[b] +
                            (s_tilde_x/s_tilde_y) * dN_dy[a] * dN_dy[b]
                        )
                        M_elem[a,b] += w * det_J * (k_gp**2 * s_tilde_x * s_tilde_y * N[a] * N[b])
                f_gp = source(x_gp, y_gp)
                for a in range(4):
                    F_elem[a] += w * det_J * f_gp * N[a]
            
            for a in range(4):
                for b in range(4):
                    K[nodes[a], nodes[b]] += K_elem[a,b]
                    M[nodes[a], nodes[b]] += M_elem[a,b]
                F[nodes[a]] += F_elem[a]
    A = K - M
    return A.tocsr(), F


def solve_helmholtz_fem(domain, frequency, velocity_array, source, alpha):
    print(f"Assembling FEM system for frequency {frequency} Hz...")
    A, F = assemble_fem_system(domain, frequency, velocity_array, source, alpha)
    print("Solving linear system...")
    U = spsolve(A, F)
    u_2d = U.reshape((domain.ny, domain.nx)).T
    X, Y = np.meshgrid(domain.x_array, domain.y_array, indexing='ij')
    source_2d = source(X, Y)
    print("FEM solution complete.")
    return u_2d, source_2d
