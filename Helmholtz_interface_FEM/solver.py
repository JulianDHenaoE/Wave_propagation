import numpy as np
from scipy.sparse.linalg import spsolve
from assembly import assemble_fem_system

def solve_helmholtz_fem(domain, frequency, velocity_array, source, alpha):
    A, F = assemble_fem_system(domain, frequency, velocity_array, source, alpha)
    U = spsolve(A, F)
    u_2d = U.reshape((domain.ny, domain.nx)).T
    return u_2d, A.shape[0]
