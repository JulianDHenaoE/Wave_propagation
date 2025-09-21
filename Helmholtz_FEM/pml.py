import numpy as np

def pml_functions(domain, alpha, frequency):
    """PML transformation functions"""
    omega = 2*np.pi*frequency
    nbl = domain.nbl
    lpml = domain.lpml
    x_array = domain.x_array
    y_array = domain.y_array

    def sigma_x(x_idx):
        if x_idx < nbl:
            return 2*np.pi*alpha*frequency*((abs(x_array[x_idx])-abs(x_array[nbl]))/lpml)**2
        elif x_idx > domain.nx - nbl - 1:
            return 2*np.pi*alpha*frequency*((abs(x_array[x_idx])-abs(x_array[domain.nx-nbl-1]))/lpml)**2
        else:
            return 0.0

    def sigma_y(y_idx):
        if y_idx < nbl:
            return 2*np.pi*alpha*frequency*((abs(y_array[y_idx])-abs(y_array[nbl]))/lpml)**2
        elif y_idx > domain.ny - nbl - 1:
            return 2*np.pi*alpha*frequency*((abs(y_array[y_idx])-abs(y_array[domain.ny-nbl-1]))/lpml)**2
        else:
            return 0.0

    def pml_transform(x_idx, y_idx):
        sx = sigma_x(x_idx)
        sy = sigma_y(y_idx)
        s_tilde_x = 1 - 1j*sx/omega
        s_tilde_y = 1 - 1j*sy/omega
        return s_tilde_x, s_tilde_y

    return pml_transform
