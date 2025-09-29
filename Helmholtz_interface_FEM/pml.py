import numpy as np

def make_pml_transform(domain, alpha, frequency):
    """
    Construye funciones de transformación PML para el dominio.
    Devuelve pml_transform(ix, iy) → (s_tilde_x, s_tilde_y).
    """
    omega = 2 * np.pi * frequency
    nbl = domain.nbl
    lpml = domain.lpml
    x_array = domain.x_array
    y_array = domain.y_array

    def sigma_x(ix):
        if ix < nbl:
            return 2*np.pi*alpha*frequency * ((abs(x_array[ix]) - abs(x_array[nbl]))/lpml)**2
        elif ix > domain.nx - nbl - 1:
            return 2*np.pi*alpha*frequency * ((abs(x_array[ix]) - abs(x_array[domain.nx-nbl-1]))/lpml)**2
        else:
            return 0.0

    def sigma_y(iy):
        if iy < nbl:
            return 2*np.pi*alpha*frequency * ((abs(y_array[iy]) - abs(y_array[nbl]))/lpml)**2
        elif iy > domain.ny - nbl - 1:
            return 2*np.pi*alpha*frequency * ((abs(y_array[iy]) - abs(y_array[domain.ny-nbl-1]))/lpml)**2
        else:
            return 0.0

    def pml_transform(ix, iy):
        sx = sigma_x(ix)
        sy = sigma_y(iy)
        s_tilde_x = 1.0 - 1j * sx / omega
        s_tilde_y = 1.0 - 1j * sy / omega
        return s_tilde_x, s_tilde_y

    return pml_transform
