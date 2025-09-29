import numpy as np

def pml_functions(domain, alpha, frequency): 
    """PML transformation functions"""
    omega = 2*np.pi*frequency
    nbl = domain.nbl # Numero de capas PML
    lpml = domain.lpml # Ancho de la PML
    x_array = domain.x_array # Coordenadas x
    y_array = domain.y_array # Coordenadas y

    def sigma_x(x_idx):    # Función sigma en x
        if x_idx < nbl: # Capa PML izquierda
            return 2*np.pi*alpha*frequency*((abs(x_array[x_idx])-abs(x_array[nbl]))/lpml)**2 # Sigma cuadrática
        elif x_idx > domain.nx - nbl - 1: # Capa PML derecha
            return 2*np.pi*alpha*frequency*((abs(x_array[x_idx])-abs(x_array[domain.nx-nbl-1]))/lpml)**2 # Sigma cuadrática
        else:
            return 0.0 # Zona interior (sin PML)

    def sigma_y(y_idx): # Función sigma en y
        if y_idx < nbl: # Capa PML inferior
            return 2*np.pi*alpha*frequency*((abs(y_array[y_idx])-abs(y_array[nbl]))/lpml)**2
        elif y_idx > domain.ny - nbl - 1: # Capa PML superior
            return 2*np.pi*alpha*frequency*((abs(y_array[y_idx])-abs(y_array[domain.ny-nbl-1]))/lpml)**2
        else: # Zona interior (sin PML)
            return 0.0 #

    def pml_transform(x_idx, y_idx): # Transformación PML en (i,j)
        sx = sigma_x(x_idx) # Sigma en x
        sy = sigma_y(y_idx) # Sigma en y
        s_tilde_x = 1 - 1j*sx/omega # s̃ₓ = 1 - iσₓ/ω
        s_tilde_y = 1 - 1j*sy/omega # s̃ᵧ = 1 - iσᵧ/ω
        return s_tilde_x, s_tilde_y # Retorna (s̃ₓ, s̃ᵧ)

    return pml_transform # Retorna la función de transformación PML