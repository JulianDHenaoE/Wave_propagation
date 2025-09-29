import numpy as np

class HelmholtzDomain:
    """Structured rectangular grid with PML padding."""
    def __init__(self, main_shape, main_extension):
        self.main_shape = main_shape  # (nx_main, ny_main)
        self.main_extension = main_extension  # (xmin, xmax, ymin, ymax)
        self.nx = self.main_shape[0]
        self.ny = self.main_shape[1]

    def pml_domain(self, nbl):
        self.nbl = nbl
        self.nx = self.main_shape[0] + 2*nbl
        self.ny = self.main_shape[1] + 2*nbl
        self.lpml = (self.main_extension[1] - self.main_extension[0])/(self.main_shape[0]-1) * nbl
        self.shape = (self.nx, self.ny)
        xmin, xmax, ymin, ymax = self.main_extension
        self.extension = (xmin - self.lpml, xmax + self.lpml,
                          ymin - self.lpml, ymax + self.lpml)
        self.x_array = np.linspace(self.extension[0], self.extension[1], self.nx)
        self.y_array = np.linspace(self.extension[2], self.extension[3], self.ny)
        self.dx = self.x_array[1] - self.x_array[0]
        self.dy = self.y_array[1] - self.y_array[0]
        self.nk = self.nx * self.ny
