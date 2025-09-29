import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Colormap personalizado para visualizar la fuente
cmap_source = LinearSegmentedColormap.from_list('source', ['green', 'white', 'purple'])

class GaussianSource:
    """
    Fuente gaussiana localizada:
    f(x,y) = A exp(-((x-x0)^2+(y-y0)^2)/(2σ^2)) * exp(i*phase)
    """
    def __init__(self, amplitude=1.0, x_pos=0.0, y_pos=0.0, sigma=0.05, phase=0.0):
        self.amplitude = amplitude
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.sigma = sigma
        self.phase = phase

    def __call__(self, x, y):
        r2 = (x - self.x_pos)**2 + (y - self.y_pos)**2
        return self.amplitude * np.exp(-r2/(2*self.sigma**2)) * np.exp(1j*self.phase)
