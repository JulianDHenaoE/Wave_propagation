import numpy as np

def calculate_wave_number(frequency, velocity):
    """
    Calcula el número de onda k = 2πf / v
    Parámetros:
        frequency : Frecuencia [Hz]
        velocity  : Velocidad de propagación [m/s]
    Retorna:
        k : número de onda [rad/m]
    """
    return 2 * np.pi * frequency / velocity


def analytical_helmholtz_solution(x, y, k, x0=0, y0=0):
    """
    Solución analítica de la ecuación de Helmholtz en 2D (fuente puntual).
    ψ(r) = e^(ikr) / (4πr)
    Parámetros:
        x, y : coordenadas donde evaluar
        k    : número de onda [rad/m]
        x0,y0: posición de la fuente
    Retorna:
        ψ(x,y): campo complejo en cada punto
    """
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_safe = np.where(r < 1e-10, 1e-10, r)  # evita división por cero
    return np.exp(1j * k * r_safe) / (4 * np.pi * r_safe)


def gaussian_source(x, y, amplitude=1, x0=0, y0=0, sigma=0.05):
    """
    Define una fuente gaussiana localizada en (x0,y0).
    Parámetros:
        x, y      : coordenadas
        amplitude : amplitud máxima
        sigma     : ancho de la gaussiana
    """
    r_squared = (x - x0)**2 + (y - y0)**2
    return amplitude * np.exp(-r_squared / (2 * sigma**2))


def create_grid(x_min, x_max, y_min, y_max, resolution):
    """
    Crea la malla cartesiana para la simulación.
    Retorna:
        X, Y: mallas 2D con coordenadas
    """
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    return np.meshgrid(x, y)