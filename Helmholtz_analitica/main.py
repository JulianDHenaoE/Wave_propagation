import numpy as np
from Helmholtz_analitica.config import (
    FREQUENCY, VELOCITY, SOURCE_AMPLITUDE, SOURCE_SIGMA,
    X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RESOLUTION
)
from Helmholtz_analitica.helmholtz_solver import (
    calculate_wave_number, analytical_helmholtz_solution,
    gaussian_source, create_grid
)
from Helmholtz_analitica.visualizar import plot_solution

def main():
    print("=== SOLUCIÓN ANALÍTICA DE HELMHOLTZ ===")

    # Configuración
    config = {
        'FREQUENCY': FREQUENCY,
        'VELOCITY': VELOCITY,
        'X_MIN': X_MIN,
        'X_MAX': X_MAX,
        'Y_MIN': Y_MIN,
        'Y_MAX': Y_MAX,
        'GRID_RESOLUTION': GRID_RESOLUTION
    }

    # Calcular número de onda
    config['k'] = calculate_wave_number(FREQUENCY, VELOCITY)
    print(f"Frecuencia: {FREQUENCY} Hz")
    print(f"Velocidad: {VELOCITY} m/s")
    print(f"Número de onda k: {config['k']:.3f} rad/m")

    # Crear malla
    X, Y = create_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RESOLUTION)
    print(f"Malla creada: {GRID_RESOLUTION}x{GRID_RESOLUTION} puntos")

    # Calcular solución
    psi = analytical_helmholtz_solution(X, Y, config['k'])
    source = gaussian_source(X, Y, SOURCE_AMPLITUDE, sigma=SOURCE_SIGMA)

    # Graficar resultados
    print("Generando gráficos...")
    plot_solution(X, Y, psi, source, config)

    print("=== COMPLETADO ===")

if __name__ == "__main__":
    main()
