# ==============================================================
# MAIN - Script principal para ejecutar la simulación
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from config import SimulationConfig
from domain import Domain
from sources import Source
from solver import MieSolver
from visualize import Visualizer

def run_simulation(regime_name, a_fraction):
    """Ejecuta una simulación para un régimen específico"""
    print(f"\n=== Ejecutando: {regime_name} ===")
    
    # Configuración
    config = SimulationConfig()
    config.a = a_fraction * config.lam  # ¡CAMBIO DIRECTAMENTE AQUÍ!
    config.Lwin = 2.5 * max(config.a, config.lam)  # Ajustar ventana
    
    # Dominio
    domain = Domain(config)
    domain.create_grid()
    
    # Fuente
    source = Source(config, domain)
    Ei = source.plane_wave()
    
    # Solver
    solver = MieSolver(config, domain, source)
    Ez_total = solver.solve()
    Ez_total, Es = solver.apply_pec_conditions(Ez_total, Ei)
    
    # Visualización
    visualizer = Visualizer(config, domain)
    fig, axs = visualizer.plot_results(Ei, Ez_total, Es)
    plt.suptitle(regime_name, fontsize=16, fontweight='bold')
    plt.savefig(f"mie_{regime_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return Ez_total, Es

def main():
    """Función principal que compara los 3 regímenes"""
    
    # Definir los tres casos a comparar
    casos = [
        ("Lambda mayor que radio", 0.25),   # λ > r (r = λ/4)
        ("Lambda igual al radio", 1.00),    # λ = r (r = λ)
        ("Lambda menor que radio", 2.00)    # λ < r (r = 2λ)
    ]
    
    resultados = {}
    
    for nombre, fraccion in casos:
        Ez_total, Es = run_simulation(nombre, fraccion)
        resultados[nombre] = {
            'Ez_total': Ez_total,
            'Es': Es,
            'a_fraction': fraccion
        }
    
    print("\n¡Comparación completada!")
    print("Resultados guardados en:")
    for nombre in resultados:
        print(f"- mie_{nombre.replace(' ', '_')}.png")

if __name__ == "__main__":
    main()