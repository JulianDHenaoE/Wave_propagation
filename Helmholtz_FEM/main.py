import numpy as np
from domain import HelmholtzDomain
from solver import solve_helmholtz_fem
from visualize import plot_solution
from config import *

def main():
    print("=== True Finite Element Method for Helmholtz Equation ===")
    domain = HelmholtzDomain(MAIN_DOMAIN_SHAPE, MAIN_DOMAIN_EXTENSION)
    domain.pml_domain(NBL)
    print(f"Domain: {domain.nx} x {domain.ny} nodes")
    print(f"Total elements: {(domain.nx-1) * (domain.ny-1)}")
    velocity_array = np.ones((domain.nx, domain.ny)) * VELOCITY
    u_fem, source_array = solve_helmholtz_fem(domain, FREQUENCY, velocity_array, SOURCE, ALPHA)
    plot_solution(domain, FREQUENCY, u_fem, source_array)

if __name__ == "__main__":
    main()
