import argparse
import numpy as np
from domain import HelmholtzDomain
from source import GaussianSource
from solver import solve_helmholtz_fem
from plotting import plot_solution

def main():
    p = argparse.ArgumentParser(description="Helmholtz 2D FEM with PML (Q1).")
    p.add_argument("--nx", type=int, default=251)
    p.add_argument("--ny", type=int, default=251)
    p.add_argument("--xmin", type=float, default=-1.0)
    p.add_argument("--xmax", type=float, default= 1.0)
    p.add_argument("--ymin", type=float, default=-1.0)
    p.add_argument("--ymax", type=float, default= 1.0)
    p.add_argument("--nbl", type=int, default=10, help="celdas de PML por lado")
    p.add_argument("--alpha", type=float, default=2.5, help="intensidad de PML")
    p.add_argument("--frequency", type=float, default=5.0)
    p.add_argument("--srcx", type=float, default=0.0)
    p.add_argument("--srcy", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.015)
    p.add_argument("--amp", type=float, default=0.02)
    args = p.parse_args()

    main_domain_shape = (args.nx - 2*args.nbl, args.ny - 2*args.nbl)
    main_domain_extension = (args.xmin, args.xmax, args.ymin, args.ymax)

    domain = HelmholtzDomain(main_domain_shape, main_domain_extension)
    domain.pml_domain(args.nbl)

    X, Y = np.meshgrid(domain.x_array, domain.y_array, indexing='ij')
    velocity_array = np.where(Y > 0.0, 1.5, 3.0).astype(float)

    src = GaussianSource(amplitude=args.amp, x_pos=args.srcx, y_pos=args.srcy,
                         sigma=args.sigma, phase=0.0)

    u_fem, ndofs = solve_helmholtz_fem(domain, args.frequency, velocity_array, src, args.alpha)

    source_array = src(X, Y)
    print(f"System size: {ndofs} unknowns on grid {domain.nx}x{domain.ny} "
          f"(elements: {(domain.nx-1)*(domain.ny-1)})")
    plot_solution(domain, args.frequency, u_fem, source_array)

if __name__ == "__main__":
    main()
