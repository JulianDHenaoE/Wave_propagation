# __init__.py dentro de Guia_rectangular
from .mesh import rect_mesh
from .analytic import kc_rect, field_TE_nodes, field_TM_nodes
from .assembly import assemble_K_M
from .eigen import solve_mode

__all__ = [
    "rect_mesh",
    "kc_rect", "field_TE_nodes", "field_TM_nodes",
    "assemble_K_M",
    "solve_mode",
]
