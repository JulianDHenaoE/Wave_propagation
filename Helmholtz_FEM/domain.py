import numpy as np
import matplotlib.pyplot as plt

##########################################################################
# Clase que guarda la geometría y malla que usarán FEM, PML y visualización
##########################################################################
class HelmholtzDomain: 
    """Clase de dominio para la generación de malla FEM"""
    def __init__(self, main_shape, main_extension): 
        # main_shape = (nx, ny), main_extension = (xmin, xmax, ymin, ymax)
        self.main_shape = main_shape
        self.main_extension = main_extension  # extensión física del dominio principal
        self.nx = self.main_shape[0]          # número de nodos en x del dominio principal
        self.ny = self.main_shape[1]          # número de nodos en y del dominio principal

    # Construcción del dominio extendido (físico + PML)
    def pml_domain(self, nbl): 
        self.nbl = nbl  # número de nodos en la PML
        # total de nodos en x e y (dominio principal + PML a ambos lados)
        self.nx = self.main_shape[0] + self.nbl*2 # número total de nodos en x
        self.ny = self.main_shape[1] + self.nbl*2 # número total de nodos en y
        # ancho físico de la PML (en coordenadas reales)
        self.lpml = (self.main_extension[1] - self.main_extension[0])/(self.main_shape[0]-1) * self.nbl
        # Fijamos la PML para que el dominio extendido sea [-1,1] x [-1,1]
        self.lpml = 0.5
        
        self.shape = (self.nx, self.ny) # número total de nodos (nx, ny)
        # extensión física total (incluye PML en los bordes)
        self.extension = tuple(
            x - self.lpml if i % 2 == 0 else x + self.lpml 
            for i, x in enumerate(self.main_extension) # xmin, xmax, ymin, ymax
        )
        # arrays de coordenadas extendidas
        self.x_array = np.linspace(self.extension[0], self.extension[1], self.nx) # desde xmin a xmax
        self.y_array = np.linspace(self.extension[2], self.extension[3], self.ny) # desde ymin a ymax
        # pasos en x e y
        self.dx = self.x_array[1] - self.x_array[0] # paso en x
        self.dy = self.y_array[1] - self.y_array[0] # paso en y
        # número total de nodos
        self.nk = self.nx * self.ny # total de nodos

    # Método para graficar el dominio físico y la PML
    def plot_domain(self):
        fig, ax = plt.subplots(figsize=(6,6))

        # Recuadro del dominio físico
        xmin, xmax, ymin, ymax = self.main_extension
        rect_fisico = plt.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            linewidth=2, edgecolor='red', facecolor='none', label="Dominio físico"
        )
        ax.add_patch(rect_fisico)

        # Recuadro del dominio extendido (físico + PML)
        xmin_ext, xmax_ext, ymin_ext, ymax_ext = self.extension
        rect_total = plt.Rectangle(
            (xmin_ext, ymin_ext), xmax_ext-xmin_ext, ymax_ext-ymin_ext,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label="Dominio extendido (con PML)"
        )
        ax.add_patch(rect_total)

        # 🔹 Sombrear la PML en los 4 bordes
        ax.fill_betweenx([ymin_ext, ymax_ext], xmin_ext, xmin, color="blue", alpha=0.2) # izquierda
        ax.fill_betweenx([ymin_ext, ymax_ext], xmax, xmax_ext, color="blue", alpha=0.2) # derecha
        ax.fill_between([xmin_ext, xmax_ext], ymin_ext, ymin, color="blue", alpha=0.2) # abajo
        ax.fill_between([xmin_ext, xmax_ext], ymax, ymax_ext, color="blue", alpha=0.2) # arriba

        # Ajustes de gráfico
        ax.set_title("Dominio físico (rojo) y PML (zona azul sombreada)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        plt.show()



# Prueba rápida si corres este archivo directamente
if __name__ == "__main__":
    main_shape = (101, 101)
    main_extension = (-0.5, 0.5, -0.5, 0.5)
    domain = HelmholtzDomain(main_shape, main_extension)
    domain.pml_domain(30)  # añade 25 nodos de PML
    domain.plot_domain()