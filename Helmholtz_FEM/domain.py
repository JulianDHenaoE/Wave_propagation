import numpy as np
import matplotlib.pyplot as plt

##########################################################################
# Clase que guarda la geometría y malla que usarán FEM, PML y visualización
##########################################################################
class HelmholtzDomain: 
    """
    Clase para la generación y gestión de dominios computacionales para 
    la ecuación de Helmholtz con capas PML (Perfectly Matched Layers).
    
    Esta clase maneja:
    - Dominio físico principal donde ocurre la física de interés
    - Dominio extendido que incluye las capas PML para absorción de ondas
    - Generación de mallas coordenadas para discretización espacial
    
    Atributos principales:
    -----------
    main_shape : tuple
        Forma del dominio principal (nx, ny) en número de nodos
    main_extension : tuple  
        Extensión física del dominio principal (xmin, xmax, ymin, ymax)
    nbl : int
        Número de nodos en cada capa PML
    lpml : float
        Ancho físico de cada capa PML en unidades del dominio
    shape : tuple
        Forma del dominio extendido (nx_total, ny_total)
    extension : tuple
        Extensión física del dominio extendido
    x_array, y_array : ndarray
        Arrays de coordenadas de la malla extendida
    dx, dy : float
        Espaciamiento entre nodos en x e y
    nk : int
        Número total de nodos en el dominio extendido
    """
    
    def __init__(self, main_shape, main_extension): 
        """
        Inicializa el dominio principal de cálculo.
        
        Parámetros:
        -----------
        main_shape : tuple (nx, ny)
            Número de nodos en x e y del dominio principal
        main_extension : tuple (xmin, xmax, ymin, ymax)
            Extensión física del dominio principal en metros
        """
        # main_shape = (nx, ny), main_extension = (xmin, xmax, ymin, ymax)
        self.main_shape = main_shape
        self.main_extension = main_extension  # extensión física del dominio principal
        self.nx = self.main_shape[0]          # número de nodos en x del dominio principal
        self.ny = self.main_shape[1]          # número de nodos en y del dominio principal
        
        print(f"✅ Dominio principal creado: {self.main_shape} nodos")
        print(f"   Extensión física: [{main_extension[0]}, {main_extension[1]}] x [{main_extension[2]}, {main_extension[3]}]")

    def pml_domain(self, nbl): 
        """
        Construye el dominio extendido incluyendo las capas PML.
        
        Parámetros:
        -----------
        nbl : int
            Número de nodos en cada capa PML (igual en los 4 bordes)
            
        Notas:
        ------
        La PML se añade simétricamente en los cuatro bordes del dominio principal.
        El dominio extendido resultante tiene dimensiones:
            (nx_principal + 2*nbl) × (ny_principal + 2*nbl)
        """
        self.nbl = nbl  # número de nodos en la PML
        
        # Total de nodos en x e y (dominio principal + PML a ambos lados)
        self.nx = self.main_shape[0] + self.nbl * 2  # número total de nodos en x
        self.ny = self.main_shape[1] + self.nbl * 2  # número total de nodos en y
        
        # Ancho físico de la PML (en coordenadas reales)
        # Cálculo basado en el espaciamiento del dominio principal
        self.lpml = (self.main_extension[1] - self.main_extension[0]) / (self.main_shape[0] - 1) * self.nbl
        
        # 🔹 FIJAMOS MANUALMENTE el ancho de PML para dominio [-1,1]×[-1,1]
        # Esto asegura que el dominio extendido sea exactamente [-1,1]×[-1,1]
        self.lpml = 0.5  # metros (para dominio principal [-0.5,0.5] extendido a [-1,1])
        
        self.shape = (self.nx, self.ny)  # número total de nodos (nx, ny)
        
        # Extensión física total (incluye PML en los bordes)
        # Se extiende simétricamente en todas las direcciones
        self.extension = tuple(
            x - self.lpml if i % 2 == 0 else x + self.lpml 
            for i, x in enumerate(self.main_extension)  # xmin, xmax, ymin, ymax
        )
        
        # Arrays de coordenadas extendidas (malla completa)
        self.x_array = np.linspace(self.extension[0], self.extension[1], self.nx)  # desde xmin_ext a xmax_ext
        self.y_array = np.linspace(self.extension[2], self.extension[3], self.ny)  # desde ymin_ext a ymax_ext
        
        # Pasos en x e y (espaciamiento uniforme)
        self.dx = self.x_array[1] - self.x_array[0]  # paso en x [m]
        self.dy = self.y_array[1] - self.y_array[0]  # paso en y [m]
        
        # Número total de nodos en el dominio extendido
        self.nk = self.nx * self.ny  # total de nodos
        
        print(f"✅ Dominio extendido con PML creado:")
        print(f"   - Nodos PML por borde: {nbl}")
        print(f"   - Ancho PML físico: {self.lpml:.3f} m")
        print(f"   - Forma total: {self.shape} nodos")
        print(f"   - Extensión total: [{self.extension[0]:.1f}, {self.extension[1]:.1f}] × [{self.extension[2]:.1f}, {self.extension[3]:.1f}]")
        print(f"   - Espaciamiento: dx={self.dx:.4f} m, dy={self.dy:.4f} m")
        print(f"   - Nodos totales: {self.nk}")

    def plot_domain(self):
        """
        Grafica el dominio físico y las capas PML para visualización.
        
        La visualización muestra:
        - Dominio físico principal (rectángulo rojo)
        - Dominio extendido con PML (rectángulo azul punteado)  
        - Regiones PML sombreadas en azul transparente
        """
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # ======================================================================
        # 1. DIBUJAR DOMINIO FÍSICO PRINCIPAL
        # ======================================================================
        xmin, xmax, ymin, ymax = self.main_extension
        
        rect_fisico = plt.Rectangle(
            (xmin, ymin), 
            xmax - xmin, 
            ymax - ymin,
            linewidth=3, 
            edgecolor='red', 
            facecolor='none', 
            label="Dominio físico principal"
        )
        ax.add_patch(rect_fisico)
        
        # ======================================================================
        # 2. DIBUJAR DOMINIO EXTENDIDO (CON PML)
        # ======================================================================
        xmin_ext, xmax_ext, ymin_ext, ymax_ext = self.extension
        
        rect_total = plt.Rectangle(
            (xmin_ext, ymin_ext), 
            xmax_ext - xmin_ext, 
            ymax_ext - ymin_ext,
            linewidth=2, 
            edgecolor='blue', 
            facecolor='none', 
            linestyle='--', 
            label="Dominio extendido (con PML)"
        )
        ax.add_patch(rect_total)
        
        # ======================================================================
        # 3. SOMBREADO DE LAS REGIONES PML
        # ======================================================================
        
        # 🔹 PML IZQUIERDA (entre xmin_ext y xmin)
        ax.fill_betweenx(
            [ymin_ext, ymax_ext],  # eje y completo
            xmin_ext,              # desde borde izquierdo extendido
            xmin,                  # hasta borde izquierdo físico
            color="blue", 
            alpha=0.15, 
            label="Capas PML"
        )
        
        # 🔹 PML DERECHA (entre xmax y xmax_ext)
        ax.fill_betweenx(
            [ymin_ext, ymax_ext],  # eje y completo  
            xmax,                  # desde borde derecho físico
            xmax_ext,              # hasta borde derecho extendido
            color="blue", 
            alpha=0.15
        )
        
        # 🔹 PML INFERIOR (entre ymin_ext y ymin)
        ax.fill_between(
            [xmin_ext, xmax_ext],  # eje x completo
            ymin_ext,              # desde borde inferior extendido
            ymin,                  # hasta borde inferior físico
            color="blue", 
            alpha=0.15
        )
        
        # 🔹 PML SUPERIOR (entre ymax y ymax_ext)
        ax.fill_between(
            [xmin_ext, xmax_ext],  # eje x completo
            ymax,                  # desde borde superior físico  
            ymax_ext,              # hasta borde superior extendido
            color="blue", 
            alpha=0.15
        )
        
        # ======================================================================
        # 4. ANOTACIONES Y ETIQUETAS
        # ======================================================================
        
        # Añadir anotaciones para claridad
        ax.text(xmin + 0.1*(xmax-xmin), ymin + 0.9*(ymax-ymin), 
               'DOMINIO\nFÍSICO', 
               ha='center', va='center', 
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.text(xmin_ext + 0.1*(xmax_ext-xmin_ext), ymin_ext + 0.1*(ymax_ext-ymin_ext), 
               'PML', 
               ha='center', va='center', 
               fontsize=10, color='blue',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
        
        # ======================================================================
        # 5. CONFIGURACIÓN FINAL DEL GRÁFICO
        # ======================================================================
        
        ax.set_title("Configuración del Dominio Computacional\nDominio Físico + Capas PML Absorbentes", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Coordenada X [m]", fontsize=12)
        ax.set_ylabel("Coordenada Y [m]", fontsize=12)
        
        # Establecer límites y aspecto
        ax.set_xlim(xmin_ext - 0.1*(xmax_ext-xmin_ext), xmax_ext + 0.1*(xmax_ext-xmin_ext))
        ax.set_ylim(ymin_ext - 0.1*(ymax_ext-ymin_ext), ymax_ext + 0.1*(ymax_ext-ymin_ext))
        ax.set_aspect('equal')
        
        # Cuadrícula y leyenda
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
        # ======================================================================
        # 6. INFORMACIÓN ADICIONAL EN CONSOLA
        # ======================================================================
        print(f"\n📊 RESUMEN DEL DOMINIO CONSTRUIDO:")
        print(f"   • Dominio físico: {self.main_shape[0]}×{self.main_shape[1]} nodos")
        print(f"   • PML: {self.nbl} nodos por borde → {self.lpml:.3f} m")
        print(f"   • Dominio total: {self.shape[0]}×{self.shape[1]} nodos")
        print(f"   • Espaciamiento: Δx={self.dx:.4f} m, Δy={self.dy:.4f} m")
        print(f"   • Razón aspecto: {self.dx/self.dy:.3f}")

    def get_meshgrid(self):
        """
        Retorna las matrices de coordenadas X, Y para toda la malla extendida.
        
        Retorna:
        --------
        X, Y : ndarray
            Matrices 2D con las coordenadas x e y de todos los nodos
        """
        return np.meshgrid(self.x_array, self.y_array, indexing='xy')


# ==============================================================================
# BLOQUE DE PRUEBA Y EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DEMOSTRACIÓN: CLASE HelmholtzDomain")
    print("=" * 70)
    
    # Parámetros del dominio principal
    main_shape = (101, 101)                    # 101×101 nodos en dominio físico
    main_extension = (-0.5, 0.5, -0.5, 0.5)   # 1.0×1.0 metros centrado en (0,0)
    
    # Crear instancia del dominio
    domain = HelmholtzDomain(main_shape, main_extension)
    
    # Añadir capas PML (30 nodos por borde)
    domain.pml_domain(30)  
    
    # Visualizar el dominio
    domain.plot_domain()
    
    # Ejemplo de uso adicional: obtener malla completa
    X, Y = domain.get_meshgrid()
    print(f"\n🔹 Malla extendida generada:")
    print(f"   - Forma de X: {X.shape}")
    print(f"   - Rango X: [{X.min():.1f}, {X.max():.1f}]")
    print(f"   - Rango Y: [{Y.min():.1f}, {Y.max():.1f}]")
    
    print("=" * 70)