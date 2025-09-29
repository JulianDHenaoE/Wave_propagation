import numpy as np

class BilinearElement:
    """
    Elemento cuadrilátero bilineal para el Método de Elementos Finitos (FEM).
    
    Esta clase implementa las funciones de forma y sus derivadas para elementos
    cuadriláteros de 4 nodos (Q4) en coordenadas naturales (ξ, η) ∈ [-1, 1]×[-1, 1].
    
    Características:
    - 4 nodos por elemento (vértices del cuadrilátero)
    - Interpolación bilineal (lineal en cada dirección)
    - Orden de convergencia: O(h²) para desplazamientos
    - Adecuado para problemas 2D de elasticidad, calor, etc.
    
    Sistema de coordenadas naturales:
        N4(-1,1) ────── N3(1,1)
          │               │
          │               │  
          │               │
        N1(-1,-1) ───── N2(1,-1)
    
    Referencia:
    -----------
    Coordinate system: (ξ, η) ∈ [-1, 1] × [-1, 1]
    """

    @staticmethod
    def shape_functions(xi, eta):
        """
        Evalúa las funciones de forma bilineales en las coordenadas naturales (ξ, η).
        
        Parámetros:
        -----------
        xi : float
            Coordenada natural ξ ∈ [-1, 1]
        eta : float
            Coordenada natural η ∈ [-1, 1]
            
        Retorna:
        --------
        N : ndarray (4,)
            Vector de funciones de forma [N₁, N₂, N₃, N₄] en el punto (ξ, η)
            
        Fórmulas:
        ---------
        N₁(ξ,η) = ¼(1 - ξ)(1 - η)   # Nodo 1: (-1,-1)
        N₂(ξ,η) = ¼(1 + ξ)(1 - η)   # Nodo 2: (1,-1)  
        N₃(ξ,η) = ¼(1 + ξ)(1 + η)   # Nodo 3: (1,1)
        N₄(ξ,η) = ¼(1 - ξ)(1 + η)   # Nodo 4: (-1,1)
        
        Propiedades:
        ------------
        1. ∑Nᵢ = 1 (partición de la unidad)
        2. Nᵢ(ξⱼ,ηⱼ) = δᵢⱼ (delta de Kronecker)
        3. Interpolación bilineal completa
        """
        # Validación de rangos (opcional, para debugging)
        if not (-1 <= xi <= 1) or not (-1 <= eta <= 1):
            print(f"⚠️ Advertencia: Punto ({xi}, {eta}) fuera del dominio de referencia [-1,1]×[-1,1]")
        
        # Cálculo de funciones de forma bilineales
        N1 = 0.25 * (1 - xi) * (1 - eta)  # Nodo 1: esquina inferior-izquierda
        N2 = 0.25 * (1 + xi) * (1 - eta)  # Nodo 2: esquina inferior-derecha
        N3 = 0.25 * (1 + xi) * (1 + eta)  # Nodo 3: esquina superior-derecha
        N4 = 0.25 * (1 - xi) * (1 + eta)  # Nodo 4: esquina superior-izquierda
        
        return np.array([N1, N2, N3, N4])

    @staticmethod
    def shape_derivatives(xi, eta):
        """
        Evalúa las derivadas de las funciones de forma respecto a ξ y η.
        
        Parámetros:
        -----------
        xi : float
            Coordenada natural ξ ∈ [-1, 1]
        eta : float
            Coordenada natural η ∈ [-1, 1]
            
        Retorna:
        --------
        dN_dxi : ndarray (4,)
            Derivadas ∂Nᵢ/∂ξ para i=1,2,3,4
        dN_deta : ndarray (4,)
            Derivadas ∂Nᵢ/∂η para i=1,2,3,4
            
        Fórmulas de derivadas:
        ----------------------
        ∂N₁/∂ξ = -¼(1 - η)    ∂N₁/∂η = -¼(1 - ξ)
        ∂N₂/∂ξ = +¼(1 - η)    ∂N₂/∂η = -¼(1 + ξ)  
        ∂N₃/∂ξ = +¼(1 + η)    ∂N₃/∂η = +¼(1 + ξ)
        ∂N₄/∂ξ = -¼(1 + η)    ∂N₄/∂η = +¼(1 - ξ)
        
        Uso en FEM:
        -----------
        Estas derivadas se usan para calcular la matriz jacobiana:
            J = [∂x/∂ξ ∂x/∂η] = [∑(∂Nᵢ/∂ξ)xᵢ ∑(∂Nᵢ/∂ξ)yᵢ]
                [∂y/∂ξ ∂y/∂η]   [∑(∂Nᵢ/∂η)xᵢ ∑(∂Nᵢ/∂η)yᵢ]
        """
        # Derivadas respecto a ξ
        dN_dxi = np.array([
            -0.25 * (1 - eta),   # ∂N₁/∂ξ
             0.25 * (1 - eta),   # ∂N₂/∂ξ
             0.25 * (1 + eta),   # ∂N₃/∂ξ
            -0.25 * (1 + eta)    # ∂N₄/∂ξ
        ])
        
        # Derivadas respecto a η  
        dN_deta = np.array([
            -0.25 * (1 - xi),    # ∂N₁/∂η
            -0.25 * (1 + xi),    # ∂N₂/∂η
             0.25 * (1 + xi),    # ∂N₃/∂η
             0.25 * (1 - xi)     # ∂N₄/∂η
        ])
        
        return dN_dxi, dN_deta

    @staticmethod
    def jacobian(xi, eta, nodal_coords):
        """
        Calcula la matriz jacobiana y su determinante para mapeo isoparamétrico.
        
        Parámetros:
        -----------
        xi, eta : float
            Coordenadas naturales
        nodal_coords : ndarray (4, 2)
            Coordenadas físicas de los 4 nodos: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Retorna:
        --------
        J : ndarray (2, 2)
            Matriz jacobiana [[∂x/∂ξ, ∂x/∂η], [∂y/∂ξ, ∂y/∂η]]
        detJ : float
            Determinante del jacobiano (área del elemento en coordenadas naturales)
        invJ : ndarray (2, 2)
            Inversa del jacobiano (para calcular derivadas en coordenadas físicas)
        """
        # Obtener derivadas en coordenadas naturales
        dN_dxi, dN_deta = BilinearElement.shape_derivatives(xi, eta)
        
        # Extraer coordenadas x e y de los nodos
        x_coords = nodal_coords[:, 0]  # [x1, x2, x3, x4]
        y_coords = nodal_coords[:, 1]  # [y1, y2, y3, y4]
        
        # Calcular componentes del jacobiano
        dx_dxi = np.dot(dN_dxi, x_coords)   # ∂x/∂ξ = Σ(∂Nᵢ/∂ξ)·xᵢ
        dx_deta = np.dot(dN_deta, x_coords) # ∂x/∂η = Σ(∂Nᵢ/∂η)·xᵢ
        dy_dxi = np.dot(dN_dxi, y_coords)   # ∂y/∂ξ = Σ(∂Nᵢ/∂ξ)·yᵢ
        dy_deta = np.dot(dN_deta, y_coords) # ∂y/∂η = Σ(∂Nᵢ/∂η)·yᵢ
        
        # Construir matriz jacobiana
        J = np.array([[dx_dxi, dx_deta],
                      [dy_dxi, dy_deta]])
        
        # Calcular determinante
        detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
        
        # Calcular inversa (si el determinante es positivo)
        if abs(detJ) > 1e-12:
            invJ = np.array([[dy_deta, -dx_deta],
                            [-dy_dxi, dx_dxi]]) / detJ
        else:
            raise ValueError(f"Jacobiano singular: det(J) = {detJ:.2e}")
            
        return J, detJ, invJ

    @staticmethod
    def test_shape_functions():
        """
        Prueba las propiedades fundamentales de las funciones de forma.
        
        Verifica:
        1. Partición de la unidad: ∑Nᵢ = 1 en todo punto
        2. Propiedad delta: Nᵢ(ξⱼ,ηⱼ) = δᵢⱼ
        3. Simetría y valores en puntos especiales
        """
        print("🧪 PROBANDO FUNCIONES DE FORMA BILINEALES")
        print("=" * 50)
        
        # Puntos de prueba (esquinas y centro)
        test_points = [
            (-1, -1, "Nodo 1"), (-1, 1, "Nodo 4"),
            (1, -1, "Nodo 2"), (1, 1, "Nodo 3"),
            (0, 0, "Centro"), (0.5, 0.5, "Punto interior")
        ]
        
        for xi, eta, desc in test_points:
            N = BilinearElement.shape_functions(xi, eta)
            sum_N = np.sum(N)
            
            print(f"\n📍 {desc} (ξ={xi}, η={eta}):")
            print(f"   N = [{N[0]:.3f}, {N[1]:.3f}, {N[2]:.3f}, {N[3]:.3f}]")
            print(f"   ∑Nᵢ = {sum_N:.6f} {'✓' if abs(sum_N - 1.0) < 1e-12 else '✗'}")
            
            # Verificar propiedad delta en nodos
            if desc.startswith("Nodo"):
                expected_delta = np.array([1.0 if f"Nodo {i+1}" == desc else 0.0 
                                         for i in range(4)])
                delta_error = np.max(np.abs(N - expected_delta))
                print(f"   Propiedad delta: error = {delta_error:.2e} {'✓' if delta_error < 1e-12 else '✗'}")


# ==============================================================================
# EJEMPLO DE USO Y PRUEBA
# ==============================================================================
if __name__ == "__main__":
    # Ejecutar pruebas de las funciones de forma
    BilinearElement.test_shape_functions()
    
    print("\n" + "=" * 60)
    print("EJEMPLO DE CÁLCULO DE JACOBIANO")
    print("=" * 60)
    
    # Definir un elemento cuadrilátero de ejemplo (cuadrado unitario)
    nodal_coords = np.array([
        [0.0, 0.0],  # Nodo 1: esquina inferior-izquierda
        [1.0, 0.0],  # Nodo 2: esquina inferior-derecha  
        [1.0, 1.0],  # Nodo 3: esquina superior-derecha
        [0.0, 1.0]   # Nodo 4: esquina superior-izquierda
    ])
    
    # Probar en el centro del elemento (ξ=0, η=0)
    xi, eta = 0.0, 0.0
    J, detJ, invJ = BilinearElement.jacobian(xi, eta, nodal_coords)
    
    print(f"📍 Punto (ξ={xi}, η={eta}):")
    print(f"   Matriz jacobiana J = ")
    print(f"   [{J[0,0]:.3f}  {J[0,1]:.3f}]")
    print(f"   [{J[1,0]:.3f}  {J[1,1]:.3f}]")
    print(f"   det(J) = {detJ:.3f} (área en coordenadas naturales)")
    print(f"   J⁻¹ = ")
    print(f"   [{invJ[0,0]:.3f}  {invJ[0,1]:.3f}]")
    print(f"   [{invJ[1,0]:.3f}  {invJ[1,1]:.3f}]")
    
    # Verificar que J⁻¹·J = I
    identity_check = invJ @ J
    print(f"   Verificación J⁻¹·J = I: error = {np.max(np.abs(identity_check - np.eye(2))):.2e}")