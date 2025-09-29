# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse.linalg import eigsh
from assembly import assemble_K_M
from mesh import rect_mesh
from analytic import kc_rect

def solve_mode(kind: str, m: int, n: int, Nx: int, Ny: int, W: float, H: float,
               k_eigs: int = 12):
    """
    Resuelve el problema de autovalores para modos electromagnéticos TE/TM 
    en una guía de onda rectangular usando el Método de Elementos Finitos (FEM).
    
    La función calcula el modo específico (m,n) seleccionando el autovalor FEM
    más cercano al valor analítico conocido, usando una estrategia de "shift"
    para mejorar la precisión y eficiencia.

    Parámetros
    ----------
    kind : str
        Tipo de modo: 'TE' (Transverso Eléctrico) o 'TM' (Transverso Magnético).
    m : int
        Índice del modo en dirección x (número de semi-variaciones).
    n : int
        Índice del modo en dirección y (número de semi-variaciones).
    Nx : int
        Número de divisiones de la malla en dirección x.
    Ny : int
        Número de divisiones de la malla en dirección y.
    W : float
        Ancho de la guía de onda en metros.
    H : float
        Alto de la guía de onda en metros.
    k_eigs : int, opcional
        Número de autovalores a calcular. Por defecto 12.

    Retorna
    -------
    kc_fem : float
        Constante de corte calculada por FEM [m⁻¹].
    kc_ana : float
        Constante de corte analítica de referencia [m⁻¹].
    err : float
        Error relativo porcentual entre FEM y analítico.
    vec : ndarray
        Autovector FEM que representa el campo modal en los nodos.
    Xy : ndarray
        Coordenadas de los nodos de la malla.
    Tri : ndarray
        Conectividad triangular de la malla.

    Notas
    -----
    El problema matemático resuelto es: (K - λM)φ = 0
    donde K es la matriz de rigidez, M la matriz de masa,
    λ = kc² es el autovalor (cuadrado del número de onda de corte),
    y φ es el autovector (campo electromagnético).
    """

    # ==========================================================================
    # ETAPA 1: GENERACIÓN DE MALLA Y MATRICES FEM
    # ==========================================================================
    
    # Generar malla triangular estructurada del dominio rectangular
    # Xy: coordenadas de nodos, Tri: conectividad, h: tamaño de elemento
    Xy, Tri, h = rect_mesh(Nx, Ny, W, H)
    
    # Ensamblar matrices globales del sistema FEM
    # K: Matriz de rigidez (representa el operador Laplaciano)
    # M: Matriz de masa (representa el término de identidad)
    K, M = assemble_K_M(Xy, Tri)


    # ==========================================================================
    # ETAPA 2: APLICACIÓN DE CONDICIONES DE FRONTERA
    # ==========================================================================
    
    # Verificar tipo de modo y aplicar condiciones de contorno apropiadas
    if kind.upper() == "TM":
        # ======================================================================
        # MODOS TM (TRANSVERSO MAGNÉTICO)
        # ======================================================================
        # Condición: Dirichlet homogénea (campo eléctrico longitudinal = 0 en bordes)
        # Físicamente: Las paredes conductoras fuerzan E_z = 0 en el contorno
        
        # Identificar nodos en el contorno (borde) de la guía
        # np.isclose() detecta nodos en las paredes con tolerancia numérica
        bnd = (np.isclose(Xy[:, 0], 0) |        # Pared izquierda (x = 0)
               np.isclose(Xy[:, 0], W) |        # Pared derecha (x = W)
               np.isclose(Xy[:, 1], 0) |        # Pared inferior (y = 0)
               np.isclose(Xy[:, 1], H))         # Pared superior (y = H)
        
        # Seleccionar solo nodos interiores (donde el campo puede ser ≠ 0)
        interior = np.where(~bnd)[0]  # ~ es el operador NOT lógico
        
    elif kind.upper() == "TE":
        # ======================================================================
        # MODOS TE (TRANSVERSO ELÉCTRICO)  
        # ======================================================================
        # Condición: Neumann natural (derivada normal del campo = 0 en bordes)
        # Físicamente: No hay corriente perpendicular en las paredes
        # En FEM, esto se implementa NO eliminando nodos del contorno
        
        # Para Neumann, todos los nodos se mantienen en el sistema
        interior = np.arange(Xy.shape[0])  # Todos los índices: [0, 1, 2, ..., N-1]
        
    else:
        raise ValueError("El parámetro 'kind' debe ser 'TE' o 'TM'.")

    # Extraer submatrices correspondientes a los grados de libertad activos
    # Kii, Mii: matrices reducidas que solo incluyen nodos interiores (TM) o todos (TE)
    Kii = K[interior][:, interior]  # Kii = K[interior, interior] (submatriz)
    Mii = M[interior][:, interior]  # Mii = M[interior, interior] (submatriz)


    # ==========================================================================
    # ETAPA 3: ESTRATEGIA DE SOLUCIÓN CON SHIFT ESPECTRAL
    # ==========================================================================
    
    # Calcular valor analítico de referencia para el modo (m,n)
    kc_ana = kc_rect(m, n, W, H)  # kc_analítico = √[(mπ/W)² + (nπ/H)²]
    
    # Usar el cuadrado del kc analítico como "shift" espectral
    # Esto mejora la eficiencia: el solver itera cerca del autovalor deseado
    sigma = kc_ana**2  # σ = kc_ana² (shift alrededor del autovalor esperado)

    # Ajustar número de autovalores a calcular según disponibilidad
    # k_req debe ser menor que la dimensión del sistema reducido
    k_req = min(max(8, k_eigs), Kii.shape[0] - 2)
    
    # ======================================================================
    # RESOLVER PROBLEMA DE AUTOVALORES GENERALIZADO: (K - λM)φ = 0
    # ======================================================================
    # eigsh: eigensolver para matrices sparse (Kii, Mii son matrices sparse)
    # k=k_req: calcular k_req autovalores
    # M=Mii: matriz de masa (problema generalizado)
    # sigma=sigma: shift espectral (buscar autovalores cerca de sigma)
    # which="LM": buscar autovalores más cercanos a sigma (Large Magnitude)
    vals, vecs = eigsh(Kii, k=k_req, M=Mii, sigma=sigma, which="LM")
    
    # Asegurar que los autovalores sean reales (el problema físico es real)
    vals = np.real(vals)
    
    # Ordenar autovalores y autovectores de menor a mayor
    # En problemas de guías de onda, los modos fundamentales tienen los menores kc
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]


    # ==========================================================================
    # ETAPA 4: POST-PROCESAMIENTO Y SELECCIÓN DEL MODO
    # ==========================================================================
    
    # Convertir autovalores λ a constantes de corte kc = √λ
    # np.clip evita problemas numéricos con posibles autovalores negativos muy pequeños
    kc_list = np.sqrt(np.clip(vals, 0.0, None))
    
    # Seleccionar el autovalor FEM más cercano al valor analítico esperado
    # Esto identifica automáticamente cuál de los k_req autovalores calculados
    # corresponde al modo (m,n) que estamos buscando
    j = int(np.argmin(np.abs(kc_list - kc_ana)))
    kc_fem = kc_list[j]  # Constante de corte calculada por FEM

    # ==========================================================================
    # ETAPA 5: RECONSTRUCCIÓN DEL CAMPO MODAL COMPLETO
    # ==========================================================================
    
    # Crear vector de campo completo para toda la malla
    vec = np.zeros(Xy.shape[0])  # Inicializar con ceros en todos los nodos
    
    # Asignar valores del autovector a los nodos interiores
    # Para TM: nodos de contorno quedan en 0 (condición Dirichlet)
    # Para TE: todos los nodos reciben valores del autovector
    vec[interior] = vecs[:, j]

    # Normalización de signo: asegurar consistencia en visualización
    # Si la suma de componentes es negativa, invertir todo el vector
    # Esto evita que el campo aparezca invertido en diferentes ejecuciones
    if np.sum(vec) < 0:
        vec = -vec

    # ==========================================================================
    # ETAPA 6: CÁLCULO DE ERROR Y PREPARACIÓN DE RESULTADOS
    # ==========================================================================
    
    # Calcular error relativo porcentual respecto al valor analítico
    err = abs(kc_fem - kc_ana) / kc_ana * 100.0

    # Retornar todos los resultados relevantes
    return kc_fem, kc_ana, err, vec, Xy, Tri


# ==============================================================================
# BLOQUE DE PRUEBA Y VALIDACIÓN
# ==============================================================================
if __name__ == "__main__":
    """
    Bloque de prueba que se ejecuta solo cuando el script corre directamente.
    Propósito: Validar el funcionamiento con casos de prueba conocidos.
    """
    
    # ==========================================================================
    # PARÁMETROS DE PRUEBA
    # ==========================================================================
    
    # Dimensiones de la guía de onda de prueba
    W, H = 1.0, 0.5  # Relación de aspecto 2:1 típica en guías rectangulares
    
    # Resolución de malla (balance entre precisión y costo computacional)
    Nx, Ny = 20, 10  # 20×10 = 200 nodos, ~342 triángulos

    print("=" * 60)
    print("VALIDACIÓN SOLVER FEM PARA MODOS EN GUÍA RECTANGULAR")
    print("=" * 60)

    # ==========================================================================
    # PRUEBA 1: MODO TE10 (MODO FUNDAMENTAL)
    # ==========================================================================
    print("\n>>> Prueba TE10 (Modo fundamental)")
    kc_fem, kc_ana, err, vec, Xy, Tri = solve_mode("TE", 1, 0, Nx, Ny, W, H)
    print(f"kc_FEM  = {kc_fem:.6f} m⁻¹")
    print(f"kc_Anal = {kc_ana:.6f} m⁻¹") 
    print(f"Error   = {err:.4f} %")
    print(f"Nodos   = {len(Xy)}, Elementos = {len(Tri)}")

    # ==========================================================================
    # PRUEBA 2: MODO TM11 (MODO SUPERIOR)
    # ==========================================================================
    print("\n>>> Prueba TM11 (Modo superior)")
    kc_fem, kc_ana, err, vec, Xy, Tri = solve_mode("TM", 1, 1, Nx, Ny, W, H)
    print(f"kc_FEM  = {kc_fem:.6f} m⁻¹")
    print(f"kc_Anal = {kc_ana:.6f} m⁻¹")
    print(f"Error   = {err:.4f} %")
    print(f"Nodos   = {len(Xy)}, Elementos = {len(Tri)}")

    # ==========================================================================
    # INFORMACIÓN ADICIONAL SOBRE LA FÍSICA DEL PROBLEMA
    # ==========================================================================
    print("\n" + "=" * 60)
    print("INFORMACIÓN FÍSICA:")
    print("=" * 60)
    print("• TE10: Campo magnético longitudinal H_z = cos(πx/W)")
    print("        Frecuencia de corte más baja (modo fundamental)")
    print("• TM11: Campo eléctrico longitudinal E_z = sin(πx/W)sin(πy/H)")
    print("        Requiere variación en ambas direcciones")
    print(f"• Dimensión guía: {W} × {H} m")
    print(f"• Frecuencia corte TE10: {(3e8 * kc_rect(1,0,W,H)/(2*np.pi))/1e9:.2f} GHz")
    print("=" * 60)