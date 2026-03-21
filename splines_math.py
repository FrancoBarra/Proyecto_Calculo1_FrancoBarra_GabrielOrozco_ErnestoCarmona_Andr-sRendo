import numpy as np

def calcular_splines_cubicos_naturales(x, y):
    """
    Implementación del algoritmo de Splines Cúbicos Naturales 
    (Basado en Burden & Faires / Chapra & Canale).
    """
    n = len(x) - 1  # Número de tramos (polinomios)
    
    # Paso 1: Inicializar arreglos para la matemática
    a = np.array(y, dtype=float)
    b = np.zeros(n)
    d = np.zeros(n)
    
    # Diferencias entre los valores de x (h)
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1] - x[i]
        
    # Paso 2: Construir el sistema de ecuaciones (Matriz Tridiagonal)
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (a[i+1] - a[i]) - (3 / h[i-1]) * (a[i] - a[i-1])
        
    # Paso 3: Resolver el sistema para encontrar los coeficientes 'c'
    l = np.ones(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)
    c = np.zeros(n+1)
    
    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
        
    # Condiciones de frontera para un Spline "Natural"
    l[n] = 1
    z[n] = 0
    c[n] = 0
    
    # Paso 4: Calcular coeficientes de atrás hacia adelante
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (a[j+1] - a[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
        
    return a[:-1], b, c[:-1], d

def evaluar_splines(x_nodos, a, b, c, d, puntos_por_tramo=10):
    """Genera los puntos (X, Y) para graficar la curva suave."""
    n = len(x_nodos) - 1
    x_curva = []
    y_curva = []
    
    for i in range(n):
        x_intermedios = np.linspace(x_nodos[i], x_nodos[i+1], puntos_por_tramo)
        
        for x_val in x_intermedios:
            delta_x = x_val - x_nodos[i]
            y_val = a[i] + b[i]*delta_x + c[i]*(delta_x**2) + d[i]*(delta_x**3)
            
            x_curva.append(x_val)
            y_curva.append(y_val)
            
    return x_curva, y_curva