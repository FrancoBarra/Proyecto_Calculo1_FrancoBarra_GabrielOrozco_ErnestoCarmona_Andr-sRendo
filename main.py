import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importamos las funciones de nuestro motor matemático
from splines_math import calcular_splines_cubicos_naturales, evaluar_splines

# 1. Leer la imagen
nombre_imagen = 'depredador.jpg'
imagen = cv2.imread(nombre_imagen)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen.")
    exit()

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. PROCESAMIENTO VISUAL (Lo que el profesor quiere ver)
# ============================================================
# Aplicamos el filtro Canny tal como lo sugiere el PDF
bordes_canny = cv2.Canny(imagen_gris, 100, 200)


# ============================================================
# 3. EXTRACCIÓN ROBUSTA DEL CONTORNO (Nuestro secreto matemático)
# ============================================================
# Usamos la máscara binaria internamente para garantizar que
# sacamos el contorno superior externo y no la malla del traje.
_, mascara = cv2.threshold(imagen_gris, 240, 255, cv2.THRESH_BINARY_INV)

puntos_x = []
puntos_y = []

# Escaneamos el área de interés (cabeza y cañón)
for x in range(200, 480):
    columna = mascara[:, x]
    indices_y = np.where(columna == 255)[0]
    
    if len(indices_y) > 0:
        puntos_x.append(x)
        puntos_y.append(indices_y[0]) 

# ============================================================
# 4. REDUCCIÓN DE PUNTOS Y CÁLCULO DE SPLINES
# ============================================================
nodos_x = puntos_x[::20]
nodos_y = puntos_y[::20]

a, b, c, d = calcular_splines_cubicos_naturales(nodos_x, nodos_y)
x_spline, y_spline = evaluar_splines(nodos_x, a, b, c, d, puntos_por_tramo=15)


# ============================================================
# 5. VISUALIZACIÓN FINAL PARA EL REPORTE
# ============================================================
plt.figure(figsize=(15, 6))

# Gráfica 1
plt.subplot(1, 3, 1)
plt.title('Imagen Original')
plt.imshow(imagen_rgb)

# Gráfica 2: ¡Volvemos a los bordes Canny!
plt.subplot(1, 3, 2)
plt.title('Detección de Bordes (Canny)')
plt.imshow(bordes_canny, cmap='gray')
plt.axvline(x=200, color='blue', linestyle='--')
plt.axvline(x=480, color='blue', linestyle='--')

# Gráfica 3: El resultado de los Splines
plt.subplot(1, 3, 3)
plt.title('Ajuste de Splines Cúbicos')
plt.imshow(imagen_rgb, alpha=0.3) # Imagen translúcida de fondo
plt.plot(x_spline, y_spline, color='blue', linewidth=4, label='Curva Spline')
plt.scatter(nodos_x, nodos_y, s=60, color='red', zorder=10, label='Nodos')

plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()