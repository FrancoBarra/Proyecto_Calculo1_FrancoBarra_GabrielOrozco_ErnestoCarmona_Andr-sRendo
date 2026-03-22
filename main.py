import cv2
import numpy as np
import matplotlib.pyplot as plt
from splines_math import calcular_splines_cubicos_naturales, evaluar_splines

# 1. LECTURA
imagen = cv2.imread('depredador.jpg')
if imagen is None: print("Error"); exit()
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. CANNY
bordes_canny = cv2.Canny(imagen_gris, 100, 200)

# 3. MASCARA + LIMPIEZA MORFOLOGICA
_, mascara = cv2.threshold(imagen_gris, 240, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=3)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

# 4. CONTORNO COMPLETO
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contorno = max(contornos, key=cv2.contourArea)
pts_all = contorno.squeeze()

# 5. SELECCION DE 60 NODOS
N_NODOS = 60
paso = len(pts_all) // N_NODOS
indices = list(range(0, len(pts_all), paso))[:N_NODOS]
nodos = pts_all[indices]
t_nodos = list(range(len(nodos)))
x_nodos = [int(p[0]) for p in nodos]
y_nodos = [int(p[1]) for p in nodos]

# 6. SPLINES PARAMETRICOS: x(t) y y(t)
ax, bx, cx, dx = calcular_splines_cubicos_naturales(t_nodos, x_nodos)
t_spline, x_spline = evaluar_splines(t_nodos, ax, bx, cx, dx, puntos_por_tramo=20)
ay, by, cy, dy = calcular_splines_cubicos_naturales(t_nodos, y_nodos)
_, y_spline = evaluar_splines(t_nodos, ay, by, cy, dy, puntos_por_tramo=20)
x_spline.append(float(x_nodos[0])); y_spline.append(float(y_nodos[0]))

# 7. VISUALIZACION
plt.figure(figsize=(18, 8))
plt.subplot(1, 3, 1); plt.title('Imagen Original'); plt.imshow(imagen_rgb)
plt.subplot(1, 3, 2); plt.title('Bordes (Canny)'); plt.imshow(bordes_canny, cmap='gray')
plt.subplot(1, 3, 3); plt.title('Contorno - Spline Parametrico (60 nodos)')
plt.imshow(imagen_rgb, alpha=0.3)
plt.plot(x_spline, y_spline, color='blue', linewidth=2.5, label='Spline parametrico')
plt.scatter(x_nodos, y_nodos, s=30, color='red', zorder=10, label='Nodos')
plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout(); plt.show()