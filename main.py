import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer la imagen
nombre_imagen = 'depredador.jpg'
imagen = cv2.imread(nombre_imagen)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen '{nombre_imagen}'. Verifica el nombre y la ubicación.")
    exit()

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

bordes = cv2.Canny(imagen_gris, 100, 200)

alto, ancho = bordes.shape
puntos_x = []
puntos_y = []


for x in range(ancho):

    columna = bordes[:, x]
    indices_y = np.where(columna == 255)[0]
    
    if len(indices_y) > 0:

        y_superior = indices_y[0]
        puntos_x.append(x)

        puntos_y.append(-y_superior) 


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Escala de Grises')
plt.imshow(imagen_gris, cmap='gray')


plt.subplot(1, 3, 2)
plt.title('Todos los Bordes (Canny)')
plt.imshow(bordes, cmap='gray')


plt.subplot(1, 3, 3)
plt.title('Puntos del Contorno Superior')

plt.scatter(puntos_x, puntos_y, s=1, color='red')
plt.grid(True)

plt.tight_layout()
plt.show()