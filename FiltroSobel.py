#CÓDIGO MONOHILO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen y convertirla a escala de grises
image_path = 'C:/Users/torre/OneDrive/Desktop/6to semestre/SISTEMAS PARALELOS/flor.jpg'
image = Image.open(image_path).convert('L')  # Convertimos a escala de grises
image_array = np.array(image)

# Definir los kernels de Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Aplicar el filtro de Sobel
def apply_sobel_filter(image, kernel):
    rows, cols = image.shape
    output = np.zeros_like(image)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i-1:i+2, j-1:j+2]
            output[i, j] = abs(np.sum(region * kernel))
    return output

# Aplicar el filtro en las direcciones X y Y
gradient_x = apply_sobel_filter(image_array, sobel_x)
gradient_y = apply_sobel_filter(image_array, sobel_y)

# Calcular la magnitud del gradiente
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255  # Normalizar a 0-255

# Mostrar la imagen original y el resultado del filtro de Sobel
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Imagen Original en Escala de Grises')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtro de Sobel - Magnitud del Gradiente')
plt.imshow(gradient_magnitude, cmap='gray')

plt.show()
