#CÓDIGO MULTIHILO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Función para aplicar el filtro de Sobel a una sección específica de la imagen
def sobel_filter_section(image_array, start_row, end_row):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    rows, cols = image_array.shape
    gradient_x = np.zeros((end_row - start_row, cols))
    gradient_y = np.zeros((end_row - start_row, cols))

    for i in range(1, end_row - start_row - 1):
        for j in range(1, cols - 1):
            region = image_array[start_row + i - 1:start_row + i + 2, j - 1:j + 2]
            gradient_x[i, j] = abs(np.sum(region * sobel_x))
            gradient_y[i, j] = abs(np.sum(region * sobel_y))

    return np.sqrt(gradient_x**2 + gradient_y**2)

# Función para aplicar el filtro de Sobel en multihilo
def sobel_filter_multithreaded(image_array, num_threads=4):
    rows, _ = image_array.shape
    chunk_size = rows // num_threads
    futures = []

    # Crear un arreglo vacío para el resultado final
    gradient_magnitude = np.zeros_like(image_array)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Dividir la imagen en partes y procesarlas en paralelo
        for i in range(num_threads):
            start_row = i * chunk_size
            end_row = (i + 1) * chunk_size if i != num_threads - 1 else rows
            futures.append(executor.submit(sobel_filter_section, image_array, start_row, end_row))

        # Combinar los resultados
        for i, future in enumerate(futures):
            start_row = i * chunk_size
            gradient_magnitude[start_row:start_row + future.result().shape[0], :] = future.result()

    return gradient_magnitude

# Cargar la imagen y convertirla a escala de grises
image_path = r'C:\Users\torre\OneDrive\Desktop\6to semestre\SISTEMAS PARALELOS\flor.jpg'  # Usamos la ruta correcta de la imagen
image = Image.open(image_path).convert('L')  # Convertimos a escala de grises
image_array = np.array(image)

# Aplicar el filtro de Sobel con multihilo
gradient_magnitude = sobel_filter_multithreaded(image_array, num_threads=4)

# Mostrar la imagen original y el resultado del filtro de Sobel
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Imagen Original en Escala de Grises')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtro de Sobel - Magnitud del Gradiente (Multihilo)')
plt.imshow(gradient_magnitude, cmap='gray')

plt.show()

