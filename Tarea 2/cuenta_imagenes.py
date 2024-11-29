import gzip
import os

# Función para calcular el número de imágenes
def contar_imagenes(ruta_imagenes):
    with gzip.open(ruta_imagenes, 'rb') as f:
        contenido = f.read()
        num_imagenes = int.from_bytes(contenido[4:8], byteorder='big')  # Bytes 4-8 contienen el número de imágenes
        return num_imagenes

# Función para calcular el número de etiquetas
def contar_etiquetas(ruta_etiquetas):
    with gzip.open(ruta_etiquetas, 'rb') as f:
        contenido = f.read()
        num_etiquetas = int.from_bytes(contenido[4:8], byteorder='big')  # Bytes 4-8 contienen el número de etiquetas
        return num_etiquetas

# Rutas digitos
ruta_digitos_imagenes_test = "./emnist-digits/emnist-digits-test-images-idx3-ubyte.gz"
ruta_digitos_etiquetas_test = "./emnist-digits/emnist-digits-test-labels-idx1-ubyte.gz"
ruta_digitos_imagenes_train = "./emnist-digits/emnist-digits-train-images-idx3-ubyte.gz"
ruta_digitos_etiquetas_train = "./emnist-digits/emnist-digits-train-labels-idx1-ubyte.gz"

# Rutas letras
ruta_letras_imagenes_test = "./emnist-letters/emnist-letters-test-images-idx3-ubyte.gz"
ruta_letras_etiquetas_test = "./emnist-letters/emnist-letters-test-labels-idx1-ubyte.gz"
ruta_letras_imagenes_train = "./emnist-letters/emnist-letters-train-images-idx3-ubyte.gz"
ruta_letras_etiquetas_train = "./emnist-letters/emnist-letters-train-labels-idx1-ubyte.gz"

# Verificar número de imágenes y etiquetas
num_imagenes_digitos_test = contar_imagenes(ruta_digitos_imagenes_test)
num_etiquetas_digitos_test = contar_etiquetas(ruta_digitos_etiquetas_test)
num_imagenes_digitos_train = contar_imagenes(ruta_digitos_imagenes_train)
num_etiquetas_digitos_train = contar_etiquetas(ruta_digitos_etiquetas_train)

num_imagenes_letras_test = contar_imagenes(ruta_letras_imagenes_test)
num_etiquetas_letras_test = contar_etiquetas(ruta_letras_etiquetas_test)
num_imagenes_letras_train = contar_imagenes(ruta_letras_imagenes_train)
num_etiquetas_letras_train = contar_etiquetas(ruta_letras_etiquetas_train)

print(f"Número de imágenes en {ruta_digitos_imagenes_test}: {num_imagenes_digitos_test}")
print(f"Número de etiquetas en {ruta_digitos_etiquetas_test}: {num_etiquetas_digitos_test}")
print(f"Número de imágenes en {ruta_digitos_imagenes_train}: {num_imagenes_digitos_train}")
print(f"Número de etiquetas en {ruta_digitos_etiquetas_train}: {num_etiquetas_digitos_train}")

print(f"Número de imágenes en {ruta_letras_imagenes_test}: {num_imagenes_letras_test}")
print(f"Número de etiquetas en {ruta_letras_etiquetas_test}: {num_etiquetas_letras_test}")
print(f"Número de imágenes en {ruta_letras_imagenes_train}: {num_imagenes_letras_train}")
print(f"Número de etiquetas en {ruta_letras_etiquetas_train}: {num_etiquetas_letras_train}")

