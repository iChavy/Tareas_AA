import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix



def cargar_datos(ruta_imagenes, ruta_etiquetas):
    """
    Carga las imágenes y etiquetas desde los archivos proporcionados.
    """
    with gzip.open(ruta_imagenes, 'rb') as archivo:
        archivo.read(4)  # Número mágico
        cantidad_imagenes = int.from_bytes(archivo.read(4), 'big')
        filas = int.from_bytes(archivo.read(4), 'big')
        columnas = int.from_bytes(archivo.read(4), 'big')
        tamano_imagen = filas * columnas
        imagenes = np.frombuffer(archivo.read(), dtype=np.uint8).reshape(cantidad_imagenes, tamano_imagen)

    with gzip.open(ruta_etiquetas, 'rb') as archivo:
        archivo.read(4)  # Número mágico
        cantidad_etiquetas = int.from_bytes(archivo.read(4), 'big')
        etiquetas = np.frombuffer(archivo.read(), dtype=np.uint8)

    return imagenes/255.0, etiquetas

def cargar_mapping(ruta_archivo):
    """
    Lee el archivo de mapping y lo almacena en un diccionario.
    """
    mapping = {}
    with open(ruta_archivo, 'r') as archivo:
        for linea in archivo:
            clave, valor = linea.strip().split()
            mapping[int(clave)] = int(valor)
    return mapping

def mostrar_imagenes_aleatorias(imagenes, etiquetas, mapping, cantidad=20, titulo="Imágenes"):
    """
    Muestra una cantidad dada de imágenes aleatorias junto con sus etiquetas usando el mapping.
    """
    indices = random.sample(range(imagenes.shape[0]), cantidad)
    imagenes_seleccionadas = imagenes[indices]
    etiquetas_seleccionadas = etiquetas[indices]

    filas = int(np.sqrt(cantidad))
    columnas = int(np.ceil(cantidad / filas))

    plt.figure(figsize=(10, 10))
    plt.suptitle(titulo, fontsize=16)
    for i, (imagen, etiqueta) in enumerate(zip(imagenes_seleccionadas, etiquetas_seleccionadas)):
        plt.subplot(filas, columnas, i + 1)
        etiqueta_mapeada = chr(mapping[etiqueta]) if etiqueta in mapping else etiqueta
        plt.imshow(imagen.reshape(28, 28), cmap='gray')
        plt.title(f"{etiqueta_mapeada}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def crear_modelo(capas_ocultas, activacion, loss, optimizador, entradas, salidas):
    modelo = Sequential()
    modelo.add(Input(shape=entradas))  # Usar Input en lugar de input_shape en Flatten
    for capa in capas_ocultas:
        modelo.add(Dense(capa, activation=activacion))
    modelo.add(Dense(salidas, activation="softmax"))
    modelo.compile(optimizer=optimizador, loss=loss, metrics=["accuracy"])
    return modelo


# Función para reinicializar pesos
def reinicializar_pesos(modelo):
    for layer in modelo.layers:
        if hasattr(layer, 'kernel_initializer') and layer.kernel is not None:
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

def entrenar_y_evaluar_modelos(modelos, x_train, y_train, x_test, y_test):
    resultados = []
    for i, modelo in enumerate(modelos):
        print(f"Entrenando Modelo MLP{i + 1}...")
        acc_totales = []
        acc_clases = []
        matrices_conf = []

        for r in range(5):
            print(f"  Repetición {r + 1}...")
            reinicializar_pesos(modelo)
            modelo.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

            predicciones = np.argmax(modelo.predict(x_test), axis=1)
            acc_total = accuracy_score(np.argmax(y_test, axis=1), predicciones)
            matriz_conf = confusion_matrix(np.argmax(y_test, axis=1), predicciones)
            acc_clase = matriz_conf.diagonal() / matriz_conf.sum(axis=1)

            acc_totales.append(acc_total)
            acc_clases.append(acc_clase)
            matrices_conf.append(matriz_conf)

        mediana_acc = np.median(acc_totales)
        resultados.append((mediana_acc, acc_clases[-1], matrices_conf[-1], acc_totales[-1]))

    return resultados

ruta_digits = "./emnist-digits"
# Concatena la ruta de los archivos con la carpeta de los dígitos con os
ruta_entrenamiento_imagenes = os.path.join(ruta_digits, "emnist-digits-train-images-idx3-ubyte.gz")
ruta_entrenamiento_etiquetas = os.path.join(ruta_digits, "emnist-digits-train-labels-idx1-ubyte.gz")
ruta_prueba_imagenes = os.path.join(ruta_digits, "emnist-digits-test-images-idx3-ubyte.gz")
ruta_prueba_etiquetas = os.path.join(ruta_digits, "emnist-digits-test-labels-idx1-ubyte.gz")
ruta_mapping = os.path.join(ruta_digits, "emnist-digits-mapping.txt")

x_entrenamiento, y_entrenamiento = cargar_datos(ruta_entrenamiento_imagenes, ruta_entrenamiento_etiquetas)
x_prueba, y_prueba = cargar_datos(ruta_prueba_imagenes, ruta_prueba_etiquetas)

mapping = cargar_mapping(ruta_mapping)

#mostrar_imagenes_aleatorias(x_entrenamiento, y_entrenamiento, mapping, cantidad=20, titulo="Imágenes del conjunto de entrenamiento")
#mostrar_imagenes_aleatorias(x_prueba, y_prueba, mapping, cantidad=20, titulo="Imágenes del conjunto de prueba")

shape_entrada = (x_entrenamiento.shape[1], )
numero_clases = len(np.unique(y_entrenamiento))

y_entrenamiento_categorico = to_categorical(y_entrenamiento, num_classes=numero_clases)
y_prueba_categorico = to_categorical(y_prueba, num_classes=numero_clases)

parametros_mlp1_exp1 = {'capas_ocultas': [256, 128], 'activacion': 'swish', 'loss': 'categorical_crossentropy'}
parametros_mlp2_exp1 = {'capas_ocultas': [256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}
parametros_mlp3_exp1 = {'capas_ocultas': [512, 256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}

MLP1_exp1 = crear_modelo(capas_ocultas=parametros_mlp1_exp1['capas_ocultas'], 
                         activacion=parametros_mlp1_exp1['activacion'], 
                         loss=parametros_mlp1_exp1['loss'], 
                         optimizador=Nadam(learning_rate=0.0003), 
                         entradas=shape_entrada, 
                         salidas=numero_clases)

MLP2_exp1 = crear_modelo(capas_ocultas=parametros_mlp2_exp1['capas_ocultas'], 
                         activacion=parametros_mlp2_exp1['activacion'], 
                         loss=parametros_mlp2_exp1['loss'], 
                         optimizador=Adam(learning_rate=0.0002), 
                         entradas=shape_entrada, 
                         salidas=numero_clases)

MLP3_exp1 = crear_modelo(capas_ocultas=parametros_mlp3_exp1['capas_ocultas'], 
                         activacion=parametros_mlp3_exp1['activacion'], 
                         loss=parametros_mlp3_exp1['loss'], 
                         optimizador=Adam(learning_rate=0.0002), 
                         entradas=shape_entrada, 
                         salidas=numero_clases)

resultados = entrenar_y_evaluar_modelos([MLP1_exp1, MLP2_exp1, MLP3_exp1], x_entrenamiento, y_entrenamiento_categorico, x_prueba, y_prueba_categorico)
