import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Nadam, Adam, AdamW
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import tensorflow as tf

#Definir semilla
np.random.seed(0)

'''
Descripción: Carga los datos de las imágenes y etiquetas desde archivos comprimidos gzip.
Entrada:
    - ruta_imagenes (str): Ruta al archivo gzip que contiene las imágenes.
    - ruta_etiquetas (str): Ruta al archivo gzip que contiene las etiquetas.
Salida:
    - imagenes: Arreglo normalizado de imágenes en escala [0, 1].
    - etiquetas: Arreglo de etiquetas correspondientes a las imágenes.
'''
def cargar_datos(ruta_imagenes, ruta_etiquetas):
    with gzip.open(ruta_imagenes, 'rb') as archivo:
        archivo.read(4) # Descarta los primeros 4 bytes del archivo porque no son relevantes
        cantidad_imagenes = int.from_bytes(archivo.read(4), 'big')
        filas = int.from_bytes(archivo.read(4), 'big')
        columnas = int.from_bytes(archivo.read(4), 'big')
        tamanyo_imagen = filas * columnas
        imagenes = np.frombuffer(archivo.read(), dtype=np.uint8).reshape(cantidad_imagenes, tamanyo_imagen)

    with gzip.open(ruta_etiquetas, 'rb') as archivo:
        archivo.read(4)
        cantidad_etiquetas = int.from_bytes(archivo.read(4), 'big')
        etiquetas = np.frombuffer(archivo.read(), dtype=np.uint8)


    return imagenes/255.0, etiquetas

'''
Descripción: Carga un mapeo de claves de un dataset a valores desde un archivo de texto
Entrada:
    - ruta_archivo (str): Ruta al archivo de texto que contiene el mapeo.
    - dataset (str): Tipo de dataset; puede ser "digits" o "letters". Por defecto es "digits".
Salida:
    - mapping (dict): Diccionario que mapea claves a valores. Para el subconjunto letters, los valores son tuplas (mayúscula, minúscula).
'''
def cargar_mapping(ruta_archivo, dataset):
    mapping = {}
    with open(ruta_archivo, 'r') as archivo:
        for linea in archivo:
            linea_aux = linea.strip().split()
            if dataset == "letters":
                clave, valor_mayusculas, valor_minusculas = map(int, linea_aux)
                mapping[clave] = (valor_mayusculas, valor_minusculas)
            else:  # Para "digits"
                clave, valor = map(int, linea_aux)
                mapping[clave] = valor
    return mapping

'''
Descripción: Muestra imágenes aleatoriamente junto con sus etiquetas en una cuadrícula.
Entrada:
    - imagenes (numpy.ndarray): Arreglo de imágenes.
    - etiquetas (numpy.ndarray): Arreglo de etiquetas de las imágenes.
    - mapping (dict): Diccionario para mapear las etiquetas.
    - cantidad (int): Número de imágenes a seleccionar y mostrar.
    - titulo (str): Título de la figura generada.
Salida: No retorna ningún valor, solo muestra la figura.
'''
def mostrar_imagenes_aleatorias(imagenes, etiquetas, mapping, cantidad, titulo):
    indices = random.sample(range(imagenes.shape[0]), cantidad)
    imagenes_seleccionadas = imagenes[indices]
    etiquetas_seleccionadas = etiquetas[indices]

    filas = int(np.sqrt(cantidad))
    columnas = int(np.ceil(cantidad / filas))

    plt.figure(figsize=(10, 10))
    plt.suptitle(titulo, fontsize=16)
    for i, (imagen, etiqueta) in enumerate(zip(imagenes_seleccionadas, etiquetas_seleccionadas)):
        plt.subplot(filas, columnas, i + 1)
        if etiqueta in mapping:
            if isinstance(mapping[etiqueta], tuple):  # Dataset EMNIST_Letters
                etiqueta_mapeada = f"{chr(mapping[etiqueta][0])}/{chr(mapping[etiqueta][1])}"
            else:  # Dataset EMNIST_Digits
                etiqueta_mapeada = chr(mapping[etiqueta])
        else:
            etiqueta_mapeada = etiqueta
        plt.imshow(imagen.reshape(64, 64), cmap='gray')  # Redimensionar a (64, 64)
        plt.title(f"{etiqueta_mapeada}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

'''
Descripción: Crea un modelo de red neuronal secuencial con las características especificadas.
Entrada:
    - capas_ocultas (list): Lista con el número de neuronas en cada capa oculta.
    - activacion (str): Función de activación a usar en las capas ocultas.
    - loss (str): Función de pérdida para el modelo.
    - optimizador (str o tf.keras.optimizers.Optimizer): Optimizador a usar.
    - entradas (tuple): Dimensión de entrada esperada por el modelo.
    - salidas (int): Número de neuronas en la capa de salida.
Salida:
    - modelo: Modelo de red neuronal.
'''
def crear_modelo(capas_ocultas, activacion, loss, optimizador, entradas, salidas):
    modelo = Sequential()
    modelo.add(Input(shape=entradas))
    for capa in capas_ocultas:
        modelo.add(Dense(capa, activation=activacion))
    modelo.add(Dense(salidas, activation="softmax"))
    modelo.compile(optimizer=optimizador, loss=loss, metrics=["accuracy"])
    return modelo


'''
Descripción: Reinicializa los pesos de todas las capas de un modelo.
Entrada:
    - modelo: MLP.
Salida:
    - No retorna ningún valor.
'''
def reinicializar_pesos(modelo):
    for layer in modelo.layers:
        if hasattr(layer, 'kernel_initializer') and layer.kernel is not None:
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

'''
Descripción: Entrena y evalúa múltiples modelos en varias repeticiones, reinicializando los pesos antes de cada repetición.
Entrada:
    - modelos (list): Lista de modelos de Keras (tf.keras.Model) a entrenar y evaluar.
    - x_entrenamiento: Datos de entrada (imágenes) para el entrenamiento.
    - y_entrenamiento: Etiquetas correspondientes a los datos de entrenamiento.
    - x_prueba: Datos de entrada (imágenes) para la evaluación.
    - y_prueba: Etiquetas correspondientes a los datos de prueba.
    - numero_clases (int): Número de clases para la clasificación.
Salida:
    - resultados (list): Lista de tuplas con los resultados para cada modelo.
    - tiempos_modelos (list): Lista con los tiempos totales de entrenamiento por modelo.
'''
def entrenar_y_evaluar_modelos(modelos, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba, numero_clases):
    resultados = []
    tiempos_modelos = [] 
    y_entrenamiento_aux = y_entrenamiento
    y_prueba_aux = y_prueba

    for i, modelo in enumerate(modelos):
        print(f"Entrenando Modelo MLP{i + 1}...")
        if modelo.loss == 'categorical_crossentropy':
            y_entrenamiento = to_categorical(y_entrenamiento, num_classes=numero_clases)
            y_prueba = to_categorical(y_prueba, num_classes=numero_clases)

        acc_totales = []
        acc_clases = []
        matrices_conf = []

        inicio_modelo = time.time() 

        for r in range(5):
            print(f"  Repetición {r + 1}...")
            reinicializar_pesos(modelo)

            modelo.fit(x_entrenamiento, y_entrenamiento, epochs=5, batch_size=32, verbose=0)
            predicciones = np.argmax(modelo.predict(x_prueba), axis=1)
            acc_total = accuracy_score(y_prueba, predicciones) if modelo.loss != 'categorical_crossentropy' else \
                        accuracy_score(np.argmax(y_prueba, axis=1), predicciones)

            matriz_conf = confusion_matrix(
                y_prueba if modelo.loss != 'categorical_crossentropy' else np.argmax(y_prueba, axis=1),
                predicciones
            )

            acc_clase = matriz_conf.diagonal() / matriz_conf.sum(axis=1)
            acc_totales.append(acc_total)
            acc_clases.append(acc_clase)
            matrices_conf.append(matriz_conf)

        fin_modelo = time.time()  
        tiempo_total = fin_modelo - inicio_modelo
        tiempos_modelos.append(tiempo_total)

        y_entrenamiento = y_entrenamiento_aux
        y_prueba = y_prueba_aux

        mediana_acc = np.median(acc_totales)
        resultados.append((mediana_acc, acc_clases[-1], matrices_conf[-1], acc_totales[-1]))

        print(f"Tiempo total para Modelo MLP{i + 1}: {tiempo_total:.2f} segundos")

    print("Tiempos totales por modelo:", tiempos_modelos)
    return resultados, tiempos_modelos

def redimensionar_y_preprocesar(imagenes, nuevo_tamano=(64, 64)):
    lado_original = int(np.sqrt(imagenes.shape[1]))  # Suponiendo imágenes cuadradas
    imagenes = imagenes.reshape(-1, lado_original, lado_original, 1)  # Escala de grises
    imagenes = tf.image.resize(imagenes, nuevo_tamano).numpy()
    return imagenes.reshape(-1, nuevo_tamano[0] * nuevo_tamano[1])  # Aplana las imágenes


ruta_letters = "./emnist-letters"
# Concatena la ruta de los archivos con la carpeta de las letras con os
ruta_entrenamiento_imagenes = os.path.join(ruta_letters, "emnist-letters-train-images-idx3-ubyte.gz")
ruta_entrenamiento_etiquetas = os.path.join(ruta_letters, "emnist-letters-train-labels-idx1-ubyte.gz")
ruta_prueba_imagenes = os.path.join(ruta_letters, "emnist-letters-test-images-idx3-ubyte.gz")
ruta_prueba_etiquetas = os.path.join(ruta_letters, "emnist-letters-test-labels-idx1-ubyte.gz")
ruta_mapping = os.path.join(ruta_letters, "emnist-letters-mapping.txt")

x_entrenamiento, y_entrenamiento = cargar_datos(ruta_entrenamiento_imagenes, ruta_entrenamiento_etiquetas)
x_prueba, y_prueba = cargar_datos(ruta_prueba_imagenes, ruta_prueba_etiquetas)

x_entrenamiento = redimensionar_y_preprocesar(x_entrenamiento, (64, 64))
x_prueba = redimensionar_y_preprocesar(x_prueba, (64, 64))

mapping = cargar_mapping(ruta_mapping, "letters")

shape_entrada = (x_entrenamiento.shape[1], )
numero_clases = len(np.unique(y_entrenamiento))

## VER SI ES POR EL ESPACIO EN BLANCO DEL MAPPING<<<<<<<<<<<<<<<<<
y_entrenamiento = y_entrenamiento - 1
y_prueba = y_prueba - 1

parametros_mlp1_exp2 = {'capas_ocultas': [256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}
parametros_mlp2_exp2 = {'capas_ocultas': [256, 128, 64], 'activacion': 'swish', 'loss': 'sparse_categorical_crossentropy'}
parametros_mlp3_exp2 = {'capas_ocultas': [512, 256, 128, 64], 'activacion': 'leaky_relu', 'loss': 'categorical_crossentropy'}

MLP1_exp2 = crear_modelo(
    capas_ocultas=parametros_mlp1_exp2['capas_ocultas'],
    activacion=parametros_mlp1_exp2['activacion'],
    loss=parametros_mlp1_exp2['loss'],
    optimizador=Nadam(learning_rate=0.0002),
    entradas=shape_entrada,
    salidas=numero_clases
)

MLP2_exp2 = crear_modelo(
    capas_ocultas=parametros_mlp2_exp2['capas_ocultas'],
    activacion=parametros_mlp2_exp2['activacion'],
    loss=parametros_mlp2_exp2['loss'],
    optimizador=AdamW(learning_rate=0.00015),
    entradas=shape_entrada,
    salidas=numero_clases
)

MLP3_exp2 = crear_modelo(
    capas_ocultas=parametros_mlp3_exp2['capas_ocultas'],
    activacion=parametros_mlp3_exp2['activacion'],
    loss=parametros_mlp3_exp2['loss'],
    optimizador=Adam(learning_rate=0.0002),
    entradas=shape_entrada,
    salidas=numero_clases
)

resultados, tiempos = entrenar_y_evaluar_modelos([MLP1_exp2, MLP2_exp2, MLP3_exp2], x_entrenamiento, y_entrenamiento, x_prueba, y_prueba, numero_clases)