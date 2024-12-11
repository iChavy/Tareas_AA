import numpy as np
import gzip
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Configurar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Función para cargar datos EMNIST desde archivos comprimidos
def cargar_datos(ruta_imagenes, ruta_etiquetas, num_imagenes):
    with gzip.open(ruta_imagenes, "rb") as f:
        imagenes = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(num_imagenes, 28, 28)
    with gzip.open(ruta_etiquetas, "rb") as f:
        etiquetas = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return imagenes / 255.0, etiquetas

# Función para crear modelos MLP
def crear_modelo(capas_ocultas, activacion, loss, learning_rate, optimizador, entradas, salidas):
    modelo = Sequential()
    modelo.add(Flatten(input_shape=entradas))
    for capa in capas_ocultas:
        modelo.add(Dense(capa, activation=activacion))
    modelo.add(Dense(salidas, activation="softmax"))
    if optimizador == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)
    elif optimizador == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    modelo.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return modelo

# Función para reinicializar pesos
def reinicializar_pesos(modelo):
    for layer in modelo.layers:
        if hasattr(layer, 'kernel_initializer') and layer.kernel is not None:
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

# Función para guardar resultados
def guardar_resultados(nombre_archivo, modelo_id, repeticion, acc_total, acc_clase, parametros=None, median_accuracy=None):
    with open(nombre_archivo, "a", encoding="utf-8") as archivo:
        archivo.write(f"Modelo: MLP{modelo_id}\n")
        if parametros:
            archivo.write(f"Parámetros: {parametros}\n")
        archivo.write(f"Repetición: Repetición {repeticion}\n")
        archivo.write(f"Accuracy total: {acc_total:.4f}\n")
        archivo.write("Accuracy por clase:\n")
        for i, acc in enumerate(acc_clase):
            archivo.write(f"  Clase {i}: {acc:.4f}\n")
        archivo.write("-" * 40 + "\n")
        if median_accuracy is not None:
            archivo.write(f"Mediana del Accuracy Total: {median_accuracy:.4f}\n")
            archivo.write("-" * 40 + "\n")

# Cargar datos de EMNIST letras
ruta_letras = "./emnist-letters"
x_train_letters, y_train_letters = cargar_datos(
    os.path.join(ruta_letras, "emnist-letters-train-images-idx3-ubyte.gz"),
    os.path.join(ruta_letras, "emnist-letters-train-labels-idx1-ubyte.gz"),
    124800,
)
x_test_letters, y_test_letters = cargar_datos(
    os.path.join(ruta_letras, "emnist-letters-test-images-idx3-ubyte.gz"),
    os.path.join(ruta_letras, "emnist-letters-test-labels-idx1-ubyte.gz"),
    20800,
)


# Convertir etiquetas a one-hot encoding
y_train_letters_one_hot = to_categorical(y_train_letters, num_classes=27)
y_test_letters_one_hot = to_categorical(y_test_letters, num_classes=27)

print("Datos cargados y etiquetas convertidas exitosamente")

# Configuraciones de parámetros para los modelos
configuraciones_parametros = [
    # Configuración 1
    [
        {'capas_ocultas': [64, 32], 'activacion': 'tanh', 'loss': 'categorical_crossentropy', 'learning_rate': 0.001, 'optimizador': 'Adam'},
        {'capas_ocultas': [128, 64, 32], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003, 'optimizador': 'Nadam'},
        {'capas_ocultas': [256, 128, 64], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002, 'optimizador': 'Adam'}
    ],
    # Configuración 2
    [
        {'capas_ocultas': [128, 64], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0004, 'optimizador': 'Nadam'},
        {'capas_ocultas': [256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0005, 'optimizador': 'Adam'},
        {'capas_ocultas': [512, 256, 128, 64], 'activacion': 'elu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003, 'optimizador': 'Nadam'}
    ],
    # Configuración 3
    [
        {'capas_ocultas': [256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0005, 'optimizador': 'Adam'},
        {'capas_ocultas': [512, 256, 128], 'activacion': 'tanh', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0004, 'optimizador': 'Nadam'},
        {'capas_ocultas': [1024, 512, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002, 'optimizador': 'Adam'}
    ],
    # Configuración 4
    [
        {'capas_ocultas': [512, 256], 'activacion': 'elu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003, 'optimizador': 'Nadam'},
        {'capas_ocultas': [256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002, 'optimizador': 'Adam'},
        {'capas_ocultas': [1024, 512, 256, 128], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0001, 'optimizador': 'Nadam'}
    ],
    # Configuración 5
    [
        {'capas_ocultas': [128, 64], 'activacion': 'tanh', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0006, 'optimizador': 'Adam'},
        {'capas_ocultas': [512, 256, 128], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003, 'optimizador': 'Nadam'},
        {'capas_ocultas': [512, 256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0004, 'optimizador': 'Adam'}
    ],
    # Configuración 6
    [
        {'capas_ocultas': [256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0007, 'optimizador': 'Adam'},
        {'capas_ocultas': [128, 64, 32], 'activacion': 'elu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0005, 'optimizador': 'Nadam'},
        {'capas_ocultas': [1024, 512, 256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003, 'optimizador': 'Adam'}
    ],
    # Configuración 7
    [
        {'capas_ocultas': [64, 32], 'activacion': 'tanh', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0008, 'optimizador': 'Nadam'},
        {'capas_ocultas': [128, 64, 32], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0006, 'optimizador': 'Adam'},
        {'capas_ocultas': [512, 256, 128, 64], 'activacion': 'elu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0004, 'optimizador': 'Nadam'}
    ],
    # Configuración 8
    [
        {'capas_ocultas': [128, 64], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0009, 'optimizador': 'Adam'},
        {'capas_ocultas': [256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0007, 'optimizador': 'Nadam'},
        {'capas_ocultas': [1024, 512, 256, 128], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0005, 'optimizador': 'Adam'}
    ],
    # Configuración 9
    [
        {'capas_ocultas': [512, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003, 'optimizador': 'Adam'},
        {'capas_ocultas': [256, 128, 64], 'activacion': 'elu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002, 'optimizador': 'Nadam'},
        {'capas_ocultas': [512, 256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0001, 'optimizador': 'Adam'}
    ],
    # Configuración 10
    [
        {'capas_ocultas': [256, 128], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0005, 'optimizador': 'Nadam'},
        {'capas_ocultas': [128, 64, 32], 'activacion': 'tanh', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0004, 'optimizador': 'Adam'},
        {'capas_ocultas': [1024, 512, 256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002, 'optimizador': 'Nadam'}
    ]
]


# Entrenar y evaluar modelos
def entrenar_y_evaluar_modelos(configuraciones, x_train, y_train, x_test, y_test, nombre_archivo):
    for config_id, configuraciones_modelos in enumerate(configuraciones, 1):
        print(f"Ejecutando configuración {config_id}...")
        for modelo_id, parametros in enumerate(configuraciones_modelos, 1):
            print(f"Entrenando Modelo MLP{modelo_id}...")
            modelo = crear_modelo(**parametros, entradas=(28, 28), salidas=27)
            acc_totales = []
            for r in range(5):  # Repetir 5 veces
                print(f"  Repetición {r + 1}...")
                reinicializar_pesos(modelo)
                modelo.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
                predicciones = np.argmax(modelo.predict(x_test), axis=1)
                acc_total = accuracy_score(np.argmax(y_test, axis=1), predicciones)
                matriz_conf = confusion_matrix(np.argmax(y_test, axis=1), predicciones)
                acc_clase = matriz_conf.diagonal() / matriz_conf.sum(axis=1)
                acc_totales.append(acc_total)
                guardar_resultados(nombre_archivo, modelo_id, r + 1, acc_total, acc_clase, parametros=parametros)
            guardar_resultados(nombre_archivo, modelo_id, 5, acc_total, acc_clase, parametros=parametros, median_accuracy=np.median(acc_totales))

entrenar_y_evaluar_modelos(configuraciones_parametros, x_train_letters, y_train_letters_one_hot, x_test_letters, y_test_letters_one_hot, "resultados_emnist_letters.txt")
