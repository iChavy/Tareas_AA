import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
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

# Cargar datos de emnist-digits
ruta_digitos = "./emnist-digits"
x_train_digits, y_train_digits = cargar_datos(
    os.path.join(ruta_digitos, "emnist-digits-train-images-idx3-ubyte.gz"),
    os.path.join(ruta_digitos, "emnist-digits-train-labels-idx1-ubyte.gz"),
    240000,
)
x_test_digits, y_test_digits = cargar_datos(
    os.path.join(ruta_digitos, "emnist-digits-test-images-idx3-ubyte.gz"),
    os.path.join(ruta_digitos, "emnist-digits-test-labels-idx1-ubyte.gz"),
    40000,
)

# Cargar datos de emnist-letters
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

print("Datos cargados exitosamente")

# Función para crear modelos MLP
def crear_modelo(capas_ocultas, activacion, loss, optimizador, entradas, salidas):
    modelo = Sequential()
    modelo.add(Flatten(input_shape=entradas))
    for capa in capas_ocultas:
        modelo.add(Dense(capa, activation=activacion))
    modelo.add(Dense(salidas, activation="softmax"))
    modelo.compile(optimizer=optimizador, loss=loss, metrics=["accuracy"])
    return modelo


###### EXPERIMENTO 1: EMNIST DIGITS ######

# Función para entrenar y evaluar modelos
def entrenar_evaluar(modelos, x_train, y_train, x_test, y_test, nombre_experimento):
    resultados = []
    for i, modelo in enumerate(modelos):
        print(f"Entrenando modelo MLP{i + 1} para {nombre_experimento}...")
        historia = modelo.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        predicciones = np.argmax(modelo.predict(x_test), axis=1)
        acc_total = accuracy_score(y_test, predicciones)
        matriz_conf = confusion_matrix(y_test, predicciones)
        print(f"Accuracy total para MLP{i + 1}: {acc_total:.4f}")
        resultados.append((acc_total, matriz_conf))
    return resultados

# Experimento 1: EMNIST Digits
MLP1 = crear_modelo(
    capas_ocultas=[128, 64],
    activacion="relu",
    loss="sparse_categorical_crossentropy",
    optimizador=Adam(),
    entradas=(28, 28),
    salidas=10  # Este número de salidas es configurable para Digits y Letters
)

MLP2 = crear_modelo(
    capas_ocultas=[256, 128, 64],
    activacion="tanh",
    loss="sparse_categorical_crossentropy",
    optimizador=Adam(),
    entradas=(28, 28),
    salidas=10  # Configurable para Digits y Letters
)

MLP3 = crear_modelo(
    capas_ocultas=[128, 128, 64, 32],
    activacion="relu",
    loss="sparse_categorical_crossentropy",
    optimizador=SGD(),
    entradas=(28, 28),
    salidas=10  # Configurable para Digits y Letters
)

resultados_digitos = entrenar_evaluar([MLP1, MLP2, MLP3], x_train_digits, y_train_digits, x_test_digits, y_test_digits, "EMNIST Digits")

'''
# Experimento 2: EMNIST Letters
resultados_letras = entrenar_evaluar(modelos, x_train_letters, y_train_letters, x_test_letters, y_test_letters, "EMNIST Letters")

# Graficar comparación de resultados
def graficar_comparacion(resultados, nombre_experimento):
    accuracies = [res[0] for res in resultados]
    modelos = [f"MLP{i + 1}" for i in range(len(resultados))]
    plt.figure(figsize=(10, 6))
    plt.bar(modelos, accuracies)
    plt.title(f"Comparación de Accuracy - {nombre_experimento}")
    plt.xlabel("Modelos")
    plt.ylabel("Accuracy Total")
    plt.ylim(0, 1)
    plt.show()

# Gráficos
graficar_comparacion(resultados_digitos, "EMNIST Digits")
graficar_comparacion(resultados_letras, "EMNIST Letters")
'''