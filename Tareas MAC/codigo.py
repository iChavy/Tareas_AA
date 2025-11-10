import numpy as np
import gzip
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt



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
def crear_modelo(capas_ocultas, activacion, loss, optimizador, entradas, salidas):
    modelo = Sequential()
    modelo.add(Flatten(input_shape=entradas))
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

# Función para mostrar imágenes
def mostrar_imagenes(imagenes, etiquetas, mapping, titulo, num_mostrar=20):
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    fig.suptitle(titulo)
    for i, ax in enumerate(axes.flat):
        idx = np.random.randint(0, len(imagenes))
        ax.imshow(imagenes[idx], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Clase: {mapping[etiquetas[idx]]}")
    plt.tight_layout()
    plt.show()

# Cargar datos de EMNIST Digits
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
# Cargardatos de EMNIST Letters
ruta_letras = "./emnist-letters"
x_train_letras, y_train_letras = cargar_datos(
    os.path.join(ruta_letras, "emnist-letters-train-images-idx3-ubyte.gz"),
    os.path.join(ruta_letras, "emnist-letters-train-labels-idx1-ubyte.gz"),
    124800,
)

x_test_letras, y_test_letras = cargar_datos(
    os.path.join(ruta_letras, "emnist-letters-test-images-idx3-ubyte.gz"),
    os.path.join(ruta_letras, "emnist-letters-test-labels-idx1-ubyte.gz"),
    20800,
)

# Cargar mapping de etiquetas para dígitos
mapping_digits = {i: str(i) for i in range(10)}

# Carga mapping de etiquetas para letras###########################################

'''
# Mostrar imágenes para comprobar la carga correcta
mostrar_imagenes(x_train_digits, y_train_digits, mapping_digits, "Entrenamiento Digits")
mostrar_imagenes(x_test_digits, y_test_digits, mapping_digits, "Prueba Digits")
'''

# Convertir etiquetas a one-hot encoding
y_train_digits_one_hot = to_categorical(y_train_digits, num_classes=10)
y_test_digits_one_hot = to_categorical(y_test_digits, num_classes=10)
y_train_letras_one_hot = to_categorical(y_train_letras, num_classes=27)
y_test_letras_one_hot = to_categorical(y_test_letras, num_classes=27)

print("Datos cargados y etiquetas convertidas exitosamente")

# Crear los modelos con sus parámetros
parametros_mlp1 = {'capas_ocultas': [256, 128], 'funcion_activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003}
parametros_mlp2 = {'capas_ocultas': [256, 128, 64], 'funcion_activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002}
parametros_mlp3 = {'capas_ocultas': [512, 256, 128], 'funcion_activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002}

MLP1 = crear_modelo(
    capas_ocultas=parametros_mlp1['capas_ocultas'],
    activacion=parametros_mlp1['funcion_activacion'],
    loss=parametros_mlp1['loss'],
    optimizador=Nadam(learning_rate=parametros_mlp1['learning_rate']),
    entradas=(28, 28),
    salidas=10
)

MLP2 = crear_modelo(
    capas_ocultas=parametros_mlp2['capas_ocultas'],
    activacion=parametros_mlp2['funcion_activacion'],
    loss=parametros_mlp2['loss'],
    optimizador=tf.keras.optimizers.AdamW(learning_rate=parametros_mlp2['learning_rate']),
    entradas=(28, 28),
    salidas=10
)

MLP3 = crear_modelo(
    capas_ocultas=parametros_mlp3['capas_ocultas'],
    activacion=parametros_mlp3['funcion_activacion'],
    loss=parametros_mlp3['loss'],
    optimizador=Adam(learning_rate=parametros_mlp3['learning_rate']),
    entradas=(28, 28),
    salidas=10
)

# Función para entrenar y evaluar modelos
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

def entrenar_y_evaluar_modelos(modelos, x_train, y_train, x_test, y_test, exp_label):
    resultados = []
    for i, modelo in enumerate(modelos):
        print(f"Entrenando Modelo {exp_label} MLP{i + 1}...")
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

#### RESULTADOS ####
resultados = entrenar_y_evaluar_modelos([MLP1, MLP2, MLP3], x_train_digits, y_train_digits_one_hot, x_test_digits, y_test_digits_one_hot)

for i, (mediana, acc_clase, matriz_conf, acc_total) in enumerate(resultados):
    print(f"\nModelo MLP{i + 1}:")
    print(f"Mediana del Accuracy Total: {mediana:.4f}")
    print(f"Accuracy Total: {acc_total:.4f}")
    print(f"Accuracy por Clase: {acc_clase}")
    print(f"Matriz de Confusión:\n{matriz_conf}")

    # Gráfico de barras de Accuracy por Clase
    plt.figure()
    plt.bar(range(len(acc_clase)), acc_clase)
    plt.title(f"Accuracy por Clase - Modelo MLP{i + 1}")
    plt.xlabel("Clases")
    plt.ylabel("Accuracy")
    plt.show()

    
