import numpy as np
import gzip
import os
from sklearn.metrics import confusion_matrix, accuracy_score
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
def cargar_datos(ruta_imagenes, ruta_etiquetas):
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

# Función para cargar mapping de etiquetas (adaptada para tres columnas)
def cargar_mapping(ruta_archivo, dataset="digits"):
    mapping = {}
    with open(ruta_archivo, 'r') as archivo:
        for linea in archivo:
            partes = linea.strip().split()
            if dataset == "letters":
                clave, valor_upper, valor_lower = map(int, partes)
                mapping[clave] = (valor_upper, valor_lower)
            else:  # Para "digits"
                clave, valor = map(int, partes)
                mapping[clave] = valor
    return mapping

def mostrar_imagenes_aleatorias(imagenes, etiquetas, mapping, cantidad=20, titulo="Imágenes"):
    """
    Muestra una cantidad dada de imágenes aleatorias junto con sus etiquetas usando el mapping.
    Incluye tanto letras mayúsculas como minúsculas si el mapping contiene tuplas.
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
        if etiqueta in mapping:
            if isinstance(mapping[etiqueta], tuple):  # Dataset "letters"
                etiqueta_mapeada = f"{chr(mapping[etiqueta][0])}/{chr(mapping[etiqueta][1])}"  # Uppercase/Lowercase
            else:  # Dataset "digits"
                etiqueta_mapeada = chr(mapping[etiqueta])
        else:
            etiqueta_mapeada = etiqueta
        plt.imshow(imagen.reshape(28, 28), cmap='gray')
        plt.title(f"{etiqueta_mapeada}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Función para crear modelos MLP
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
            modelo.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)

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
    return resultados

'''
##########EXPERIMENTO 1 ########

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

# Convertir etiquetas a one-hot encoding
y_train_digits_one_hot = to_categorical(y_train_digits, num_classes=10)
y_test_digits_one_hot = to_categorical(y_test_digits, num_classes=10)

# Crear los modelos con sus parámetros
parametros_mlp1_exp1 = {'capas_ocultas': [256, 128], 'funcion_activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003}
parametros_mlp2_exp1 = {'capas_ocultas': [256, 128, 64], 'funcion_activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002}
parametros_mlp3_exp1 = {'capas_ocultas': [512, 256, 128], 'funcion_activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002}

MLP1_exp1 = crear_modelo(**parametros_mlp1_exp1, optimizador=Nadam(learning_rate=parametros_mlp1_exp1['learning_rate']), entradas=(28, 28), salidas=10)
MLP2_exp1 = crear_modelo(**parametros_mlp2_exp1, optimizador=Adam(learning_rate=parametros_mlp2_exp1['learning_rate']), entradas=(28, 28), salidas=10)
MLP3_exp1 = crear_modelo(**parametros_mlp3_exp1, optimizador=Adam(learning_rate=parametros_mlp3_exp1['learning_rate']), entradas=(28, 28), salidas=10)

##########RESULTADOS EXPERIMENTO 1###########



resultados_exp1 = entrenar_y_evaluar_modelos([MLP1_exp1, MLP2_exp1, MLP3_exp1], x_train_digits, y_train_digits_one_hot, x_test_digits, y_test_digits_one_hot, "EXP1")

for i, (mediana, acc_clase, matriz_conf, acc_total) in enumerate(resultados_exp1):
    print(f"\nModelo EXP1 MLP{i + 1}:")
    print(f"Mediana del Accuracy Total: {mediana:.4f}")
    print(f"Accuracy Total: {acc_total:.4f}")
    print(f"Accuracy por Clase: {acc_clase}")
    print(f"Matriz de Confusión:\n{matriz_conf}")
'''
ruta_digits = "./emnist-digits"
# Concatena la ruta de los archivos con la carpeta de los dígitos con os
ruta_entrenamiento_imagenes = os.path.join(ruta_digits, "emnist-digits-train-images-idx3-ubyte.gz")
ruta_entrenamiento_etiquetas = os.path.join(ruta_digits, "emnist-digits-train-labels-idx1-ubyte.gz")
ruta_prueba_imagenes = os.path.join(ruta_digits, "emnist-digits-test-images-idx3-ubyte.gz")
ruta_prueba_etiquetas = os.path.join(ruta_digits, "emnist-digits-test-labels-idx1-ubyte.gz")
ruta_mapping = os.path.join(ruta_digits, "emnist-digits-mapping.txt")

x_entrenamiento, y_entrenamiento = cargar_datos(ruta_entrenamiento_imagenes, ruta_entrenamiento_etiquetas)
x_prueba, y_prueba = cargar_datos(ruta_prueba_imagenes, ruta_prueba_etiquetas)

mapping = cargar_mapping(ruta_mapping, "digits")

shape_entrada = (x_entrenamiento.shape[1], )
numero_clases = len(np.unique(y_entrenamiento))

y_entrenamiento_categorico = to_categorical(y_entrenamiento, num_classes=numero_clases)
y_prueba_categorico = to_categorical(y_prueba, num_classes=numero_clases)

parametros_mlp1_exp1 = {'capas_ocultas': [256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}
parametros_mlp2_exp1 = {'capas_ocultas': [256, 128, 64], 'activacion': 'swish', 'loss': 'categorical_crossentropy'}
parametros_mlp3_exp1 = {'capas_ocultas': [512, 256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}

MLP1_exp1 = crear_modelo(
    capas_ocultas=parametros_mlp1_exp1['capas_ocultas'],
    activacion=parametros_mlp1_exp1['activacion'],
    loss=parametros_mlp1_exp1['loss'],
    optimizador=Nadam(learning_rate=0.0002),
    entradas=shape_entrada,
    salidas=numero_clases
)

MLP2_exp1 = crear_modelo(
    capas_ocultas=parametros_mlp2_exp1['capas_ocultas'],
    activacion=parametros_mlp2_exp1['activacion'],
    loss=parametros_mlp2_exp1['loss'],
    optimizador=AdamW(learning_rate=0.00015),
    entradas=shape_entrada,
    salidas=numero_clases
)

MLP3_exp1 = crear_modelo(
    capas_ocultas=parametros_mlp3_exp1['capas_ocultas'],
    activacion=parametros_mlp3_exp1['activacion'],
    loss=parametros_mlp3_exp1['loss'],
    optimizador=Adam(learning_rate=0.0002),
    entradas=shape_entrada,
    salidas=numero_clases
)
resultados = entrenar_y_evaluar_modelos([MLP1_exp1, MLP2_exp1, MLP3_exp1], x_entrenamiento, y_entrenamiento_categorico, x_prueba, y_prueba_categorico)

median_diggits = []
for i, (mediana_acc, acc_clase, matriz_conf, acc_totales) in enumerate(resultados):
    print(f"\nModelo MLP{i + 1}:")
    print(f"Mediana 5 iteraciones: {mediana_acc:.4f}")
    print(f"Accuracy por clase: {acc_clase}")

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(acc_clase)), acc_clase)
    plt.xlabel("Clases")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy por clase para MLP{i + 1}")
    plt.show()

    print(f"Accuracy total del modelo: {acc_totales}")
    print(f"Matriz de confusión:\n{matriz_conf}")
    median_diggits.append(mediana_acc)

print(f"\nMediana Diggits: {np.median(median_diggits):.4f}")

##########EXPERIMENTO 2#############
'''
# Cargar datos de EMNIST Letters
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

# Cargar mapping de etiquetas para letras
# print(os.path.join(ruta_letras, "emnist-letters-mapping.txt"))
mapping_letters = cargar_mapping(os.path.join(ruta_letras, "emnist-letters-mapping.txt"))


# Mostrar imágenes para comprobar la carga correcta
mostrar_imagenes(x_train_letras, y_train_letras, mapping_letters, "Entrenamiento Letters")
mostrar_imagenes(x_test_letras, y_test_letras, mapping_letters, "Prueba Letters")

# Convertir etiquetas a one-hot encoding
y_train_letras_one_hot = to_categorical(y_train_letras, num_classes=27)
y_test_letras_one_hot = to_categorical(y_test_letras, num_classes=27)

# Usar los mejores parámetros identificados
# Usar los mejores parámetros identificados (corrección en el nombre del campo 'funcion_activacion')
parametros_mlp1_exp2 = {'capas_ocultas': [256, 128], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0003}
parametros_mlp2_exp2 = {'capas_ocultas': [128, 64, 32], 'activacion': 'swish', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0004}
parametros_mlp3_exp2 = {'capas_ocultas': [512, 256, 128, 64], 'activacion': 'relu', 'loss': 'categorical_crossentropy', 'learning_rate': 0.0002}

# Crear modelos para el experimento 2 con los mejores parámetros
MLP1_exp2 = crear_modelo(
    capas_ocultas=parametros_mlp1_exp2['capas_ocultas'],
    activacion=parametros_mlp1_exp2['activacion'],
    loss=parametros_mlp1_exp2['loss'],
    optimizador=Adam(learning_rate=parametros_mlp1_exp2['learning_rate']),
    entradas=(28, 28),
    salidas=27
)

MLP2_exp2 = crear_modelo(
    capas_ocultas=parametros_mlp2_exp2['capas_ocultas'],
    activacion=parametros_mlp2_exp2['activacion'],
    loss=parametros_mlp2_exp2['loss'],
    optimizador=Nadam(learning_rate=parametros_mlp2_exp2['learning_rate']),
    entradas=(28, 28),
    salidas=27
)

MLP3_exp2 = crear_modelo(
    capas_ocultas=parametros_mlp3_exp2['capas_ocultas'],
    activacion=parametros_mlp3_exp2['activacion'],
    loss=parametros_mlp3_exp2['loss'],
    optimizador=Adam(learning_rate=parametros_mlp3_exp2['learning_rate']),
    entradas=(28, 28),
    salidas=27
)



##########RESULTADOS EXPERIMENTO 2#########
resultados_exp2 = entrenar_y_evaluar_modelos([MLP1_exp2, MLP2_exp2, MLP3_exp2], x_train_letras, y_train_letras_one_hot, x_test_letras, y_test_letras_one_hot, "EXP2")

for i, (mediana, acc_clase, matriz_conf, acc_total) in enumerate(resultados_exp2):
    print(f"\nModelo EXP2 MLP{i + 1}:")
    print(f"Mediana del Accuracy Total: {mediana:.4f}")
    print(f"Accuracy Total: {acc_total:.4f}")
    print(f"Accuracy por Clase: {acc_clase}")
    print(f"Matriz de Confusión:\n{matriz_conf}")
'''

ruta_letters = "./emnist-letters"
# Concatena la ruta de los archivos con la carpeta de las letras con os
ruta_entrenamiento_imagenes = os.path.join(ruta_letters, "emnist-letters-train-images-idx3-ubyte.gz")
ruta_entrenamiento_etiquetas = os.path.join(ruta_letters, "emnist-letters-train-labels-idx1-ubyte.gz")
ruta_prueba_imagenes = os.path.join(ruta_letters, "emnist-letters-test-images-idx3-ubyte.gz")
ruta_prueba_etiquetas = os.path.join(ruta_letters, "emnist-letters-test-labels-idx1-ubyte.gz")
ruta_mapping = os.path.join(ruta_letters, "emnist-letters-mapping.txt")

x_entrenamiento, y_entrenamiento = cargar_datos(ruta_entrenamiento_imagenes, ruta_entrenamiento_etiquetas)
x_prueba, y_prueba = cargar_datos(ruta_prueba_imagenes, ruta_prueba_etiquetas)

mapping = cargar_mapping(ruta_mapping, "letters")

shape_entrada = (x_entrenamiento.shape[1], )
numero_clases = len(np.unique(y_entrenamiento))

y_entrenamiento = y_entrenamiento - 1
y_prueba = y_prueba - 1

y_entrenamiento_categorico = to_categorical(y_entrenamiento, num_classes=numero_clases)
y_prueba_categorico = to_categorical(y_prueba, num_classes=numero_clases)

parametros_mlp1_exp2 = {'capas_ocultas': [16], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}
parametros_mlp2_exp2 = {'capas_ocultas': [16], 'activacion': 'relu', 'loss': 'categorical_crossentropy'}
parametros_mlp3_exp2 = {'capas_ocultas': [16], 'activacion': 'swish', 'loss': 'categorical_crossentropy'}

MLP1_exp2 = crear_modelo(capas_ocultas=parametros_mlp1_exp2['capas_ocultas'], 
                         activacion=parametros_mlp1_exp2['activacion'], 
                         loss=parametros_mlp1_exp2['loss'], 
                         optimizador=Adam(learning_rate=0.0002), 
                         entradas=shape_entrada, 
                         salidas=numero_clases)

MLP2_exp2 = crear_modelo(capas_ocultas=parametros_mlp2_exp2['capas_ocultas'], 
                         activacion=parametros_mlp2_exp2['activacion'], 
                         loss=parametros_mlp2_exp2['loss'], 
                         optimizador=Adam(learning_rate=0.0003), 
                         entradas=shape_entrada, 
                         salidas=numero_clases)

MLP3_exp2 = crear_modelo(capas_ocultas=parametros_mlp3_exp2['capas_ocultas'], 
                         activacion=parametros_mlp3_exp2['activacion'], 
                         loss=parametros_mlp3_exp2['loss'], 
                         optimizador=Nadam(learning_rate=0.0002), 
                         entradas=shape_entrada, 
                         salidas=numero_clases)

entrenar_y_evaluar_modelos([MLP1_exp2, MLP2_exp2, MLP3_exp2], x_entrenamiento, y_entrenamiento_categorico, x_prueba, y_prueba_categorico)

median_diggits = []
for i, (mediana_acc, acc_clase, matriz_conf, acc_totales) in enumerate(resultados):
    print(f"\nModelo MLP{i + 1}:")
    print(f"Mediana 5 iteraciones: {mediana_acc:.4f}")
    print(f"Accuracy por clase: {acc_clase}")

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(acc_clase)), acc_clase)
    plt.xlabel("Clases")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy por clase para MLP{i + 1}")
    plt.show()

    print(f"Accuracy total del modelo: {acc_totales}")
    print(f"Matriz de confusión:\n{matriz_conf}")
    median_diggits.append(mediana_acc)

print(f"\nMediana Letters: {np.median(median_diggits):.4f}")