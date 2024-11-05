import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Rutas de las carpetas de datos
ruta_entrenamiento = 'training_set'
ruta_prueba = 'valid_set'

# Función para cargar imágenes y convertirlas a un tamaño unificado
def cargar_imagenes(ruta, tamanyo=(64, 64)):
    etiquetas = []
    imagenes = []
    for clase in ['cats', 'dogs']:
        ruta_clase = os.path.join(ruta, clase)
        etiqueta = 0 if clase == 'cats' else 1
        for archivo in os.listdir(ruta_clase):
            # Ignora todos los archivos que no sean .jpg
            if not archivo.lower().endswith(('.jpg')):
                continue
            
            ruta_imagen = os.path.join(ruta_clase, archivo)
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            
            # Comprobar si la imagen se ha cargado correctamente
            if imagen is None:
                print(f"No se pudo cargar la imagen: {ruta_imagen}")
                continue
            
            # Redimensionar la imagen
            imagen = cv2.resize(imagen, tamanyo)
            imagenes.append(imagen.flatten())
            etiquetas.append(etiqueta)
            
    print("Imágenes cargadas correctamente")
    return np.array(imagenes), np.array(etiquetas)


# Cargar los datos de entrenamiento y prueba
imagenes_entrenamiento, etiquetas_entrenamiento = cargar_imagenes(ruta_entrenamiento)
imagenes_prueba, etiquetas_prueba = cargar_imagenes(ruta_prueba)

# ==============================
# EXPERIMENTO 1: SVM CON KERNEL RBF Y LINEAL
# ==============================

# Pipeline para SVM con kernel RBF sin PCA
pipeline_rbf = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('svm', SVC(kernel='rbf', gamma='scale'))
])

# Entrenamiento y evaluación con kernel RBF
pipeline_rbf.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
predicciones_rbf = pipeline_rbf.predict(imagenes_prueba)
accuracy_rbf = accuracy_score(etiquetas_prueba, predicciones_rbf)

# Pipeline para SVM con kernel lineal sin PCA
pipeline_lineal = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('svm', SVC(kernel='linear'))
])

# Entrenamiento y evaluación con kernel lineal
pipeline_lineal.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
predicciones_lineal = pipeline_lineal.predict(imagenes_prueba)
accuracy_lineal = accuracy_score(etiquetas_prueba, predicciones_lineal)

print("Fin del experimento 1")

# ==============================
# EXPERIMENTO 2: REDUCCIÓN DE DIMENSIONALIDAD CON PCA A 256 COMPONENTES
# ==============================

# Pipeline para SVM con kernel RBF y PCA
pipeline_rbf_pca = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('pca', PCA(n_components=256)),  # Reducción de dimensionalidad
    ('svm', SVC(kernel='rbf', gamma='scale'))
])

# Entrenamiento y evaluación con kernel RBF con PCA
pipeline_rbf_pca.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
predicciones_rbf_pca = pipeline_rbf_pca.predict(imagenes_prueba)
accuracy_rbf_pca = accuracy_score(etiquetas_prueba, predicciones_rbf_pca)

# Pipeline para SVM con kernel lineal y PCA
pipeline_lineal_pca = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('pca', PCA(n_components=256)),  # Reducción de dimensionalidad
    ('svm', SVC(kernel='linear'))
])

# Entrenamiento y evaluación con kernel lineal con PCA
pipeline_lineal_pca.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
predicciones_lineal_pca = pipeline_lineal_pca.predict(imagenes_prueba)
accuracy_lineal_pca = accuracy_score(etiquetas_prueba, predicciones_lineal_pca)

print("Fin del experimento 2")

# ==============================
# RESULTADOS Y GRÁFICAS
# ==============================

# Resultados en un diccionario
resultados = {
    'Modelo': ['SVM RBF', 'SVM Lineal', 'SVM RBF con PCA', 'SVM Lineal con PCA'],
    'Accuracy Total': [accuracy_rbf, accuracy_lineal, accuracy_rbf_pca, accuracy_lineal_pca]
}

# Crear DataFrame para mostrar en tabla
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

# Gráfico de barras para comparar accuracy
plt.figure(figsize=(10, 6))
plt.bar(df_resultados['Modelo'], df_resultados['Accuracy Total'])
plt.xlabel('Modelo')
plt.ylabel('Accuracy')
plt.title('Accuracy total por modelo')
plt.show()

# Reporte de clasificación para cada modelo
print("Reporte de clasificación SVM RBF:")
print(classification_report(etiquetas_prueba, predicciones_rbf, target_names=['Cats', 'Dogs']))
print("Reporte de clasificación SVM Lineal:")
print(classification_report(etiquetas_prueba, predicciones_lineal, target_names=['Cats', 'Dogs']))
print("Reporte de clasificación SVM RBF con PCA:")
print(classification_report(etiquetas_prueba, predicciones_rbf_pca, target_names=['Cats', 'Dogs']))
print("Reporte de clasificación SVM Lineal con PCA:")
print(classification_report(etiquetas_prueba, predicciones_lineal_pca, target_names=['Cats', 'Dogs']))

print("Fin de todos los experimentos")
