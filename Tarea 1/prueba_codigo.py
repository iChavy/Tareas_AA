import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Rutas de las carpetas de datos
ruta_entrenamiento = 'training_set'
ruta_prueba = 'valid_set'

# Función para cargar imágenes y convertirlas a un tamaño unificado
def cargar_imagenes(ruta, tamaño=(64, 64)):
    etiquetas = []
    imagenes = []
    for clase in ['cats', 'dogs']:
        ruta_clase = os.path.join(ruta, clase)
        etiqueta = 0 if clase == 'cats' else 1
        for archivo in os.listdir(ruta_clase):
            ruta_imagen = os.path.join(ruta_clase, archivo)
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            imagen = cv2.resize(imagen, tamaño)
            imagenes.append(imagen.flatten())
            etiquetas.append(etiqueta)
    return np.array(imagenes), np.array(etiquetas)

# Cargar los datos de entrenamiento y prueba
imagenes_entrenamiento, etiquetas_entrenamiento = cargar_imagenes(ruta_entrenamiento)
imagenes_prueba, etiquetas_prueba = cargar_imagenes(ruta_prueba)

# Experimento 1: imágenes unificadas como vectores
# Entrenamiento y prueba con kernel RBF
modelo_rbf = SVC(kernel='rbf')
modelo_rbf.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
predicciones_rbf = modelo_rbf.predict(imagenes_prueba)
accuracy_rbf = accuracy_score(etiquetas_prueba, predicciones_rbf)

# Entrenamiento y prueba con kernel lineal
modelo_lineal = SVC(kernel='linear')
modelo_lineal.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
predicciones_lineal = modelo_lineal.predict(imagenes_prueba)
accuracy_lineal = accuracy_score(etiquetas_prueba, predicciones_lineal)

# Experimento 2: Reducción de dimensionalidad con PCA a 256 componentes
pca = PCA(n_components=256)
imagenes_entrenamiento_pca = pca.fit_transform(imagenes_entrenamiento)
imagenes_prueba_pca = pca.transform(imagenes_prueba)

# SVM con kernel RBF sobre datos reducidos
modelo_rbf_pca = SVC(kernel='rbf')
modelo_rbf_pca.fit(imagenes_entrenamiento_pca, etiquetas_entrenamiento)
predicciones_rbf_pca = modelo_rbf_pca.predict(imagenes_prueba_pca)
accuracy_rbf_pca = accuracy_score(etiquetas_prueba, predicciones_rbf_pca)

# SVM con kernel lineal sobre datos reducidos
modelo_lineal_pca = SVC(kernel='linear')
modelo_lineal_pca.fit(imagenes_entrenamiento_pca, etiquetas_entrenamiento)
predicciones_lineal_pca = modelo_lineal_pca.predict(imagenes_prueba_pca)
accuracy_lineal_pca = accuracy_score(etiquetas_prueba, predicciones_lineal_pca)

# Resultados y Gráficas
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

