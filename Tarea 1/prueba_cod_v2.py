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
from sklearn.model_selection import GridSearchCV

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
# EXPERIMENTO 1: OPTIMIZACIÓN DEL MODELO SVM SIN PCA
# ==============================

# Definir el rango de parámetros a probar para el modelo SVM sin PCA
parametros_sin_pca = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [1, 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}

# Pipeline para SVM sin PCA
pipeline_svm_sin_pca = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('svm', SVC())
])

# GridSearchCV para encontrar los mejores parámetros sin PCA
grid_search_sin_pca = GridSearchCV(pipeline_svm_sin_pca, parametros_sin_pca, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_sin_pca.fit(imagenes_entrenamiento, etiquetas_entrenamiento)

# Mostrar los mejores parámetros y precisión obtenidos sin PCA
print("Mejores parámetros sin PCA:", grid_search_sin_pca.best_params_)
print("Mejor precisión sin PCA (validación cruzada):", grid_search_sin_pca.best_score_)

# Evaluar el mejor modelo sin PCA en el conjunto de prueba
mejor_modelo_sin_pca = grid_search_sin_pca.best_estimator_
predicciones_sin_pca = mejor_modelo_sin_pca.predict(imagenes_prueba)
accuracy_sin_pca = accuracy_score(etiquetas_prueba, predicciones_sin_pca)

print("Accuracy en conjunto de prueba sin PCA:", accuracy_sin_pca)
print("Reporte de clasificación para el mejor modelo sin PCA:")
print(classification_report(etiquetas_prueba, predicciones_sin_pca, target_names=['Cats', 'Dogs']))
print("fin experimento 1")

# ==============================
# EXPERIMENTO 2: OPTIMIZACIÓN DEL MODELO SVM CON PCA
# ==============================

# Definir el rango de parámetros a probar para el modelo SVM con PCA
parametros_con_pca = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [1, 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}

# Pipeline para SVM con PCA
pipeline_svm_con_pca = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('pca', PCA(n_components=256)),   # Reducción de dimensionalidad
    ('svm', SVC())
])

# GridSearchCV para encontrar los mejores parámetros con PCA
grid_search_con_pca = GridSearchCV(pipeline_svm_con_pca, parametros_con_pca, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_con_pca.fit(imagenes_entrenamiento, etiquetas_entrenamiento)

# Mostrar los mejores parámetros y precisión obtenidos con PCA
print("Mejores parámetros con PCA:", grid_search_con_pca.best_params_)
print("Mejor precisión con PCA (validación cruzada):", grid_search_con_pca.best_score_)

# Evaluar el mejor modelo con PCA en el conjunto de prueba
mejor_modelo_con_pca = grid_search_con_pca.best_estimator_
predicciones_con_pca = mejor_modelo_con_pca.predict(imagenes_prueba)
accuracy_con_pca = accuracy_score(etiquetas_prueba, predicciones_con_pca)

print("Accuracy en conjunto de prueba con PCA:", accuracy_con_pca)
print("Reporte de clasificación para el mejor modelo con PCA:")
print(classification_report(etiquetas_prueba, predicciones_con_pca, target_names=['Cats', 'Dogs']))

# ==============================
# RESULTADOS Y GRÁFICAS
# ==============================

# Resultados en un diccionario
resultados = {
    'Modelo': ['SVM Optimizado sin PCA', 'SVM Optimizado con PCA'],
    'Accuracy Total': [accuracy_sin_pca, accuracy_con_pca]
}

# Crear DataFrame para mostrar en tabla
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

# Gráfico de barras para comparar accuracy
plt.figure(figsize=(10, 6))
plt.bar(df_resultados['Modelo'], df_resultados['Accuracy Total'])
plt.xlabel('Modelo')
plt.ylabel('Accuracy')
plt.title('Accuracy total por modelo (Optimizado)')
plt.show()

print("Optimización completa.")