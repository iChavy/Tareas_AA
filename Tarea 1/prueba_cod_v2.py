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
import time

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
'''
# ==============================
# EXPERIMENTO 1: OPTIMIZACIÓN DEL MODELO SVM SIN PCA
# ==============================

# Definir el rango de parámetros a probar para el modelo SVM sin PCA
parametros_sin_pca = {
    #'svm__C': [0.1, 0.5, 1, 5],
    #'svm__gamma': [1, 0.1, 0.01, 0.001],
    #'svm__C': [1, 5, 10, 50, 100],
    #'svm__gamma': [0.0001, 0.00001, 0.00005, 0.000005],
    #'svm__C': [0.8, 1, 1.5, 2, 3],
    #'svm__gamma': [0.001, 0.005, 0.00005],
    #'svm__C': [0.5, 0.8, 1, 1.2, 1.5, 2, 5, 10],
    #'svm__gamma': [0.001, 0.0008, 5e-05, 3e-05],
    #'svm__C': [0.9, 1, 1.1, 1.2, 1.5],
    #'svm__gamma': [0.0007, 0.0008, 0.0009, 0.001],
    #'svm__C': [0.8, 0.9, 1, 1.1, 1.2],
    #'svm__gamma': [0.0006, 0.00065, 0.0007, 0.00075],
    #'svm__C': [0.95, 1, 1.02, 1.05],
    #'svm__gamma': [0.0006, 0.00062, 0.000625, 0.00063, 0.00065],
    'svm__C': [1],
    #'svm__gamma': [0.0006, 0.00062, 0.000625, 0.0006251, 0.0006252, 0.0006253, 0.0006254, 0.0006255, 0.0006256, 0.0006257, 0.0006258, 0.0006259, 0.000626],
    'svm__gamma': [0.000625, 0.0006251],
    'svm__kernel': ['rbf', 'linear'],
    'svm__tol': [1e-3],
    'svm__shrinking': [True]
}

# Pipeline para SVM sin PCA
pipeline_svm_sin_pca = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('svm', SVC())
])

# GridSearchCV para encontrar los mejores parámetros sin PCA
grid_search_sin_pca = GridSearchCV(pipeline_svm_sin_pca, parametros_sin_pca, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
# CV es validacion cruzada, si es 3 son 3 folds
    #En GridSearchCV, .fit() no solo ajusta el modelo una vez, sino que busca automáticamente la mejor combinación de hiperparámetros. Esto implica:

    #Entrenar el modelo con múltiples combinaciones de parámetros en un proceso de validación cruzada.
    #Al finalizar, selecciona la mejor combinación y ajusta el modelo final con esta configuración óptima.

grid_search_sin_pca.fit(imagenes_entrenamiento, etiquetas_entrenamiento)

# Mostrar los mejores parámetros y precisión obtenidos sin PCA
print("Mejores parámetros sin PCA:", grid_search_sin_pca.best_params_)
print("Mejor precisión sin PCA (validación cruzada):", grid_search_sin_pca.best_score_)

#### imprime el accuaracy con sus parametros
# Obtener y mostrar los resultados detallados de cada combinación de parámetros
print("\nResultados detallados de cada combinación de hiperparámetros:")
resultados_cv = grid_search_sin_pca.cv_results_

# Abrir archivo en modo adjunto para guardar los resultados con codificación UTF-8
with open("accuracy.txt", "a", encoding="utf-8") as file:
    for i in range(len(resultados_cv['params'])):
        parametros = resultados_cv['params'][i]
        mean_accuracy = resultados_cv['mean_test_score'][i]  # Promedio de accuracy en la validación cruzada
        std_accuracy = resultados_cv['std_test_score'][i]    # Desviación estándar del accuracy
        tiempo_total = resultados_cv['mean_fit_time'][i]     # Tiempo promedio de ajuste en segundos

        # Imprimir en consola
        print(f"[CV] Parámetros: {parametros}; Tiempo total: {tiempo_total:.2f} min; Accuracy promedio: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        # Guardar en archivo con codificación UTF-8
        file.write(f"[CV] Parámetros: {parametros}; Tiempo total: {tiempo_total:.2f} min; Accuracy promedio: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")

# Evaluar el mejor modelo sin PCA en el conjunto de prueba
mejor_modelo_sin_pca = grid_search_sin_pca.best_estimator_
predicciones_sin_pca = mejor_modelo_sin_pca.predict(imagenes_prueba)
accuracy_sin_pca = accuracy_score(etiquetas_prueba, predicciones_sin_pca)

# Imprimir y guardar el accuracy del mejor modelo
print("Accuracy en conjunto de prueba sin PCA:", accuracy_sin_pca)
print("Reporte de clasificación para el mejor modelo sin PCA:")
print(classification_report(etiquetas_prueba, predicciones_sin_pca, target_names=['Cats', 'Dogs']))

# Guardar también el mejor resultado en el archivo con codificación UTF-8
with open("accuracy.txt", "a", encoding="utf-8") as file:
    file.write("\nMejores parámetros sin PCA:\n")
    file.write(str(grid_search_sin_pca.best_params_) + "\n")
    file.write(f"Mejor precisión sin PCA (validación cruzada): {grid_search_sin_pca.best_score_:.4f}\n")
    file.write(f"Accuracy en conjunto de prueba sin PCA: {accuracy_sin_pca:.4f}\n")

print("fin experimento 1, archivo listo")
'''
# ==============================
# EXPERIMENTO 2: OPTIMIZACIÓN DEL MODELO SVM CON PCA
# ==============================

# Definir el rango de parámetros a probar para el modelo SVM con PCA
parametros_con_pca = {
    #'svm__C': [0.1, 1, 10, 100],
    #'svm__gamma': [1, 0.1, 0.01, 0.001],
    #'svm__C': [5, 10, 12, 15],
    #'svm__gamma': [0.0008, 0.001, 0.0012, 0.0015],
    #'svm__C': [10, 15, 18, 20],
    #'svm__gamma': [0.0006, 0.0008, 0.001, 0.0012],
    #'svm__C': [16, 18, 20, 22],
    #'svm__gamma': [0.0006, 0.0007, 0.0008, 0.0009],
    #'svm__C': [15, 16, 17],
    #'svm__gamma': [0.00075, 0.0008, 0.00085],
    #'svm__C': [16.5, 17, 17.5],
    #'svm__gamma': [0.00078, 0.0008, 0.00082],
    #'svm__C': [16.8, 17, 17.2],
    #'svm__gamma': [0.00081, 0.00082, 0.00083],
    #'svm__C': [16.9, 17, 17.1],
    #'svm__gamma': [0.000815, 0.00082, 0.000825],
    #'svm__C': [0.1, 1, 10, 50, 100],     
    #'svm__gamma': [0.01, 0.001, 0.0001, 0.0005, 'scale', 'auto'],
    #'svm__C': [0.05, 0.08, 0.1, 0.2, 0.5],        # Alrededor del valor óptimo encontrado
    #'svm__gamma': [5e-5, 0.00008, 0.0001, 0.00015],
    #'svm__C': [0.4, 0.5, 0.6, 0.7],         # Ajustando alrededor de 0.5
    #'svm__gamma': [4e-05, 5e-05, 6e-05, 7e-05],
    #'svm__C': [0.35, 0.4, 0.45, 0.5],
    #'svm__gamma': [5e-05, 6e-05, 6.5e-05, 7e-05],
    #'svm__C': [0.38, 0.4, 0.42, 0.43],
    #'svm__gamma': [6.3e-05, 6.5e-05, 6.7e-05, 7e-05], ### al parecer son los mejores valores
    #'svm__C': [0.36, 0.38, 0.39, 0.40],
    #'svm__gamma': [6.6e-05, 6.7e-05, 6.8e-05, 6.9e-05],
    #'svm__C': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    #'svm__gamma': [5e-05, 6.5e-05, 7.5e-05, 0.0001, 0.0002, 0.0003],
    #'svm__C': [0.25, 0.35, 0.4, 0.45, 0.6, 0.7],
    #'svm__gamma': [4e-05, 5.5e-05, 7e-05, 8.5e-05, 0.00012, 0.00015],
    #'svm__C': [0.5, 0.75, 1.0],
    #'svm__gamma': [3.5e-05, 4.5e-05, 6.5e-05, 7.5e-05],
    'svm__C': [0.35, 0.4, 0.45],
    'svm__gamma': [6.5e-05, 6.7e-05, 7e-05],
    'svm__kernel': ['sigmoid']
}

# Pipeline para SVM con PCA
pipeline_svm_con_pca = Pipeline([
    ('escalador', StandardScaler()),  # Escala los datos
    ('pca', PCA(n_components=256)),   # Reducción de dimensionalidad
    ('svm', SVC())
])

# GridSearchCV para encontrar los mejores parámetros con PCA
grid_search_con_pca = GridSearchCV(pipeline_svm_con_pca, parametros_con_pca, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Medir el tiempo de ejecución
start_time = time.time()
grid_search_con_pca.fit(imagenes_entrenamiento, etiquetas_entrenamiento)
end_time = time.time()

# Mostrar y guardar los mejores parámetros y precisión obtenidos con PCA
mejores_parametros = grid_search_con_pca.best_params_
mejor_precision = grid_search_con_pca.best_score_
print("Mejores parámetros con PCA:", mejores_parametros)
print("Mejor precisión con PCA (validación cruzada):", mejor_precision)

# Evaluar el mejor modelo con PCA en el conjunto de prueba
mejor_modelo_con_pca = grid_search_con_pca.best_estimator_
predicciones_con_pca = mejor_modelo_con_pca.predict(imagenes_prueba)
accuracy_con_pca = accuracy_score(etiquetas_prueba, predicciones_con_pca)

print("Accuracy en conjunto de prueba con PCA:", accuracy_con_pca)
print("Reporte de clasificación para el mejor modelo con PCA:")
print(classification_report(etiquetas_prueba, predicciones_con_pca, target_names=['Cats', 'Dogs']))

# Guardar los resultados en el archivo exp2.txt
file_path = "exp2.txt"
with open(file_path, "a", encoding="utf-8") as file:  # Abre en modo adjunto (agregar)
    file.write("Parámetros probados con PCA:\n")
    resultados_cv = grid_search_con_pca.cv_results_
    
    for i in range(len(resultados_cv['params'])):
        parametros = resultados_cv['params'][i]
        mean_accuracy = resultados_cv['mean_test_score'][i]
        std_accuracy = resultados_cv['std_test_score'][i]
        tiempo_total = resultados_cv['mean_fit_time'][i]
        
        # Guardar cada combinación de parámetros y el tiempo
        file.write(f"[CV] Parámetros: {parametros}; Tiempo total: {tiempo_total:.2f} s; Accuracy promedio: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n")
    
    # Guardar los mejores resultados al final
    file.write("\nMejores parámetros con PCA:\n")
    file.write(str(mejores_parametros) + "\n")
    file.write(f"Mejor precisión con PCA (validación cruzada): {mejor_precision:.4f}\n")
    file.write(f"Accuracy en conjunto de prueba con PCA: {accuracy_con_pca:.4f}\n")
    file.write(f"Tiempo de ejecución total: {(end_time - start_time) / 60:.2f} minutos\n")
    
print("Experimento 2 completado, archivo 'exp2.txt' listo")

'''
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
'''