import scipy.io
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carga el archivo .mat
data = scipy.io.loadmat('emnist-letters.mat')

# Carga los vectores de red convolucional de entrenamiento y prueba
train_embeddings = pd.read_csv('train_embeddings.csv')
test_embeddings = pd.read_csv('test_embeddings.csv')

# Elimina la columna de label de las imágenes
train_embeddings = train_embeddings.iloc[:, :-1]
test_embeddings = test_embeddings.iloc[:, :-1]

# Seed
np.random.seed(0)

data = data['dataset']

# Separa los conjuntos de datos de entrenamiento y prueba
train = data['train']
test = data['test']

train_images = train[0][0]['images'][0][0]
train_labels = train[0][0]['labels'][0][0]

test_images = test[0][0]['images'][0][0]
test_labels = test[0][0]['labels'][0][0]

# Cantidad de clases
n_labels = 26

# Arreglos para almacenar los datos de entrenamiento y prueba seleccionados
data_train_images = []
data_test_images = []
data_train_labels = np.array([])
data_test_labels = np.array([])

# Arreglos para almacenar los datos de entrenamiento y prueba seleccionados de los vectores de red convolucional
data_train_embeddings = []
data_test_embeddings = []

# Seleccionar 1000 datos de entrenamiento y 100 datos de prueba por cada clase
for label in range(1, n_labels + 1):
    # Obtener los índices de las imágenes de la clase actual
    train_index = np.where(train_labels == label)[0]
    test_index = np.where(test_labels == label)[0]
    
    # Obtiene los índices de 1000 datos de entrenamiento y 100 datos de prueba al azar
    random_train_index = np.random.choice(train_index, size=1000, replace=False)
    random_test_index = np.random.choice(test_index, size=100, replace=False)

    # Obtiene las imágenes de entrenamiento y prueba seleccionadas
    train_selected = [train_images[i] for i in random_train_index]
    test_selected = [test_images[i] for i in random_test_index]
    
    # Obtiene los vectores de red convolucional de entrenamiento y prueba seleccionados
    train_embeddings_selected = [train_embeddings.iloc[i] for i in random_train_index]
    test_embeddings_selected = [test_embeddings.iloc[i] for i in random_test_index]

    # Almacena las imágenes seleccionadas
    data_train_images.extend(train_selected)
    data_test_images.extend(test_selected)
    
    # Almacena los vectores de red convolucional seleccionados
    data_train_embeddings.extend(train_embeddings_selected)
    data_test_embeddings.extend(test_embeddings_selected)

    # Almacena las etiquetas de las imágenes seleccionadas
    data_train_labels = np.append(data_train_labels, np.repeat(label, 1000))
    data_test_labels = np.append(data_test_labels, np.repeat(label, 100))

# Modelos SVM
models = [
    Pipeline([
        ('svm', SVC(kernel='rbf'))
    ]),
    Pipeline([
        ('pca', PCA(n_components=128)),
        ('svm', SVC(kernel='rbf'))
    ]),
    Pipeline([
        ('svm', SVC(kernel='rbf'))
    ])
]

# Experimentos realizados
models_name = ["Experimento 1", "Experimento 2", "Experimento 3"]

model_index = 0
# Lista para almacenar las matrices de confusión por modelo
confusion_matrix_models = []

# Entrenamiento y evaluación de los modelos
for model in models:
    # Conjunto de entrenamiento con imágenes como vectores
    if model_index != 2:
        model.fit(data_train_images, data_train_labels)
        predictions = model.predict(data_test_images)
    
    # Conjunto de entrenamiento con vectores de red convolucional
    else:
        model.fit(data_train_embeddings, data_train_labels)
        predictions = model.predict(data_test_embeddings)
    
    # Matriz de confusión    
    confusion_matrix_model = confusion_matrix(predictions, data_test_labels)
    confusion_matrix_models.append(confusion_matrix_model)
    
    # Reporte de clasificación
    report = classification_report(predictions, data_test_labels)
    
    # Accuracy total
    accuracy = np.mean(predictions == data_test_labels)
    
    print(models_name[model_index])
    print("Reporte de clasificación: ")
    print(report)
    print("Accuracy total: ", accuracy)
    model_index += 1

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gráficos de las matrices de confusión
for i, (matrix, model_name) in enumerate(zip(confusion_matrix_models, models_name)):
    ax = sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', ax=axes[i])
    ax.set_title(model_name)
    ax.set_xlabel('Etiqueta Predicha')
    ax.set_ylabel('Etiqueta Real')

plt.tight_layout()
plt.show()

accuracies_diagonal = []
# Selección de la diagonal de la matriz de confusión para calcular el accuracy por clase
for confusion_matrix_model in confusion_matrix_models:
    diagonal = np.diag(confusion_matrix_model)
    total = confusion_matrix_model.sum(axis=1)
    accuracy_diagonal_model = diagonal / total
    accuracies_diagonal.append(accuracy_diagonal_model)


fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(1, 27)
bar_width = 0.25
opacity = 0.8

# Gráfico de barras comparando el accuracy de los modelos
for i, (accuracy_diagonal_model, model_name) in enumerate(zip(accuracies_diagonal, models_name)):
    plt.bar(index + i * bar_width, accuracy_diagonal_model, bar_width, alpha=opacity, label=model_name)

plt.xlabel('Clase')
plt.ylabel('Accuracy')
plt.title('Accuracy por clase')
plt.xticks(index + bar_width, range(1, 27))
plt.legend()

plt.tight_layout()
plt.show()