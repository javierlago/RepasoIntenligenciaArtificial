import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_loss_curve(train_losses, val_losses=None):
    """
    Plotea la curva de p�rdida (loss) durante el entrenamiento.

    Par�metros:
    - train_losses: lista de p�rdidas del conjunto de entrenamiento por �poca.
    - val_losses: (opcional) lista de p�rdidas del conjunto de validaci�n por �poca.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_accuracy(train_accs, val_accs=None):
    """
    Plotea la curva de precisi�n (accuracy) durante el entrenamiento.

    Par�metros:
    - train_accs: lista de accuracies del conjunto de entrenamiento por �poca.
    - val_accs: (opcional) lista de accuracies del conjunto de validaci�n por �poca.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train Accuracy')
    if val_accs:
        plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Muestra la matriz de confusi�n a partir de los valores reales y predichos.

    Par�metros:
    - y_true: etiquetas verdaderas.
    - y_pred: etiquetas predichas por el modelo.
    - class_names: lista de nombres de clases (opcional).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_true, y_pred):
    """
    Muestra una gr�fica de dispersi�n para comparar valores reales y predichos
    en un problema de regresi�n.

    Par�metros:
    - y_true: valores reales del target.
    - y_pred: valores predichos por el modelo.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression: True vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
