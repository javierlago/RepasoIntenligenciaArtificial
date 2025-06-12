from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def classification_metrics(y_true, y_pred):
    """
    Calcula métricas de clasificación dadas etiquetas reales y predichas.

    Devuelve:
    - accuracy, precision, recall, f1 (macro)
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def regression_metrics(y_true, y_pred):
    """
    Calcula métricas de regresión dadas salidas reales y predichas.

    Devuelve:
    - MSE, RMSE, MAE, R²
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
