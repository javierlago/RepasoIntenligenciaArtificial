# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
import numpy as np
from sklearn.metrics import  mean_absolute_error, r2_score
def scale_datasets(datasets, method='minmax', return_scaler=False):
    """
    Escala múltiples datasets usando el tipo de escalador indicado,
    ajustando sobre el primer dataset.

    Parámetros:
    -----------
    datasets : list or tuple of np.ndarray
        Conjuntos de datos a escalar. El primero se usará para ajustar el scaler.
    
    method : str
        Tipo de escalado: 'minmax' o 'standard'.
    
    return_scaler : bool
        Si True, también devuelve el scaler ajustado.

    Retorna:
    --------
    Lista de datasets escalados (y opcionalmente el scaler)
    """
    if not datasets:
        raise ValueError("No se proporcionaron datasets para escalar.")

    # Selección del escalador
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Método de escalado no válido. Usa 'minmax' o 'standard'.")

    # Ajustar sobre el primer dataset y transformar todos
    scaled = [scaler.fit_transform(datasets[0])]
    for X in datasets[1:]:
        scaled.append(scaler.transform(X))

    if return_scaler:
        return scaled, scaler
    return scaled

def plot_confusion(y_true, y_pred, label_map, title="Matriz de Confusión", figsize=(10,8)):
    """
    Dibuja una matriz de confusión usando etiquetas legibles.
    
    Parámetros:
    ------------
    y_true : array-like
        Etiquetas reales
    y_pred : array-like
        Etiquetas predichas
    label_map : dict
        Diccionario que mapea clase (int) a nombre (str)
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    labels = [label_map[i] for i in sorted(label_map)]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model_metrics_clasisification(y_true, y_pred, average='macro', title='Evaluación del modelo'):
    """
    Calcula y muestra métricas clave de clasificación:
    accuracy, precision, recall y f1-score con media configurable.

    Parámetros:
    ------------
    y_true : array-like
        Etiquetas reales del conjunto de datos.
    y_pred : array-like
        Etiquetas predichas por el modelo.
    average : str
        Tipo de media: 'macro', 'micro', 'weighted' (por defecto: 'macro').
    title : str
        Título opcional que se imprime antes de las métricas.

    Retorna:
    --------
    metrics_dict : dict
        Diccionario con las métricas calculadas.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    print(f"\n📊 {title}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision ({average}): {prec:.4f}")
    print(f"Recall ({average}):    {rec:.4f}")
    print(f"F1-score ({average}):  {f1:.4f}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }


def evaluate_regression_metrics(y_true, y_pred, title='📈 Evaluación del modelo de regresión'):
    """
    Calcula y muestra métricas estándar para problemas de regresión:
    RMSE, MAE y R².

    Parámetros:
    ------------
    y_true : array-like
        Valores reales del conjunto de datos.
    y_pred : array-like
        Valores predichos por el modelo.
    title : str
        Título opcional que se imprime antes de las métricas.

    Retorna:
    --------
    metrics_dict : dict
        Diccionario con las métricas calculadas.
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{title}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_regression_metrics(metrics_dict, title='📊 Métricas del modelo de regresión'):
    """
    Genera un gráfico de barras con las métricas de regresión: RMSE, MAE y R².

    Parámetros:
    ------------
    metrics_dict : dict
        Diccionario con las métricas calculadas (rmse, mae, r2).
    title : str
        Título del gráfico.
    """
    # Separar nombres y valores
    metric_names = ['RMSE', 'MAE', 'R²']
    metric_values = [metrics_dict['rmse'], metrics_dict['mae'], metrics_dict['r2']]

    # Plot
    plt.figure(figsize=(8, 4))
    bars = plt.barh(metric_names, metric_values, color='skyblue')
    plt.xlabel('Valor')
    plt.title(title)

    # Añadir etiquetas en las barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center')

    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()

def split_data(X, y, val_size=0.2, test_size=0.2, random_state=42, stratify=True):
    """
    Divide X e y en conjuntos de entrenamiento, validación y test.

    Parámetros:
    ------------
    X : array-like
        Datos de entrada (features)
    y : array-like
        Etiquetas
    val_size : float
        Proporción del total que se usará para validación
    test_size : float
        Proporción del total que se usará para test
    random_state : int
        Semilla para la división
    stratify : bool
        Si True, mantiene proporciones de clases

    Retorna:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    stratify_labels = y if stratify else None

    # Paso 1: separamos test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )

    # Paso 2: separamos train y val del resto
    val_ratio = val_size / (1 - test_size)  # porque lo aplicamos sobre X_temp
    stratify_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=stratify_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

from torch.utils.data import TensorDataset, DataLoader
import torch

def create_dataloaders(X_train, y_train, 
                       X_val=None, y_val=None, 
                       X_test=None, y_test=None, 
                       batch_size=64, task='regression'):
    """
    Crea dataloaders para regresión o clasificación.
    
    Parámetros:
    ------------
    task : str
        'regression' o 'classification'
    """

    def to_tensor_dataset(X, y):
        def ensure_numpy(arr):
            return arr if isinstance(arr, np.ndarray) else np.array(arr)

        X = ensure_numpy(X)
        y = ensure_numpy(y)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        if task == 'classification':
            y_tensor = torch.tensor(y, dtype=torch.long)
        elif task == 'regression':
            y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        else:
            raise ValueError("task debe ser 'regression' o 'classification'")

        return TensorDataset(X_tensor, y_tensor)


    dataloaders = {
        'train': DataLoader(to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    }

    if X_val is not None and y_val is not None:
        dataloaders['val'] = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    if X_test is not None and y_test is not None:
        dataloaders['test'] = DataLoader(to_tensor_dataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return dataloaders


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu', log_every=1):
    """
    Entrena un modelo de red neuronal en PyTorch y evalúa en cada epoch.

    Parámetros:
    ------------
    model : nn.Module
        Modelo a entrenar
    train_loader : DataLoader
        Dataloader de entrenamiento
    val_loader : DataLoader
        Dataloader de validación
    criterion : función de pérdida
    optimizer : optimizador
    epochs : int
        Número de épocas
    device : str
        'cpu' o 'cuda'
    log_every : int
        Cada cuántas epochs se imprime el log (default: 1)

    Retorna:
    --------
    metrics : dict
        Diccionario con listas de métricas por epoch
    """

    model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []

    print(f"\n🚀 Comenzando entrenamiento ({epochs} epochs)...\n")

    for epoch in range(epochs):
        # --- Entrenamiento ---
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Métricas de entrenamiento
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds, average='macro')
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # --- Validación ---
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # Guardar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Mostrar log
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

    print("\n✅ Entrenamiento completado.")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_recalls': train_recalls,
        'val_recalls': val_recalls,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s
    }
    
    
    


def train_regression_model(model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu', log_every=1):
    model.to(device)

    train_losses, val_losses = [], []
    train_r2s, val_r2s = [], []

    print(f"\n🚀 Entrenando red neuronal de regresión ({epochs} epochs)...\n")

    for epoch in range(epochs):
        # --- Entrenamiento ---
        model.train()
        running_loss = 0.0
        all_preds, all_targets = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_r2 = r2_score(all_targets, all_preds)

        # --- Validación ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_r2 = r2_score(val_targets, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)

        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")

    print("\n✅ Entrenamiento completado.")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }


def plot_training_metrics(metrics_dict, title='Entrenamiento del modelo'):
    """
    Dibuja curvas de pérdida, accuracy y f1-score durante el entrenamiento.

    Parámetros:
    ------------
    metrics_dict : dict
        Diccionario retornado por train_model()
    title : str
        Título del gráfico
    """
    epochs = range(1, len(metrics_dict['train_losses']) + 1)

    plt.figure(figsize=(18, 5))

    # Gráfico de pérdida
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics_dict['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics_dict['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Gráfico de accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics_dict['train_accs'], label='Train Acc')
    plt.plot(epochs, metrics_dict['val_accs'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Gráfico de F1-score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics_dict['train_f1s'], label='Train F1')
    plt.plot(epochs, metrics_dict['val_f1s'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('F1-score')
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def validate_model(model, dataloader, average='macro'):
    """
    Evalúa un modelo PyTorch en un dataloader (validación o test).

    Detecta automáticamente si hay GPU y calcula métricas comunes.

    Parámetros:
    ------------
    model : nn.Module
        Modelo ya entrenado
    dataloader : DataLoader
        Conjunto de datos sobre el que validar
    average : str
        Tipo de media: 'macro', 'micro', 'weighted' (default: 'macro')

    Retorna:
    --------
    metrics : dict
        Diccionario con accuracy, precision, recall y f1
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average=average)
    rec = recall_score(all_labels, all_preds, average=average)
    f1 = f1_score(all_labels, all_preds, average=average)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
def plot_comparison_metrics(metrics1, metrics2, label1='Modelo 1', label2='Modelo 2'):
    """
    Compara dos modelos en un gráfico de barras lado a lado.

    Parámetros:
    ------------
    metrics1 : dict
        Métricas del primer modelo
    metrics2 : dict
        Métricas del segundo modelo
    label1 : str
        Nombre del primer modelo
    label2 : str
        Nombre del segundo modelo
    """
    import numpy as np

    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values1 = [metrics1['accuracy'], metrics1['precision'], metrics1['recall'], metrics1['f1_score']]
    values2 = [metrics2['accuracy'], metrics2['precision'], metrics2['recall'], metrics2['f1_score']]

    x = np.arange(len(labels))  # etiquetas
    width = 0.35

    plt.figure(figsize=(9, 5))
    bars1 = plt.bar(x - width/2, values1, width, label=label1, color='lightgreen')
    bars2 = plt.bar(x + width/2, values2, width, label=label2, color='lightcoral')

    plt.ylabel('Valor')
    plt.ylim(0, 1)
    plt.title('Comparación de modelos')
    plt.xticks(x, labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Añadir valores encima de las barras
    for bar in bars1 + bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_model_metrics(metrics, model_name='Modelo'):
    """
    Muestra las métricas (accuracy, precision, recall, f1) de un modelo en un gráfico de barras.

    Parámetros:
    ------------
    metrics : dict
        Diccionario con las métricas (output de validate_model o evaluate_model_metrics)
    model_name : str
        Nombre del modelo para el título del gráfico
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 1)
    plt.title(f'Métricas de {model_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Mostrar los valores arriba de las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    
    
def plot_training_curves(metrics, title_prefix='Red Neuronal'):
    """
    Dibuja las curvas de entrenamiento y validación de pérdida y R².

    Parámetros:
    ------------
    metrics : dict
        Diccionario de métricas devuelto por train_regression_model().
    title_prefix : str
        Texto que se antepone a los títulos de los gráficos.
    """

    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics['val_losses'], label='Val Loss')
    plt.title(f'{title_prefix} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    # R²
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_r2s'], label='Train R²')
    plt.plot(epochs, metrics['val_r2s'], label='Val R²')
    plt.title(f'{title_prefix} - R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def compare_models_plot(nn_metrics, rf_metrics):
    """
    Dibuja una comparación de métricas entre Red Neuronal y Random Forest.
    """
    metric_names = ['RMSE', 'MAE', 'R²']
    nn_values = [nn_metrics['rmse'], nn_metrics['mae'], nn_metrics['r2']]
    rf_values = [rf_metrics['rmse'], rf_metrics['mae'], rf_metrics['r2']]

    x = range(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], rf_values, width=width, label='Random Forest', color='skyblue')
    plt.bar([i + width/2 for i in x], nn_values, width=width, label='Red Neuronal', color='salmon')

    plt.xticks(x, metric_names)
    plt.ylabel("Valor")
    plt.title("Comparación de métricas en Test")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    


def validate_regression_model(model, dataloader):
    """
    Evalúa un modelo PyTorch de regresión en un dataloader.

    Retorna RMSE, MAE y R².

    Parámetros:
    ------------
    model : nn.Module
        Modelo ya entrenado
    dataloader : DataLoader
        Datos de validación o test

    Retorna:
    --------
    metrics : dict
        Diccionario con RMSE, MAE y R²
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(batch_y.numpy())

    # Calcular métricas
    rmse = root_mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
def compare_models_plot(nn_metrics, rf_metrics, title='Comparación en Test Set'):
    """
    Dibuja una comparación de métricas entre Red Neuronal y Random Forest.
    """
    metric_names = ['RMSE', 'MAE', 'R²']
    nn_values = [nn_metrics['rmse'], nn_metrics['mae'], nn_metrics['r2']]
    rf_values = [rf_metrics['rmse'], rf_metrics['mae'], rf_metrics['r2']]

    x = range(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], rf_values, width=width, label='Random Forest', color='skyblue')
    plt.bar([i + width/2 for i in x], nn_values, width=width, label='Red Neuronal', color='salmon')

    for i, val in enumerate(rf_values):
        plt.text(i - width/2, val + 0.01, f'{val:.2f}', ha='center')
    for i, val in enumerate(nn_values):
        plt.text(i + width/2, val + 0.01, f'{val:.2f}', ha='center')

    plt.xticks(x, metric_names)
    plt.ylabel("Valor")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()