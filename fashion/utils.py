# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score

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

def evaluate_model_metrics(y_true, y_pred, average='macro', title='Evaluación del modelo'):
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

def create_dataloaders(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, batch_size=64):
    """
    Convierte arrays de NumPy en TensorDatasets y devuelve DataLoaders para entrenamiento, validación y test.

    Parámetros:
    ------------
    X_train, y_train : array-like
        Datos de entrenamiento.
    X_val, y_val : array-like, opcional
        Datos de validación.
    X_test, y_test : array-like, opcional
        Datos de test.
    batch_size : int
        Tamaño de los batches (por defecto 64).

    Retorna:
    --------
    dataloaders : dict
        Diccionario con los dataloaders: 'train', 'val', 'test'
    """

    def to_tensor_dataset(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

    dataloaders = {}

    # Entrenamiento
    train_dataset = to_tensor_dataset(X_train, y_train)
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validación
    if X_val is not None and y_val is not None:
        val_dataset = to_tensor_dataset(X_val, y_val)
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Test
    if X_test is not None and y_test is not None:
        test_dataset = to_tensor_dataset(X_test, y_test)
        dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    
    
    
    import matplotlib.pyplot as plt

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
import matplotlib.pyplot as plt

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
