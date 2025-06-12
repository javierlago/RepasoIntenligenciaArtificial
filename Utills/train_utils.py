import torch
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score, r2_score, accuracy_score


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

    print(f"\n?? Comenzando entrenamiento ({epochs} epochs)...\n")

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

    print("\n? Entrenamiento completado.")

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


def fit_regression(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    """
    Entrena una red de regresión durante varias épocas con validación.

    Devuelve:
    - Diccionario con listas de métricas por época:
        - train_losses, val_losses
        - train_r2s, val_r2s
    """
    train_losses, val_losses = [], []
    train_r2s, val_r2s = [], []

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        r2 = r2_score(y_true, y_pred)
        train_losses.append(epoch_loss)
        train_r2s.append(r2)

        # Validación
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_r2 = r2_score(y_true, y_pred)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }
