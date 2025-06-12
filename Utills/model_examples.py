import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Modelo 1: CNN optimizado para CIFAR-10
# --------------------------------------------------
class CIFAR10_CNN(nn.Module):
    """
    Red convolucional optimizada para clasificaci�n de im�genes CIFAR-10.
    Arquitectura t�pica:
    - 2 bloques Conv2D + ReLU + MaxPool
    - 1 capa oculta totalmente conectada
    - Salida con logits (sin softmax)

    Usar con CrossEntropyLoss.
    """
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 8, 8)
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Logits

# --------------------------------------------------
# Modelo 2: MLP para clasificaci�n tabular (e.g. tiroides)
# --------------------------------------------------
class TabularClassifier(nn.Module):
    """
    MLP para clasificaci�n binaria o multiclase en datos tabulares.

    - Entrada -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Salida
    """
    def __init__(self, input_dim, output_dim):
        super(TabularClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Sin softmax
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# Modelo 3: MLP para regresi�n (e.g. California Housing)
# --------------------------------------------------
class HousingRegressor(nn.Module):
    """
    MLP para regresi�n en datos como California Housing.

    - Entrada -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(1)
    """
    def __init__(self, input_dim):
        super(HousingRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# Ahora tienes 3 modelos espec�ficos, listos para adaptar en examen:
# - CIFAR10_CNN: im�genes
# - TabularClassifier: clasificaci�n m�dica, etc.
# - HousingRegressor: regresi�n tipo housing
