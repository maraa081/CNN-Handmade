"""
model.py — Assembleur du réseau CNN-Handmade

Construit le graphe de couches et orchestre :
    forward → backward → update → training loop

TODO
"""


class CNN:
    """
    Réseau CNN complet pour MNIST.

    Architecture prévue :
        Conv2D(1→32, 3, pad=1) → ReLU → MaxPool2D(2)
        Conv2D(32→64, 3, pad=1) → ReLU → MaxPool2D(2)
        Flatten
        Dense(3136→128) → ReLU
        Dense(128→10)
        Softmax + CrossEntropyLoss

    TODO : à assembler une fois Dense, Softmax et CrossEntropy prêts.
    """

    def __init__(self):
        self.layers = []

    def add(self, layer):
        """Ajoute une couche au réseau."""
        self.layers.append(layer)

    def forward(self, x):
        """Passe x à travers toutes les couches."""
        raise NotImplementedError("CNN.forward — à implémenter")

    def backward(self, loss_grad):
        """Rétropropage le gradient à travers toutes les couches."""
        raise NotImplementedError("CNN.backward — à implémenter")

    def update(self, lr):
        """Met à jour tous les paramètres."""
        raise NotImplementedError("CNN.update — à implémenter")

    def train(self, train_loader, epochs=10, lr=0.01):
        """Boucle d'entraînement complète."""
        raise NotImplementedError("CNN.train — à implémenter")

    def evaluate(self, test_loader):
        """Calcule l'accuracy sur le jeu de test."""
        raise NotImplementedError("CNN.evaluate — à implémenter")

    def __repr__(self):
        layers_str = "\n  ".join(str(l) for l in self.layers)
        return f"CNN(\n  {layers_str}\n)"
