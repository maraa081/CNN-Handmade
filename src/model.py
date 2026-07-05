"""
model.py — Assembleur du réseau CNN-Handmade

Orchestre le cycle complet :
    forward → backward → update → training loop → evaluation

Utilisation typique :
    model = CNN()
    model.add(Conv2D(1, 32, 3, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, 64, 3, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(3136, 128))
    model.add(ReLU())
    model.add(Dense(128, 10))

    model.train(train_loader, epochs=5, lr=0.01)
    acc = model.evaluate(test_loader)
"""

import numpy as np
from layers import (
    Conv2D, MaxPool2D, ReLU, Flatten, Dense,
)
from losses import CrossEntropyLoss


class CNN:
    """
    Réseau CNN complet pour MNIST.

    Architecture recommandée :
        Conv2D(1→32, 3, pad=1) → ReLU → MaxPool2D(2)
        Conv2D(32→64, 3, pad=1) → ReLU → MaxPool2D(2)
        Flatten
        Dense(3136→128) → ReLU
        Dense(128→10)
    """

    def __init__(self):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()

    def add(self, layer):
        """Ajoute une couche au réseau."""
        self.layers.append(layer)

    def forward(self, x):
        """
        Passe l'entrée x à travers toutes les couches.

        x : (N, C, H, W) — images en channels_first
        retourne : logits (N, C) — scores bruts avant softmax
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        Rétropropage le gradient à travers toutes les couches,
        de la dernière à la première.

        grad : gradient de la loss par rapport aux logits (N, C)
        retourne : gradient par rapport à l'entrée
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr):
        """
        Met à jour tous les paramètres apprenables.

        lr : learning rate (pas d'apprentissage)
        """
        for layer in self.layers:
            layer.update(lr)

    def train(self, train_loader, epochs=10, lr=0.01, verbose=True):
        """
        Boucle d'entraînement complète.

        Pour chaque epoch :
            1. Forward : passer les batches dans le réseau → logits
            2. Loss : CrossEntropyLoss(logits, y_true)
            3. Backward : rétropropager le gradient
            4. Update : mettre à jour les poids

        Args :
            train_loader : DataLoader avec les batches (channels_first)
            epochs       : nombre de passes complètes sur les données
            lr           : learning rate
            verbose      : afficher la progression ?

        Retourne :
            dict avec l'historique : {'loss': [...], 'accuracy': [...]}
        """
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                N = batch_x.shape[0]

                # ── Forward ──
                logits = self.forward(batch_x)

                # ── Loss ──
                loss = self.loss_fn.forward(logits, batch_y)

                # ── Backward ──
                grad = self.loss_fn.backward()
                self.backward(grad)

                # ── Update ──
                self.update(lr)

                # Stats
                epoch_loss += loss * N
                total += N
                correct += (np.argmax(logits, axis=1) == np.argmax(batch_y, axis=1)).sum()

            avg_loss = epoch_loss / total
            acc = correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)

            if verbose:
                print(f"  Època {epoch + 1}/{epochs}  ─  loss: {avg_loss:.4f}  ─  accuracy: {acc:.4f}")

        return history

    def evaluate(self, test_loader):
        """
        Calcule l'accuracy sur un ensemble de test.

        Args :
            test_loader : DataLoader (ne mélange pas, shuffle=False)

        Retourne :
            float entre 0 et 1 (proportion de bonnes réponses)
        """
        correct = 0
        total = 0

        for batch_x, batch_y in test_loader:
            logits = self.forward(batch_x)
            total += batch_x.shape[0]
            correct += (np.argmax(logits, axis=1) == np.argmax(batch_y, axis=1)).sum()

        return correct / total

    def __repr__(self):
        layers_str = "\n  ".join(str(l) for l in self.layers)
        return f"CNN(\n  {layers_str}\n)"
