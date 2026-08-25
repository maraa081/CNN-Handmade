"""
model.py — Assembleur du réseau CNN-Handmade

Orchestre le cycle complet :
    forward -> backward -> update -> training loop -> evaluation

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
    Conv2D, MaxPool2D, ReLU, Flatten, Dense, Dropout,
)
from losses import CrossEntropyLoss
from optimizers import SGD


class CNN:
    """
    Réseau CNN complet pour MNIST.

    Architecture recommandée :
        Conv2D(1->32, 3, pad=1) -> ReLU -> MaxPool2D(2)
        Conv2D(32->64, 3, pad=1) -> ReLU -> MaxPool2D(2)
        Flatten
        Dense(3136->128) -> ReLU
        Dense(128->10)

    Optimiseurs disponibles (via optimizers.py) :
        SGD       — descente de gradient classique (défaut)
        Momentum  — SGD avec élan (lisse les oscillations)
        Adam      — lr adaptatif + momentum (recommendé pour tuning)
    """

    def __init__(self, optimizer=None):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else SGD(lr=0.01)

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

    def update(self, lr=None):
        """
        Met à jour tous les paramètres via l'optimiseur.

        Délègue la mise à jour à self.optimizer.update(layers, lr).
        Chaque optimiseur (SGD, Momentum, Adam) applique sa propre règle.

        Args :
            lr : learning rate (optionnel — utilise celui de l'optimiseur par défaut)
        """
        self.optimizer.update(self.layers, lr)

    def train_mode(self):
        """
        Passe toutes les couches en mode entraînement.

        Nécessaire pour les couches qui se comportent différemment
        en entraînement et en évaluation (Dropout activé vs désactivé).
        """
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval_mode(self):
        """
        Passe toutes les couches en mode évaluation.

        Désactive le Dropout, la normalisation par lots, etc.
        À appeler AVANT evaluate() ou predict().
        """
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def train(self, train_loader, epochs=10, lr=None, verbose=True):
        """
        Boucle d'entraînement complète.

        Pour chaque epoch :
            1. Forward : passer les batches dans le réseau -> logits
            2. Loss : CrossEntropyLoss(logits, y_true)
            3. Backward : rétropropager le gradient
            4. Update : mettre à jour les poids via l'optimiseur

        Args :
            train_loader : DataLoader avec les batches (channels_first)
            epochs       : nombre de passes complètes sur les données
            lr           : learning rate (None = utilise celui de l'optimiseur)
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

                # -- Mode entraînement (dropout actif, etc.) --
                self.train_mode()

                # -- Forward --
                logits = self.forward(batch_x)

                # -- Loss --
                loss = self.loss_fn.forward(logits, batch_y)

                # -- Backward --
                grad = self.loss_fn.backward()
                self.backward(grad)

                # -- Update --
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
                print(f"  Epoch {epoch + 1}/{epochs}  -  loss: {avg_loss:.4f}  -  accuracy: {acc:.4f}")

        return history

    def evaluate(self, test_loader):
        """
        Calcule l'accuracy sur un ensemble de test.

        Passe le modèle en mode évaluation (dropout désactivé)
        avant de calculer l'accuracy.

        Args :
            test_loader : DataLoader (ne mélange pas, shuffle=False)

        Retourne :
            float entre 0 et 1 (proportion de bonnes réponses)
        """
        self.eval_mode()
        correct = 0
        total = 0

        for batch_x, batch_y in test_loader:
            logits = self.forward(batch_x)
            total += batch_x.shape[0]
            correct += (np.argmax(logits, axis=1) == np.argmax(batch_y, axis=1)).sum()

        return correct / total

    def save_weights(self, path):
        """
        Sauvegarde tous les poids du réseau dans un fichier .npz.

        Chargeable plus tard avec load_weights() pour éviter
        de réentraîner le modèle.

        Args :
            path : chemin du fichier .npz (ex: "model_weights.npz")
        """
        import os
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'kernels'):
                params[f'conv_{i}_kernels'] = layer.kernels
                params[f'conv_{i}_bias'] = layer.bias
            elif hasattr(layer, 'W'):
                params[f'dense_{i}_W'] = layer.W
                params[f'dense_{i}_b'] = layer.b
        np.savez(path, **params)
        size = os.path.getsize(path)
        print(f"\n   Poids sauvegardés -> {path}  ({size / 1024:.1f} Ko)")

    def load_weights(self, path):
        """
        Charge les poids depuis un fichier .npz dans l'architecture
        actuelle du modèle.

        Le modèle doit AVOIR LA MÊME ARCHITECTURE qu'au moment
        de la sauvegarde (mêmes couches dans le même ordre).

        Args :
            path : chemin du fichier .npz (ex: "model_weights.npz")
        """
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'kernels'):
                layer.kernels = data[f'conv_{i}_kernels']
                layer.bias = data[f'conv_{i}_bias']
            elif hasattr(layer, 'W'):
                layer.W = data[f'dense_{i}_W']
                layer.b = data[f'dense_{i}_b']
        print(f"   Poids chargés depuis -> {path}")

    def predict(self, x):
        """
        Prédiction rapide pour une image ou un batch.

        Passe le modèle en mode évaluation (dropout désactivé)
        avant de prédire.

        Args :
            x : image (1, 28, 28) ou batch (N, 1, 28, 28) en channels_first

        Retourne :
            - si une seule image : le chiffre prédit (int)
            - si un batch : tableau des prédictions (int)
        """
        self.eval_mode()
        single = x.ndim == 3
        if single:
            x = x[np.newaxis, :]
        logits = self.forward(x)
        preds = np.argmax(logits, axis=1)
        return int(preds[0]) if single else preds

    def __repr__(self):
        layers_str = "\n  ".join(str(l) for l in self.layers)
        return f"CNN(\n  {layers_str}\n)"
