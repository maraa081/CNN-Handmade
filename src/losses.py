"""
losses.py — Fonctions de perte et softmax

Modules à venir :
    Softmax          — normalisation exponentielle (probabilités)
    CrossEntropyLoss — perte d'entropie croisée (avec softmax intégré)
"""

import numpy as np


class Softmax:
    """
    Softmax : normalise un vecteur de scores en distribution de probabilités.

    Forward : p_i = exp(x_i) / sum_j exp(x_j)
    Backward : combiné avec CrossEntropy pour stabilité numérique

    TODO
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        """x : (N, C)  →  retourne (N, C)"""
        raise NotImplementedError("Softmax.forward — à implémenter")

    def backward(self, grad_output):
        raise NotImplementedError("Softmax.backward — à implémenter")

    def update(self, lr):
        pass

    def __repr__(self):
        return "Softmax()"


class CrossEntropyLoss:
    """
    Perte d'entropie croisée (souvent combinée avec Softmax backward).

    L = - 1/N * sum_n sum_c y_true[n,c] * log(y_pred[n,c] + epsilon)

    TODO
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred : (N, C) — logits ou probas
        y_true : (N, C) — one-hot
        retourne : float (la loss moyenne sur le batch)
        """
        raise NotImplementedError("CrossEntropyLoss.forward — à implémenter")

    def backward(self):
        """
        Retourne le gradient de la loss par rapport à y_pred.
        (N, C)
        """
        raise NotImplementedError("CrossEntropyLoss.backward — à implémenter")

    def __repr__(self):
        return "CrossEntropyLoss()"
