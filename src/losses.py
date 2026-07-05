"""
losses.py — Fonctions de perte et softmax

Softmax          — normalisation exponentielle (probabilités)
CrossEntropyLoss — perte d'entropie croisée (sur les logits, stable)

Note : CrossEntropyLoss intègre softmax en interne pour la stabilité
numérique (évite les log(0) et les débordements exponentiels).
"""

import numpy as np


class Softmax:
    """
    Softmax : normalise un vecteur de scores en distribution de probabilités.

    Forward : p_i = exp(x_i - max) / sum_j exp(x_j - max)

    Astuce de stabilité : on soustrait le max de chaque échantillon avant
    l'exponentielle pour éviter les débordements (exp(1000) = inf).

    Backward : gradient individuel du softmax (Jacobienne).
    En pratique, on utilise plutôt CrossEntropyLoss qui combine softmax + loss.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        """
        x : (N, C) — logits
        retourne : (N, C) — probabilités (chaque ligne somme à 1)
        """
        self.input = x

        # Stabilité : soustraire le max de chaque échantillon
        x_shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.output = exp_x / exp_x.sum(axis=1, keepdims=True)

        return self.output

    def backward(self, grad_output):
        """
        grad_output : (N, C) — gradient de la perte par rapport aux probas softmax
        retourne : (N, C) — gradient par rapport aux logits d'entrée

        Jacobienne du softmax : p_i * (delta_ij - p_j)
        Appliquée au gradient : dL/dx_i = sum_j dL/dp_j * p_i * (delta_ij - p_j)
                                    = p_i * (dL/dp_i - sum_j dL/dp_j * p_j)
        """
        p = self.output  # (N, C)
        N, C = p.shape

        # dL/dx = p * (dL/dp - sum(dL/dp * p, axis=1, keepdims=True))
        dp = grad_output
        dx = p * (dp - (dp * p).sum(axis=1, keepdims=True))

        return dx

    def update(self, lr):
        """Softmax n'a pas de paramètres."""
        pass

    def __repr__(self):
        return "Softmax()"


class CrossEntropyLoss:
    """
    Perte d'entropie croisée, avec softmax intégré pour la stabilité.

    Prend des **logits** (scores bruts) en entrée, pas des probabilités.
    Le softmax est appliqué en interne de façon stable.

    Forward (logits → loss) :
        p = softmax(logits)
        L = -1/N * sum_n sum_c y_true[n,c] * log(p[n,c] + eps)

    Backward :
        dL/dlogits = (p - y_true) / N     [formule stable combinée]

    Pourquoi c'est mieux que Softmax + CrossEntropy séparés ?
        - Évite log(0) car p > 0 (softmax garantit des probas > 0)
        - Le gradient combiné (p - y_true) évite de calculer la Jacobienne
          complète du softmax et de la multiplier par dL/dp
    """

    def __init__(self):
        self.logits = None      # entrée brute (pour backward si besoin)
        self.y_pred = None      # softmax(logits)
        self.y_true = None      # one-hot

    def forward(self, logits, y_true):
        """
        logits : (N, C) — scores bruts avant softmax
        y_true : (N, C) — one-hot encoding
        retourne : float — la loss moyenne sur le batch
        """
        self.logits = logits
        self.y_true = y_true

        # Softmax stable
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.y_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy : -1/N * sum(y_true * log(y_pred))
        # On ajoute 1e-15 pour éviter log(0) (au cas où)
        eps = 1e-15
        loss = -np.sum(y_true * np.log(self.y_pred + eps)) / logits.shape[0]

        return loss

    def backward(self):
        """
        Retourne le gradient de la loss par rapport aux logits.

        La formule magique (softmax + cross-entropy combinés) :
            dL/dlogits = (softmax(logits) - y_true) / N

        Pourquoi c'est si simple ?
            - La cross-entropy seule donne dL/dp = - y_true / p
            - La Jacobienne du softmax donne dp/dlogits
            - Le produit simplifie à p - y_true (par échantillon)
            - On divise par N pour la moyenne du batch
        """
        N = self.logits.shape[0]
        return (self.y_pred - self.y_true) / N

    def accuracy(self, logits, y_true):
        """
        Calcule l'accuracy à partir des logits et des one-hot.

        logits : (N, C)
        y_true : (N, C) — one-hot
        retourne : float entre 0 et 1
        """
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y_true, axis=1)
        return np.mean(preds == labels)

    def __repr__(self):
        return "CrossEntropyLoss()"
