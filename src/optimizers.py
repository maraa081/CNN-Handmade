"""
optimizers.py — Optimiseurs pour CNN-Handmade

Chaque optimiseur implémente une méthode update(layers, lr=None)
qui parcourt les couches du réseau et applique la mise à jour
des paramètres (kernels/bias pour Conv2D, W/b pour Dense).

Optimiseurs disponibles :
    SGD       — descente de gradient classique
    Momentum  — SGD avec momentum (accumule la direction)
    Adam      — Adaptive Moment Estimation (momentum + RMSprop)
"""

import numpy as np


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SGD — Vanilla Stochastic Gradient Descent                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SGD:
    """
    Descente de gradient stochastique standard.

    θ ← θ - lr * ∇θ

    C'est l'optimiseur de base, celui utilisé par défaut jusqu'ici.
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layers, lr=None):
        """
        Applique la mise à jour SGD à toutes les couches.

        Args :
            layers : liste de couches (Conv2D, Dense, etc.)
            lr     : learning rate optionnel (sinon utilise self.lr)
        """
        lr = lr if lr is not None else self.lr

        for layer in layers:
            if hasattr(layer, 'kernels'):       # Conv2D
                layer.kernels -= lr * layer.d_kernels
                layer.bias    -= lr * layer.d_bias
            elif hasattr(layer, 'W'):           # Dense
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
            # Autres couches (ReLU, MaxPool, Flatten) : pas de paramètres

    def __repr__(self):
        return f"SGD(lr={self.lr})"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Momentum — SGD avec élan                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Momentum:
    """
    SGD avec momentum.

    v ← α * v - lr * ∇θ
    θ ← θ + v

    Au lieu de suivre uniquement le gradient local, on accumule
    une "vitesse" qui lisse les oscillations et accélère dans
    les directions stables.

    α (momentum) typique : 0.9
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}          # dictionnaire des vitesses

    def _ensure_shape(self, key, param):
        """Crée un buffer de vitesse pour un paramètre s'il n'existe pas."""
        if key not in self.v:
            self.v[key] = np.zeros_like(param)

    def update(self, layers, lr=None):
        """
        Applique la mise à jour Momentum à toutes les couches.

        La première fois, initialise les buffers de vitesse.
        """
        lr = lr if lr is not None else self.lr
        mom = self.momentum

        for i, layer in enumerate(layers):
            if hasattr(layer, 'kernels'):       # Conv2D
                # ── kernels ──
                wk = f'conv_{i}_kernels'
                self._ensure_shape(wk, layer.kernels)
                self.v[wk] = mom * self.v[wk] - lr * layer.d_kernels
                layer.kernels += self.v[wk]

                # ── bias ──
                wb = f'conv_{i}_bias'
                self._ensure_shape(wb, layer.bias)
                self.v[wb] = mom * self.v[wb] - lr * layer.d_bias
                layer.bias += self.v[wb]

            elif hasattr(layer, 'W'):           # Dense
                # ── W ──
                wk = f'dense_{i}_W'
                self._ensure_shape(wk, layer.W)
                self.v[wk] = mom * self.v[wk] - lr * layer.dW
                layer.W += self.v[wk]

                # ── b ──
                wb = f'dense_{i}_b'
                self._ensure_shape(wb, layer.b)
                self.v[wb] = mom * self.v[wb] - lr * layer.db
                layer.b += self.v[wb]

    def reset(self):
        """Réinitialise les vitesses (utile pour un nouvel entraînement)."""
        self.v = {}

    def __repr__(self):
        return f"Momentum(lr={self.lr}, momentum={self.momentum})"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Adam — Adaptive Moment Estimation                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Adam:
    """
    Optimiseur Adam : combine Momentum + RMSprop.

    m ← β1 * m + (1 - β1) * g          ← estimation du 1er moment (moyenne)
    v ← β2 * v + (1 - β2) * g²         ← estimation du 2e moment (variance)

    m̂ ← m / (1 - β1^t)                  ← correction de biais
    v̂ ← v / (1 - β2^t)

    θ ← θ - lr * m̂ / (√v̂ + ε)

    Pourquoi Adam est génial :
        - Learning rate adaptatif par paramètre (via v)
        - Momentum qui accélère dans les bonnes directions
        - Fonctionne bien avec des lr plus petits (0.001 par défaut)
        - Robuste — moins besoin de tuner le lr

    β1 (momentum) typique : 0.9
    β2 (RMSprop)  typique : 0.999
    ε  (stabilité)         : 1e-8
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}          # premier moment (moyenne des gradients)
        self.v = {}          # deuxième moment (variance des gradients)
        self.t = 0           # pas de temps

    def _ensure_moments(self, key, param):
        """Crée les buffers de moment pour un paramètre si besoin."""
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)

    def update(self, layers, lr=None):
        """
        Applique la mise à jour Adam à toutes les couches.

        Note : le learning rate d'Adam est généralement plus bas
        que SGD (~0.001 au lieu de ~0.01).
        """
        lr = lr if lr is not None else self.lr
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps

        for i, layer in enumerate(layers):
            if hasattr(layer, 'kernels'):       # Conv2D
                self._apply_adam(f'conv_{i}', layer.kernels, layer.d_kernels, lr, b1, b2, eps)
                self._apply_adam(f'conv_{i}', layer.bias, layer.d_bias, lr, b1, b2, eps)

            elif hasattr(layer, 'W'):           # Dense
                self._apply_adam(f'dense_{i}', layer.W, layer.dW, lr, b1, b2, eps)
                self._apply_adam(f'dense_{i}', layer.b, layer.db, lr, b1, b2, eps)

    def _apply_adam(self, prefix, param, grad, lr, b1, b2, eps):
        """
        Applique une étape Adam à un paramètre.

        Workflow :
            1. met à jour les moments m, v
            2. corrige le biais
            3. applique la mise à jour
        """
        key_w = f'{prefix}_W' if param.ndim > 1 else f'{prefix}_b'
        self._ensure_moments(key_w, param)
        t = self.t

        # Mise à jour des moments
        self.m[key_w] = b1 * self.m[key_w] + (1 - b1) * grad
        self.v[key_w] = b2 * self.v[key_w] + (1 - b2) * (grad ** 2)

        # Correction de biais
        m_hat = self.m[key_w] / (1 - b1 ** t)
        v_hat = self.v[key_w] / (1 - b2 ** t)

        # Mise à jour
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def reset(self):
        """Réinitialise les moments (important entre plusieurs entraînements)."""
        self.m = {}
        self.v = {}
        self.t = 0

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"
