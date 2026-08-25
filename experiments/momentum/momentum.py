#!/usr/bin/env python3
"""
--------------------------------------------------------------------------
   MOMENTUM — SGD avec élan
--------------------------------------------------------------------------

  Optimiseur : Momentum (SGD + élan)
  Paramètres  : lr + momentum (α)

  Théorie :
      v <- α·v - lr·∇θ
      θ <- θ + v

  Au lieu de suivre uniquement le gradient local, on accumule une
  "vitesse" qui lisse les oscillations et accélère dans les directions
  stables. Résultat : convergence plus rapide et plus stable.

  α (momentum) typique : 0.9  (0.0 = vanilla SGD, 0.99 = très agressif)

  Lancement : python3 experiments/momentum/momentum.py
  Résultat   : experiments/momentum/momentum.png
--------------------------------------------------------------------------
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

# Chemin vers src/
ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, join(ROOT, "src"))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN
from optimizers import Momentum


# +==========================================================================+
# |   RÉGLAGES                                                            |
# +==========================================================================+

LEARNING_RATE = 0.01       # <- Pas d'apprentissage
MOMENTUM_VAL = 0.9          # <- Coefficient d'élan (0.9 = défaut, 0.0 = SGD pur)
BATCH_SIZE    = 64
EPOCHS        = 5
DATA_LIMIT    = 2000

# -- Optimiseur -------------------------------------------------------------
#   Momentum avec élan réglable. Essaie différentes valeurs :
#     MOMENTUM_VAL = 0.0  -> équivalent SGD
#     MOMENTUM_VAL = 0.5  -> élan modéré
#     MOMENTUM_VAL = 0.9  -> élan fort (recommandé)
#     MOMENTUM_VAL = 0.99 -> élan très fort (peut overshooter)
OPT = Momentum(lr=LEARNING_RATE, momentum=MOMENTUM_VAL)

# ===========================================================================

# -- Données --
print(" Chargement MNIST…")
loader = MNISTLoader()
(x_train, y_train), (x_test, y_test) = loader.load(join(ROOT, "data"))
if DATA_LIMIT:
    x_train, y_train = x_train[:DATA_LIMIT], y_train[:DATA_LIMIT]
train_loader = preprocess_pipeline(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = preprocess_pipeline(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
print(f"  OK {x_train.shape[0]} train / {x_test.shape[0]} test")

# -- Architecture --
model = CNN(optimizer=OPT)
model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
model.add(ReLU())
model.add(MaxPool2D(2))
model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
model.add(ReLU())
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(3136, 128))
model.add(ReLU())
model.add(Dense(128, 10))

print(f"\n Réseau :\n{model}")
print(f"  Optimiseur : {model.optimizer}")

# -- Entraînement --
print(f"\n{'=' * 50}")
print(f"  Entraînement — {EPOCHS} epochs")
print(f"{'=' * 50}\n")

t_start = time.time()
history = model.train(train_loader, epochs=EPOCHS, verbose=True)
elapsed = time.time() - t_start

m, s = divmod(elapsed, 60)
print(f"\n[time]  {int(m)}m {int(s)}s")

# -- Évaluation --
print(f"\n{'=' * 50}")
print(f" Évaluation sur {x_test.shape[0]} images test…")
test_acc = model.evaluate(test_loader)
print(f"{'=' * 50}\n")
print(f" Accuracy : {test_acc:.4f}  ({test_acc * 100:.1f}%)")

# -- Graphique --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["loss"], marker="o", linewidth=2, markersize=6)
ax1.set_title(f"Loss — {model.optimizer}", fontsize=11)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(history["accuracy"], marker="s", linewidth=2, markersize=6, color="green")
ax2.set_title(f"Accuracy — Test final : {test_acc:.1%}", fontsize=11)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
out = join(dirname(abspath(__file__)), "momentum.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n Graphique : {out}")

# -- Sauvegarde des poids --
weights_path = join(dirname(abspath(__file__)), "momentum_weights.npz")
model.save_weights(weights_path)
