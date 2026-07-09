#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔥 ADAM — Adaptive Moment Estimation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Optimiseur : Adam (Momentum + RMSprop)
  Paramètres  : lr, beta1, beta2, eps

  Théorie :
      m ← β1·m + (1-β1)·g          (moyenne des gradients)
      v ← β2·v + (1-β2)·g²         (variance des gradients)
      θ ← θ - lr · m̂ / (√v̂ + ε)    (mise à jour normalisée)

  Pourquoi Adam est génial :
    - Learning rate adaptatif par paramètre (via v)
    - Momentum qui accélère dans les bonnes directions
    - Fonctionne avec des lr plus petits (0.001 par défaut)
    - Beaucoup plus robuste — moins besoin de tuner le lr

  Lancement : python3 experiments/adam/tune_cnn.py
  Résultat   : experiments/adam/tune_result.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
from optimizers import Adam


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⛽ RÉGLAGES                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LEARNING_RATE = 0.001      # ← Adam utilise typiquement un lr plus petit (0.001)
BETA1        = 0.9          # ← Décroissance du 1er moment (gradient moyen)
BETA2        = 0.999        # ← Décroissance du 2e moment (gradient carré)
EPS          = 1e-8         # ← Stabilité numérique (évite division par zéro)

BATCH_SIZE    = 64
EPOCHS        = 5
DATA_LIMIT    = 2000

# ── Optimiseur ─────────────────────────────────────────────────────────────
#   Adam. Tu peux jouer avec beta1 et beta2 pour voir l'effet :
#     BETA1 = 0.9  → momentum standard
#     BETA1 = 0.95 → moins d'influence du momentum
#     BETA2 = 0.99 → variance plus "longue mémoire" (peut osciller)
#     BETA2 = 0.999 → défaut, bonne stabilité
OPT = Adam(lr=LEARNING_RATE, beta1=BETA1, beta2=BETA2, eps=EPS)

# ═══════════════════════════════════════════════════════════════════════════

# ── Données ──
print("📥 Chargement MNIST…")
loader = MNISTLoader()
(x_train, y_train), (x_test, y_test) = loader.load(join(ROOT, "data"))
if DATA_LIMIT:
    x_train, y_train = x_train[:DATA_LIMIT], y_train[:DATA_LIMIT]
train_loader = preprocess_pipeline(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = preprocess_pipeline(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
print(f"  ✔ {x_train.shape[0]} train / {x_test.shape[0]} test")

# ── Architecture ──
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

print(f"\n🧠 Réseau :\n{model}")
print(f"⚙️  Optimiseur : {model.optimizer}")

# ── Entraînement ──
print(f"\n{'═' * 50}")
print(f"🏋️  Entraînement — {EPOCHS} epochs")
print(f"{'═' * 50}\n")

t_start = time.time()
history = model.train(train_loader, epochs=EPOCHS, verbose=True)
elapsed = time.time() - t_start

m, s = divmod(elapsed, 60)
print(f"\n⏱️  {int(m)}m {int(s)}s")

# ── Évaluation ──
print(f"\n{'═' * 50}")
print(f"📊 Évaluation sur {x_test.shape[0]} images test…")
test_acc = model.evaluate(test_loader)
print(f"{'═' * 50}\n")
print(f"🎯 Accuracy : {test_acc:.4f}  ({test_acc * 100:.1f}%)")

# ── Graphique ──
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
out = join(dirname(abspath(__file__)), "tune_result.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n📁 Graphique : {out}")

# ── Sauvegarde des poids ──
weights_path = join(dirname(abspath(__file__)), "model_weights.npz")
model.save_weights(weights_path)
