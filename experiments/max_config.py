#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔥 MAX CONFIG — Toutes les optimisations combinées
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Ce modèle cumule TOUT ce qu'on a implémenté :

    ✅ Adam          — lr adaptatif + momentum
    ✅ Dropout       — régularisation par désactivation aléatoire
    ✅ L2 Weight Decay — pénalise les gros poids

  Le but : atteindre les meilleures performances possibles sur MNIST
  avec notre réseau fait main.

  Lancement : python3 experiments/max_config.py
  Résultat   : max_config.png (courbes) + max_config_weights.npz (poids)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(ROOT, "src"))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense, Dropout
from model import CNN
from optimizers import Adam


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⛽ RÉGLAGES                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Optimiseur : Adam + L2 Weight Decay ──
LEARNING_RATE = 0.001       # Adam typique
BETA1        = 0.9           # Momentum
BETA2        = 0.999         # RMSprop
EPS          = 1e-8          # Stabilité
WEIGHT_DECAY = 0.0005        # L2 régularisation (0.0005 = léger, efficace)

# ── Dropout ──
DROPOUT_RATE = 0.5           # Désactive 50% des neurones après le premier Dense

# ── Entraînement ──
BATCH_SIZE = 64
EPOCHS     = 5
DATA_LIMIT = None            # None = tout MNIST (60000 images)

# ═══════════════════════════════════════════════════════════════════════════

OUT_DIR = dirname(abspath(__file__))

OPT = Adam(lr=LEARNING_RATE, beta1=BETA1, beta2=BETA2, eps=EPS, weight_decay=WEIGHT_DECAY)


# ── Données ──
print("📥 Chargement MNIST…")
loader = MNISTLoader()
(x_train, y_train), (x_test, y_test) = loader.load(join(ROOT, "data"))
if DATA_LIMIT:
    x_train, y_train = x_train[:DATA_LIMIT], y_train[:DATA_LIMIT]
train_loader = preprocess_pipeline(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = preprocess_pipeline(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
print(f"  ✔ {x_train.shape[0]} train / {x_test.shape[0]} test")

# ── Architecture "Max Config" ──
model = CNN(optimizer=OPT)
model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
model.add(ReLU())
model.add(MaxPool2D(2))                          # 28×28 → 14×14

model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
model.add(ReLU())
model.add(MaxPool2D(2))                          # 14×14 → 7×7

model.add(Flatten())                             # → 3136

model.add(Dense(3136, 128))
model.add(ReLU())
model.add(Dropout(p=DROPOUT_RATE))               # ← Dropout après la première Dense
model.add(Dense(128, 10))

# ── Résumé ──
print(f"\n🧠 Réseau Max Config :\n{model}")
print(f"⚙️  Optimiseur : Adam + L2(λ={WEIGHT_DECAY})")
print(f"🎲 Dropout : p={DROPOUT_RATE}")
print(f"📦 Données : batch={BATCH_SIZE}, epochs={EPOCHS}, data={'TOUT' if DATA_LIMIT is None else DATA_LIMIT}")

# ── Entraînement ──
print(f"\n{'═' * 55}")
print(f"🏋️  Entraînement")
print(f"{'═' * 55}\n")

t_start = time.time()
history = model.train(train_loader, epochs=EPOCHS, verbose=True)
elapsed = time.time() - t_start

m, s = divmod(elapsed, 60)
print(f"\n⏱️  Temps : {int(m)}m {int(s)}s")

# ── Évaluation ──
print(f"\n{'═' * 55}")
print(f"📊 Évaluation sur {x_test.shape[0]} images test…")
test_acc = model.evaluate(test_loader)
print(f"{'═' * 55}\n")
print(f"🎯 Accuracy finale : {test_acc:.4f}  ({test_acc * 100:.1f}%)")

# ── Graphique ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["loss"], marker="o", linewidth=2, markersize=6, color="#2ca02c")
ax1.set_title("Loss (entraînement)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(history["accuracy"], marker="s", linewidth=2, markersize=6, color="#2ca02c")
ax2.set_title(f"Accuracy — Test : {test_acc:.1%}", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

plt.suptitle("Max Config — Adam + Dropout + L2", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

graph_path = join(ROOT, "max_config.png")
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
print(f"📊 Graphique → {graph_path}")

# ── Sauvegarde des poids ──
weights_path = join(ROOT, "max_config_weights.npz")
model.save_weights(weights_path)

# ── Récap ──
print(f"\n{'═' * 55}")
print(f"  🔥 MAX CONFIG — RÉSULTATS")
print(f"{'═' * 55}")
print(f"  Optimiseur : Adam + L2(λ={WEIGHT_DECAY})")
print(f"  Dropout    : p={DROPOUT_RATE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Data       : {'60000' if DATA_LIMIT is None else str(DATA_LIMIT)} images")
print(f"  Test acc   : {test_acc * 100:.2f}%")
print(f"  Temps      : {int(m)}m {int(s)}s")
print(f"{'═' * 55}")
