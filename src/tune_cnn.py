#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎮   TUNE CNN — Joue avec les paramètres du réseau !
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Lance le script :   python3 src/tune_cnn.py

  Tout se règle dans la section "⛽ RÉGLAGES" juste en dessous.
  Le résultat (graphique) est sauvegardé dans tune_result.png.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sauvegarde en fichier (pas besoin d'écran)
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

# Débufferisation
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = dirname(dirname(abspath(__file__)))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⛽ RÉGLAGES — Modifie ces valeurs pour voir l'effet sur l'entraînement ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LEARNING_RATE = 0.01      # ← Pas d'apprentissage (0.1, 0.01, 0.001, 0.0001…)
BATCH_SIZE    = 64         # ← Taille des lots (32, 64, 128, 256…)
EPOCHS        = 5          # ← Nombre de passages sur les données (5, 10, 20…)
DATA_LIMIT    = 2000       # ← Nbre d'images utilisé (2000 = rapide ; mets None pour tout)

# ═══════════════════════════════════════════════════════════════════════════
#  Modifie aussi l'architecture du réseau plus bas si tu veux :
#     - Les lignes "Conv2D(1→32)" → change les filtres
#     - Les lignes "Dense(…→128)" → change les neurones
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────
#  📥 CHARGEMENT
# ──────────────────────────────────────────────────────────────────────────

print("📥 Chargement de MNIST…")
t0 = time.time()
loader = MNISTLoader()
(x_train, y_train), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))

# Sous-ensemble optionnel (pour aller vite)
if DATA_LIMIT is not None:
    x_train = x_train[:DATA_LIMIT]
    y_train = y_train[:DATA_LIMIT]

train_loader = preprocess_pipeline(x_train, y_train,
                                   batch_size=BATCH_SIZE, shuffle=True)
test_loader  = preprocess_pipeline(x_test, y_test,
                                   batch_size=BATCH_SIZE, shuffle=False)

n_batches = len(train_loader)
print(f"  ✔ Images : {x_train.shape[0]} train / {x_test.shape[0]} test")
print(f"  ✔ Format : {next(iter(train_loader))[0].shape}")
print(f"  ✔ ~{n_batches} batches par epoch")


# ──────────────────────────────────────────────────────────────────────────
#  🔧 ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────
#  Couches disponibles :
#     Conv2D(canaux_entrée, canaux_sortie, taille_ kernel, pad=?)
#     MaxPool2D(facteur)
#     ReLU()
#     Flatten()
#     Dense(neurones_entrée, neurones_sortie)
# ──────────────────────────────────────────────────────────────────────────

model = CNN()

model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))   # 28×28 → 32 canaux
model.add(ReLU())
model.add(MaxPool2D(2))                                     # → 14×14

model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))  # 14×14 → 64 canaux
model.add(ReLU())
model.add(MaxPool2D(2))                                     # → 7×7

model.add(Flatten())                                        # vecteur de 3136

model.add(Dense(3136, 128))                                 # 3136 → 128 neurones
model.add(ReLU())
model.add(Dense(128, 10))                                   # → 10 chiffres

print(f"\n🧠 Réseau :\n{model}")
print(f"⚡ {n_batches * EPOCHS} passages forward/backward au total")


# ──────────────────────────────────────────────────────────────────────────
#  🏋️  ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 50}")
print(f"🏋️  Entraînement")
print(f"    lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}")
print(f"{'═' * 50}\n")

t_start = time.time()
history = model.train(train_loader, epochs=EPOCHS, lr=LEARNING_RATE, verbose=True)
t_elapsed = time.time() - t_start

m, s = divmod(t_elapsed, 60)
print(f"\n⏱️  Temps d'entraînement : {int(m)}m {int(s)}s")


# ──────────────────────────────────────────────────────────────────────────
#  📊 ÉVALUATION
# ──────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 50}")
print(f"📊 Évaluation sur {x_test.shape[0]} images de test…")
test_acc = model.evaluate(test_loader)
print(f"{'═' * 50}\n")
print(f"🎯 Accuracy : {test_acc:.4f}  ({test_acc * 100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────
#  📈 GRAPHIQUE → tune_result.png
# ──────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["loss"], marker="o", linewidth=2, markersize=6)
ax1.set_title(f"Loss (entraînement)\nlr={LEARNING_RATE}, batch={BATCH_SIZE}", fontsize=12)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(history["accuracy"], marker="s", linewidth=2, markersize=6, color="green")
ax2.set_title(f"Accuracy (entraînement)\nTest final : {test_acc:.1%}", fontsize=12)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
output_path = join(ROOT_DIR, "tune_result.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"📁 Graphique sauvegardé : {output_path}")


# ──────────────────────────────────────────────────────────────────────────
#  🧠  CONSEILS
# ──────────────────────────────────────────────────────────────────────────
#
#  LEARNING RATE :
#    0.01   → bon équilibre pour commencer
#    0.1    → apprend vite, mais peut être instable
#    0.001  → stable, mais lent (besoin de + d'epochs)
#
#  BATCH SIZE :
#    32     → + de mises à jour par epoch (bruit régularisant)
#    64     → bon compromis (recommandé)
#    128    → plus rapide, parfois moins bonne généralisation
#
#  EPOCHS :
#    5      → premier aperçu rapide
#    10     → pour voir converger
#    20+    → pour aller chercher les ~99%
#
#  DATA_LIMIT :
#    2000   → entraînement rapide (~1 min)
#    5000   → un peu plus long (~3 min)
#    None   → tout MNIST (60000 images, ~15-20 min)
#
#  ASTUCE : essaie DATA_LIMIT=None avec lr=0.001 et EPOCHS=10
#           Le CNN peut atteindre ~99% sur MNIST !
#
# ═══════════════════════════════════════════════════════════════════════════
