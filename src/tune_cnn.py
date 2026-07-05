#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎮   TUNE CNN — Joue avec les paramètres du réseau !
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tu n'as besoin de toucher qu'aux paramètres dans la section "⛽ RÉGLAGES".
  Lance le script :   python3 src/tune_cnn.py
  Magic ✨

  Ce que tu peux changer :
     - LEARNING_RATE  → rapidité d'apprentissage (trop haut = instable, trop bas = lent)
     - BATCH_SIZE     → nombre d'images vues à la fois (32, 64, 128…)
     - EPOCHS         → nombre de passages sur toutes les données
     - ARCHITECTURE   → tu peux modifier les couches du réseau plus bas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # ← change en "Agg" si pas d'écran (SSH/serveur)
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

# Forcer l'affichage immédiat dans le terminal (pas de buffer)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# ── Chemin racine du projet ──
ROOT_DIR = dirname(dirname(abspath(__file__)))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⛽ RÉGLAGES — Change ces valeurs pour voir l'effet sur l'entraînement  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LEARNING_RATE = 0.01   # ← Pas d'apprentissage (essaie 0.1, 0.01, 0.001, 0.0001)
BATCH_SIZE    = 64     # ← Taille des lots (32, 64, 128, 256… plus gros = +stable mais +lent)
EPOCHS        = 5      # ← Nombre de répétitions sur les données (5, 10, 20…)

# ═══════════════════════════════════════════════════════════════════════════
#  Tu peux aussi modifier l'architecture du réseau plus bas :
#     - ligne "Conv2D(1→32)…"  → changer le nombre de filtres
#     - ligne "Dense(128→10)"  → changer la taille de la couche cachée
# ═══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────
#  CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────

print("📥 Chargement de MNIST…")
loader = MNISTLoader()
(x_train, y_train), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))

# ── Construction des DataLoaders ──
train_loader = preprocess_pipeline(x_train, y_train,
                                   batch_size=BATCH_SIZE, shuffle=True)
test_loader  = preprocess_pipeline(x_test, y_test,
                                   batch_size=BATCH_SIZE, shuffle=False)

# Affiche un petit échantillon pour vérifier
print(f"\n📦 Batch d'entraînement : {next(iter(train_loader))[0].shape}")
print(f"📦 Batch de test        : {next(iter(test_loader))[0].shape}")
print(f"🧮 Total entraînement   : {x_train.shape[0]} images")
print(f"🧮 Total test           : {x_test.shape[0]} images")


# ──────────────────────────────────────────────────────────────────────────
#  🔧 ARCHITECTURE DU RÉSEAU
# ──────────────────────────────────────────────────────────────────────────
#  Modifie les lignes ci-dessous pour changer la structure du CNN.
#  Chaque ligne est une couche, lues dans l'ordre.
#
#  Couches disponibles :
#     Conv2D(canaux_entrée, canaux_sortie, taille_kernel, pad=?)
#     MaxPool2D(facteur)
#     ReLU()
#     Flatten()
#     Dense(neurones_entrée, neurones_sortie)
# ──────────────────────────────────────────────────────────────────────────

model = CNN()

model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))   # 28×28 → 32 canaux de 28×28
model.add(ReLU())
model.add(MaxPool2D(2))                                     # 28×28 → 14×14

model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))  # 14×14 → 64 canaux de 14×14
model.add(ReLU())
model.add(MaxPool2D(2))                                     # 14×14 → 7×7

model.add(Flatten())                                        # (64, 7, 7) → vecteur de 3136

model.add(Dense(3136, 128))                                 # 3136 → 128 neurones
model.add(ReLU())
model.add(Dense(128, 10))                                   # 128 → 10 classes (chiffres 0-9)

print(f"\n🧠 Réseau construit :\n{model}")


# ──────────────────────────────────────────────────────────────────────────
#  🏋️ ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 50}")
print(f"🏋️  Entraînement — lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}")
print(f"{'═' * 50}\n")

history = model.train(train_loader, epochs=EPOCHS, lr=LEARNING_RATE, verbose=True)


# ──────────────────────────────────────────────────────────────────────────
#  📊 ÉVALUATION
# ──────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 50}")
print(f"📊 Évaluation sur le jeu de test…")
print(f"{'═' * 50}")

test_acc = model.evaluate(test_loader)
print(f"\n🎯 Accuracy sur le test : {test_acc:.4f}  ({test_acc * 100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────
#  📈 GRAPHIQUE DE LA LOSS ET DE L'ACCURACY
# ──────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history["loss"], marker="o", linewidth=2, markersize=6)
plt.title(f"Loss (entraînement)\nlr={LEARNING_RATE}, batch={BATCH_SIZE}", fontsize=12)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], marker="s", linewidth=2, markersize=6, color="green")
plt.title(f"Accuracy (entraînement)\nTest final : {test_acc:.1%}", fontsize=12)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ──────────────────────────────────────────────────────────────────────────
#  🧠 CONSEILS POUR JOUER AVEC LES PARAMÈTRES
# ──────────────────────────────────────────────────────────────────────────
#
#   LEARNING RATE :
#     - 0.01  → bon équilibre pour commencer
#     - 0.1   → apprend vite mais peut "sauter" par-dessus la solution (instable)
#     - 0.001 → apprend lentement mais stable, il faut + d'epochs
#     - 0.0001 → très lent, probablement pas assez pour 5 epochs
#
#   BATCH SIZE :
#     - 32    → plus de mises à jour par epoch, un peu de bruit (régularise)
#     - 64    → bon compromis (défaut recommandé)
#     - 128   → plus stable, plus rapide, mais généralise parfois moins bien
#     - 256   → risque de "sur-apprentissage" (memorise plutôt qu'apprend)
#
#   EPOCHS :
#     - 5     → donne une première idée rapide
#     - 10    → pour voir la convergence
#     - 20+   → pour atteindre le maximum de performance
#
#   ASTUCE : essaie de trouver la meilleure accuracy possible !
#   Un CNN comme celui-ci peut atteindre ~99% sur MNIST.
#
# ═══════════════════════════════════════════════════════════════════════════
