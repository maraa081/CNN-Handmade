"""
forward_trace.py — Trace complète du cheminement d'une image dans le réseau.

Prend une image réelle de MNIST et la fait passer à travers chaque couche,
en affichant pour chaque étape :
  - Les dimensions
  - Des statistiques (min, max, mean, std)
  - Un échantillon de valeurs
  - La sauvegarde des feature maps en image

Usage :
    python3 traces/forward_trace.py
    Les images de trace sont sauvegardées dans traces/out/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # pas besoin d'affichage
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

# Chemin racine
ROOT_DIR = dirname(dirname(abspath(__file__)))
OUT_DIR = join(dirname(abspath(__file__)), 'out')
os.makedirs(OUT_DIR, exist_ok=True)

# Import des couches depuis cnn.py
from cnn import (
    MNISTLoader, normalize, add_channel_dim, to_one_hot, DataLoader,
    Conv2D, MaxPool2D, ReLU, Flatten
)


# ────────────────────────────────────────────────────────────────────────────
# 1️⃣  CHARGEMENT — On prend UNE image
# ────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("📦 ÉTAPE 1 — CHARGEMENT MNIST")
print("=" * 70)

loader = MNISTLoader()
(x_train, y_train), (x_test, y_test) = loader.load(join(ROOT_DIR, 'data'))

# On choisit un chiffre précis (le premier "3" du train)
target_digit = 3
idx = np.where(y_train == target_digit)[0][5]  # 6e occurrence du chiffre 3

image_raw = x_train[idx]  # (28, 28)
label = y_train[idx]

print(f"  Image sélectionnée : index {idx}, label = {label}")
print(f"  Shape brute        : {image_raw.shape}")
print(f"  Type               : {image_raw.dtype}")
print(f"  Pixel min          : {image_raw.min()}")
print(f"  Pixel max          : {image_raw.max()}")

# Sauvegarde image originale
plt.figure(figsize=(3, 3))
plt.imshow(image_raw, cmap='gray')
plt.title(f"Original — label: {label}", fontsize=10)
plt.axis('off')
plt.tight_layout()
plt.savefig(join(OUT_DIR, '01_original.png'), dpi=100)
plt.close()
print(f"  → Sauvegardée : traces/out/01_original.png")

# Aperçu des premières valeurs de pixels
print(f"\n  Aperçu (coins haut-gauche 8×8) :")
print(f"  {image_raw[:8, :8]}")


# ────────────────────────────────────────────────────────────────────────────
# 2️⃣  PRÉTRAITEMENT
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🧹 ÉTAPE 2 — PRÉTRAITEMENT (normalisation + reshape)")
print("=" * 70)

# Normalisation [0, 255] → [0, 1]
image_norm = normalize(image_raw)  # (28, 28)
print(f"  → normalize() : min={image_norm.min():.4f}, max={image_norm.max():.4f}")

# Ajout canal + batch : (28,28) → (1, 28, 28, 1)
image_bch = add_channel_dim(image_norm[np.newaxis, ...])  # (1, 28, 28, 1)
print(f"  → add_channel_dim() → shape : {image_bch.shape}")

# Format channels_first pour Conv2D : (1, 28, 28, 1) → (1, 1, 28, 28)
image_input = image_bch.transpose(0, 3, 1, 2)
print(f"  → transpose (N, H, W, C → N, C, H, W) : {image_input.shape}")
print(f"  → Prêt pour Conv2D !")


# ────────────────────────────────────────────────────────────────────────────
# 3️⃣  CONV2D #1  (1 → 4 canaux, kernel 3×3, pad=1)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🔲 ÉTAPE 3 — CONV2D #1  (1 → 4, kernel=3, stride=1, pad=1)")
print("=" * 70)

conv1 = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, pad=1)
print(f"  Couche : {conv1}")
print(f"  Poids   : kernels shape = {conv1.kernels.shape}")
print(f"  Biais   : bias shape = {conv1.bias.shape}")

# Un coup d'œil aux kernels
print(f"\n  Kernel #0 (aplati, 9 valeurs) :")
print(f"    {conv1.kernels[0, 0].round(3).tolist()}")

feature_map_1 = conv1.forward(image_input)
print(f"\n  → Sortie shape : {feature_map_1.shape}  (N, C, H, W)")
print(f"  Min : {feature_map_1.min():.4f}  Max : {feature_map_1.max():.4f}")
print(f"  Mean: {feature_map_1.mean():.4f}  Std: {feature_map_1.std():.4f}")

# Afficher les 4 feature maps
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for c in range(4):
    axes[c].imshow(feature_map_1[0, c], cmap='viridis')
    axes[c].set_title(f"FM #{c}", fontsize=9)
    axes[c].axis('off')
plt.suptitle("Conv2D #1 — Feature Maps (4 canaux)", fontsize=12)
plt.tight_layout()
plt.savefig(join(OUT_DIR, '02_conv1_feature_maps.png'), dpi=120)
plt.close()
print(f"  → Feature maps sauvegardées : traces/out/02_conv1_feature_maps.png")

# Aperçu valeurs d'une feature map
print(f"\n  Aperçu feature map #0 (coin 6×6) :")
print(f"  {feature_map_1[0, 0, :6, :6].round(2)}")


# ────────────────────────────────────────────────────────────────────────────
# 4️⃣  RELU #1
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⚡ ÉTAPE 4 — RELU #1")
print("=" * 70)

relu1 = ReLU()
after_relu1 = relu1.forward(feature_map_1)

print(f"  Shape : {after_relu1.shape} (inchangée)")
print(f"  Avant ReLU  — min={feature_map_1.min():.4f}  max={feature_map_1.max():.4f}")
print(f"  Après ReLU  — min={after_relu1.min():.4f}  max={after_relu1.max():.4f}")
print(f"  Négatifs tués : {(feature_map_1 < 0).sum()} → 0")
print(f"  Valeurs activées : {after_relu1.sum():.2f} (somme totale)")

fig, axes = plt.subplots(2, 4, figsize=(12, 5))
for c in range(4):
    axes[0, c].imshow(feature_map_1[0, c], cmap='viridis')
    axes[0, c].set_title(f"Avant ReLU #{c}", fontsize=8)
    axes[0, c].axis('off')
    axes[1, c].imshow(after_relu1[0, c], cmap='viridis')
    axes[1, c].set_title(f"Après ReLU #{c}", fontsize=8)
    axes[1, c].axis('off')
plt.suptitle("ReLU #1 — Avant / Après", fontsize=12)
plt.tight_layout()
plt.savefig(join(OUT_DIR, '03_relu1.png'), dpi=120)
plt.close()
print(f"  → Sauvegardée : traces/out/03_relu1.png")


# ────────────────────────────────────────────────────────────────────────────
# 5️⃣  MAXPOOL #1  (2×2, stride=2)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🔽 ÉTAPE 5 — MAXPOOL #1  (2×2, stride=2)")
print("=" * 70)

pool1 = MaxPool2D(pool_size=2, stride=2)
print(f"  Couche : {pool1}")

after_pool1 = pool1.forward(after_relu1)

print(f"  Entrée  : {after_relu1.shape}  →  28×28")
print(f"  Sortie  : {after_pool1.shape}  →  14×14")
print(f"  Min : {after_pool1.min():.4f}  Max : {after_pool1.max():.4f}")

# Vérif : première fenêtre 2×2 d'une feature map
print(f"\n  Vérification fenêtre top-left, FM #0 :")
first_window = after_relu1[0, 0, :2, :2]
print(f"    Fenêtre 2×2 :")
print(f"      {first_window.round(3)}")
print(f"    Max retenu : {after_pool1[0, 0, 0, 0]:.4f}")
print(f"    Max réel    : {first_window.max():.4f}")
print(f"    ✅ Match !" if abs(after_pool1[0, 0, 0, 0] - first_window.max()) < 1e-6 else "    ❌ Problème !")

fig, axes = plt.subplots(2, 4, figsize=(12, 5))
for c in range(4):
    axes[0, c].imshow(after_relu1[0, c], cmap='viridis')
    axes[0, c].set_title(f"Avant Pool #{c}\n(28×28)", fontsize=8)
    axes[0, c].axis('off')
    axes[1, c].imshow(after_pool1[0, c], cmap='viridis')
    axes[1, c].set_title(f"Après Pool #{c}\n(14×14)", fontsize=8)
    axes[1, c].axis('off')
plt.suptitle("MaxPool #1 — 28×28 → 14×14", fontsize=12)
plt.tight_layout()
plt.savefig(join(OUT_DIR, '04_pool1.png'), dpi=120)
plt.close()
print(f"  → Sauvegardée : traces/out/04_pool1.png")


# ────────────────────────────────────────────────────────────────────────────
# 6️⃣  CONV2D #2  (4 → 8 canaux, kernel 3×3, pad=1)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🔲 ÉTAPE 6 — CONV2D #2  (4 → 8, kernel=3, stride=1, pad=1)")
print("=" * 70)

conv2 = Conv2D(in_channels=4, out_channels=8, kernel_size=3, stride=1, pad=1)
print(f"  Couche : {conv2}")
print(f"  Poids   : kernels shape = {conv2.kernels.shape}")

feature_map_2 = conv2.forward(after_pool1)
print(f"  → Sortie shape : {feature_map_2.shape}")
print(f"  Min : {feature_map_2.min():.4f}  Max : {feature_map_2.max():.4f}")
print(f"  Mean: {feature_map_2.mean():.4f}  Std: {feature_map_2.std():.4f}")

# Afficher les 8 feature maps
fig, axes = plt.subplots(2, 4, figsize=(12, 5))
for c in range(8):
    axes[c // 4, c % 4].imshow(feature_map_2[0, c], cmap='viridis')
    axes[c // 4, c % 4].set_title(f"FM #{c}", fontsize=8)
    axes[c // 4, c % 4].axis('off')
plt.suptitle("Conv2D #2 — Feature Maps (8 canaux, 14×14)", fontsize=12)
plt.tight_layout()
plt.savefig(join(OUT_DIR, '05_conv2_feature_maps.png'), dpi=120)
plt.close()
print(f"  → Feature maps sauvegardées : traces/out/05_conv2_feature_maps.png")


# ────────────────────────────────────────────────────────────────────────────
# 7️⃣  RELU #2
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⚡ ÉTAPE 7 — RELU #2")
print("=" * 70)

relu2 = ReLU()
after_relu2 = relu2.forward(feature_map_2)
print(f"  Shape : {after_relu2.shape}")
print(f"  Négatifs tués : {(feature_map_2 < 0).sum()} valeurs mises à 0")
print(f"  Activation totale : {after_relu2.sum():.2f}")


# ────────────────────────────────────────────────────────────────────────────
# 8️⃣  MAXPOOL #2  (2×2, stride=2)  → 14×14 → 7×7
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🔽 ÉTAPE 8 — MAXPOOL #2  (2×2, stride=2)  → 7×7")
print("=" * 70)

pool2 = MaxPool2D(pool_size=2, stride=2)
after_pool2 = pool2.forward(after_relu2)

print(f"  Entrée  : {after_relu2.shape}  →  14×14")
print(f"  Sortie  : {after_pool2.shape}  →   7×7")
print(f"  Min : {after_pool2.min():.4f}  Max : {after_pool2.max():.4f}")

fig, axes = plt.subplots(2, 4, figsize=(12, 5))
for c in range(8):
    axes[c // 4, c % 4].imshow(after_pool2[0, c], cmap='viridis')
    axes[c // 4, c % 4].set_title(f"FM #{c} (7×7)", fontsize=8)
    axes[c // 4, c % 4].axis('off')
plt.suptitle("MaxPool #2 — 8 feature maps 7×7", fontsize=12)
plt.tight_layout()
plt.savefig(join(OUT_DIR, '06_pool2_final_features.png'), dpi=120)
plt.close()
print(f"  → Feature maps finales : traces/out/06_pool2_final_features.png")


# ────────────────────────────────────────────────────────────────────────────
# 9️⃣  FLATTEN  → vecteur de 8×7×7 = 392
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📐 ÉTAPE 9 — FLATTEN  (8×7×7 → 392)")
print("=" * 70)

flatten = Flatten()
flattened = flatten.forward(after_pool2)

print(f"  Entrée  : {after_pool2.shape}")
print(f"  Sortie  : {flattened.shape}")
print(f"  Vérif   : 8 × 7 × 7 = {8 * 7 * 7}")
print(f"  Min : {flattened.min():.4f}  Max : {flattened.max():.4f}  Mean : {flattened.mean():.4f}")
print(f"  Premières 10 valeurs : {flattened[0, :10].round(4).tolist()}")


# ────────────────────────────────────────────────────────────────────────────
# 🔟  RÉCAPITULATIF — Évolution des dimensions
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📊 RÉCAPITULATIF — Évolution des dimensions")
print("=" * 70)

steps = [
    ("Image originale",                image_raw.shape,      "(H, W)"),
    ("Normalisée [0,1]",               image_norm.shape,     "(H, W)"),
    ("+ Canal + Batch",                image_bch.shape,      "(N, H, W, C)"),
    ("→ Channels First",               image_input.shape,    "(N, C, H, W)"),
    ("Conv2D #1 (1→4, k=3, p=1)",     feature_map_1.shape,  ""),
    ("ReLU #1",                        after_relu1.shape,    "inchangée"),
    ("MaxPool #1 (2×2, s=2)",          after_pool1.shape,    "28→14"),
    ("Conv2D #2 (4→8, k=3, p=1)",     feature_map_2.shape,  ""),
    ("ReLU #2",                        after_relu2.shape,    "inchangée"),
    ("MaxPool #2 (2×2, s=2)",          after_pool2.shape,    "14→7"),
    ("Flatten",                        flattened.shape,      "(N, 392)"),
]

print(f"  {'Étape':<35} {'Shape':<22} {'Notes'}")
print(f"  {'─'*35} {'─'*22} {'─'*20}")
for name, shape, note in steps:
    print(f"  {name:<35} {str(shape):<22} {note}")

print(f"\n  ✅ 1 image MNIST (28×28) → vecteur de 392 features après 2 blocs Conv+Pool !")


# ────────────────────────────────────────────────────────────────────────────
# 🖼️  Image récapitulative : tout le pipeline en une image
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🎨 Génération de l'image récapitulative du pipeline")
print("=" * 70)

# Image récap — layout custom avec add_gridspec
fig = plt.figure(figsize=(20, 9))
gs = fig.add_gridspec(3, 10, hspace=0.35, wspace=0.25)

# ── Rangée 0 : Original → Conv1 FM → Pool1 ──
ax = fig.add_subplot(gs[0, 0])
ax.imshow(image_raw, cmap='gray')
ax.set_title("Original 28×28 ×1", fontsize=8)
ax.axis('off')

for c in range(4):
    ax = fig.add_subplot(gs[0, 2 + c])
    ax.imshow(feature_map_1[0, c], cmap='viridis')
    ax.set_title(f"Conv1 FM#{c} 28×28", fontsize=7)
    ax.axis('off')

ax = fig.add_subplot(gs[0, 7])
ax.imshow(after_pool1[0, 0], cmap='viridis')
ax.set_title("Pool1 14×14", fontsize=8)
ax.axis('off')

ax = fig.add_subplot(gs[0, 8])
ax.imshow(feature_map_2[0, 0], cmap='viridis')
ax.set_title("Conv2 14×14", fontsize=8)
ax.axis('off')

# ── Rangée 1 : texte du pipeline ──
ax = fig.add_subplot(gs[1, :])
ax.text(0.5, 0.5,
    "Cnv1(1→4)  →  ReLU  →  Pool1(2×2)  →  Cnv2(4→8)  →  ReLU  →  Pool2(2×2)  →  Flatten",
    ha='center', va='center', fontsize=14, fontweight='bold', color='#333')
ax.axis('off')

# ── Rangée 2 : 8 final feature maps + vecteur ──
for c in range(8):
    ax = fig.add_subplot(gs[2, c])
    ax.imshow(after_pool2[0, c], cmap='viridis')
    ax.set_title(f"Final FM#{c} 7×7", fontsize=7)
    ax.axis('off')

# Vecteur aplati
ax = fig.add_subplot(gs[2, 8:10])
vals_2d = flattened[0].reshape(14, 28)  # 14×28 = 392
ax.imshow(vals_2d, cmap='viridis', aspect='auto')
ax.set_title("Flatten → 392 features", fontsize=8)
ax.axis('off')

plt.suptitle("CNN-Handmade — Pipeline complet : 28×28 → 392 features (chiffre 3)",
             fontsize=14, fontweight='bold')
plt.savefig(join(OUT_DIR, '00_pipeline_recap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → Image récap : traces/out/00_pipeline_recap.png")

print("\n✅ Trace terminée ! Tous les résultats dans traces/out/")
