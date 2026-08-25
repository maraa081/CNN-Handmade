#!/usr/bin/env python3
"""
fgsm.py — Attaque FGSM (Fast Gradient Sign Method) sur CNN-Handmade

FGSM (Goodfellow, 2014) : une seule étape de gradient pour tromper le modèle.

    x_adv = clip(x + eps * sign(grad_x), 0, 1)

Le gradient est calculé PAR RAPPORT À L'ENTRÉE (l'image), pas par rapport
aux poids. Mon backward() le retourne directement — aucun framework requis.

Usage :
    python3 adversarial/scripts/fgsm.py --dataset mnist --weights model_weights_full.npz
    python3 adversarial/scripts/fgsm.py --dataset emnist --weights emnist_letters_weights_full.npz
    python3 adversarial/scripts/fgsm.py --targeted --target 3
    python3 adversarial/scripts/fgsm.py --eps 0.05 0.1 0.2 0.3 --n 1000

Sorties :
    - Résumé chiffré dans le terminal (accuracy, taux de flip, confiance)
    - Visualisations dans adversarial/results/
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))  # → CNN-Handmade/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, join(ROOT_DIR, "scripts"))

from data import MNISTLoader, EMNISTLoader, normalize, add_channel_dim
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN
from download_emnist import ensure_data

LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ──────────────────────────────────────────────────────────────────────────
#  Modèle & données
# ──────────────────────────────────────────────────────────────────────────

def build_model(dataset):
    """Même architecture que l'entraînement, sortie 10 (MNIST) ou 26 (EMNIST)."""
    num_classes = 10 if dataset == "mnist" else 26
    model = CNN()
    model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(3136, 128))
    model.add(ReLU())
    model.add(Dense(128, num_classes))
    return model


def load_data(dataset, n):
    """Charge n images de test, normalisées, prêtes pour le réseau."""
    if dataset == "mnist":
        loader = MNISTLoader()
        (_, _), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))
    else:
        ensure_data()
        loader = EMNISTLoader("letters")
        (_, _), (x_test, y_test) = loader.load(join(ROOT_DIR, "data", "emnist"))

    rng = np.random.RandomState(42)
    idx = rng.choice(len(x_test), size=min(n, len(x_test)), replace=False)
    x = x_test[idx]
    y = y_test[idx]

    # Normalisation + channels_first (N, 1, 28, 28)
    x = normalize(x)
    x = add_channel_dim(x)            # (N, 28, 28, 1)
    x = x.transpose(0, 3, 1, 2)       # (N, 1, 28, 28)
    return x, y


def label_str(dataset, idx):
    return str(idx) if dataset == "mnist" else LETTERS[idx]


# ──────────────────────────────────────────────────────────────────────────
#  L'attaque
# ──────────────────────────────────────────────────────────────────────────

def fgsm(model, x, y_onehot, eps, targeted=False):
    """
    Génère les versions adverses de x.

    Args :
        model    : CNN entraîné
        x        : batch d'images (N, 1, 28, 28), normalisées
        y_onehot : one-hot de la classe de référence (vraie ou cible)
        eps      : amplitude du bruit
        targeted : True → minimiser la loss (forcer la classe cible)
                   False → maximiser la loss (éloigner de la vraie classe)

    Retourne : x_adv (N, 1, 28, 28), bruit utilisé (pour visualisation)
    """
    # Forward + loss
    logits = model.forward(x)
    model.loss_fn.forward(logits, y_onehot)

    # Gradient de la loss PAR RAPPORT À L'ENTRÉE
    grad = model.loss_fn.backward()
    dx = model.backward(grad)          # (N, 1, 28, 28)

    # Direction : +sign(grad) pour non-ciblé, -sign(grad) pour ciblé
    direction = np.sign(dx)
    if targeted:
        direction = -direction

    noise = eps * direction
    x_adv = np.clip(x + noise, 0.0, 1.0)
    return x_adv, noise


# ──────────────────────────────────────────────────────────────────────────
#  Évaluation
# ──────────────────────────────────────────────────────────────────────────

def accuracy(model, x, y):
    preds = model.predict(x)
    return (preds == y).mean()


def run_attack(model, x, y, eps, targeted, target_class):
    """Attaque le batch avec un epsilon donné → (accuracy, flip_rate, conf_moy)."""
    # Nombre de classes = largeur des logits (après un forward)
    logits_probe = model.forward(x[:1])
    C = logits_probe.shape[1]

    if targeted:
        y_ref = np.tile(np.eye(C)[target_class], (len(x), 1))
    else:
        y_ref = np.eye(C)[y]

    x_adv, noise = fgsm(model, x, y_ref, eps, targeted)

    preds_clean = model.predict(x)
    preds_adv = model.predict(x_adv)

    acc_adv = (preds_adv == y).mean()
    # flip = prédiction changée (par rapport à la prédiction propre)
    flip = (preds_adv != preds_clean).mean()
    # succès ciblé = la prédiction est devenue la classe cible
    targeted_success = (preds_adv == target_class).mean() if targeted else 0.0

    return acc_adv, flip, targeted_success, x_adv, noise


# ──────────────────────────────────────────────────────────────────────────
#  Visualisations
# ──────────────────────────────────────────────────────────────────────────

def save_grid(x, y, x_adv, noise, eps, dataset, out_path, n_show=8):
    """Comparatif : original / bruit (×10) / attaqué, avec prédictions."""
    model = None  # les prédictions sont déjà connues par l'appelant via x_adv
    n = min(n_show, len(x))
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    for i in range(n):
        for j, (img, title) in enumerate([
            (x[i, 0], "Original"),
            (noise[i, 0] * 10, f"Bruit ×10 (ε={eps})"),
            (x_adv[i, 0], "Attaqué"),
        ]):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    plt.suptitle(f"FGSM ε={eps} — {dataset.upper()}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   🖼️  {out_path}")


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attaque FGSM sur CNN-Handmade")
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--weights", default=None, help="fichier .npz des poids")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--n", type=int, default=500, help="images attaquées")
    parser.add_argument("--targeted", action="store_true", help="attaque ciblée")
    parser.add_argument("--target", type=int, default=3, help="classe cible (ciblée)")
    args = parser.parse_args()

    # ── Modèle ──
    model = build_model(args.dataset)
    if args.weights is None:
        args.weights = "models/model_weights_full.npz" if args.dataset == "mnist" \
                       else "models/emnist_letters_weights.npz"
    weights_path = join(ROOT_DIR, args.weights)
    model.load_weights(weights_path)
    print(f"🧠 Modèle chargé : {args.dataset.upper()} ← {args.weights}")

    # ── Données ──
    x, y = load_data(args.dataset, args.n)
    acc_clean = accuracy(model, x, y)
    print(f"📦 {len(x)} images de test — accuracy propre : {acc_clean:.1%}")

    # ── Boucle sur les epsilons ──
    print(f"\n⚔️  Attaque FGSM {'ciblée (→ ' + label_str(args.dataset, args.target) + ')' if args.targeted else 'non ciblée'}")
    print(f"{'ε':>6} | {'acc attaqué':>12} | {'flip':>6} | {'ciblé':>8} | {'perte acc':>10}")
    print("-" * 55)

    results = []
    for eps in args.eps:
        acc_adv, flip, tgt_success, x_adv, noise = run_attack(
            model, x, y, eps, args.targeted, args.target)
        loss = acc_clean - acc_adv
        results.append((eps, acc_adv, flip, tgt_success))
        print(f"{eps:>6.2f} | {acc_adv:>12.1%} | {flip:>6.1%} | {tgt_success:>8.1%} | {loss:>10.1%}")

        # Visualisation
        out_dir = join(ROOT_DIR, "adversarial", "results")
        import os
        os.makedirs(out_dir, exist_ok=True)
        tag = "tgt" if args.targeted else "untgt"
        save_grid(x, y, x_adv, noise, eps, args.dataset,
                  join(out_dir, f"fgsm_{tag}_eps{eps}.png"))

    # ── Graphique accuracy vs eps ──
    eps_list = [r[0] for r in results]
    acc_list = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps_list, acc_list, marker="o", linewidth=2)
    ax.axhline(acc_clean, color="green", linestyle="--", label=f"propre ({acc_clean:.1%})")
    ax.set_xlabel("ε (amplitude du bruit)")
    ax.set_ylabel("Accuracy sur images attaquées")
    ax.set_title(f"FGSM {args.dataset.upper()} — accuracy vs ε")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    curve_path = join(ROOT_DIR, "adversarial", "results", f"fgsm_curve_{tag}.png")
    plt.savefig(curve_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n📈 Courbe → {curve_path}")


if __name__ == "__main__":
    main()
