#!/usr/bin/env python3
"""
transfer.py — Transfert d'attaque entre modèles (boîte noire)

Un exemple adversarial généré contre un modèle A trompe souvent aussi un
autre modèle B, même si B a été entraîné séparément. C'est la
**transferabilité** (Papernot et al., 2016) : c'est ce qui rend les
attaques dangereuses en pratique — un attaquant n'a pas besoin d'accéder
au modèle cible, il attaque un modèle public similaire et réutilise les
exemples.

Ce script :
1. Charge un modèle SOURCE (attaqué, gradient accessible)
2. Génère des exemples adverses contre lui (FGSM ou PGD)
3. Les rejoue sur un modèle CIBLE (sans gradient)
4. Mesure la transferabilité : acc de la cible sur les exemples adverses

Usage :
    python3 adversarial/scripts/transfer.py --src model_weights_full.npz --dst max_config_weights.npz
    python3 adversarial/scripts/transfer.py --src ... --dst ... --attack pgd --eps 0.1 0.3
    python3 adversarial/scripts/transfer.py --list-models   # montre les modèles dispo

Sorties :
    - Résumé chiffré dans le terminal (acc source vs acc cible, taux de transfert)
    - Visualisations dans adversarial/results/
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath, basename

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))  # → CNN-Handmade/
sys.path.insert(0, ROOT_DIR)

from adversarial.scripts.fgsm import build_model, load_data, accuracy, save_grid
from adversarial.scripts.pgd import pgd

# Modèles MNIST entraînés différemment (même architecture, sauf max_config
# qui ajoute un Dropout) → bons candidats pour le transfert
KNOWN_MODELS = {
    "full":      "models/model_weights_full.npz",     # entraînement complet
    "classic":   "models/model_weights.npz",          # entraînement classique
    "max_config": "models/max_config_weights.npz",    # Adam + Dropout + L2
}


def build_mnist_model(with_dropout=False):
    """Même architecture que l'entraînement, avec ou sans Dropout."""
    from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense, Dropout
    from model import CNN
    m = CNN()
    m.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
    m.add(ReLU())
    m.add(MaxPool2D(2))
    m.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
    m.add(ReLU())
    m.add(MaxPool2D(2))
    m.add(Flatten())
    m.add(Dense(3136, 128))
    m.add(ReLU())
    if with_dropout:
        m.add(Dropout(p=0.5))
    m.add(Dense(128, 10))
    return m


def load_mnist_model(path):
    """Charge un modèle MNIST en détectant l'architecture (avec/sans Dropout)."""
    m = build_mnist_model()
    try:
        m.load_weights(path)
    except KeyError:
        m = build_mnist_model(with_dropout=True)
        m.load_weights(path)
    return m


# ──────────────────────────────────────────────────────────────────────────
#  Attaque : FGSM ou PGD (toujours non ciblée ici)
# ──────────────────────────────────────────────────────────────────────────

def attack(model, x, y, eps, attack_type, steps=20, rng=None):
    """Génère les exemples adverses contre `model`. Retourne x_adv, noise."""
    C = model.forward(x[:1]).shape[1]
    y_ref = np.eye(C)[y]

    if attack_type == "fgsm":
        logits = model.forward(x)
        model.loss_fn.forward(logits, y_ref)
        grad = model.loss_fn.backward()
        dx = model.backward(grad)
        noise = eps * np.sign(dx)
        x_adv = np.clip(x + noise, 0.0, 1.0)
        return x_adv, noise

    # PGD
    return pgd(model, x, y_ref, eps, steps=steps, rng=rng)


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transfert d'attaque entre modèles")
    parser.add_argument("--src", default="full", help="modèle source (gradient dispo)")
    parser.add_argument("--dst", default="max_config", help="modèle cible (boîte noire)")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--n", type=int, default=500, help="images attaquées")
    parser.add_argument("--steps", type=int, default=20, help="itérations PGD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-models", action="store_true",
                        help="affiche les modèles connus et sort")
    args = parser.parse_args()

    if args.list_models:
        print("Modèles MNIST disponibles (même architecture, entraînements différents) :")
        for name, path in KNOWN_MODELS.items():
            print(f"   {name:<12} -> {path}")
        sys.exit(0)

    # Résolution des noms courts → fichiers
    def resolve(name):
        return join(ROOT_DIR, KNOWN_MODELS.get(name, name))

    src_path, dst_path = resolve(args.src), resolve(args.dst)
    print(f"[SRC] {basename(src_path)}")
    print(f"[DST] {basename(dst_path)}")

    # ── Modèles ──
    src_model = load_mnist_model(src_path)
    dst_model = load_mnist_model(dst_path)

    # ── Données ──
    x, y = load_data("mnist", args.n)
    acc_src = accuracy(src_model, x, y)
    acc_dst = accuracy(dst_model, x, y)
    print(f"[DATA] {len(x)} images -- acc source {acc_src:.1%} | acc cible {acc_dst:.1%}")

    rng = np.random.RandomState(args.seed)

    # ── Boucle sur les epsilons ──
    print(f"\n[ATTACK] {args.attack.upper()} contre {basename(src_path)} -> test sur {basename(dst_path)}")
    print(f"{'eps':>6} | {'acc src adv':>12} | {'acc dst adv':>12} | {'transfert':>10}")
    print("-" * 55)

    results = []
    out_dir = join(ROOT_DIR, "adversarial", "results")
    import os
    os.makedirs(out_dir, exist_ok=True)

    for eps in args.eps:
        x_adv, noise = attack(src_model, x, y, eps, args.attack, args.steps, rng)

        acc_src_adv = accuracy(src_model, x_adv, y)
        acc_dst_adv = accuracy(dst_model, x_adv, y)

        # Taux de transfert : parmi les images que la SOURCE s'est trompée,
        # combien trompent aussi la CIBLE (et qui étaient bien prédites par la cible)
        preds_src_clean = src_model.predict(x)
        preds_src_adv = src_model.predict(x_adv)
        preds_dst_clean = dst_model.predict(x)
        preds_dst_adv = dst_model.predict(x_adv)

        correct_src = preds_src_clean == y          # bien prédites par la source
        fooled_src = (preds_src_adv != y) & correct_src   # source trompée par l'attaque
        transfer = (preds_dst_adv != preds_dst_clean)     # cible a changé d'avis
        # Sur les images où la source a été trompée ET la cible était correcte :
        denom = (fooled_src & (preds_dst_clean == y)).sum()
        num = (fooled_src & (preds_dst_clean == y) & transfer).sum()
        transfer_rate = num / denom if denom > 0 else float("nan")

        results.append((eps, acc_src_adv, acc_dst_adv, transfer_rate))
        print(f"{eps:>6.2f} | {acc_src_adv:>12.1%} | {acc_dst_adv:>12.1%} | {transfer_rate:>10.1%}")

        save_grid(x, y, x_adv, noise, eps, "mnist",
                  join(out_dir, f"transfer_{args.attack}_eps{eps}_src{args.src}_dst{args.dst}.png"))

    # ── Courbe : acc source vs acc cible ──
    eps_list = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eps_list, [r[1] for r in results], marker="o", linewidth=2,
            label=f"acc source ({basename(src_path)})")
    ax.plot(eps_list, [r[2] for r in results], marker="s", linewidth=2, color="crimson",
            label=f"acc cible ({basename(dst_path)})")
    ax.axhline(acc_dst, color="green", linestyle="--", label=f"cible propre ({acc_dst:.1%})")
    ax.set_xlabel("eps (amplitude du bruit)")
    ax.set_ylabel("Accuracy sur exemples adverses")
    ax.set_title(f"Transfert {args.attack.upper()} -- {basename(src_path)} -> {basename(dst_path)}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    curve_path = join(out_dir, f"transfer_curve_{args.attack}_src{args.src}_dst{args.dst}.png")
    plt.savefig(curve_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Courbe -> {curve_path}")


if __name__ == "__main__":
    main()
