#!/usr/bin/env python3
"""
pgd.py — Attaque PGD (Projected Gradient Descent) sur CNN-Handmade

PGD (Madry et al., 2018) : version itérative de FGSM, considérée comme
l'attaque L_inf la plus forte ("gold standard" des attaques adversariales).

    x_0   = x + U(-eps, eps)              # démarrage aléatoire dans la boule
    x_{t+1} = clip( x_t + alpha * sign(grad_x), x - eps, x + eps )   # pas de gradient
    x_{t+1} = clip( x_{t+1}, 0, 1 )       # projection dans l'espace image

Chaque étape fait un petit pas de gradient (alpha), puis *projette* le
résultat dans la boule L_inf de rayon eps autour de l'original. C'est la
projection qui donne son nom à l'attaque : on ne sort jamais de la boule.

Pourquoi c'est plus fort que FGSM :
- FGSM fait UN seul grand pas : il peut "dépasser" l'optimum et rater.
- PGD fait plusieurs petits pas, chacun recalcule le gradient au point
  courant : il converge vers un vrai maximum local de la loss.

Usage :
    python3 adversarial/scripts/pgd.py --dataset mnist --weights model_weights_full.npz
    python3 adversarial/scripts/pgd.py --eps 0.1 0.2 0.3 --steps 40 --n 500
    python3 adversarial/scripts/pgd.py --no-random-start --seed 0
    python3 adversarial/scripts/pgd.py --compare          # courbe FGSM vs PGD

Sorties :
    - Résumé chiffré dans le terminal (accuracy, taux de flip)
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

# On réutilise le code commun de fgsm.py (modèle, données, visualisation)
from adversarial.scripts.fgsm import build_model, load_data, accuracy, label_str, save_grid


# ──────────────────────────────────────────────────────────────────────────
#  L'attaque
# ──────────────────────────────────────────────────────────────────────────

def pgd(model, x, y_onehot, eps, steps=20, alpha=None, random_start=True, rng=None):
    """
    Attaque PGD L_inf : plusieurs pas de gradient projetés dans la boule eps.

    Args :
        model        : CNN entraîné
        x            : batch d'images (N, 1, 28, 28), normalisées [0, 1]
        y_onehot     : one-hot de la classe de référence (vraie ou cible)
        eps          : rayon de la boule L_inf
        steps        : nombre d'itérations
        alpha        : taille du pas (défaut : eps / 4, comme Madry et al.)
        random_start : True → démarrer depuis x + U(-eps, eps) (reco Madry)
                       False → démarrer depuis x (déterministe)
        rng          : générateur aléatoire (reproductibilité)

    Retourne : x_adv (N, 1, 28, 28), bruit final (pour visualisation)
    """
    if rng is None:
        rng = np.random.RandomState()
    if alpha is None:
        alpha = eps / 4.0

    # Borne inf/sup de la boule L_inf autour de chaque image
    x_min = np.clip(x - eps, 0.0, 1.0)
    x_max = np.clip(x + eps, 0.0, 1.0)

    # Démarrage : bruit uniforme dans la boule (ou x propre)
    if random_start:
        x_adv = x + rng.uniform(-eps, eps, size=x.shape)
    else:
        x_adv = x.copy()
    x_adv = np.clip(x_adv, x_min, x_max)

    for _ in range(steps):
        # Gradient de la loss PAR RAPPORT À L'ENTRÉE au point courant
        logits = model.forward(x_adv)
        model.loss_fn.forward(logits, y_onehot)
        grad = model.loss_fn.backward()
        dx = model.backward(grad)

        # Pas de gradient signé
        x_adv = x_adv + alpha * np.sign(dx)

        # Projection : dans la boule (x_min, x_max) puis dans [0, 1]
        x_adv = np.clip(x_adv, x_min, x_max)
        x_adv = np.clip(x_adv, 0.0, 1.0)

    return x_adv, x_adv - x


# ──────────────────────────────────────────────────────────────────────────
#  Évaluation
# ──────────────────────────────────────────────────────────────────────────

def run_attack(model, x, y, eps, steps, alpha, random_start, rng, targeted, target_class):
    """Attaque le batch avec un epsilon donné → (acc_adv, flip, targeted_success, x_adv, noise)."""
    logits_probe = model.forward(x[:1])
    C = logits_probe.shape[1]

    if targeted:
        y_ref = np.tile(np.eye(C)[target_class], (len(x), 1))
    else:
        y_ref = np.eye(C)[y]

    x_adv, noise = pgd(model, x, y_ref, eps, steps, alpha, random_start, rng)

    preds_clean = model.predict(x)
    preds_adv = model.predict(x_adv)

    acc_adv = (preds_adv == y).mean()
    flip = (preds_adv != preds_clean).mean()
    targeted_success = (preds_adv == target_class).mean() if targeted else 0.0

    return acc_adv, flip, targeted_success, x_adv, noise


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attaque PGD sur CNN-Handmade")
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--weights", default=None, help="fichier .npz des poids")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--n", type=int, default=500, help="images attaquées")
    parser.add_argument("--steps", type=int, default=20, help="itérations PGD")
    parser.add_argument("--alpha", type=float, default=None, help="pas (défaut eps/4)")
    parser.add_argument("--no-random-start", action="store_true",
                        help="démarrer depuis x (déterministe) au lieu de bruit uniforme")
    parser.add_argument("--seed", type=int, default=42, help="seed aléatoire")
    parser.add_argument("--targeted", action="store_true", help="attaque ciblée")
    parser.add_argument("--target", type=int, default=3, help="classe cible (ciblée)")
    parser.add_argument("--compare", action="store_true",
                        help="lance aussi FGSM et trace la courbe FGSM vs PGD")
    args = parser.parse_args()

    # ── Modèle ──
    model = build_model(args.dataset)
    if args.weights is None:
        args.weights = "models/model_weights_full.npz" if args.dataset == "mnist" \
                       else "models/emnist_letters_weights.npz"
    weights_path = join(ROOT_DIR, args.weights)
    model.load_weights(weights_path)
    print(f"[MODEL] {args.dataset.upper()} <- {args.weights}")

    # ── Données ──
    x, y = load_data(args.dataset, args.n)
    acc_clean = accuracy(model, x, y)
    print(f"[DATA] {len(x)} images de test -- accuracy propre : {acc_clean:.1%}")

    rng = np.random.RandomState(args.seed)

    # ── Boucle sur les epsilons ──
    tag = "tgt" if args.targeted else "untgt"
    rs_tag = "rs" if not args.no_random_start else "det"
    print(f"\n[ATTACK] PGD {args.steps} steps, alpha=eps/4, "
          f"start={'random' if not args.no_random_start else 'deterministe'}")
    print(f"{'eps':>6} | {'acc attaque':>12} | {'flip':>6} | {'cible':>8} | {'perte acc':>10}")
    print("-" * 55)

    results = []
    for eps in args.eps:
        acc_adv, flip, tgt_success, x_adv, noise = run_attack(
            model, x, y, eps, args.steps, args.alpha,
            not args.no_random_start, rng, args.targeted, args.target)
        loss = acc_clean - acc_adv
        results.append((eps, acc_adv, flip, tgt_success))
        print(f"{eps:>6.2f} | {acc_adv:>12.1%} | {flip:>6.1%} | {tgt_success:>8.1%} | {loss:>10.1%}")

        out_dir = join(ROOT_DIR, "adversarial", "results")
        import os
        os.makedirs(out_dir, exist_ok=True)
        save_grid(x, y, x_adv, noise, eps, args.dataset,
                  join(out_dir, f"pgd_{tag}_{rs_tag}_eps{eps}_s{args.steps}.png"))

    # ── Courbes ──
    out_dir = join(ROOT_DIR, "adversarial", "results")

    # Courbe PGD seule
    eps_list = [r[0] for r in results]
    acc_list = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps_list, acc_list, marker="o", linewidth=2, color="crimson",
            label=f"PGD ({args.steps} steps)")
    ax.axhline(acc_clean, color="green", linestyle="--", label=f"propre ({acc_clean:.1%})")
    ax.set_xlabel("eps (rayon de la boule L_inf)")
    ax.set_ylabel("Accuracy sur images attaquees")
    ax.set_title(f"PGD {args.dataset.upper()} -- accuracy vs eps")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    curve_path = join(out_dir, f"pgd_curve_{tag}_{rs_tag}_s{args.steps}.png")
    plt.savefig(curve_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Courbe -> {curve_path}")

    # ── Comparaison FGSM vs PGD ──
    if args.compare:
        from adversarial.scripts.fgsm import run_attack as fgsm_run
        print("\n[COMPARE] Lancement FGSM sur les memes epsilons...")
        fgsm_acc = []
        for eps in args.eps:
            acc_adv, _, _, _, _ = fgsm_run(model, x, y, eps, False, args.target)
            fgsm_acc.append(acc_adv)
            print(f"   FGSM eps={eps:.2f} -> acc {acc_adv:.1%}")

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(eps_list, fgsm_acc, marker="o", linewidth=2, label="FGSM (1 step)")
        ax.plot(eps_list, acc_list, marker="s", linewidth=2, color="crimson",
                label=f"PGD ({args.steps} steps)")
        ax.axhline(acc_clean, color="green", linestyle="--", label=f"propre ({acc_clean:.1%})")
        ax.set_xlabel("eps (amplitude du bruit)")
        ax.set_ylabel("Accuracy sur images attaquees")
        ax.set_title(f"FGSM vs PGD -- {args.dataset.upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        cmp_path = join(out_dir, f"fgsm_vs_pgd_{tag}_{rs_tag}_s{args.steps}.png")
        plt.savefig(cmp_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Comparaison -> {cmp_path}")


if __name__ == "__main__":
    main()
