#!/usr/bin/env python3
"""
eval_defended.py — Évaluation du modèle défendu SANS ré-entraînement.

Charge les poids déjà entraînés et produit :
  1. [EVAL] Comparaison ÉQUITABLE (FGSM) : défendu vs standard entraîné
     sur les MÊMES 5000 images (standard_same_data.npz)
  2. [EVAL] PGD (20 steps) : défendu vs modèle standard complet
     (model_weights_full.npz, la référence historique)

Usage :
    python3 adversarial/scripts/eval_defended.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))  # -> CNN-Handmade/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))

from model import CNN
from adversarial.scripts.fgsm import build_model, load_data, accuracy
from adversarial.scripts.pgd import pgd

EPS_LIST = [0.05, 0.1, 0.2, 0.3]
PGD_STEPS = 20


def fgsm_batch(model, x, y, eps):
    logits = model.forward(x)
    C = logits.shape[1]
    model.loss_fn.forward(logits, np.eye(C)[y])
    grad = model.loss_fn.backward()
    dx = model.backward(grad)
    return np.clip(x + eps * np.sign(dx), 0.0, 1.0)


def robustness_fgsm(model, x, y, eps_list):
    res = {"clean": accuracy(model, x, y)}
    for eps in eps_list:
        res[eps] = accuracy(model, fgsm_batch(model, x, y, eps), y)
    return res


def robustness_pgd(model, x, y, eps_list, steps=PGD_STEPS):
    res = {}
    rng = np.random.RandomState(42)
    C = model.forward(x[:1]).shape[1]
    y_ref = np.eye(C)[y]
    for eps in eps_list:
        x_adv, _ = pgd(model, x, y_ref, eps, steps=steps, rng=rng)
        res[eps] = accuracy(model, x_adv, y)
    return res


def main():
    # -- Données de test (500 images, comme les runs précédents) --
    x_te, y_te = load_data("mnist", 500)
    print(f"[DATA] {len(x_te)} images de test")

    # -- Modèles --
    adv_model = build_model("mnist")
    adv_model.load_weights(join(ROOT_DIR, "models", "defend_mnist_weights.npz"))
    print("[LOAD] défendu            -> models/defend_mnist_weights.npz")

    same_model = build_model("mnist")
    same_model.load_weights(join(ROOT_DIR, "models", "standard_same_data.npz"))
    print("[LOAD] standard (mêmes données) -> models/standard_same_data.npz")

    full_model = build_model("mnist")
    full_model.load_weights(join(ROOT_DIR, "models", "model_weights_full.npz"))
    print("[LOAD] standard complet   -> models/model_weights_full.npz")

    # -- 1) Comparaison ÉQUITABLE : mêmes données d'entraînement --
    print("\n[EVAL] Robustesse FGSM (500 images de test)")
    adv_res = robustness_fgsm(adv_model, x_te, y_te, EPS_LIST)
    same_res = robustness_fgsm(same_model, x_te, y_te, EPS_LIST)

    print("\n[EVAL] Comparaison ÉQUITABLE (mêmes données d'entraînement)")
    print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
    print("-" * 42)
    for eps in ["clean"] + EPS_LIST:
        s_acc, d_acc = same_res[eps], adv_res[eps]
        print(f"{str(eps):>6} | {s_acc:>10.1%} | {d_acc:>10.1%} | {d_acc - s_acc:>+8.1%}")

    # -- 2) Éval PGD : défendu vs standard complet (référence) --
    print("\n[EVAL] Robustesse PGD (20 steps, random start)")
    adv_pgd = robustness_pgd(adv_model, x_te, y_te, EPS_LIST)
    full_pgd = robustness_pgd(full_model, x_te, y_te, EPS_LIST)
    print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
    print("-" * 42)
    for eps in EPS_LIST:
        print(f"{str(eps):>6} | {full_pgd[eps]:>10.1%} | {adv_pgd[eps]:>10.1%} | {adv_pgd[eps] - full_pgd[eps]:>+8.1%}")

    # -- Courbes --
    out_dir = join(ROOT_DIR, "adversarial", "results")
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Courbe équitable FGSM
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(EPS_LIST, [same_res[e] for e in EPS_LIST], marker="o", linewidth=2,
            label=f"standard (clean {same_res['clean']:.1%})")
    ax.plot(EPS_LIST, [adv_res[e] for e in EPS_LIST], marker="s", linewidth=2, color="green",
            label=f"défendu (clean {adv_res['clean']:.1%})")
    ax.set_xlabel("eps (amplitude du bruit FGSM)")
    ax.set_ylabel("Accuracy sous attaque")
    ax.set_title("Modèle défendu vs standard -- MÊMES données (comparaison équitable)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fair_path = join(out_dir, "defend_fair_fgsm.png")
    plt.savefig(fair_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Courbe équitable -> {fair_path}")

    # Courbe PGD
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(EPS_LIST, [full_pgd[e] for e in EPS_LIST], marker="o", linewidth=2,
            label="standard complet (PGD)")
    ax.plot(EPS_LIST, [adv_pgd[e] for e in EPS_LIST], marker="s", linewidth=2, color="green",
            label="défendu (PGD)")
    ax.set_xlabel("eps (amplitude du bruit PGD)")
    ax.set_ylabel("Accuracy sous attaque")
    ax.set_title("Modèle défendu vs standard complet -- PGD 20 steps")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    pgd_path = join(out_dir, "defend_curve_pgd.png")
    plt.savefig(pgd_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Courbe PGD -> {pgd_path}")

    print("\n[DONE] Évaluation terminée.")


if __name__ == "__main__":
    main()
