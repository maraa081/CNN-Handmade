#!/usr/bin/env python3
"""
defend.py — Adversarial Training (la défense) sur CNN-Handmade

Principe (Goodfellow et al. 2015, Madry et al. 2018) : on réentraîne le
modèle en lui montrant PENDANT l'entraînement des exemples adverses.
Le modèle apprend à être robuste : les gradients de l'attaquant ne le
trompent plus car il a "vu" ce type de bruit.

Boucle d'entraînement :
    pour chaque batch (x, y) :
        x_adv = FGSM(modèle_courant, x, y, eps)   # attaque à la volée
        entraîne sur [x ; x_adv]                   # propre + adverses

On attaque avec le modèle EN TRAIN DE s'entraîner : les exemples adverses
suivent l'évolution du modèle (c'est ce qui rend la défense forte).

Usage :
    python3 adversarial/scripts/defend.py                     # défaut : 5000 img, 3 epochs
    python3 adversarial/scripts/defend.py --n-train 10000 --epochs 5 --eps 0.15
    python3 adversarial/scripts/defend.py --quick             # test rapide (800 img, 1 epoch)
    python3 adversarial/scripts/defend.py --eval-pgd          # évalue aussi sous PGD

Sorties :
    - Poids du modèle défendu : defend_mnist_weights.npz (racine du repo)
    - Comparatif robustesse vs modèle standard dans le terminal + results/
"""

import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))  # -> CNN-Handmade/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import MNISTLoader, normalize, add_channel_dim
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN

from adversarial.scripts.fgsm import build_model, load_data, accuracy
from adversarial.scripts.pgd import pgd


# --------------------------------------------------------------------------
#  FGSM sur un batch (réutilisée à chaque étape d'entraînement)
# --------------------------------------------------------------------------

def fgsm_batch(model, x, y, eps):
    """Exemples adverses FGSM pour un batch (non ciblé)."""
    logits = model.forward(x)
    C = logits.shape[1]
    model.loss_fn.forward(logits, np.eye(C)[y])
    grad = model.loss_fn.backward()
    dx = model.backward(grad)
    return np.clip(x + eps * np.sign(dx), 0.0, 1.0)


# --------------------------------------------------------------------------
#  Entraînement adversarial
# --------------------------------------------------------------------------

def train_adversarial(model, x_train, y_train, epochs, batch_size, eps, lr=0.01, verbose=True):
    """
    Boucle d'entraînement avec exemples adverses générés à la volée.

    Chaque batch est attaqué (FGSM, eps donné) puis on entraîne sur
    [batch propre ; batch adversarial] concaténés.
    """
    N = len(x_train)
    rng = np.random.RandomState(0)
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        # Shuffle
        idx = rng.permutation(N)
        epoch_loss, correct, total = 0.0, 0, 0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            bx = x_train[batch_idx]
            by = y_train[batch_idx]
            by_oh = np.eye(10)[by]

            # 1) Attaque du batch avec le modèle courant
            model.eval_mode()
            bx_adv = fgsm_batch(model, bx, by, eps)
            model.train_mode()

            # 2) Entraînement sur propre + adverses
            cx = np.concatenate([bx, bx_adv], axis=0)
            cy = np.concatenate([by_oh, by_oh], axis=0)

            logits = model.forward(cx)
            loss = model.loss_fn.forward(logits, cy)
            grad = model.loss_fn.backward()
            model.backward(grad)
            model.update(lr)

            epoch_loss += loss * len(cx)
            total += len(cx)
            correct += (np.argmax(logits, axis=1) == np.argmax(cy, axis=1)).sum()
            n_batches += 1

        history["loss"].append(epoch_loss / total)
        history["accuracy"].append(correct / total)
        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs} -- loss: {epoch_loss / total:.4f} -- acc: {correct / total:.4f}")

    return history


# --------------------------------------------------------------------------
#  Évaluation robustesse
# --------------------------------------------------------------------------

def robustness(model, x, y, eps_list, use_pgd=False, steps=20):
    """Accuracy propre + accuracy sous attaque pour chaque eps."""
    results = {}
    results["clean"] = accuracy(model, x, y)
    rng = np.random.RandomState(42)
    for eps in eps_list:
        if use_pgd:
            C = model.forward(x[:1]).shape[1]
            y_ref = np.eye(C)[y]
            x_adv, _ = pgd(model, x, y_ref, eps, steps=steps, rng=rng)
        else:
            x_adv = fgsm_batch(model, x, y, eps)
        results[eps] = accuracy(model, x_adv, y)
    return results


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Adversarial training sur CNN-Handmade")
    parser.add_argument("--n-train", type=int, default=5000, help="images d'entraînement")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.15,
                        help="amplitude FGSM pendant l'entraînement")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true", help="test rapide (800 img, 1 epoch)")
    parser.add_argument("--eval-pgd", action="store_true", help="évalue aussi sous PGD (long)")
    parser.add_argument("--baseline", action="store_true",
                        help="entraîne aussi un modèle STANDARD sur les mêmes données (comparaison équitable)")
    parser.add_argument("--load", default=None,
                        help="ne pas entraîner : évaluer ce modèle déjà entraîné (robustesse FGSM + PGD)")
    parser.add_argument("--out", default="models/defend_mnist_weights.npz")
    args = parser.parse_args()

    if args.quick:
        args.n_train, args.epochs = 800, 1

    # -- Données --
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(x_train), size=min(args.n_train, len(x_train)), replace=False)
    x_tr = normalize(add_channel_dim(x_train[idx])).transpose(0, 3, 1, 2)
    y_tr = y_train[idx]
    print(f"[DATA] {len(x_tr)} images d'entraînement (batch {args.batch}, {args.epochs} epochs, eps={args.eps})")

    x_te, y_te = load_data("mnist", 500)
    print(f"[DATA] {len(x_te)} images de test")

    # -- Modèle défendu (même architecture) --
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
    model.add(Dense(128, 10))

    # -- Mode évaluation seule : charger un modèle déjà entraîné --
    if args.load:
        load_path = join(ROOT_DIR, args.load)
        print(f"[LOAD] Évaluation seule du modèle -> {load_path}")
        model.load_weights(load_path)
        x_te, y_te = load_data("mnist", 500)
        eps_list = [0.05, 0.1, 0.2, 0.3]

        print("\n[EVAL] Robustesse FGSM (500 images de test)")
        adv_res = robustness(model, x_te, y_te, eps_list)
        print(f"{'eps':>6} | {'défendu':>10}")
        print("-" * 22)
        for eps in ["clean"] + eps_list:
            print(f"{str(eps):>6} | {adv_res[eps]:>10.1%}")

        if args.eval_pgd:
            print("\n[EVAL] Robustesse PGD (20 steps)")
            adv_pgd = robustness(model, x_te, y_te, eps_list, use_pgd=True)
            print(f"{'eps':>6} | {'défendu PGD':>14}")
            print("-" * 26)
            for eps in eps_list:
                print(f"{str(eps):>6} | {adv_pgd[eps]:>14.1%}")

        # Courbe comparée vs standard
        std_model = build_model("mnist")
        std_model.load_weights(join(ROOT_DIR, "models", "model_weights_full.npz"))
        std_res = robustness(std_model, x_te, y_te, eps_list)
        out_dir = join(ROOT_DIR, "adversarial", "results")
        import os
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(eps_list, [std_res[e] for e in eps_list], marker="o", linewidth=2,
                label=f"standard (clean {std_res['clean']:.1%})")
        ax.plot(eps_list, [adv_res[e] for e in eps_list], marker="s", linewidth=2, color="green",
                label=f"défendu (clean {adv_res['clean']:.1%})")
        ax.set_xlabel("eps (amplitude du bruit)")
        ax.set_ylabel("Accuracy sous attaque")
        ax.set_title("Modèle défendu vs standard -- MNIST")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        curve_path = join(out_dir, "defend_curve_fgsm.png")
        plt.savefig(curve_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n[PLOT] Courbe -> {curve_path}")
        return

    # -- Entraînement adversarial --
    print("\n[ADV-TRAIN] Entraînement avec exemples adverses...")
    t0 = time.time()
    history = train_adversarial(model, x_tr, y_tr, args.epochs, args.batch, args.eps, args.lr)
    m, s = divmod(time.time() - t0, 60)
    print(f"[ADV-TRAIN] Terminé en {int(m)}m{int(s)}s")

    # -- Sauvegarde --
    out_path = join(ROOT_DIR, args.out)
    model.save_weights(out_path)
    print(f"[SAVE] Poids défendus -> {out_path}")

    # -- Baseline équitable : modèle standard entraîné sur les MÊMES données --
    if args.baseline:
        print("\n[BASELINE] Entraînement d'un modèle standard sur les mêmes images...")
        std_same = CNN()
        std_same.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
        std_same.add(ReLU())
        std_same.add(MaxPool2D(2))
        std_same.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
        std_same.add(ReLU())
        std_same.add(MaxPool2D(2))
        std_same.add(Flatten())
        std_same.add(Dense(3136, 128))
        std_same.add(ReLU())
        std_same.add(Dense(128, 10))

        from model import CNN as _  # noqa

        N = len(x_tr)
        rng2 = np.random.RandomState(0)
        for epoch in range(args.epochs):
            idx = rng2.permutation(N)
            epoch_loss, correct, total = 0.0, 0, 0
            for start in range(0, N, args.batch):
                batch_idx = idx[start:start + args.batch]
                bx = x_tr[batch_idx]
                by = np.eye(10)[y_tr[batch_idx]]
                logits = std_same.forward(bx)
                loss = std_same.loss_fn.forward(logits, by)
                grad = std_same.loss_fn.backward()
                std_same.backward(grad)
                std_same.update(args.lr)
                epoch_loss += loss * len(bx)
                total += len(bx)
                correct += (np.argmax(logits, axis=1) == np.argmax(by, axis=1)).sum()
            print(f"  Epoch {epoch + 1}/{args.epochs} -- loss: {epoch_loss / total:.4f} -- acc: {correct / total:.4f}")
        std_same.save_weights(join(ROOT_DIR, "models", "standard_same_data.npz"))
        print(f"[SAVE] Baseline standard -> models/standard_same_data.npz")

        std_res = robustness(std_same, x_te, y_te, eps_list)
        print("\n[EVAL] Comparaison ÉQUITABLE (mêmes données d'entraînement)")
        print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
        print("-" * 42)
        for eps in ["clean"] + eps_list:
            s_acc, d_acc = std_res[eps], adv_res[eps]
            print(f"{str(eps):>6} | {s_acc:>10.1%} | {d_acc:>10.1%} | {d_acc - s_acc:>+8.1%}")

    # -- Comparaison robustesse : modèle standard (full) vs défendu --
    eps_list = [0.05, 0.1, 0.2, 0.3]
    print("\n[EVAL] Robustesse FGSM (500 images de test)")

    std_model = build_model("mnist")
    std_model.load_weights(join(ROOT_DIR, "models", "model_weights_full.npz"))

    std_res = robustness(std_model, x_te, y_te, eps_list)
    adv_res = robustness(model, x_te, y_te, eps_list)

    print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
    print("-" * 42)
    for eps in ["clean"] + eps_list:
        s_acc, d_acc = std_res[eps], adv_res[eps]
        print(f"{str(eps):>6} | {s_acc:>10.1%} | {d_acc:>10.1%} | {d_acc - s_acc:>+8.1%}")

    # -- Évaluation PGD (optionnelle, plus longue) --
    if args.eval_pgd:
        print("\n[EVAL] Robustesse PGD (20 steps)")
        std_pgd = robustness(std_model, x_te, y_te, eps_list, use_pgd=True)
        adv_pgd = robustness(model, x_te, y_te, eps_list, use_pgd=True)
        print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
        print("-" * 42)
        for eps in eps_list:
            print(f"{str(eps):>6} | {std_pgd[eps]:>10.1%} | {adv_pgd[eps]:>10.1%} | {adv_pgd[eps] - std_pgd[eps]:>+8.1%}")

    # -- Courbe comparée --
    out_dir = join(ROOT_DIR, "adversarial", "results")
    import os
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eps_list, [std_res[e] for e in eps_list], marker="o", linewidth=2,
            label=f"standard (clean {std_res['clean']:.1%})")
    ax.plot(eps_list, [adv_res[e] for e in eps_list], marker="s", linewidth=2, color="green",
            label=f"adversarial training (clean {adv_res['clean']:.1%})")
    ax.set_xlabel("eps (amplitude du bruit FGSM)")
    ax.set_ylabel("Accuracy sous attaque")
    ax.set_title(f"Adversarial training -- MNIST (eps train={args.eps})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    curve_path = join(out_dir, "defend_curve_fgsm.png")
    plt.savefig(curve_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Courbe -> {curve_path}")

    # Historique d'entraînement
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], marker="o", linewidth=2, label="loss")
    ax.set_title("Adversarial training -- loss par epoch")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = join(out_dir, "defend_history.png")
    plt.savefig(hist_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Historique -> {hist_path}")


if __name__ == "__main__":
    main()
