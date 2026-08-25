#!/usr/bin/env python3
"""
harden.py — Version durcie du CNN : se défendre contre TOUTES les attaques
précédentes (FGSM, PGD, attaques ciblées, transfert).

On combine trois couches de défense :

  1. Adversarial training (FGSM ou PGD) : réentraîner le modèle avec des
     exemples adverses generes a la volee. C'est LA defense de reference
     (Goodfellow 2015, Madry 2018). Le modele apprend a etre robuste
     parce qu'il a "vu" les attaques pendant l'entrainement.

  2. Feature squeezing (Xu et al. 2018) : reduire la profondeur de bits des
     pixels avant l'inference (8 bits -> 3-4 bits). Les perturbations
     adversariales sont des ecarts minuscules : la quantification les
     ecrase. Defense d'entree, sans retrain, complementaire a la 1.

  3. Evaluation multi-attaques : on verifie la robustesse contre FGSM,
     PGD, attaques CIBLEES et transfert depuis un autre modele. Une
     defense qui ne tient que contre FGSM n'est pas une defense.

Usage :
    # Entrainement + evaluation complete (PGD adversarial training, par defaut)
    python3 adversarial/scripts/harden.py --n-train 5000 --epochs 3

    # Adversarial training FGSM (plus rapide, moins robuste)
    python3 adversarial/scripts/harden.py --attack fgsm --n-train 5000 --epochs 3

    # Evaluation seule d'un modele deja dureci (sans retrain)
    python3 adversarial/scripts/harden.py --load models/defend_pgd_mnist_weights.npz

    # Test rapide (800 images, 1 epoch)
    python3 adversarial/scripts/harden.py --quick

Sorties :
    - Poids dureis : models/defend_pgd_mnist_weights.npz (ou --out)
    - Tableau de robustesse complet dans le terminal
    - Courbes dans adversarial/results/
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

from model import CNN
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from data import MNISTLoader, normalize, add_channel_dim
from adversarial.scripts.fgsm import build_model, load_data, accuracy
from adversarial.scripts.pgd import pgd

EPS_LIST = [0.05, 0.1, 0.2, 0.3]
PGD_STEPS_EVAL = 20


# --------------------------------------------------------------------------
#  Attaques sur batch (pour l'entraînement et l'évaluation)
# --------------------------------------------------------------------------

def fgsm_batch(model, x, y, eps):
    """FGSM non ciblée sur un batch."""
    logits = model.forward(x)
    C = logits.shape[1]
    model.loss_fn.forward(logits, np.eye(C)[y])
    grad = model.loss_fn.backward()
    dx = model.backward(grad)
    return np.clip(x + eps * np.sign(dx), 0.0, 1.0)


def pgd_batch(model, x, y, eps, steps=PGD_STEPS_EVAL, rng=None):
    """PGD non ciblée sur un batch (random start)."""
    C = model.forward(x[:1]).shape[1]
    y_ref = np.eye(C)[y]
    x_adv, _ = pgd(model, x, y_ref, eps, steps=steps, rng=rng)
    return x_adv


# --------------------------------------------------------------------------
#  Feature squeezing — défense d'entrée (Xu et al. 2018)
# --------------------------------------------------------------------------

def feature_squeeze(x, bits=3):
    """
    Réduit la profondeur de bits de l'image : 8 bits -> `bits` bits.

    Principe : une perturbation adversarial ε·sign(∇) est un écart très
    petit sur chaque pixel. En ne gardant que 2^bits niveaux de gris, on
    écrase tout écart inférieur à 1/(2^bits - 1) : le bruit disparaît,
    l'image utile reste (le modèle n'a pas besoin de 256 niveaux pour
    reconnaître un chiffre).

    x: batch (N, 1, 28, 28) normalisé [0, 1] -> même forme, quantifié.
    """
    levels = 2 ** bits
    return np.round(x * (levels - 1)) / (levels - 1)


# --------------------------------------------------------------------------
#  Entraînement adversarial (FGSM ou PGD)
# --------------------------------------------------------------------------

def train_adversarial(model, x_train, y_train, epochs, batch_size, eps,
                      attack="pgd", pgd_steps=7, lr=0.01, verbose=True):
    """
    Boucle d'entraînement avec exemples adverses générés à la volée.

    attack="fgsm" : 1 étape de gradient (rapide, robustesse moyenne).
    attack="pgd"  : attaque itérative PGD pendant l'entraînement (Madry
    et al. 2018) -> robustesse bien supérieure, c'est le gold standard.

    Chaque batch est attaqué avec le modèle EN TRAIN de s'entraîner, puis
    on entraîne sur [batch propre ; batch adversarial] concaténés.
    """
    N = len(x_train)
    rng = np.random.RandomState(0)
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        idx = rng.permutation(N)
        epoch_loss, correct, total = 0.0, 0, 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            bx = x_train[batch_idx]
            by = y_train[batch_idx]
            by_oh = np.eye(10)[by]

            # 1) Attaque du batch avec le modèle courant
            model.eval_mode()
            if attack == "pgd":
                bx_adv = pgd_batch(model, bx, by, eps, steps=pgd_steps, rng=rng)
            else:
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

        history["loss"].append(epoch_loss / total)
        history["accuracy"].append(correct / total)
        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs} -- loss: {epoch_loss / total:.4f} -- acc: {correct / total:.4f}")

    return history


# --------------------------------------------------------------------------
#  Évaluation robustesse
# --------------------------------------------------------------------------

def eval_attack(model, x, y, attack, eps, steps=PGD_STEPS_EVAL, rng=None,
                targeted=False, target_class=None, squeeze_bits=None):
    """Accuracy sous une attaque donnée (optionnellement + feature squeeze)."""
    if squeeze_bits is not None:
        x_in = feature_squeeze(x, squeeze_bits)
    else:
        x_in = x

    if targeted:
        C = model.forward(x_in[:1]).shape[1]
        y_tgt = np.full(len(y), target_class)
        y_ref = np.eye(C)[y_tgt]
        if attack == "pgd":
            x_adv, _ = pgd(model, x_in, y_ref, eps, steps=steps, rng=rng)
        else:
            logits = model.forward(x_in)
            model.loss_fn.forward(logits, y_ref)
            grad = model.loss_fn.backward()
            dx = model.backward(grad)
            x_adv = np.clip(x_in + eps * np.sign(dx), 0.0, 1.0)
    else:
        if attack == "pgd":
            x_adv = pgd_batch(model, x_in, y, eps, steps=steps, rng=rng)
        else:
            x_adv = fgsm_batch(model, x_in, y, eps)

    return accuracy(model, x_adv, y)


def robustness_report(model, x, y, eps_list=EPS_LIST, steps=PGD_STEPS_EVAL,
                      squeeze_bits=None, label="modèle"):
    """
    Tableau de robustesse complet contre toutes les attaques précédentes :
    FGSM, PGD, FGSM ciblée, PGD ciblée. Retourne un dict {attack: {eps: acc}}.
    """
    rng = np.random.RandomState(42)
    report = {}

    print(f"\n[EVAL] {label} (clean: {accuracy(model, x, y):.1%})"
          + (f", feature squeeze {squeeze_bits} bits" if squeeze_bits else ""))

    for atk_name, atk_cfg in [
        ("fgsm",      dict(attack="fgsm", targeted=False)),
        ("pgd",       dict(attack="pgd",  targeted=False)),
        ("fgsm_cible", dict(attack="fgsm", targeted=True, target_class=3)),
        ("pgd_cible", dict(attack="pgd",  targeted=True, target_class=3)),
    ]:
        report[atk_name] = {}
        for eps in eps_list:
            report[atk_name][eps] = eval_attack(
                model, x, y, eps=eps, steps=steps, rng=rng,
                squeeze_bits=squeeze_bits, **atk_cfg)

    # Affichage
    print(f"{'eps':>6} | {'FGSM':>7} | {'PGD':>7} | {'FGSM cibl.':>10} | {'PGD cibl.':>10}")
    print("-" * 52)
    for eps in eps_list:
        print(f"{str(eps):>6} | {report['fgsm'][eps]:>7.1%} | {report['pgd'][eps]:>7.1%} "
              f"| {report['fgsm_cible'][eps]:>10.1%} | {report['pgd_cible'][eps]:>10.1%}")

    return report


def transfer_report(src_model, dst_model, x, y, eps_list=EPS_LIST,
                    steps=PGD_STEPS_EVAL, label_src="source", label_dst="cible"):
    """
    Attaque par transfert : on génère des exemples adverses avec le modèle
    SOURCE (gradient disponible) et on les teste sur la CIBLE (boîte noire).
    C'est l'attaque la plus réaliste : l'attaquant n'a pas accès au modèle.
    """
    rng = np.random.RandomState(42)
    print(f"\n[EVAL] Transfert {label_src} -> {label_dst} "
          f"(cible clean: {accuracy(dst_model, x, y):.1%})")
    print(f"{'eps':>6} | {'acc cible (FGSM src)':>20}")
    print("-" * 30)
    transfer = {}
    for eps in eps_list:
        # Exemples adverses générés contre la source
        x_adv = fgsm_batch(src_model, x, y, eps)
        acc = accuracy(dst_model, x_adv, y)
        transfer[eps] = acc
        print(f"{str(eps):>6} | {acc:>20.1%}")
    return transfer


# --------------------------------------------------------------------------
#  Courbes
# --------------------------------------------------------------------------

def plot_curves(report_std, report_def, eps_list, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for atk, color, marker in [("fgsm", "tab:blue", "o"), ("pgd", "tab:red", "s")]:
        ax.plot(eps_list, [report_def[atk][e] for e in eps_list],
                marker=marker, linewidth=2, color=color,
                label=f"défendu -- {atk}")
    for atk, color, marker in [("fgsm", "tab:blue", "o"), ("pgd", "tab:red", "s")]:
        ax.plot(eps_list, [report_std[atk][e] for e in eps_list],
                marker=marker, linewidth=2, color=color, linestyle="--",
                label=f"standard -- {atk}")
    ax.set_xlabel("eps (amplitude du bruit)")
    ax.set_ylabel("Accuracy sous attaque")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out_path}")


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Version durcie du CNN (défenses)")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.3,
                        help="amplitude d'attaque pendant l'entraînement")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd",
                        help="type d'adversarial training (pgd recommandé)")
    parser.add_argument("--pgd-steps-train", type=int, default=7,
                        help="itérations PGD pendant l'entraînement")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true", help="test rapide")
    parser.add_argument("--load", default=None,
                        help="évaluer un modèle déjà durci sans ré-entraîner")
    parser.add_argument("--squeeze", type=int, default=None,
                        help="feature squeeze à l'éval (bits, ex: 3)")
    parser.add_argument("--out", default="models/defend_pgd_mnist_weights.npz")
    parser.add_argument("--no-transfer", action="store_true",
                        help="saute l'éval transfert (plus rapide)")
    parser.add_argument("--no-eval", action="store_true",
                        help="saute toute l'évaluation (test d'entraînement seul)")
    args = parser.parse_args()

    if args.quick:
        args.n_train, args.epochs = 800, 1
        args.no_transfer = True
        args.no_eval = True

    # -- Données --
    x_te, y_te = load_data("mnist", 500)
    print(f"[DATA] {len(x_te)} images de test")

    loader = MNISTLoader()
    (x_train, y_train), _ = loader.load(join(ROOT_DIR, "data"))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(x_train), size=min(args.n_train, len(x_train)), replace=False)
    x_tr = normalize(add_channel_dim(x_train[idx])).transpose(0, 3, 1, 2)
    y_tr = y_train[idx]
    print(f"[DATA] {len(x_tr)} images d'entraînement (batch {args.batch}, "
          f"{args.epochs} epochs, attack={args.attack}, eps={args.eps})")

    # -- Modèle (même architecture que partout ailleurs) --
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

    # -- Mode évaluation seule --
    if args.load:
        model.load_weights(join(ROOT_DIR, args.load))
        print(f"[LOAD] Modèle durci -> {args.load}")

        report = robustness_report(model, x_te, y_te, label="défendu")
        if args.squeeze:
            report_sq = robustness_report(model, x_te, y_te,
                                          squeeze_bits=args.squeeze,
                                          label=f"défendu + squeeze {args.squeeze} bits")
        if not args.no_transfer:
            std_model = build_model("mnist")
            std_model.load_weights(join(ROOT_DIR, "models", "model_weights_full.npz"))
            transfer_report(std_model, model, x_te, y_te)
        return

    # -- Entraînement adversarial --
    print(f"\n[ADV-TRAIN] Adversarial training ({args.attack}) avec exemples adverses...")
    t0 = time.time()
    history = train_adversarial(model, x_tr, y_tr, args.epochs, args.batch,
                                args.eps, attack=args.attack,
                                pgd_steps=args.pgd_steps_train, lr=args.lr)
    m, s = divmod(time.time() - t0, 60)
    print(f"[ADV-TRAIN] Terminé en {int(m)}m{int(s)}s")

    out_path = join(ROOT_DIR, args.out)
    model.save_weights(out_path)
    print(f"[SAVE] Poids durcis -> {out_path}")

    if args.no_eval:
        print("[SKIP] Évaluation sautée (--no-eval)")
        return

    # -- Évaluation complète --
    report_def = robustness_report(model, x_te, y_te, label="défendu")

    std_model = build_model("mnist")
    std_model.load_weights(join(ROOT_DIR, "models", "model_weights_full.npz"))
    report_std = robustness_report(std_model, x_te, y_te, label="standard (full)")

    # Tableau comparatif
    print("\n[EVAL] RÉSUMÉ — standard vs défendu (FGSM)")
    print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
    print("-" * 42)
    for eps in EPS_LIST:
        s_acc, d_acc = report_std["fgsm"][eps], report_def["fgsm"][eps]
        print(f"{str(eps):>6} | {s_acc:>10.1%} | {d_acc:>10.1%} | {d_acc - s_acc:>+8.1%}")

    print("\n[EVAL] RÉSUMÉ — standard vs défendu (PGD)")
    print(f"{'eps':>6} | {'standard':>10} | {'défendu':>10} | {'gain':>8}")
    print("-" * 42)
    for eps in EPS_LIST:
        s_acc, d_acc = report_std["pgd"][eps], report_def["pgd"][eps]
        print(f"{str(eps):>6} | {s_acc:>10.1%} | {d_acc:>10.1%} | {d_acc - s_acc:>+8.1%}")

    # Transfert : attaque générée sur le standard -> testée sur le défendu
    if not args.no_transfer:
        transfer_report(std_model, model, x_te, y_te,
                        label_src="standard", label_dst="défendu")

    # Courbes
    out_dir = join(ROOT_DIR, "adversarial", "results")
    import os
    os.makedirs(out_dir, exist_ok=True)
    plot_curves(report_std, report_def, EPS_LIST,
                f"Version durcie vs standard -- adversarial training {args.attack}",
                join(out_dir, f"harden_curve_{args.attack}.png"))

    # Historique
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], marker="o", linewidth=2, label="loss")
    ax.set_title(f"Adversarial training {args.attack} -- loss par epoch")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(out_dir, f"harden_history_{args.attack}.png"), dpi=130,
                bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Historique -> adversarial/results/harden_history_{args.attack}.png")

    print("\n[DONE] Version durcie entraînée et évaluée.")


if __name__ == "__main__":
    main()
