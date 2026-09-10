#!/usr/bin/env python3
"""
harden2.py — Version durcie v2 : pousser la frontiere precision/robustesse.

Pourquoi une v2 ? La v1 (harden.py) plafonne a 67% propre / 24% sous FGSM eps 0.3.
Ce n'est PAS une limite de la methode, c'est un manque de budget :
  - 5000 images sur 60000 disponibles
  - 3 epochs (une convergence demande 15-30 epochs)
  - entrainement depuis zero (on a deja un modele propre a 98.6%)
  - pas de decroissance du learning rate
  - pas d'augmentation de donnees

Les leviers implementes ici :
  1. --warm-start      : partir du modele propre deja entraine (gain immediat)
  2. --n-train 60000   : tout le jeu d'entrainement
  3. --lr-drop         : decroissance du lr (indispensable pour converger)
  4. --loss trades     : TRADES (Zhang 2019), meilleure frontiere que PGD-AT
  5. --augment         : translations aleatoires de +-2 px
  6. --val / --save-best : selection du meilleur modele sur la robustesse
                         reelle, pas sur l'accuracy propre
  7. --report          : evaluation honnete (PGD multi-restarts), pour ne pas
                         se raconter d'histoires sur sa propre robustesse

Usage :
    # Entrainement complet recommande (a lancer sur la machine la plus rapide)
    python3 adversarial/scripts/harden2.py --warm-start models/model_weights_full.npz

    # Comparaison TRADES
    python3 adversarial/scripts/harden2.py --warm-start models/model_weights_full.npz --loss trades

    # Evaluation seule d'un modele sauvegarde (avec multi-restarts)
    python3 adversarial/scripts/harden2.py --report models/harden2_best.npz --restarts 3

    # Test rapide (verifie que tout tourne)
    python3 adversarial/scripts/harden2.py --quick
"""

import argparse
import os
import sys
import time
from os.path import abspath, dirname, join

import numpy as np

import matplotlib
matplotlib.use("Agg")

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))  # -> CNN-Handmade/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import MNISTLoader, normalize, add_channel_dim  # noqa: E402
from adversarial.scripts.fgsm import build_model, load_data, accuracy  # noqa: E402

EPS_EVAL = [0.05, 0.1, 0.2, 0.3]


# --------------------------------------------------------------------------
#  Utilitaires numeriques
# --------------------------------------------------------------------------

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kl_divergence(p, q, eps=1e-12):
    """KL(p || q), ligne par ligne. Renvoie un vecteur (N,)."""
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=1)


def clip_gradients(model, max_norm):
    """Ecrête la norme L2 globale des gradients (evite l'explosion).

    Indispensable pour TRADES : sur un modele deja convergé, le terme KL est
    beaucoup plus gros que le terme CE -> sans ecrêtage, la premiere mise a jour
    detruit les poids.
    """
    cibles = []
    for l in model.layers:
        if hasattr(l, "kernels"):
            cibles += [l.d_kernels, l.d_bias]
        elif hasattr(l, "W"):
            cibles += [l.dW, l.db]
    total = sum(float(np.sum(g * g)) for g in cibles if g is not None)
    norme = float(np.sqrt(total))
    if max_norm and norme > max_norm:
        f = max_norm / (norme + 1e-12)
        for g in cibles:
            if g is not None:
                g *= f
    return norme


def augment_shift(x, max_shift=2, rng=None):
    """Translation aleatoire de chaque image de +-max_shift pixels (bords noirs)."""
    rng = rng or np.random
    n, c, h, w = x.shape
    out = np.zeros_like(x)
    for i in range(n):
        dy = rng.randint(-max_shift, max_shift + 1)
        dx = rng.randint(-max_shift, max_shift + 1)
        out[i, :, max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
            x[i, :, max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


# --------------------------------------------------------------------------
#  Attaques sur batch
# --------------------------------------------------------------------------

def fgsm_batch(model, x, y, eps, random_start=False, rng=None):
    """FGSM non ciblee, avec demarrage aleatoire optionnel (Fast-AT, Wong 2020)."""
    if random_start:
        rng = rng or np.random
        delta = rng.uniform(-eps, eps, size=x.shape).astype(np.float32)
        x0 = np.clip(x + delta, 0.0, 1.0)
    else:
        x0 = x
    logits = model.forward(x0)
    C = logits.shape[1]
    model.loss_fn.forward(logits, np.eye(C)[y])
    dx = model.backward(model.loss_fn.backward())
    return np.clip(x0 + eps * np.sign(dx), 0.0, 1.0)


def pgd_batch(model, x, y, eps, steps=5, alpha=None, rng=None):
    """PGD L-inf non ciblee avec demarrage aleatoire (Madry 2018)."""
    rng = rng or np.random
    alpha = alpha or eps / 4.0
    C = model.forward(x[:1]).shape[1]
    y_oh = np.eye(C)[y]
    x_adv = np.clip(x + rng.uniform(-eps, eps, size=x.shape).astype(np.float32), 0.0, 1.0)
    for _ in range(steps):
        model.loss_fn.forward(model.forward(x_adv), y_oh)
        dx = model.backward(model.loss_fn.backward())
        x_adv = np.clip(x_adv + alpha * np.sign(dx), x - eps, x + eps)
        x_adv = np.clip(x_adv, 0.0, 1.0)
    return x_adv


def attaque(model, x, y, eps, attack, steps, rng=None):
    if attack == "fgsm":
        return fgsm_batch(model, x, y, eps, rng=rng)
    if attack == "fgsm-rs":
        return fgsm_batch(model, x, y, eps, random_start=True, rng=rng)
    return pgd_batch(model, x, y, eps, steps=steps, rng=rng)


# --------------------------------------------------------------------------
#  Entrainement
# --------------------------------------------------------------------------

def entrainer(model, x_tr, y_tr, x_val, y_val, args):
    N = len(x_tr)
    rng = np.random.RandomState(args.seed)
    historique = []
    meilleur = -1.0
    lr = args.lr

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Decroissance du learning rate
        if args.lr_drop:
            paliers = [float(p) for p in args.lr_drop.split(",")]
            if any(abs(epoch / args.epochs - p) < 0.5 / args.epochs for p in paliers):
                lr *= 0.1
                print(f"  [LR] epoch {epoch} : lr -> {lr:.5f}")

        idx = rng.permutation(N)
        perte, n_vus = 0.0, 0

        for start in range(0, N, args.batch):
            bi = idx[start:start + args.batch]
            bx, by = x_tr[bi], y_tr[bi]
            if args.augment:
                bx = augment_shift(bx, args.shift, rng)

            # 1) attaque du batch avec le modele courant
            model.eval_mode()
            bx_adv = attaque(model, bx, by, args.eps, args.attack,
                             args.pgd_steps, rng=rng)
            model.train_mode()

            # 2) gradient combine (propre + adversarial)
            if args.loss == "trades":
                z_adv = model.forward(bx_adv)          # activations ecrasees
                z = model.forward(bx)                  # activations propres
                p, q = softmax(z), softmax(z_adv)
                kl_vec = kl_divergence(p, q)           # (N,) par echantillon
                kl = kl_vec.mean()
                y_oh = np.eye(p.shape[1])[by]
                ce = -np.log(p[np.arange(len(by)), by] + 1e-12).mean()
                # d/dz [ CE + beta*KL ] : (p - y) + beta * p * (log(p/q) - KL)
                grad = ((p - y_oh)
                        + args.beta * p * (np.log(p + 1e-12)
                                           - np.log(q + 1e-12) - kl_vec[:, None]))
                perte += (ce + args.beta * kl) * len(by)
            else:
                if args.mix < 1.0:
                    n_adv = int(round(len(bx) * args.mix))
                    cx = np.concatenate([bx, bx_adv[:n_adv]], axis=0)
                    cy = np.concatenate([by, by[:n_adv]], axis=0)
                else:
                    cx, cy = bx_adv, by
                z = model.forward(cx)
                perte += model.loss_fn.forward(z, np.eye(z.shape[1])[cy]) * len(by)
                grad = model.loss_fn.backward()

            model.backward(grad)
            clip_gradients(model, args.clip)
            model.update(lr)
            n_vus += len(by)

        # 3) validation : accuracy propre ET accuracy sous attaque
        acc_clean = accuracy(model, x_val, y_val)
        acc_rob = accuracy(model, pgd_batch(model, x_val, y_val, args.eps,
                                            steps=args.val_steps, rng=rng), y_val)
        dt = time.time() - t0
        reste = (args.epochs - epoch) * dt / 60
        print(f"  Epoch {epoch:>2}/{args.epochs} | loss {perte / n_vus:6.4f} | "
              f"val clean {acc_clean:6.2%} | val PGD{args.val_steps} {acc_rob:6.2%} | "
              f"{dt / 60:5.1f} min | reste ~{reste:4.0f} min")

        historique.append((epoch, perte / n_vus, acc_clean, acc_rob, lr))

        # 4) sauvegarde. On selectionne sur la ROBUSTESSE, pas sur le propre.
        model.save_weights(args.out + "_last.npz")
        if acc_rob > meilleur:
            meilleur = acc_rob
            model.save_weights(args.out)
            print(f"           -> meilleur modele sauvegarde (val PGD {acc_rob:.2%})")

    return historique


# --------------------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------------------

def evaluer(model, x_te, y_te, eps_list, steps, restarts=1, label="modele"):
    """Accuracy sous FGSM et PGD, avec multi-restarts pour PGD."""
    res = {"fgsm": {}, "pgd": {}}
    for eps in eps_list:
        res["fgsm"][eps] = accuracy(model, fgsm_batch(model, x_te, y_te, eps), y_te)
        accs = []
        for r in range(restarts):
            rng = np.random.RandomState(1000 + r)
            accs.append(accuracy(model, pgd_batch(model, x_te, y_te, eps,
                                                  steps=steps, rng=rng), y_te))
        res["pgd"][eps] = min(accs)  # le pire cas = la vraie robustesse
    return res


def rapport(model, x_te, y_te, eps_list, steps, restarts, label):
    print(f"\n[EVAL] {label} ({len(x_te)} images, PGD {steps} pas, {restarts} restart(s))")
    print(f"{'eps':>6} | {'FGSM':>8} | {'PGD (pire cas)':>15}")
    print("-" * 36)
    res = evaluer(model, x_te, y_te, eps_list, steps, restarts, label)
    for eps in eps_list:
        print(f"{eps:>6} | {res['fgsm'][eps]:>8.1%} | {res['pgd'][eps]:>15.1%}")
    return res


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Version durcie v2 (frontiere precision/robustesse)")
    p.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--val", type=int, default=1000, help="images de validation (robustesse)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.3, help="eps d'entrainement")
    p.add_argument("--attack", choices=["pgd", "fgsm", "fgsm-rs"], default="pgd")
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--val-steps", type=int, default=10)
    p.add_argument("--loss", choices=["pgdat", "trades"], default="pgdat")
    p.add_argument("--beta", type=float, default=6.0, help="poids KL (TRADES)")
    p.add_argument("--mix", type=float, default=0.5,
                   help="part d'exemples adverses dans le batch (1.0 = Madry pur)")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--shift", type=int, default=2)
    p.add_argument("--lr", type=float, default=None,
                   help="defaut: 0.005 (pgdat) ou 0.001 (trades)")
    p.add_argument("--clip", type=float, default=1.0,
                   help="norme L2 max des gradients (0 = desactive)")
    p.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 (0 = desactive)")
    p.add_argument("--lr-drop", default="0.5,0.8", help="epochs (fractions) ou lr x0.1")
    p.add_argument("--warm-start", default="auto",
                   help="poids initiaux, ou 'auto' (modele propre du dataset), ou 'none'")
    p.add_argument("--out", default="models/harden2_best.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default=None, help="evaluation seule d'un modele")
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--eval-n", type=int, default=500, help="images de test pour l'eval finale")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_train, args.epochs, args.val = 800, 1, 100
        args.pgd_steps, args.val_steps = 2, 2
        args.lr_drop = ""

    if args.lr is None:
        args.lr = 0.001 if args.loss == "trades" else 0.005
        print(f"[LR] defaut pour {args.loss} -> lr {args.lr}")

    x_te, y_te = load_data(args.dataset, args.eval_n)

    # -- Evaluation seule --
    if args.report:
        model = build_model(args.dataset)
        model.load_weights(join(ROOT_DIR, args.report))
        print(f"[LOAD] {args.report}")
        rapport(model, x_te, y_te, EPS_EVAL, args.eval_steps, args.restarts, "modele durci")
        return

    # -- Donnees --
    loader = MNISTLoader()
    (x_all, y_all), _ = loader.load(join(ROOT_DIR, "data"))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(x_all), size=min(args.n_train + args.val, len(x_all)),
                     replace=False)
    x_sel = normalize(add_channel_dim(x_all[idx])).transpose(0, 3, 1, 2)
    y_sel = y_all[idx]
    x_val, y_val = x_sel[:args.val], y_sel[:args.val]
    x_tr, y_tr = x_sel[args.val:], y_sel[args.val:]

    print(f"[DATA] {len(x_tr)} train / {len(x_val)} val / {len(x_te)} test")
    print(f"[CONF] attack={args.attack} steps={args.pgd_steps} eps={args.eps} "
          f"loss={args.loss} mix={args.mix} lr={args.lr} clip={args.clip} "
          f"epochs={args.epochs} augment={args.augment}")

    # -- Modele --
    model = build_model(args.dataset)
    if args.optimizer != "sgd":
        from optimizers import Adam, Momentum
        cls = Momentum if args.optimizer == "momentum" else Adam
        kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
        model.optimizer = cls(**kwargs)
        print(f"[OPT] optimiseur -> {args.optimizer} (lr={args.lr}, wd={args.weight_decay})")
    elif args.weight_decay:
        from optimizers import SGD
        model.optimizer = SGD(lr=args.lr, weight_decay=args.weight_decay)

    if args.warm_start == "auto":
        auto = {"mnist": "models/model_weights_full.npz",
                "emnist": "models/emnist_letters_weights_full.npz"}[args.dataset]
        args.warm_start = auto if os.path.exists(join(ROOT_DIR, auto)) else "none"
        print(f"[WARM] auto -> {args.warm_start}")
    if args.warm_start not in (None, "none", ""):
        model.load_weights(join(ROOT_DIR, args.warm_start))
        acc0 = accuracy(model, x_te, y_te)
        print(f"[WARM] depart depuis {args.warm_start} (clean test {acc0:.2%})")

    # -- Entrainement --
    t0 = time.time()
    entrainer(model, x_tr, y_tr, x_val, y_val, args)
    m, s = divmod(time.time() - t0, 60)
    print(f"\n[TRAIN] termine en {int(m)} min {int(s)} s")

    # -- Evaluation finale --
    model.load_weights(args.out)
    rapport(model, x_te, y_te, EPS_EVAL, args.eval_steps, args.restarts,
            f"harden2 ({args.loss}, eps={args.eps})")


if __name__ == "__main__":
    main()
