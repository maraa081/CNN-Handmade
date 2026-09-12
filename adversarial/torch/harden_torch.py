#!/usr/bin/env python3
"""
harden_torch.py — Meme chose que harden2.py, mais en PyTorch.

Meme architecture, memes attaques, memes recettes, memes options. La seule
difference est le moteur de calcul : autograd + noyaux optimises, donc
utilisable sur GPU quand le modele grossit.

    python3 adversarial/torch/harden_torch.py --quick              # deja que tout tourne
    python3 adversarial/torch/harden_torch.py --parite             # verifie l'equivalence avec NumPy
    python3 adversarial/torch/harden_torch.py --n-train 60000 --epochs 15 --pgd-steps 20
    python3 adversarial/torch/harden_torch.py --report models/harden2_aug_pgdat.npz --restarts 3

Le mode `--parite` est le plus important a lancer en premier : il charge le
MEME fichier de poids dans les deux implementations, calcule l'accuracy propre
et sous attaque, et compare. Si les deux divergent, il y a un bug.

Choix du peripherique : automatique (cuda -> directml -> cpu), forcable avec
    --device cuda | dml | cpu
Sous ROCm (carte AMD), PyTorch expose le GPU via l'API "cuda".
"""

import argparse
import os
import sys
import time
from os.path import abspath, dirname, join

import numpy as np

try:
    import torch
except ImportError:
    print("=" * 68)
    print("  PyTorch n'est pas installe sur ce Python.")
    print("=" * 68)
    print(f"\n  Python utilise : {sys.executable}")
    print(f"  Version        : {sys.version.split()[0]}\n")
    print("  Installation (CPU, le plus simple et le plus fiable) :")
    print("    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("  GPU AMD sous Windows :  pip install torch-directml")
    print("  GPU AMD (le plus rapide) : ROCm sous WSL2")
    print()
    print("  Attention : PyTorch ne supporte pas toutes les versions de Python")
    print("  (3.10 a 3.12 est le plus sur ; 3.14 n'est pas supporte).")
    print("  Detail complet : adversarial/torch/README.md, section 6.\n")
    sys.exit(1)

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, dirname(abspath(__file__)))

from modele import CNN, charger_npz, sauver_npz, nb_parametres          # noqa: E402
from entrainement import (charger_train, charger_test, entrainer,       # noqa: E402
                          rapport, CONFIG_AUG)

EPS_EVAL = [0.05, 0.1, 0.2, 0.3]


def choisir_device(demande="auto"):
    if demande == "cpu":
        return torch.device("cpu")
    if demande == "dml":
        import torch_directml
        return torch_directml.device()
    if demande == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


# --------------------------------------------------------------------------
#  Mode parite : verifie que PyTorch et NumPy donnent le meme resultat
# --------------------------------------------------------------------------

def mode_parite(chemin, device, n=200):
    """Charge le meme .npz dans les deux implementations et compare."""
    print("=" * 68)
    print("  PARITE NumPy <-> PyTorch")
    print("=" * 68)
    print(f"\nPoids : {chemin}")

    x_te, y_te = charger_test("mnist", n)
    print(f"Test  : {len(x_te)} images\n")

    # --- PyTorch ---
    modele = CNN().to(device)
    charger_npz(modele, join(ROOT_DIR, chemin))
    modele.eval()
    from attaques import attaque, accuracy as acc_torch
    xt, yt = x_te.to(device), y_te.to(device)
    res_t = {"clean": acc_torch(modele, xt, yt)}
    for eps in (0.1, 0.3):
        res_t[f"fgsm{eps}"] = acc_torch(modele, attaque(modele, xt, yt, eps, "fgsm"), yt)
    res_t["pgd0.2"] = acc_torch(modele, attaque(modele, xt, yt, 0.2, "pgd", 20), yt)

    # --- NumPy (version faite main) ---
    from adversarial.scripts.fgsm import build_model, load_data, accuracy as acc_np
    from adversarial.scripts.harden import fgsm_batch, pgd_batch
    x_np, y_np = load_data("mnist", n)
    m_np = build_model("mnist")
    m_np.load_weights(join(ROOT_DIR, chemin))
    res_n = {"clean": acc_np(m_np, x_np, y_np)}
    for eps in (0.1, 0.3):
        res_n[f"fgsm{eps}"] = acc_np(m_np, fgsm_batch(m_np, x_np, y_np, eps), y_np)
    res_n["pgd0.2"] = acc_np(
        m_np, pgd_batch(m_np, x_np, y_np, 0.2, steps=20,
                        rng=np.random.RandomState(0)), y_np)

    print(f"{'mesure':>10} | {'PyTorch':>9} | {'NumPy':>9} | {'ecart':>8}")
    print("-" * 46)
    for cle in res_t:
        e = abs(res_t[cle] - res_n[cle])
        print(f"{cle:>10} | {res_t[cle]:>9.2%} | {res_n[cle]:>9.2%} | {e:>8.2%}")
    ecart_max = max(abs(res_t[c] - res_n[c]) for c in res_t)
    print()
    print(f"Ecart maximal : {ecart_max:.2%}")
    print("(les PGD utilisent des graines aleatoires differentes entre les deux "
          "implementations :\n un petit ecart a eps=0.2 est normal. FGSM et clean "
          "doivent coller.)")
    return ecart_max


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Version durcie v2 en PyTorch (autograd)")
    p.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256,
                   help="256 conseille sur GPU (64 sous-utilise la carte)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--attack", choices=["pgd", "fgsm", "fgsm-rs", "apgd-ce", "apgd-dlr"],
                   default="pgd",
                   help="attaque a l'ENTRAINEMENT : pgd (defaut), fgsm, fgsm-rs,\n"
                        "ou apgd-ce / apgd-dlr (pas adaptatif, demarrage aleatoire ;\n"
                        "plus couteux, mais c'est APGD-DLR qui resiste le moins\n"
                        "bien a l'evaluation)")
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--pgd-alpha", type=float, default=None,
                   help="taille du pas de PGD a l'ENTRAINEMENT (defaut : eps/4).\n"
                        "eps/10 avec 10-20 pas donne une attaque d'entrainement\n"
                        "plus fine, donc une robustesse moins specifique")
    p.add_argument("--val-steps", type=int, default=10)
    p.add_argument("--loss", choices=["pgdat", "trades"], default="pgdat")
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--beta-warmup", type=float, default=0.3,
                   help="fraction du run sur laquelle beta monte de 10%% a 100%% (0 = pas de rampe)")
    p.add_argument("--mix", type=float, default=0.5)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--aug-fort", action="store_true")
    p.add_argument("--aug-config", default="")
    p.add_argument("--shift", type=int, default=2)
    p.add_argument("--lr", type=float, default=None,
                   help="defaut: 0.05 (pgdat) ou 0.01 (trades)")
    p.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="momentum")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--lr-drop", default="0.5,0.8")
    p.add_argument("--warm-start", default="auto")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "dml"])
    p.add_argument("--threads", type=int, default=0, help="threads CPU (0 = auto)")
    p.add_argument("--out", default="models/harden_torch_best.pt")
    p.add_argument("--npz", default=None,
                   help="sauvegarder AUSSI les poids au format .npz (compatible NumPy)")
    p.add_argument("--large", action="store_true",
                   help="modele large (~1,7M parametres au lieu de 421k) : test de\n"
                        "l'hypothese 'plus de capacite = plus de robustesse'")
    p.add_argument("--resume", default=None,
                   help="reprendre un run interrompu depuis <out>_last.pt")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="epoch de depart (0 = deduit du checkpoint)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default=None)
    p.add_argument("--parite", action="store_true")
    p.add_argument("--parite-n", type=int, default=200)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--eval-n", type=int, default=500)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    lr_donne = args.lr          # None si l'utilisateur n'a pas force --lr

    if args.quick:
        args.n_train, args.epochs, args.val = 800, 1, 100
        args.pgd_steps, args.val_steps = 2, 2
        args.lr_drop = ""
        args.eval_steps, args.eval_n, args.restarts = 5, 200, 1

    if args.threads:
        torch.set_num_threads(args.threads)

    device = choisir_device(args.device)

    # -- Mode parite --
    if args.parite:
        chemin = args.warm_start
        if chemin in ("auto", None, "none"):
            chemin = "models/model_weights_full.npz"
        mode_parite(chemin, device, args.parite_n)
        return

    print("=" * 68)
    print("  HARDEN TORCH (autograd)")
    print("=" * 68)
    print(f"Peripherique : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    x_te, y_te = charger_test(args.dataset, args.eval_n)

    # -- Evaluation seule --
    if args.report:
        modele = CNN().to(device)
        if args.report.endswith(".pt"):
            modele.load_state_dict(torch.load(join(ROOT_DIR, args.report),
                                              map_location=device))
        else:
            charger_npz(modele, join(ROOT_DIR, args.report))
        modele.eval()
        rapport(modele, x_te.to(device), y_te.to(device), EPS_EVAL,
                args.eval_steps, args.restarts, f"modele durci ({args.report})", device)
        return

    # -- Donnees --
    train, val = charger_train(args.n_train, args.val, args.dataset)
    print(f"[DATA] {len(train[0])} train / {len(val[0])} val / {len(x_te)} test")

    if args.lr is None:
        args.lr = 0.002 if args.loss == "trades" else 0.05
        print(f"[LR] defaut pour {args.loss} -> lr {args.lr}")

    if args.augment:
        cfg = dict(CONFIG_AUG)
        cfg["translation"] = args.shift
        if args.aug_fort:
            cfg.update({"rotation": 15.0, "bruit_p": 0.03, "cutout": 8,
                        "zoom": 0.12, "prob": 0.6})
        for morceau in filter(None, args.aug_config.split(",")):
            cle, _, valeur = morceau.partition("=")
            cle = cle.strip()
            if cle in cfg:
                cfg[cle] = type(cfg[cle])(valeur)
        args.aug_cfg = cfg
        print("[AUG] " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    else:
        args.aug_cfg = None

    print(f"[CONF] attack={args.attack} steps={args.pgd_steps} eps={args.eps} "
          f"loss={args.loss} mix={args.mix} lr={args.lr} clip={args.clip} "
          f"batch={args.batch} epochs={args.epochs} augment={args.augment}")

    # -- Modele --
    modele = CNN(large=args.large).to(device)
    print(f"[MODEL] {nb_parametres(modele):,} parametres".replace(",", " "))

    # -- Reprise d'un run interrompu (veille du PC, arret manuel...) --
    # Le checkpoint complet est ecrit a chaque epoch dans <out>_last.pt.
    if args.resume:
        chemin_resume = (args.resume if os.path.isabs(args.resume)
                         else join(ROOT_DIR, args.resume))
        if not os.path.exists(chemin_resume):
            print(f"[ERREUR] checkpoint introuvable : {chemin_resume}")
            return 1
        ck = torch.load(chemin_resume, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            modele.load_state_dict(ck["model"])
            print(f"[RESUME] {chemin_resume} (epoch {ck.get('epoch', '?')}, "
                  f"lr {ck.get('lr', '?')})")
            if lr_donne is None and ck.get("lr") is not None:
                args.lr = float(ck["lr"])
                print(f"[RESUME] lr repris du checkpoint -> {args.lr:.5f}")
            if args.start_epoch <= 0:
                args.start_epoch = int(ck.get("epoch", 0)) + 1
            args.meilleur_init = float(ck.get("meilleur", -1.0))
        else:
            # Ancien format : state_dict seul. On reprend les poids, et l'epoch
            # de depart doit alors etre donnee a la main (--start-epoch).
            modele.load_state_dict(ck)
            print(f"[RESUME] {chemin_resume} (ancien format : poids seuls)")
            print("[RESUME] [warn] epoch et lr inconnus : preciser --start-epoch et --lr")
        if args.start_epoch <= 0:
            args.start_epoch = 1
        print(f"[RESUME] reprise a l'epoch {args.start_epoch} / {args.epochs}")

    if args.resume:
        pass
    elif args.warm_start == "auto":
        auto = {"mnist": "models/model_weights_full.npz",
                "emnist": "models/emnist_letters_weights_full.npz"}[args.dataset]
        args.warm_start = auto if os.path.exists(join(ROOT_DIR, auto)) else "none"
        print(f"[WARM] auto -> {args.warm_start}")
    if not args.resume and args.warm_start not in (None, "none", ""):
        charger_npz(modele, join(ROOT_DIR, args.warm_start))
        from attaques import accuracy as acc_torch
        modele.eval()
        print(f"[WARM] clean test : {acc_torch(modele, x_te.to(device), y_te.to(device)):.2%}")

    # -- Optimiseur --
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(modele.parameters(), lr=args.lr,
                              momentum=0.0, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        opt = torch.optim.SGD(modele.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(modele.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    print(f"[OPT] {args.optimizer}")

    # -- Entrainement --
    t0 = time.time()
    entrainer(modele, opt, train, val, args, device)
    m, s = divmod(time.time() - t0, 60)
    print(f"\n[TRAIN] termine en {int(m)} min {int(s)} s")

    # -- Sauvegarde .npz compatible NumPy --
    best = CNN(large=args.large).to("cpu")
    best.load_state_dict(torch.load(args.out, map_location="cpu"))
    if args.npz:
        sauver_npz(best, join(ROOT_DIR, args.npz))
        print(f"[NPZ] poids aussi sauvegardes -> {args.npz} (chargeable en NumPy)")

    # -- Evaluation finale --
    best = best.to(device)
    rapport(best, x_te.to(device), y_te.to(device), EPS_EVAL,
            args.eval_steps, args.restarts, f"harden_torch ({args.loss}, eps={args.eps})",
            device)


if __name__ == "__main__":
    main()
