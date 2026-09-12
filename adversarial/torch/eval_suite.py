#!/usr/bin/env python3
"""eval_suite.py - suite d'attaques complete sur un modele entraine.

C'est l'outil qui transforme un chiffre en PREUVE. Un modele annonce comme
robuste doit survivre a PLUSIEURS familles d'attaques, pas seulement a celle
contre laquelle il a ete entraine.

Trois familles sont passees :

  1. A GRADIENT (white-box) : FGSM, PGD multi-restarts, APGD (CE et DLR).
     L'attaquant connait le modele et remonte son gradient.
  2. SANS GRADIENT (black-box, score-based) : Square Attack et NES.
     L'attaquant ne voit que les SCORES de sortie. Sert de controle
     independant : une defense qui ne tient que face aux attaques a gradient
     est suspecte.
  3. DECISION-BASED (black-box, label seul) : Boundary Attack.
     L'attaquant ne voit que la classe predite. Il mesure la distance L2
     necessaire pour tromper le modele.

Usage
-----
    # Suite complete sur le modele durci (long : comptez 30-60 min sur CPU)
    python3 adversarial/torch/eval_suite.py --weights models/harden2_aug_pgdat_120ep.pt

    # Verification rapide de la chaine
    python3 adversarial/torch/eval_suite.py --weights models/....pt --quick

    # Seulement les attaques sans gradient
    python3 adversarial/torch/eval_suite.py --famille blackbox

Sortie : un tableau par famille + le PIRE CAS tous types confondus.
"""

import argparse
import os
import sys
import time
from os.path import abspath, dirname, isabs, join

import torch

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, join(ROOT_DIR, "adversarial", "torch"))

from adversarial.torch.modele import CNN, charger_npz          # noqa: E402
from adversarial.torch.attaques import fgsm, pgd, accuracy    # noqa: E402
from adversarial.torch.attaques_avancees import (              # noqa: E402
    apgd, square, nes, cw_l2, boundary,
)
from adversarial.torch.entrainement import charger_test        # noqa: E402

EPS_DEFAUT = [0.1, 0.2, 0.3]


def charger_modele(chemin, device, dataset):
    """Charge un modele depuis un .pt (state_dict ou checkpoint) ou un .npz.

    L'architecture (standard ou large) est DEDUITE de la forme des poids :
    la premiere convolution a 32 canaux en standard et 64 en large. Comme ca,
    aucun risque de se tromper de taille en evaluant un modele sauvegarde.
    """
    num_classes = 26 if dataset == "emnist" else 10
    if chemin.endswith(".npz"):
        modele = CNN(num_classes=num_classes).to(device)   # .npz = standard
        charger_npz(modele, chemin)
    else:
        ck = torch.load(chemin, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        large = int(sd["conv1.weight"].shape[0]) > 32
        modele = CNN(num_classes=num_classes, large=large).to(device)
        modele.load_state_dict(sd)
        if large:
            print("[MODEL] architecture LARGE detectee (~1,7M parametres)")
    modele.eval()
    return modele


def ligne(nom, res, eps_list):
    return f"  {nom:<22} | " + " | ".join(f"{res[e]:>8.1%}" for e in eps_list)


def entete(eps_list, titre):
    print(f"\n{titre}")
    print(f"  {'attaque':<22} | " + " | ".join(f"{('eps=' + str(e)):>8}" for e in eps_list))
    print("  " + "-" * (23 + 11 * len(eps_list)))


def main():
    p = argparse.ArgumentParser(description="Suite d'attaques complete (white-box + black-box)")
    p.add_argument("--weights", default="models/harden2_aug_pgdat_120ep.pt",
                   help="modele a evaluer (.pt ou .npz)")
    p.add_argument("--dataset", choices=["mnist", "kmnist", "emnist"], default="mnist",
                   help="mnist par defaut ; kmnist = kana japonais (meme format, 10 classes)")
    p.add_argument("--n", type=int, default=500, help="images de test")
    p.add_argument("--eps", nargs="+", type=float, default=EPS_DEFAUT)
    p.add_argument("--famille", choices=["tout", "whitebox", "blackbox", "l2"], default="tout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "dml"])
    p.add_argument("--quick", action="store_true", help="budget reduit (verification de la chaine)")

    # budgets (surchargeables)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-restarts", type=int, default=3)
    p.add_argument("--pgd-alpha", type=float, default=None,
                   help="taille du pas de PGD (defaut : eps/4). Un pas plus petit\n"
                        "(eps/10) avec plus de pas donne une attaque plus fine")
    p.add_argument("--apgd-steps", type=int, default=100)
    p.add_argument("--apgd-restarts", type=int, default=2)
    p.add_argument("--square-steps", type=int, default=500)
    p.add_argument("--square-restarts", type=int, default=1)
    p.add_argument("--nes-steps", type=int, default=20)
    p.add_argument("--nes-samples", type=int, default=10)
    p.add_argument("--cw-steps", type=int, default=100)
    p.add_argument("--boundary-steps", type=int, default=200)
    args = p.parse_args()

    if args.quick:
        args.n, args.eps = 100, [0.3]
        args.pgd_steps, args.pgd_restarts = 10, 1
        args.apgd_steps, args.apgd_restarts = 20, 1
        args.square_steps, args.nes_steps, args.nes_samples = 100, 5, 5
        args.cw_steps, args.boundary_steps = 20, 50

    from adversarial.torch.harden_torch import choisir_device
    device = choisir_device(args.device)

    chemin = args.weights if isabs(args.weights) else join(ROOT_DIR, args.weights)
    if not os.path.exists(chemin):
        print(f"[ERREUR] modele introuvable : {chemin}")
        return 1

    print("=" * 72)
    print("  SUITE D'ATTAQUES - evaluation multi-familles")
    print("=" * 72)
    print(f"[LOAD] {args.weights}")
    modele = charger_modele(chemin, device, args.dataset)
    x, y = charger_test(args.dataset, args.n)
    x, y = x.to(device), y.to(device)
    print(f"[DATA] {len(x)} images de test | peripherique : {device}")

    propre = accuracy(modele, x, y)
    print(f"[EVAL] precision propre : {propre:.1%}")

    resultats = {}          # (famille, nom, eps) -> accuracy
    t_total = time.time()

    def chrono(t0, nom):
        print(f"    -> {nom} termine en {time.time() - t0:.1f} s")

    # ---------------- 1) Attaques a GRADIENT ----------------
    if args.famille in ("tout", "whitebox"):
        entete(args.eps, "[1/3] ATTAQUES A GRADIENT (white-box)")

        for eps in args.eps:
            t0 = time.time()
            xa = fgsm(modele, x, y, eps)
            resultats[("grad", "FGSM (1 pas)", eps)] = accuracy(modele, xa, y)

            pires = []
            for r in range(args.pgd_restarts):
                torch.manual_seed(1000 + r)
                pires.append(accuracy(modele, pgd(modele, x, y, eps, args.pgd_steps,
                                                  alpha=args.pgd_alpha), y))
            resultats[("grad", f"PGD-{args.pgd_steps} ({args.pgd_restarts} rest.)", eps)] = min(pires)

            xa = apgd(modele, x, y, eps, loss="ce", steps=args.apgd_steps,
                      restarts=args.apgd_restarts, seed=2000)
            resultats[("grad", f"APGD-CE ({args.apgd_steps})", eps)] = accuracy(modele, xa, y)

            xa = apgd(modele, x, y, eps, loss="dlr", steps=args.apgd_steps,
                      restarts=args.apgd_restarts, seed=3000)
            resultats[("grad", f"APGD-DLR ({args.apgd_steps})", eps)] = accuracy(modele, xa, y)
            chrono(t0, f"eps={eps}")

        for nom in ["FGSM (1 pas)", f"PGD-{args.pgd_steps} ({args.pgd_restarts} rest.)",
                    f"APGD-CE ({args.apgd_steps})", f"APGD-DLR ({args.apgd_steps})"]:
            print(ligne(nom, {e: resultats[("grad", nom, e)] for e in args.eps}, args.eps))

    # ---------------- 2) Attaques SANS GRADIENT ----------------
    if args.famille in ("tout", "blackbox"):
        entete(args.eps, "[2/3] ATTAQUES SANS GRADIENT (black-box, score-based)")

        for eps in args.eps:
            t0 = time.time()
            xa = square(modele, x, y, eps, steps=args.square_steps,
                        restarts=args.square_restarts, seed=4000)
            resultats[("bb", f"Square ({args.square_steps})", eps)] = accuracy(modele, xa, y)

            xa = nes(modele, x, y, eps, steps=args.nes_steps, samples=args.nes_samples, seed=5000)
            resultats[("bb", f"NES ({args.nes_steps}x{args.nes_samples})", eps)] = accuracy(modele, xa, y)
            chrono(t0, f"eps={eps}")

        for nom in [f"Square ({args.square_steps})", f"NES ({args.nes_steps}x{args.nes_samples})"]:
            print(ligne(nom, {e: resultats[("bb", nom, e)] for e in args.eps}, args.eps))

    # ---------------- 3) Attaques L2 (distance minimale) ----------------
    if args.famille in ("tout", "l2"):
        eps_max = max(args.eps)
        print(f"\n[3/3] ATTAQUES L2 (distance minimale pour tromper le modele)")
        print(f"  {'attaque':<22} | {'succes':>8} | {'distance L2 moyenne':>20}")
        print("  " + "-" * 56)

        t0 = time.time()
        xa, dist = cw_l2(modele, x, y, steps=args.cw_steps)
        trompe = (modele(xa).argmax(dim=1) != y)
        d_moy = dist[trompe].mean().item() if bool(trompe.any()) else float("nan")
        print(f"  {'CW-L2':<22} | {trompe.float().mean().item():>8.1%} | {d_moy:>20.3f}")
        resultats[("l2", "CW-L2", eps_max)] = trompe.float().mean().item()
        chrono(t0, "CW-L2")

        t0 = time.time()
        xa, dist = boundary(modele, x, y, steps=args.boundary_steps, seed=6000)
        trompe = (modele(xa).argmax(dim=1) != y)
        d_moy = dist[trompe].mean().item() if bool(trompe.any()) else float("nan")
        print(f"  {'Boundary':<22} | {trompe.float().mean().item():>8.1%} | {d_moy:>20.3f}")
        resultats[("l2", "Boundary", eps_max)] = trompe.float().mean().item()
        chrono(t0, "Boundary")

        print("\n  Note : ces deux attaques mesurent une DISTANCE, pas une precision.")
        print("  Le taux de succes est la part des images trompees par l'attaque.")

    # ---------------- Verdict ----------------
    linf = {k: v for k, v in resultats.items() if k[0] in ("grad", "bb") and k[2] == max(args.eps)}
    if linf:
        pire = min(linf.values())
        qui = [k[1] for k, v in linf.items() if v == pire][0]
        print("\n" + "=" * 72)
        print(f"  PIRE CAS a eps={max(args.eps)} : {pire:.1%}   (attaque : {qui})")
        print(f"  Precision propre                : {propre:.1%}")
        print("=" * 72)

    m, s = divmod(time.time() - t_total, 60)
    print(f"\n[TEMPS] suite terminee en {int(m)} min {int(s)} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
