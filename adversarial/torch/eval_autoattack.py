#!/usr/bin/env python3
"""eval_autoattack.py - le juge officiel : AutoAttack (Croce & Hein).

C'est l'item n°1 du perimetre "avant de publier" : notre suite est une
reimplementation maison. Tant que les chiffres ne sont pas croises avec le
paquet de reference, aucun nombre n'est opposable devant une relecture.

AutoAttack enchaine quatre attaques et prend la pire :
  - APGD-CE  (pas adaptatif, momentum, restarts)
  - APGD-DLR (meme chose avec la Difference of Logits Ratio, insensible a la
    saturation de la CE)
  - FAB      (Fast Adaptive Boundary : minimisation de la distance Lp)
  - Square   (boite noire, sans gradient)
Et "version=plus" ajoute deux attaques supplementaires (APGD-targeted et
Square avec plus de requetes) : a reserver aux chiffres d'annonce.

Usage
-----
    # installation (une fois)
    pip install git+https://github.com/fra31/auto-attack

    # chiffre opposable sur tout le jeu de test
    python -u adversarial/torch/eval_autoattack.py \
        --weights models/bande_cible50.pt --device cuda --n 10000

    # version longue (chiffres d'annonce, plus lente)
    python -u adversarial/torch/eval_autoattack.py \
        --weights models/bande_cible50.pt --device cuda --n 10000 --version plus

Comment lire le resultat : `precision robuste` est un PIRE CAS sur les quatre
attaques. Si elle est proche de ce que donne notre suite, nos chiffres sont
validés. Si elle est nettement plus basse, notre suite sous-estimait la
vulnerabilite (ce qui est le cas typique quand un modele a une surface
rugueuse : cf. le rapport gradient/aleatoire de `audit_masquage.py`).
"""

import argparse
import sys
import time
from os.path import abspath, dirname, join

import torch

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, join(ROOT_DIR, "adversarial", "torch"))

from adversarial.torch.entrainement import charger_test        # noqa: E402
from adversarial.torch.eval_suite import charger_modele        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    p.add_argument("--eps", type=float, default=0.3, help="budget L-infini")
    p.add_argument("--n", type=int, default=10000, help="images de test (10000 = le jeu complet)")
    p.add_argument("--bs", type=int, default=250, help="taille de lot")
    p.add_argument("--version", choices=["standard", "plus", "rand"], default="standard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "dml"])
    args = p.parse_args()

    try:
        from autoattack import AutoAttack
    except ImportError:
        print("=" * 70)
        print("  AutoAttack n'est pas installe.")
        print("=" * 70)
        print("\n  Installation :\n")
        print("    pip install git+https://github.com/fra31/auto-attack\n")
        return 1

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    modele = charger_modele(args.weights, device, args.dataset)
    x, y = charger_test(args.dataset, args.n)
    x, y = x.to(device), y.to(device)

    print("=" * 70)
    print("  AUTOATTACK (le juge officiel)")
    print("=" * 70)
    print(f"  modele   : {args.weights}")
    print(f"  images   : {len(x)} | eps={args.eps} (L-infini) | version={args.version}")
    print(f"  appareil : {device}")
    print(f"  attaques : APGD-CE, APGD-DLR, FAB, Square"
          + (" (+ APGD-targeted, Square-plus)" if args.version == "plus" else ""))

    # Version de la lib : les API ont bouge entre les versions, on se protege.
    try:
        import autoattack as _aa
        print(f"  autoattack : {getattr(_aa, '__version__', 'inconnue')}")
    except Exception:                                          # noqa: BLE001
        pass

    adversaire = AutoAttack(modele, norm="Linf", eps=args.eps, version=args.version,
                            device=device, seed=args.seed)
    t0 = time.time()
    x_adv = adversaire.run_standard_evaluation(x, y, bs=args.bs)
    dt = time.time() - t0

    # `run_standard_evaluation` imprime deja la precision apres chaque attaque ;
    # on refait le calcul ici pour le chiffre final, avec la meme definition que
    # notre suite (argmax == y).
    with torch.no_grad():
        robuste = (modele(x_adv).argmax(1) == y).float().mean().item()
        propre = (modele(x).argmax(1) == y).float().mean().item()

    print("\n" + "=" * 70)
    print("  RESULTAT (a recopier dans memoire.md et la model card)")
    print("=" * 70)
    print(f"  precision propre      : {propre:.2%}")
    print(f"  precision ROBUSTE     : {robuste:.2%}   (pire cas AutoAttack, eps={args.eps})")
    print(f"  duree                 : {dt / 60:.1f} min")
    print(f"  images                : {len(x)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
