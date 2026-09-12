#!/usr/bin/env python3
"""diag_attaque_interne.py - une attaque interne apprend-elle quelque chose ?

Compare, sur un meme lot d'images, ce que valent plusieurs facons de fabriquer
l'exemple adverse d'un batch d'ENTRAINEMENT :

  1. le DEPART ALEATOIRE seul, c'est-a-dire du bruit uniforme dans la boule eps
  2. APGD-CE avec retour="dernier"  (ce que fait l'entrainement corrige : le
     dernier point de la marche, comme PGD)
  3. APGD-CE avec retour="meilleur" (semantique d'AutoAttack pour l'EVALUATION :
     le pire point de la trajectoire, depart aleatoire inclus)
  4. PGD avec un pas fin eps/10 (la recette du run v4)
  5. PGD-10 avec le pas eps/4 (la metrique de suivi affichee chaque epoch)

Pour chacune : la cross-entropy moyenne, la precision du modele et |delta|inf.

Lecture : ln(10) = 2.303. Une ligne a CE ~ 2.30 et precision ~10% veut dire que
le modele repond n'importe quoi (uniforme) sur ces images : elles ne lui
apprennent RIEN. Si la ligne 1 (le simple depart aleatoire) est aussi
destructrice que les attaques, alors l'attaque qui renvoie le "meilleur point"
renvoie ce bruit - et un modele entraine la-dessus reste bloque.

Mesure du 2026-09-12 (run v5 casse) : CE adv 2.32 = ln(10) et val PGD10 figee a
11.6% pendant 12 epochs, alors que le meme modele et la meme recette donnaient
91.0% avec une attaque interne PGD. Detail : adversarial/memoire.md.

    python3 adversarial/torch/diag_attaque_interne.py --weights models/harden_v5_apgd_ce.pt
    python3 adversarial/torch/diag_attaque_interne.py --weights models/harden_v4_pgd20.pt
"""

import argparse
import math
import sys
from os.path import abspath, dirname, isabs, join

import torch
import torch.nn.functional as F

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, join(ROOT_DIR, "adversarial", "torch"))

from adversarial.torch.attaques import pgd, accuracy          # noqa: E402
from adversarial.torch.attaques_avancees import apgd          # noqa: E402
from adversarial.torch.entrainement import charger_test       # noqa: E402
from adversarial.torch.harden_torch import choisir_device     # noqa: E402
from adversarial.torch.eval_suite import charger_modele       # noqa: E402


def decrire(nom, modele, x, xa, y):
    with torch.no_grad():
        ce = F.cross_entropy(modele(xa), y).item()
        acc = accuracy(modele, xa, y)
    delta = (xa - x).abs().flatten(1).max(1).values.mean().item()
    print(f"  {nom:<36} | CE {ce:6.3f} | acc {acc:6.2%} | |delta|inf {delta:.3f}")


def main():
    p = argparse.ArgumentParser(description="L'attaque interne d'entrainement est-elle informative ?")
    p.add_argument("--weights", default="models/harden_v5_apgd_ce.pt")
    p.add_argument("--n", type=int, default=256, help="images de test")
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "dml"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = choisir_device(args.device)
    chemin = args.weights if isabs(args.weights) else join(ROOT_DIR, args.weights)
    modele = charger_modele(chemin, device, "mnist")
    x, y = charger_test("mnist", args.n)
    x, y = x.to(device), y.to(device)

    print("=" * 78)
    print("  DIAGNOSTIC DE L'ATTAQUE INTERNE")
    print("=" * 78)
    print(f"[LOAD] {args.weights}  |  {len(x)} images  |  eps={args.eps}  |  peripherique {device}")
    with torch.no_grad():
        print(f"[REF]  precision propre du modele : {accuracy(modele, x, y):.2%}")
    print(f"[REF]  ln(10) = {math.log(10):.3f} = reponse uniforme du modele\n")

    # 1) le depart aleatoire seul (candidat du retour "meilleur")
    b = torch.empty_like(x, device="cpu").uniform_(-args.eps, args.eps).to(device)
    decrire("1. depart aleatoire seul (bruit)", modele, x, (x + b).clamp(0.0, 1.0), y)

    # 2) et 3) APGD, les deux semantiques de retour
    xa_dernier = apgd(modele, x, y, args.eps, loss="ce", steps=args.steps,
                      restarts=1, random_start=True, seed=1234, retour="dernier")
    decrire(f"2. APGD-CE-{args.steps} retour=dernier", modele, x, xa_dernier, y)
    xa_meilleur = apgd(modele, x, y, args.eps, loss="ce", steps=args.steps,
                       restarts=1, random_start=True, seed=1234, retour="meilleur")
    decrire(f"3. APGD-CE-{args.steps} retour=meilleur", modele, x, xa_meilleur, y)

    # 4) et 5) PGD, pas fin (recette v4) et pas eps/4 (metrique de suivi)
    decrire(f"4. PGD-{args.steps} pas eps/10 (v4)", modele, x,
            pgd(modele, x, y, args.eps, steps=args.steps, alpha=args.eps / 10.0), y)
    decrire("5. PGD-10 pas eps/4 (suivi)", modele, x,
            pgd(modele, x, y, args.eps, steps=10), y)

    print("\n  Comment lire : si les lignes 1 et 3 sont au niveau de ln(10) alors")
    print("  l'attaque qui garde le meilleur point renvoie le bruit de depart, et")
    print("  l'entrainement n'apprend rien sur la moitie adverse du batch.")
    print("  La ligne 2 (retour=dernier) doit etre nettement en dessous.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
