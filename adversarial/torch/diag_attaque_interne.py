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
apprennent RIEN. Mesure du 2026-09-12 (run v5) : CE adv 2.33 = ln(10) et val
PGD10 figee a 11.6% pendant 12 epochs, alors que le meme modele et la meme
recette donnaient 91.0% avec une attaque interne PGD (pas fin eps/10).

Deux colonnes de structure ont ete ajoutees apres ce diagnostic :

  - 'borne' = part de pixels pousses a la limite de la boule (|delta| = eps).
    Un pas de 2*eps (APGD) saute au coin des le premier pas ; un pas eps/10
    (PGD, recette v4) y arrive progressivement, donc une plus grande part de
    pixels reste a l'interieur : la perturbation est plus douce, donc
    apprenable par le modele.
  - 'signes opposes' = part de pixels voisins dont la perturbation change de
    signe. 50% = masque binaire aleatoire (illisible), nettement moins = champ
    de signe coherent.

ATTENTION (mesure du 12/09, moteur NumPy) : le bruit uniforme n'est PAS une
attaque. Sur le modele standard non robuste (model_weights_full.npz) : propre
98.5%, bruit uniforme eps=0.3 97.0%, FGSM eps=0.3 2.1%. La ligne 1 a ~90-97%
est donc normale et n'indique rien.

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


def decrire(nom, modele, x, xa, y, eps):
    with torch.no_grad():
        ce = F.cross_entropy(modele(xa), y).item()
        acc = accuracy(modele, xa, y)
    delta = xa - x
    moyen = delta.abs().mean().item()
    # fraction de pixels pousses a la borne de la boule (|delta| = eps)
    frac_borne = (delta.abs() >= 0.99 * eps).float().mean().item()
    # structure : part de pixels voisins dont le signe differe (50% = bruit pur,
    # moins = champ de signe coherent, donc perturbation "lisible")
    s = torch.sign(delta)
    flips = (s[:, :, :, 1:] != s[:, :, :, :-1]).float().mean().item()
    flips += (s[:, :, 1:, :] != s[:, :, :-1, :]).float().mean().item()
    print(f"  {nom:<33} | CE {ce:6.3f} | acc {acc:6.2%} | |d|moy {moyen:.3f} | "
          f"borne {frac_borne:5.1%} | signes opposes {flips / 2:5.1%}")


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
    decrire("1. depart aleatoire seul (bruit)", modele, x, (x + b).clamp(0.0, 1.0), y, args.eps)

    # 2) et 3) APGD, les deux semantiques de retour
    xa_dernier = apgd(modele, x, y, args.eps, loss="ce", steps=args.steps,
                      restarts=1, random_start=True, seed=1234, retour="dernier")
    decrire(f"2. APGD-CE-{args.steps} retour=dernier", modele, x, xa_dernier, y, args.eps)
    xa_meilleur = apgd(modele, x, y, args.eps, loss="ce", steps=args.steps,
                       restarts=1, random_start=True, seed=1234, retour="meilleur")
    decrire(f"3. APGD-CE-{args.steps} retour=meilleur", modele, x, xa_meilleur, y, args.eps)

    # 4) et 5) PGD, pas fin (recette v4) et pas eps/4 (metrique de suivi)
    decrire(f"4. PGD-{args.steps} pas eps/10 (v4)", modele, x,
            pgd(modele, x, y, args.eps, steps=args.steps, alpha=args.eps / 10.0), y, args.eps)
    decrire("5. PGD-10 pas eps/4 (suivi)", modele, x,
            pgd(modele, x, y, args.eps, steps=10), y, args.eps)

    print("\n  Comment lire :")
    print("  - 'CE' : si elle vaut ln(10) = 2.30, le modele repond uniformement sur ces")
    print("    images : il n'apprend rien de la moitie adverse du batch.")
    print("  - '|d|moy' et 'pixels a la borne' : a quel point une attaque pousse TOUS les")
    print("    pixels a la limite de la boule. Un pas de 2*eps (APGD) saute au coin des")
    print("    le premier pas ; un pas eps/10 (PGD v4) y arrive progressivement.")
    print("  - 'signes opposes voisins' : 50% = masque de bruit pur (illisible pour le")
    print("    modele), nettement moins = perturbation coherente, donc apprenable.")
    print("  Comparer 2 et 4 : ce sont les deux seules que l'ENTRAINEMENT utilise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
