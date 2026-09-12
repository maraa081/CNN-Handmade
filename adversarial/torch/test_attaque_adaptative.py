#!/usr/bin/env python3
"""Verification de l'attaque interne a budget adaptatif (attaque_adaptative.py).

A lancer AVANT tout run de 20 minutes : les tests tournent sur un petit
modele et quelques images, ils finissent en quelques secondes sur CPU.

    python -u adversarial/torch/test_attaque_adaptative.py

Quatre choses sont verifiees :
  1. la selection du pas k* (premier passage au-dessus de la cible) ;
  2. l'equivalence avec pgd() quand --cible est desactive (k* = K) ;
  3. le COUT : nombre de passages avant+arriere identique a pgd() ;
  4. le drapeau `plafond` et le journal (resume / alerte).
"""

import sys
from os.path import abspath, dirname, join

import torch
import torch.nn.functional as F

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, dirname(abspath(__file__)))

from attaques import pgd                                             # noqa: E402
from attaque_adaptative import pgd_bande, cible_effective, JournalBande  # noqa: E402

ECHECS = []


def verifier(nom, condition, detail=""):
    etat = "OK  " if condition else "FAIL"
    print(f"  [{etat}] {nom}" + (f" : {detail}" if detail else ""))
    if not condition:
        ECHECS.append(nom)


class PetitModele(torch.nn.Module):
    """Un modele jouet : conv 1x1 + lineaire sur 28x28 aplati, 10 classes."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 28 * 28, 10)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty(p.shape).uniform_(-0.05, 0.05, generator=g))

    def forward(self, x):
        h = torch.relu(self.conv(x))
        return self.fc(h.flatten(1))


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    modele = PetitModele().eval()
    x = torch.rand(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    eps, steps, alpha = 0.3, 20, 0.03

    print("\n[1] Selection du pas k*")
    # IMPORTANT : hors de torch.no_grad() -- l'attaque a besoin du graphe pour
    # calculer le gradient par rapport a l'image (erreur attrapee le 12/09 :
    # sous no_grad les logits n'ont pas de grad_fn et torch.autograd.grad leve
    # "element 0 of tensors does not require grad").
    _, res_cible = pgd_bande(modele, x, y, eps, steps, alpha, cible=0.5)
    traj = res_cible["tromperie_trajectoire"]
    k = res_cible["k"]
    if k < steps:
        avant = traj[k - 1] if k > 0 else 0.0
        verifier("k* est le PREMIER pas au-dessus de la cible",
                 traj[k] >= 0.5 and avant < 0.5,
                 f"k*={k}, d(k*-1)={avant:.2f}, d(k*)={traj[k]:.2f}")
    else:
        verifier("cible atteinte en <= K pas (ou plafond signale)",
                 res_cible["plafond"], f"k*={k}, plafond={res_cible['plafond']}")
    verifier("budget = k* x alpha", abs(res_cible["budget"] - k * alpha) < 1e-9,
             f"{res_cible['budget']:.3f}")
    verifier("budget dans la zone sure (<= 2 eps)",
             res_cible["budget_eps"] <= 2.0 + 1e-9,
             f"{res_cible['budget_eps']:.2f} eps")

    print("\n[2] Equivalence avec pgd() quand la cible est desactivee")
    torch.manual_seed(7)
    avec_pgd = pgd(modele, x, y, eps, steps, alpha)
    torch.manual_seed(7)
    avec_bande, res_fixe = pgd_bande(modele, x, y, eps, steps, alpha, cible=None)
    verifier("k* = K quand cible=None", res_fixe["k"] == steps, f"k*={res_fixe['k']}")
    verifier("points identiques a pgd()",
             torch.allclose(avec_pgd, avec_bande, atol=0.0),
             f"ecart max {(avec_pgd - avec_bande).abs().max().item():.2e}")

    print("\n[3] Cout : meme nombre de passages avant+arriere que pgd()")
    compteur = {"n": 0}

    def hook(_module, _entree, _sortie):
        compteur["n"] += 1

    h = modele.register_forward_hook(hook)
    # pgd() : une passe avant par pas (le gradient vient du meme graphe).
    compteur["n"] = 0
    pgd(modele, x, y, eps, steps, alpha)
    n_pgd = compteur["n"]
    # pgd_bande() : meme structure, la difficulte est lue sur les logits
    # deja calcules -> aucune passe avant supplementaire.
    compteur["n"] = 0
    pgd_bande(modele, x, y, eps, steps, alpha, cible=0.5)
    n_bande = compteur["n"]
    h.remove()
    verifier("passages avant : bande == pgd", n_pgd == n_bande,
             f"pgd={n_pgd}, bande={n_bande} (attendu {steps})")
    verifier("et == nombre de pas", n_pgd == steps, f"pgd={n_pgd}")

    print("\n[4] Plafond et journal")
    # cible impossible a atteindre en K pas -> plafond
    _, res_haut = pgd_bande(modele, x, y, eps, steps, alpha, cible=1.01)
    verifier("cible hors de portee -> plafond signale", res_haut["plafond"])
    verifier("cible hors de portee -> k* = K", res_haut["k"] == steps)
    j = JournalBande()
    for _ in range(3):
        j.ajouter(res_cible)
    j.ajouter(res_haut)
    verifier("resume non vide", bool(j.resume()), j.resume())
    verifier("alerte au-dela du seuil de plafond",
             bool(j.alerte(seuil=0.2)) and not j.alerte(seuil=0.9))

    print("\n[5] Warmup de la cible")
    verifier("cible au depart", abs(cible_effective(0.5, 1, 120, 0.5, 0.2) - 0.2) < 0.01)
    verifier("cible atteinte a la fin", abs(cible_effective(0.5, 120, 120, 0.5, 0.2) - 0.5) < 1e-9)
    verifier("rampe=0 -> cible constante", cible_effective(0.5, 5, 120, 0.0, 0.2) == 0.5)

    print("\n" + "=" * 68)
    if ECHECS:
        print(f"  {len(ECHECS)} test(s) en echec : {', '.join(ECHECS)}")
        return 1
    print("  Tous les tests passent : le module est pret pour un run.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
