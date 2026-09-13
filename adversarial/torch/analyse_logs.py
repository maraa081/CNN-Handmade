#!/usr/bin/env python3
"""Analyse des trajectoires d'entrainement (taux de tromperie de l'attaque
interne et CE adverse, epoch par epoch) pour tester le mecanisme propose dans
l'article (section 4.4) : "un depart a budget bas garde l'attaque interne
informative".

Usage : python3 adversarial/torch/analyse_logs.py [dossier_logs]
"""

import os
import re
import sys

MOTIF = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|.*?"
    r"CE propre\s+([\d.]+)\s*\|\s*CE adv\s+([\d.]+)\s*\|\s*attaque\s+([\d.]+)%"
    r".*?val clean\s+([\d.]+)%.*?val PGD\d+\s+([\d.]+)%")


def lire(chemin):
    lignes = []
    with open(chemin, encoding="utf-8", errors="ignore") as f:
        for ligne in f:
            m = MOTIF.search(ligne)
            if m:
                lignes.append(dict(
                    epoch=int(m.group(1)), total=int(m.group(2)),
                    ce_propre=float(m.group(3)), ce_adv=float(m.group(4)),
                    attaque=float(m.group(5)), val_clean=float(m.group(6)),
                    val_pgd=float(m.group(7))))
    return lignes


def decrire(nom, lignes):
    if not lignes:
        return None
    def a(frac):
        i = min(len(lignes) - 1, max(0, round(frac * (len(lignes) - 1))))
        return lignes[i]
    deb, q1, moitie, q3, fin = a(0), a(0.25), a(0.5), a(0.75), a(1.0)
    return dict(nom=nom, n=len(lignes), deb=deb, q1=q1, moitie=moitie, q3=q3, fin=fin)


def main():
    dossier = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "logs")
    fichiers = sorted(f for f in os.listdir(dossier) if f.endswith(".log"))
    resumes = []
    for f in fichiers:
        lignes = lire(os.path.join(dossier, f))
        r = decrire(f.replace(".log", ""), lignes)
        if r:
            resumes.append(r)

    print("Trajectoires : a chaque run, la valeur a 0%, 25%, 50%, 75% et 100% du")
    print("run. 'attaque' = part du lot adverse que l'attaque interne a fait")
    print("basculer ; 'CE adv' = perte adverse ; 'PGD10' = robustesse de validation")
    print("(attaque faible) ; ln(10) = 2.303 = niveau du hasard sur 10 classes.\n")
    entete = (f"{'run':<28}{'epoch':>6}{'attaque':>9}{'CE adv':>9}"
              f"{'CE prop':>9}{'val cln':>9}{'val PGD':>9}")
    for r in resumes:
        print(f"=== {r['nom']} ({r['n']} epochs lus)")
        print(entete)
        for etape, e in (("debut", r["deb"]), ("25%", r["q1"]), ("50%", r["moitie"]),
                         ("75%", r["q3"]), ("fin", r["fin"])):
            print(f"{etape:<28}{e['epoch']:>6}{e['attaque']:>8.1f}%"
                  f"{e['ce_adv']:>9.3f}{e['ce_propre']:>9.3f}"
                  f"{e['val_clean']:>8.1f}%{e['val_pgd']:>8.1f}%")
        print()

    print("SYNTHESE (comparaison des familles)\n")
    print(f"{'run':<28}{'attaque deb':>12}{'attaque fin':>12}{'CEadv deb':>11}"
          f"{'CEadv fin':>11}{'PGD fin':>9}")
    for r in resumes:
        print(f"{r['nom']:<28}{r['deb']['attaque']:>11.1f}%{r['fin']['attaque']:>11.1f}%"
              f"{r['deb']['ce_adv']:>11.3f}{r['fin']['ce_adv']:>11.3f}"
              f"{r['fin']['val_pgd']:>8.1f}%")


if __name__ == "__main__":
    main()
