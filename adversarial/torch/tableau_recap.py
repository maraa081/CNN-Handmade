#!/usr/bin/env python3
"""tableau_recap.py - tableau final pret a coller, a partir des JSON d'evaluation.

Probleme resolu : a la fin d'une serie de runs (KMNIST, puis les variantes), on
recopiait les chiffres a la main depuis les logs -- lent, et une faute de frappe
sur un pourcentage suffit a raconter une autre histoire. Ici, chaque
`eval_suite.py --json ...` ecrit ses resultats une fois, et ce script les relit
pour imprimer UN tableau Markdown, trie, pret a coller dans le README ou dans la
model card.

Usage
-----
    # Tous les JSON presents dans adversarial/results/logs/ (defaut)
    python3 adversarial/torch/tableau_recap.py

    # Seulement certains fichiers
    python3 adversarial/torch/tableau_recap.py chemin/a.json chemin/b.json

    # Export tableur (point-virgule, virgule decimale francaise)
    python3 adversarial/torch/tableau_recap.py --csv recap.csv

    # Regarder un autre budget que le plus grand (par defaut : le max des eps)
    python3 adversarial/torch/tableau_recap.py --eps 0.2

Aucune dependance a torch : ce script ne lit que du JSON.
"""

import argparse
import glob
import json
import os
import re
import sys
from os.path import abspath, basename, dirname, exists, join

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
DEFAUT = join(ROOT_DIR, "adversarial", "results", "logs", "*.json")

# Ordre d'affichage des colonnes d'attaque. Le prefixe suffit : le nom complet
# contient les parametres ("PGD-20 (3 rest.)", "Square (3000)"), et on veut les
# voir -- c'est justement ce qui rend deux colonnes comparables ou non.
ORDRE = ("FGSM", "PGD", "APGD-CE", "APGD-DLR", "Square", "NES")

# Une entree est consideree comme un resultat de cette suite si elle a ce champ.
CHAMP_FORMAT = "format"


def rang(nom):
    """Cle de tri d'une attaque : famille, puis nombre de pas croissant, puis nom.

    Le nombre est ce qui rend l'ordre lisible : Square-500 avant Square-3000
    (l'attaque la moins longue d'abord), sans dependre du tri alphabetique.
    Les inconnues vont a la fin.
    """
    majuscule = nom.upper()
    nombres = [int(m) for m in re.findall(r"\d+", nom)]
    pas = nombres[0] if nombres else 0
    for i, prefixe in enumerate(ORDRE):
        if majuscule.startswith(prefixe.upper()):
            return (i, pas, nom)
    return (len(ORDRE), pas, nom)


def lire(chemin):
    """Charge un JSON d'evaluation, ou None si ce n'en est pas un."""
    try:
        with open(chemin, encoding="utf-8") as f:
            etat = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[SKIP] {chemin} : illisible ({e})")
        return None
    if not isinstance(etat, dict) or CHAMP_FORMAT not in etat:
        print(f"[SKIP] {chemin} : pas un JSON d'eval_suite.py / eval_autoattack.py")
        return None
    etat.setdefault("source", "suite")
    etat["_fichier"] = chemin
    return etat


def cle_modele(etat):
    """Identifie le modele evalue, pour rapprocher suite maison et AutoAttack."""
    return basename(etat.get("modele") or etat.get("libelle") or "?")


def eps_choisi(etat, demande):
    """Le budget a lire : celui demande, sinon le plus grand du fichier."""
    dispo = sorted(float(e) for e in etat.get("eps", []))
    if not dispo:
        return None
    if demande is None:
        return dispo[-1]
    for e in dispo:
        if abs(e - demande) < 1e-9:
            return e
    return None


def valeur(etat, nom_attaque, eps):
    """Accuracy d'une attaque au budget donne, ou None."""
    table = etat.get("attaques", {}).get(nom_attaque, {})
    for cle, v in table.items():
        if abs(float(cle) - eps) < 1e-9:
            return v
    return None


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def pct_gras(v):
    return "-" if v is None else f"**{v * 100:.1f}%**"


def construire(lignes, eps):
    """Assemble les colonnes et les cellules de chaque ligne."""
    noms = sorted({n for l in lignes for n in l["etat"].get("attaques", {})},
                  key=rang)
    tableau = []
    for l in lignes:
        etat = l["etat"]
        cel = {n: valeur(etat, n, eps) for n in noms}
        vals = [v for v in cel.values() if v is not None]
        pire_js = etat.get("pire_cas") or {}
        eps_max = max(float(e) for e in etat.get("eps", [eps]))
        if abs(eps - eps_max) < 1e-9 and pire_js.get("accuracy") is not None:
            # le verdict ecrit par eval_suite.py (familles gradient + sans gradient)
            pire_val, pire_att = pire_js["accuracy"], pire_js.get("attaque", "?")
        else:
            pire_val, pire_att = (min(vals) if vals else None), "min des colonnes"
        tableau.append({
            "dataset": etat.get("dataset", "?"),
            "libelle": etat.get("libelle") or etat.get("modele", "?"),
            "modele": etat.get("modele", ""),
            "n": etat.get("n_images", "?"),
            "propre": etat.get("propre"),
            "attaques": cel,
            "pire": pire_val,
            "pire_attaque": pire_att,
        })
    # Tri : jeu de donnees, puis pire cas decroissant (le meilleur en haut).
    tableau.sort(key=lambda r: (r["dataset"], -(r["pire"] if r["pire"] is not None else -1)))
    for i, r in enumerate(tableau, 1):
        r["rang"] = i
    return noms, tableau


def entete_markdown(noms, eps, avec_officiel):
    cols = ["#", "Jeu", "Modele", "Images", "Propre"] + list(noms)
    cols += ["Pire cas maison"] + (["AutoAttack (officiel)"] if avec_officiel else [])
    return "| " + " | ".join(cols) + " |\n" + "|" + "|".join(["---"] * len(cols)) + "|"


def ligne_markdown(r, noms, eps, avec_officiel):
    cols = [str(r["rang"]), r["dataset"], r["libelle"], str(r["n"]),
            pct_gras(r["propre"])]
    cols += [pct(r["attaques"][n]) for n in noms]
    cols += [pct_gras(r["pire"])]
    if avec_officiel:
        officiel = (r.get("officiel") or {}).get("pire_cas", {}).get("accuracy")
        cols += [pct_gras(officiel)]
    return "| " + " | ".join(cols) + " |"


def note(r, noms, eps):
    """Dit QUI a produit le pire cas : le chiffre seul peut tromper."""
    if r["pire_attaque"] and r["pire_attaque"] != "min des colonnes":
        return f"  {r['libelle']:<28} pire cas {pct(r['pire']):>7}  " \
               f"(eps={eps:.2f}, attaque : {r['pire_attaque']})"
    return f"  {r['libelle']:<28} pire cas {pct(r['pire']):>7}  (eps={eps:.2f})"


def ecrire_csv(chemin, noms, tableau, eps, avec_officiel):
    entetes = ["rang", "dataset", "libelle", "modele", "images", "eps", "propre"]
    entetes += [f"{n}" for n in noms] + ["pire_cas_maison"]
    entetes += ["autoattack_officiel"] if avec_officiel else []
    entetes += ["pire_cas_attaque"]
    lignes = [";".join(entetes)]
    for r in tableau:
        cel = [str(r["rang"]), r["dataset"], r["libelle"], r["modele"], str(r["n"]),
               f"{eps:.2f}", f"{r['propre'] * 100:.1f}".replace(".", ",")]
        cel += ["-" if r["attaques"][n] is None else f"{r['attaques'][n] * 100:.1f}".replace(".", ",")
                for n in noms]
        cel += ["-" if r["pire"] is None else f"{r['pire'] * 100:.1f}".replace(".", ",")]
        if avec_officiel:
            off = (r.get("officiel") or {}).get("pire_cas", {}).get("accuracy")
            cel += ["-" if off is None else f"{off * 100:.1f}".replace(".", ",")]
        cel += [str(r["pire_attaque"])]
        lignes.append(";".join(cel))
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print(f"[CSV] {len(tableau)} ligne(s) ecrite(s) -> {chemin}")


def main():
    p = argparse.ArgumentParser(description="Tableau recapitulatif des evaluations")
    p.add_argument("fichiers", nargs="*", help="JSON d'eval_suite.py (defaut : results/logs/*.json)")
    p.add_argument("--eps", type=float, default=None,
                   help="budget L-infini a lire (defaut : le plus grand de chaque fichier)")
    p.add_argument("--csv", default="", help="exporte aussi le tableau en CSV")
    p.add_argument("--titre", default="", help="titre affiche au-dessus du tableau")
    args = p.parse_args()

    chemins = args.fichiers or sorted(glob.glob(DEFAUT))
    if not chemins:
        print(f"[ERREUR] aucun JSON trouve (cherche : {DEFAUT})")
        print("  Un fichier JSON est produit par :")
        print("    python3 adversarial/torch/eval_suite.py --weights ... --json <chemin>.json")
        return 1

    lignes = []
    officiels = {}
    for chemin in chemins:
        chemin = chemin if os.path.isabs(chemin) else join(ROOT_DIR, chemin)
        if not exists(chemin):
            print(f"[SKIP] {chemin} : introuvable")
            continue
        etat = lire(chemin)
        if etat is None:
            continue
        if etat["source"] == "autoattack":
            officiels[cle_modele(etat)] = etat
            continue
        eps = eps_choisi(etat, args.eps)
        if eps is None:
            print(f"[SKIP] {chemin} : budget {args.eps} absent du fichier")
            continue
        etat["_eps"] = eps
        lignes.append({"etat": etat})

    if not lignes:
        print("[ERREUR] aucun resultat exploitable.")
        return 1

    # Budget commun, pour que toutes les lignes du tableau soient comparables :
    # on prend le plus grand eps present dans TOUS les fichiers (sauf si --eps
    # impose un budget precis).
    if args.eps is not None:
        eps_commun = args.eps
    else:
        communs = [float(x) for x in lignes[0]["etat"]["eps"]]
        for l in lignes[1:]:
            communs = [e for e in communs
                       if any(abs(e - float(y)) < 1e-9 for y in l["etat"]["eps"])]
        if not communs:
            print("[ERREUR] les fichiers n'ont aucun budget en commun.")
            return 1
        eps_commun = max(communs)

    noms, tableau = construire(lignes, eps_commun)

    # Rapproche les resultats AutoAttack du modele correspondant (meme chemin de
    # poids). Le chiffre officiel n'est PAS melange au pire cas maison : il est
    # sur 10000 images, l'autre sur 500. On les montre cote a cote.
    for r in tableau:
        r["officiel"] = officiels.get(basename(r["modele"]))
    avec_officiel = bool(officiels)
    if officiels:
        orphelins = [k for k in officiels if not any(basename(r["modele"]) == k for r in tableau)]
        for k in orphelins:
            print(f"[note] AutoAttack pour {k} : aucun resultat de suite maison a rapprocher")

    titre = args.titre or "Recapitulatif des evaluations (suite multi-familles maison)"
    print(f"### {titre}\n")
    print(f"Budget : eps={eps_commun:.2f} (L-infini). Plus le chiffre est bas, plus le modele")
    print(f"casse. ``Pire cas maison`` = minimum de la ligne sur {tableau[0]['n']} images ;"
          + (" ``AutoAttack (officiel)`` = eval_autoattack.py sur 10000 images"
             " (le chiffre d'annonce).\n" if avec_officiel else "\n"))
    print(entete_markdown(noms, eps_commun, avec_officiel))
    for r in tableau:
        print(ligne_markdown(r, noms, eps_commun, avec_officiel))
    print()
    print("Pire cas et attaque qui l'a produit :")
    for r in tableau:
        print(note(r, noms, eps_commun))
        off = (r.get("officiel") or {}).get("pire_cas", {})
        if off.get("accuracy") is not None:
            print(f"  {'':<28} officiel AutoAttack {pct(off['accuracy']):>7}  "
                  f"(eps={off.get('eps'):.2f}, {off.get('attaque')}, "
                  f"{r['officiel'].get('n_images')} images)")
    print()
    print("Lecture : deux lignes ne sont comparables que si leurs colonnes portent les memes")
    print("parametres (voir les nombres entre parentheses). Un chiffre de la suite maison est")
    print("optimiste d'environ 3 points sur 500 images ; le chiffre d'annonce est celui")
    print("d'AutoAttack sur 10000 images (eval_autoattack.py).")

    if args.csv:
        ecrire_csv(args.csv if os.path.isabs(args.csv) else join(ROOT_DIR, args.csv),
                   noms, tableau, eps_commun, avec_officiel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
