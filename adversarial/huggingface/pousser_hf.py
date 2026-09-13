#!/usr/bin/env python3
"""pousser_hf.py - publie la carte du modele et l'espace de demo sur Hugging Face.

Pourquoi ce script : les poids `.pt` sont ignores par git (`models/*.pt`), donc
le depot GitHub ne peut pas les heberger. Hugging Face est leur place (1,7 Mo
par modele, versionnes), et le meme compte heberge l'espace de demonstration.

Le script est IDEMPOTENT et PRUDENT : par defaut il ne fait rien, il affiche le
plan (quels fichiers, vers quel depot, lesquels manquent). Il faut `--envoyer`
pour ecrire chez Hugging Face. Il refuse d'envoyer si un poids manque, plutot
que de publier un depot a moitie rempli.

Usage
-----
    # 1. Regarder ce qui partirait (rien n'est envoye)
    python3 adversarial/huggingface/pousser_hf.py

    # 2. Envoyer (jeton en variable d'environnement, jamais sur la ligne de
    #    commande : la ligne de commande reste dans l'historique du shell)
    HF_TOKEN=hf_xxx python3 adversarial/huggingface/pousser_hf.py --envoyer

    # Envoyer seulement la carte, ou seulement l'espace
    HF_TOKEN=hf_xxx python3 adversarial/huggingface/pousser_hf.py --envoyer --seulement carte
    HF_TOKEN=hf_xxx python3 adversarial/huggingface/pousser_hf.py --envoyer --seulement espace

Prerequis : `pip install huggingface_hub` et un jeton avec droit d'ECRITURE
(https://huggingface.co/settings/tokens). Le compte et les noms de depots sont
dans le bloc REGLAGES ci-dessous.
"""

import argparse
import os
import sys
from os.path import abspath, dirname, exists, isabs, join

# --------------------------------------------------------------------------
#  REGLAGES : a confirmer avec Maraa avant le premier envoi
# --------------------------------------------------------------------------

DEPOT_MODELE = "maraa081/mnist-cnn-robuste"
DEPOT_ESPACE = "maraa081/mnist-robuste-demo"

# (nom dans le depot, chemin local, description courte)
# Le nom dans le depot doit correspondre EXACTEMENT aux cles de FICHIERS dans
# espace/app.py, sinon l'espace ne trouve pas ses poids.
POIDS = [
    ("bande_cible50.pt", "models/bande_cible50.pt",
     "champion : budget adaptatif, 98.85% propre / 91.25% officiel"),
    ("a6_plan_budget.pt", "models/a6_plan_budget.pt",
     "plan 0.2 -> 2 eps, 99.17% propre / 82.40% officiel"),
    ("a8_plan_0p2_1.pt", "models/a8_plan_0p2_1.pt",
     "plan 0.2 -> 1 eps, 99.27% propre / 78.25% officiel"),
    ("abl_a_eps10.pt", "models/abl_a_eps10.pt",
     "reference : constante 2 eps, 99.53% propre / 50.71% officiel"),
    ("standard.pt", "models/standard.pt",
     "modele non defendu (temoin)"),
]

ICI = dirname(abspath(__file__))          # adversarial/huggingface/
ROOT = dirname(dirname(ICI))              # racine du depot
CARTE = join(ICI, "README.md")            # carte du modele (README.md du depot HF)
ESPACE = join(ICI, "espace")              # fichiers du Space


def resoudre(chemin):
    return chemin if isabs(chemin) else join(ROOT, chemin)


def main():
    p = argparse.ArgumentParser(description="Publie la carte et l'espace sur Hugging Face")
    p.add_argument("--envoyer", action="store_true",
                   help="ecrit reellement chez Hugging Face (sans ce flag : simulation)")
    p.add_argument("--seulement", choices=["carte", "espace"], default=None,
                   help="publier seulement la carte du modele ou seulement l'espace")
    p.add_argument("--revision", default=None, help="branche du depot (defaut : main)")
    args = p.parse_args()

    # ---- Etat des lieux, avant toute chose ----
    print("Plan de publication")
    print(f"  carte    : {CARTE}")
    print(f"  espace   : {ESPACE}")
    print(f"  depot    : {DEPOT_MODELE} (+ espace {DEPOT_ESPACE})")
    print()
    manquants = []
    total = 0
    for nom, local, description in POIDS:
        chemin = resoudre(local)
        if exists(chemin):
            taille = os.path.getsize(chemin)
            total += taille
            print(f"  [OK]    {nom:<20} {taille / 1e6:5.2f} Mo  <- {local}")
        else:
            manquants.append((nom, local))
            print(f"  [MANQUE] {nom:<20}        -  <- {local} ({description})")
    print(f"\n  total : {total / 1e6:.2f} Mo sur {len(POIDS)} fichier(s)")

    if manquants:
        print("\n[ARRET] poids manquants, rien n'a ete envoye :")
        for nom, local in manquants:
            print(f"  {nom} <- {local}")
        print("  Corriger le bloc POIDS (nom exact du fichier dans models/) ou "
              "retirer le modele de la liste.")
        return 1

    if not args.envoyer:
        print("\n[SIMULATION] rien n'a ete envoye. Relancer avec --envoyer "
              "et HF_TOKEN dans l'environnement.")
        return 0

    jeton = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not jeton:
        print("[ARRET] aucun jeton : mettre HF_TOKEN dans l'environnement "
              "(jamais sur la ligne de commande).")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ARRET] huggingface_hub absent : pip install huggingface_hub")
        return 1

    api = HfApi(token=jeton)
    qui = api.whoami()
    print(f"\n[HF] connecte : {qui.get('name')}")

    if args.seulement in (None, "carte"):
        api.create_repo(DEPOT_MODELE, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=CARTE, path_in_repo="README.md",
                        repo_id=DEPOT_MODELE, revision=args.revision)
        print(f"[HF] carte envoyee -> {DEPOT_MODELE}/README.md")
        for nom, local, _ in POIDS:
            api.upload_file(path_or_fileobj=resoudre(local), path_in_repo=nom,
                            repo_id=DEPOT_MODELE, revision=args.revision)
            print(f"[HF] poids envoye -> {DEPOT_MODELE}/{nom}")

    if args.seulement in (None, "espace"):
        api.create_repo(DEPOT_ESPACE, repo_type="space", exist_ok=True,
                        space_sdk="gradio")
        api.upload_folder(folder_path=ESPACE, repo_id=DEPOT_ESPACE,
                          repo_type="space", revision=args.revision,
                          ignore_patterns=["__pycache__/*", "*.pyc"])
        print(f"[HF] espace publie -> {DEPOT_ESPACE}")
        print(f"     l'espace lit les poids dans {DEPOT_MODELE} "
              "(variable MODELE_HF, defaut dans app.py)")

    print("\n[FIN] fait. Verifier les deux pages dans le navigateur avant "
          "d'annoncer quoi que ce soit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
