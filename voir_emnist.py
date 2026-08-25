#!/usr/bin/env python3
"""
voir_emnist.py — Affiche les lettres EMNIST (idéal dans Spyder ou IPython)

Usage :
    python3 voir_emnist.py              # grille aléatoire de 20 lettres
    python3 voir_emnist.py --letter a   # toutes les variantes d'écriture de 'a'
    python3 voir_emnist.py --letter z --n 30   # 30 variantes de 'z'

Dans Spyder : ouvre le fichier et lance-le (F5), la grille s'affiche dans les Plots.
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import EMNISTLoader
from download_emnist import ensure_data

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def grille_aleatoire(x, y, n=20):
    """n lettres tirées au hasard, avec leur label en titre."""
    n_cols = 5
    n_rows = n // n_cols + (1 if n % n_cols else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.RandomState(42)
    indices = rng.choice(len(x), size=n, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"'{LETTERS[y[idx]]}'", fontsize=14)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("EMNIST Letters — échantillon aléatoire", fontsize=15)
    plt.tight_layout()
    plt.show()


def grille_par_lettre(x, y, lettre, n=24):
    """n variantes d'écriture d'une même lettre (pour voir la diversité)."""
    if lettre not in LETTERS:
        print(f"FAIL Lettre invalide : '{lettre}'. Choisis dans : {LETTERS}")
        sys.exit(1)

    idx_lettre = LETTERS.index(lettre)
    indices = np.where(y == idx_lettre)[0]

    n = min(n, len(indices))
    rng = np.random.RandomState(7)
    picks = rng.choice(indices, size=n, replace=False)

    n_cols = 6
    n_rows = n // n_cols + (1 if n % n_cols else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, idx in enumerate(picks):
        ax = axes[i]
        ax.imshow(x[idx], cmap="gray")
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Variantes de la lettre '{lettre}' ({n} exemples)", fontsize=15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Afficher les lettres EMNIST")
    parser.add_argument("--letter", type=str, default=None,
                        help="afficher les variantes d'une lettre précise (a-z)")
    parser.add_argument("--n", type=int, default=24, help="nombre d'exemples")
    args = parser.parse_args()

    ensure_data()
    loader = EMNISTLoader("letters")
    (x_train, y_train), _ = loader.load(join(ROOT_DIR, "data", "emnist"))
    print(f" {len(x_train)} lettres chargées")

    if args.letter:
        grille_par_lettre(x_train, y_train, args.letter.lower(), args.n)
    else:
        grille_aleatoire(x_train, y_train, n=20)
