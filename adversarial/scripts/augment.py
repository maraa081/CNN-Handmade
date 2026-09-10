#!/usr/bin/env python3
"""
augment.py — Augmentation de donnees aleatoire pour MNIST / EMNIST.

Objectif : "brusquer" artificiellement les images d'entrainement pour que le
modele apprenne des invariants (position, orientation, epaisseur du trait) au
lieu de memoriser chaque pixel.

IMPORTANT
---------
- Rien n'est ecrit sur le disque, rien n'est envoye nulle part : les
  transformations sont appliquees en memoire, a la volee, pendant
  l'entrainement. Cout : quelques millisecondes par batch.
- A n'appliquer QUE sur le train. Jamais sur le test ni sur la validation :
  sinon on ne mesure plus les performances reelles.
- L'augmentation ne remplace PAS l'adversarial training. Elle aide la precision
  propre et la robustesse aux transformations "naturelles" (rotation, deplacement,
  bruit impulsionnel). Les attaques L-inf bornees demandent toujours PGD-AT.

Transformations disponibles
---------------------------
- rotation   : petite rotation aleatoire (bilineaire, fond noir)
- translation: decalage aleatoire de quelques pixels
- zoom        : leger agrandissement / retrecissement
- bruit       : pixels parasites ajoutes LA OU IL N'Y EN A PAS (fond noir)
                + quelques pixels retires (sel et poivre)
- cutout      : petit carre efface (force le modele a ne pas dependre d'une zone)
- epaississement : dilatation legere du trait (variation de style d'ecriture)

Demonstration visuelle :
    python3 adversarial/scripts/augment.py            -> results/augmentation_samples.png
    python3 adversarial/scripts/augment.py --n 12 --force
"""

import argparse
import sys
from os.path import abspath, dirname, join

import numpy as np

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))


# --------------------------------------------------------------------------
#  Configuration par defaut
# --------------------------------------------------------------------------

CONFIG_DEFAUT = {
    "rotation": 12.0,      # degres max (0 = desactive)
    "translation": 2,      # pixels max
    "zoom": 0.10,          # +/- 10 % d'echelle
    "bruit_p": 0.02,       # part de pixels parasites ajoutes
    "bruit_intensite": (0.5, 1.0),
    "cutout": 6,           # cote max du carre efface (0 = desactive)
    "epaisseur": 0.3,      # probabilite de varier l'epaisseur du trait (0 = off)
    "epaisseur_melange": 0.6,  # intensite de l'effet (0 = rien, 1 = plein)
    "prob": 0.5,           # probabilite d'appliquer chaque transformation
}


# --------------------------------------------------------------------------
#  Outils de sampling
# --------------------------------------------------------------------------

def _bilineaire(image, u, v):
    """Echantillonne `image` (h, w) aux coordonnees flottantes u, v.

    Hors des bornes -> 0 (le fond de MNIST est noir, ca evite les bords blancs
    qu'on obtiendrait avec un clip).
    """
    h, w = image.shape
    valide = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    du = u - u0
    dv = v - v0
    out = ((1 - du) * (1 - dv) * image[v0, u0]
           + du * (1 - dv) * image[v0, u1]
           + (1 - du) * dv * image[v1, u0]
           + du * dv * image[v1, u1])
    return np.where(valide, out, 0.0)


def _transforme_affine(x, matrices, rng_seed=None):
    """Applique une matrice 2x2 (rotation + zoom) a chaque image du batch.

    x : (N, C, H, W) normalise [0, 1].
    matrices : tableau (N, 2, 2).
    """
    n, c, h, w = x.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))  # jj = colonnes, ii = lignes
    jjc, iic = jj - cx, ii - cy
    out = np.empty_like(x)
    for k in range(n):
        M = matrices[k]
        u = M[0, 0] * jjc + M[0, 1] * iic + cx
        v = M[1, 0] * jjc + M[1, 1] * iic + cy
        for ch in range(c):
            out[k, ch] = _bilineaire(x[k, ch], u, v)
    return out


# --------------------------------------------------------------------------
#  Transformations
# --------------------------------------------------------------------------

def rotation(x, degre_max, rng):
    n = x.shape[0]
    a = np.deg2rad(rng.uniform(-degre_max, degre_max, size=n))
    cos, sin = np.cos(a), np.sin(a)
    M = np.zeros((n, 2, 2))
    M[:, 0, 0], M[:, 0, 1] = cos, sin
    M[:, 1, 0], M[:, 1, 1] = -sin, cos
    return _transforme_affine(x, M)


def zoom_et_rotation(x, degre_max, zoom_max, rng):
    """Rotation + echelle en un seul passage (plus economique)."""
    n = x.shape[0]
    a = np.deg2rad(rng.uniform(-degre_max, degre_max, size=n))
    s = 1.0 + rng.uniform(-zoom_max, zoom_max, size=n)
    cos, sin = np.cos(a) * s, np.sin(a) * s
    M = np.zeros((n, 2, 2))
    M[:, 0, 0], M[:, 0, 1] = cos, sin
    M[:, 1, 0], M[:, 1, 1] = -sin, cos
    return _transforme_affine(x, M)


def translation(x, decalage_max, rng):
    """Decalage entier : plus rapide et sans interpolation (aucun flou ajoute)."""
    n, c, h, w = x.shape
    out = np.zeros_like(x)
    dy = rng.randint(-decalage_max, decalage_max + 1, size=n)
    dx = rng.randint(-decalage_max, decalage_max + 1, size=n)
    for k in range(n):
        sy0, sy1 = max(dy[k], 0), h + min(dy[k], 0)
        sx0, sx1 = max(dx[k], 0), w + min(dx[k], 0)
        out[k, :, sy0:sy1, sx0:sx1] = x[k, :, sy0 - dy[k]:sy1 - dy[k],
                                          sx0 - dx[k]:sx1 - dx[k]]
    return out


def bruit_impulsionnel(x, p, intensite, rng):
    """Sel et poivre.

    - "sel" : des pixels allumes la ou il n'y en a pas (fond noir -> valeur haute).
      C'est exactement l'idee "des ecarts pas presents de base".
    - "poivre" : quelques pixels du trait eteints.

    C'est plus proche d'une vraie perturbation d'ecriture (tache d'encre,
    scanner bruite) qu'un bruit gaussien uniforme.
    """
    n, c, h, w = x.shape
    out = x.copy()
    n_sel = int(round(p * h * w))
    n_poivre = int(round(p * h * w * 0.5))
    for k in range(n):
        img = out[k, 0]
        fond = img < 0.2                      # pixels de fond
        for _ in range(n_sel):
            i, j = rng.randint(h), rng.randint(w)
            if fond[i, j]:
                img[i, j] = rng.uniform(*intensite)
        for _ in range(n_poivre):
            i, j = rng.randint(h), rng.randint(w)
            if img[i, j] > 0.2:               # pixel du trait
                img[i, j] = 0.0
    return np.clip(out, 0.0, 1.0)


def cutout(x, taille_max, rng):
    """Efface un carre aleatoire (Regularization, DeVries & Taylor 2017)."""
    n, c, h, w = x.shape
    out = x.copy()
    for k in range(n):
        t = rng.randint(taille_max // 2, taille_max + 1)
        y0, x0 = rng.randint(h - t), rng.randint(w - t)
        out[k, :, y0:y0 + t, x0:x0 + t] = 0.0
    return out


def varier_epaisseur(x, rng, melange=0.6):
    """Epaissit OU amincit le trait (variation de style d'ecriture).

    Element structurant en croix (4 voisins) et non un carre 3x3 : une
    dilatation 3x3 double quasiment l'encre (mesure : x1.80), ce qui detruit
    la forme du chiffre. Avec la croix et `melange`, l'effet est progressif.

    Une image sur deux est epaissie, l'autre amincie.
    melange=1.0 -> effet plein ; melange=0.0 -> aucun effet.
    """
    n, c, h, w = x.shape
    pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    out = x.copy()
    for k in range(n):
        voisins = [pad[k, :, 0:h, 1:w + 1], pad[k, :, 2:h + 2, 1:w + 1],
                   pad[k, :, 1:h + 1, 0:w], pad[k, :, 1:h + 1, 2:w + 2]]
        if rng.rand() < 0.5:                       # epaissir
            extreme = np.maximum.reduce([x[k]] + voisins)
            out[k] = x[k] + melange * (extreme - x[k])
        else:                                      # amincir
            extreme = np.minimum.reduce([x[k]] + voisins)
            out[k] = x[k] - melange * (x[k] - extreme)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------
#  Pipeline
# --------------------------------------------------------------------------

def augmenter(x, rng=None, config=None):
    """Applique un pipeline aleatoire a un batch (N, C, H, W) normalise [0, 1].

    Chaque transformation est tiree independamment avec la probabilite `prob`.
    Entree et sortie ont la meme forme (N, C, H, W).
    """
    rng = rng or np.random
    cfg = dict(CONFIG_DEFAUT)
    if config:
        cfg.update(config)

    if x.ndim == 3:
        x = x[:, None, :, :]
        mono = True
    else:
        mono = False

    if cfg["rotation"] or cfg["zoom"]:
        x = zoom_et_rotation(x, cfg["rotation"], cfg["zoom"], rng)
    if cfg["translation"] and rng.rand() < cfg["prob"]:
        x = translation(x, cfg["translation"], rng)
    if cfg["epaisseur"] and rng.rand() < cfg["epaisseur"]:
        x = varier_epaisseur(x, rng, cfg["epaisseur_melange"])
    if cfg["bruit_p"] and rng.rand() < cfg["prob"]:
        x = bruit_impulsionnel(x, cfg["bruit_p"], cfg["bruit_intensite"], rng)
    if cfg["cutout"] and rng.rand() < cfg["prob"] * 0.5:
        x = cutout(x, cfg["cutout"], rng)

    return x[:, 0] if mono else x


def jeu_augmente(x, n_copies=1, rng=None, config=None):
    """Genere `n_copies` versions augmentees d'un jeu complet (mode hors ligne).

    Utile pour inspecter a l'oeil, ou pour sauvegarder un jeu enrichi sur disque
    si on veut entrainer sans refaire l'augmentation a chaque epoch.
    """
    rng = rng or np.random.RandomState(0)
    lots = [x]
    for _ in range(n_copies):
        lots.append(augmenter(x.copy(), rng=rng, config=config))
    return np.concatenate(lots, axis=0)


# --------------------------------------------------------------------------
#  Demonstration visuelle
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Augmentation de donnees : demo visuelle")
    p.add_argument("--n", type=int, default=10, help="nombre d'images a montrer")
    p.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="applique tout sans probabilite")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data import MNISTLoader, normalize, add_channel_dim

    loader = MNISTLoader()
    (x_all, y_all), _ = loader.load(join(ROOT_DIR, "data"))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(x_all), size=args.n, replace=False)
    x = normalize(add_channel_dim(x_all[idx])).transpose(0, 3, 1, 2)
    y = y_all[idx]

    cfg = {"prob": 1.0} if args.force else {}
    xa = augmenter(x.copy(), rng=rng, config=cfg)

    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 1.4, 3.4))
    for k in range(args.n):
        axes[0, k].imshow(x[k, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(str(y[k]), fontsize=9)
        axes[1, k].imshow(xa[k, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle("Augmentation aleatoire : original (haut) vs augmente (bas)")
    fig.tight_layout()

    out_dir = join(ROOT_DIR, "adversarial", "results")
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = join(out_dir, "augmentation_samples.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[PLOT] {out}")

    # Statistiques : montrer que le signal reste exploitable
    print(f"\nDensite de trait (part de pixels allumes) : "
          f"{float((x > 0.2).mean()):.3f} -> {float((xa > 0.2).mean()):.3f}")
    print(f"Intensite moyenne                         : "
          f"{float(x.mean()):.4f} -> {float(xa.mean()):.4f}")
    print(f"Ecart moyen par pixel                     : "
          f"{float(np.abs(xa - x).mean()):.4f}")
    print("\n(rappel : ce n'est PAS une perturbation adversarial. On deplace et "
          "on ajoute de l'encre,\n pas un bruit borne en norme L-inf sur les pixels.)")


if __name__ == "__main__":
    main()
