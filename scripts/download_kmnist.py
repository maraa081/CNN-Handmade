#!/usr/bin/env python3
"""
download_kmnist.py -- Prepare les donnees KMNIST (Kuzushiji-MNIST)

Kuzushiji-MNIST : 70000 images de kana japonais manuscrits (hiragana),
28x28, 10 classes. Prefixe "KMNIST" ci-dessous par symetrie avec EMNIST :
c'est la meme famille de taches (classification de caracteres manuscrits).

Distribution officielle CODH (universite de Tohoku), en 4 fichiers .npz :
    kmnist-train-imgs.npz    (60000, 28, 28) uint8
    kmnist-train-labels.npz  (60000,)        uint8
    kmnist-test-imgs.npz     (10000, 28, 28) uint8
    kmnist-test-labels.npz   (10000,)        uint8

Marche sur Windows / macOS / Linux. Utilise numpy pour verifier la forme
et les plages de valeurs des tableaux telecharges.

Priorite de recuperation :
  1. Si data/kmnist/kmnist-*.npz existent deja et sont valides -> rien a faire
  2. Sinon -> telechargement depuis le site CODH

Usage :
    python3 download_kmnist.py
"""

import os
import sys
import urllib.request
from os.path import join, dirname, abspath, exists, basename

import numpy as np

ROOT = dirname(dirname(abspath(__file__)))  # -> CNN-Handmade/
DATA_DIR = join(ROOT, "data", "kmnist")
BASE_URL = "https://codh.rois.ac.jp/kmnist/dataset/kmnist/"

# (nom du fichier officiel, forme attendue)
FILES = [
    ("kmnist-train-imgs.npz", (60000, 28, 28)),
    ("kmnist-train-labels.npz", (60000,)),
    ("kmnist-test-imgs.npz", (10000, 28, 28)),
    ("kmnist-test-labels.npz", (10000,)),
]

# Plages de valeurs attendues selon le type de fichier
IMG_RANGE = (0, 255)
LABEL_RANGE = (0, 9)

# Classes officielles, dans l'ordre des labels 0-9
CLASSES = ("o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo")


def check_file(path, expected_shape, verbose=False):
    """
    Verifie qu'un fichier .npz existe, contient un seul tableau, a la forme
    attendue et des valeurs dans les plages attendues.

    Les 4 fichiers officiels stockent chacun un unique tableau sous la cle
    'arr_0'. Un fichier telecharge a moitie (ou tronque) echoue ici, ce qui
    evite de repartir avec des donnees corrompues.

    Retourne :
        bool : True si le fichier est present et valide
    """
    if not exists(path):
        return False

    name = basename(path)
    is_image = "imgs" in name
    lo, hi = IMG_RANGE if is_image else LABEL_RANGE

    try:
        with np.load(path) as data:
            keys = list(data.keys())
            if len(keys) != 1:
                if verbose:
                    print(f"  ERREUR : {name} contient {len(keys)} tableaux (attendu 1)")
                return False
            array = data[keys[0]]
            shape = array.shape
            vmin, vmax = int(array.min()), int(array.max())
            ndim = array.ndim
            dtype = array.dtype
    except Exception as e:
        if verbose:
            print(f"  ERREUR : {name} illisible ({e})")
        return False

    if shape != expected_shape:
        if verbose:
            print(f"  ERREUR : {name} a la forme {shape} (attendu {expected_shape})")
        return False

    if vmin < lo or vmax > hi:
        if verbose:
            print(f"  ERREUR : {name} a des valeurs hors de [{lo}, {hi}] : min={vmin} max={vmax}")
        return False

    if verbose:
        print(f"  [OK] {name}  shape={shape}  dtype={dtype}  ndim={ndim}  valeurs {vmin}-{vmax}")
    return True


def already_downloaded(verbose=False):
    """True si les 4 fichiers sont presents et valides."""
    return all(check_file(join(DATA_DIR, name), shape, verbose) for name, shape in FILES)


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, done * 100 // total_size)
        sys.stdout.write(f"\r  Telechargement : {pct}% ({done // 1024} Ko / {total_size // 1024} Ko)")
        sys.stdout.flush()


def download_file(name, expected_shape):
    """
    Telecharge un fichier .npz depuis CODH, le valide, puis le met en place.

    On ecrit d'abord dans un fichier temporaire '.part' et on ne renomme
    qu'apres validation : un telechargement interrompu ne laisse jamais
    un .npz invalide a la place du fichier final.
    """
    url = BASE_URL + name
    dest = join(DATA_DIR, name)
    tmp = dest + ".part"

    print(f"  -> {name} ...")
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
    except Exception as e:
        print(f"\n  ERREUR : {e}")
        print(f"  URL tentee : {url}")
        if exists(tmp):
            os.remove(tmp)
        return False
    print()

    if not check_file(tmp, expected_shape, verbose=True):
        print(f"  ERREUR : {name} telecharge mais invalide - fichier rejete")
        os.remove(tmp)
        return False

    os.replace(tmp, dest)
    return True


def download_all():
    """Telecharge les 4 fichiers dans data/kmnist/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[1/2] Telechargement de KMNIST depuis CODH (~21 Mo au total)...")
    print(f"      source : {BASE_URL}")

    for name, expected_shape in FILES:
        if not download_file(name, expected_shape):
            print("  Astuce : le jeu de donnees est tres diffuse (miroirs GitHub,")
            print("           Kaggle). Place a la main les 4 .npz dans :")
            print(f"           {DATA_DIR}")
            return False
    return True


def print_summary():
    """Affiche un resume des donnees chargees : formes, labels, repartition."""
    print("\n[2/2] Resume des donnees KMNIST :")
    arrays = {}
    for name, expected_shape in FILES:
        with np.load(join(DATA_DIR, name)) as data:
            arrays[name] = data[list(data.keys())[0]]

    x_train = arrays["kmnist-train-imgs.npz"]
    y_train = arrays["kmnist-train-labels.npz"]
    x_test = arrays["kmnist-test-imgs.npz"]
    y_test = arrays["kmnist-test-labels.npz"]

    print(f"  x_train : {x_train.shape}  {x_train.dtype}  valeurs {x_train.min()}-{x_train.max()}")
    print(f"  y_train : {y_train.shape}  {y_train.dtype}  labels {y_train.min()}-{y_train.max()}")
    print(f"  x_test  : {x_test.shape}   {x_test.dtype}  valeurs {x_test.min()}-{x_test.max()}")
    print(f"  y_test  : {y_test.shape}   {y_test.dtype}  labels {y_test.min()}-{y_test.max()}")

    print(f"  10 classes : {' '.join(CLASSES)}")

    counts_train = np.bincount(y_train.astype(np.int64), minlength=len(CLASSES))
    counts_test = np.bincount(y_test.astype(np.int64), minlength=len(CLASSES))
    print(f"  Repartition train : {counts_train.min()}-{counts_train.max()} images par classe")
    print(f"  Repartition test  : {counts_test.min()}-{counts_test.max()} images par classe")


def ensure_data(verbose=True):
    """Appele par train_kmnist.py : garantit que les donnees existent."""
    if already_downloaded(verbose=verbose):
        if verbose:
            print("[OK] Donnees KMNIST deja presentes dans data/kmnist/")
        return True

    ok = download_all()
    if ok and already_downloaded(verbose=False):
        if verbose:
            print("[OK] Donnees KMNIST pretes !")
        return True

    print("\nERREUR : impossible de preparer les donnees.")
    return False


if __name__ == "__main__":
    ok = ensure_data()
    if ok:
        print_summary()
    sys.exit(0 if ok else 1)
