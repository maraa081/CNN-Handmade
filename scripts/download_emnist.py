#!/usr/bin/env python3
"""
download_emnist.py — Prépare les données EMNIST Letters

Marche sur Windows / macOS / Linux. Utilise uniquement la bibliothèque
standard Python (pas besoin de numpy).

Priorité de récupération :
  1. Si data/emnist/emnist-letters-*.idx* existent déjà  -> rien à faire
  2. Sinon, si data/emnist_letters.zip est présent        -> extraction locale
  3. Sinon -> téléchargement depuis le site NIST (gzip.zip, ~561 Mo, plus lent)

Usage :
    python3 download_emnist.py
"""

import os
import sys
import gzip
import shutil
import zipfile
import urllib.request
from os.path import join, dirname, abspath, exists

ROOT = dirname(dirname(abspath(__file__)))  # -> CNN-Handmade/
DATA_DIR = join(ROOT, "data", "emnist")
LOCAL_ZIP = join(ROOT, "data", "emnist_letters.zip")
NIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
NIST_ZIP_PATH = join(DATA_DIR, "gzip.zip")

FILES = [
    "emnist-letters-train-images-idx3-ubyte",
    "emnist-letters-train-labels-idx1-ubyte",
    "emnist-letters-test-images-idx3-ubyte",
    "emnist-letters-test-labels-idx1-ubyte",
]


def already_downloaded():
    return all(exists(join(DATA_DIR, f)) for f in FILES)


def extract_gz(gz_path, out_path):
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def extract_from_local_zip():
    print("[1/2] Extraction depuis data/emnist_letters.zip ...")
    with zipfile.ZipFile(LOCAL_ZIP) as z:
        names = z.namelist()
        for f in FILES:
            gz_name = f + ".gz"
            member = next((n for n in names if n.endswith(gz_name)), None)
            if member is None:
                print(f"  ERREUR : {gz_name} introuvable dans le zip")
                return False
            z.extract(member, DATA_DIR)
            extract_gz(join(DATA_DIR, member), join(DATA_DIR, f))
            print(f"  + {f}")
    return True


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, done * 100 // total_size)
        sys.stdout.write(f"\r  Telechargement : {pct}% ({done // (1024 * 1024)} Mo / {total_size // (1024 * 1024)} Mo)")
        sys.stdout.flush()


def download_from_nist():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[1/2] Telechargement du zip complet EMNIST depuis le site NIST (~561 Mo)...")
    print("      (lent, le site NIST est capricieux - patience !)")
    try:
        urllib.request.urlretrieve(NIST_ZIP_URL, NIST_ZIP_PATH, _progress)
    except Exception as e:
        print(f"\n  ERREUR : {e}")
        print("  Telecharge le zip manuellement depuis https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip")
        print(f"  puis place-le dans : {NIST_ZIP_PATH}")
        return False
    print()

    print("[2/2] Extraction des fichiers letters...")
    with zipfile.ZipFile(NIST_ZIP_PATH) as z:
        names = z.namelist()
        for f in FILES:
            gz_name = f + ".gz"
            member = next((n for n in names if n.endswith(gz_name)), None)
            if member is None:
                print(f"  ERREUR : {gz_name} introuvable dans le zip")
                return False
            z.extract(member, DATA_DIR)
            extract_gz(join(DATA_DIR, member), join(DATA_DIR, f))
            print(f"  + {f}")
    return True


def ensure_data(verbose=True):
    """Appelé par train_emnist.py : garantit que les données existent."""
    if already_downloaded():
        if verbose:
            print("[OK] Donnees EMNIST deja presentes dans data/emnist/")
        return True
    if exists(LOCAL_ZIP):
        ok = extract_from_local_zip()
    else:
        ok = download_from_nist()
    if ok and already_downloaded():
        if verbose:
            print("[OK] Donnees EMNIST pretes !")
        return True
    print("\nERREUR : impossible de preparer les donnees.")
    return False


if __name__ == "__main__":
    ok = ensure_data()
    sys.exit(0 if ok else 1)
