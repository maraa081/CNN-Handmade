#!/usr/bin/env python3
"""
train_kmnist.py -- Entrainement du CNN sur KMNIST (kana japonais manuscrits)

Meme reseau from-scratch que MNIST / EMNIST, applique aux 10 classes de
Kuzushiji-MNIST (hiragana : o, ki, su, tsu, na, ha, ma, ya, re, wo).

Usage :
    python3 train_kmnist.py                              # Rapide : 5000 images, 3 epochs
    python3 train_kmnist.py --limit 1000 --epochs 1      # Encore plus rapide
    python3 train_kmnist.py --full                       # Complet : 60000 images
    python3 train_kmnist.py --full --epochs 15
    python3 train_kmnist.py --samples                    # Affiche des kana KMNIST

Donnees : data/kmnist/kmnist-*.npz  (telechargees depuis CODH, universite de Tohoku)
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(dirname(abspath(__file__)))  # -> CNN-Handmade/
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import KMNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN
from download_kmnist import ensure_data

NUM_CLASSES = 10

# Classes officielles KMNIST, dans l'ordre des labels 0-9
CLASSES = ("o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo")


def build_model():
    """Meme architecture que MNIST / EMNIST, sortie 10 (10 kana)."""
    model = CNN()
    model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(3136, 128))
    model.add(ReLU())
    model.add(Dense(128, NUM_CLASSES))
    return model


def show_samples(x, y, n=16, title="Kana KMNIST", out="kmnist_samples.png"):
    """Affiche n kana avec leur classe pour verifier l'orientation des images."""
    n_cols, n_rows = 4, n // 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
    indices = np.random.choice(len(x), size=n, replace=False)
    for i, idx in enumerate(indices):
        ax = axes[i // n_cols][i % n_cols]
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"'{CLASSES[y[idx]]}'", fontsize=12)
        ax.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = join(ROOT_DIR, out)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" Echantillons -> {path}")


def get_option(name, default=None):
    """Lit une option de la forme '--name valeur' sur la ligne de commande."""
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return default


if __name__ == "__main__":
    data_dir = join(ROOT_DIR, "data", "kmnist")

    # -- S'assurer que les donnees existent (telecharge si besoin) --
    if not ensure_data():
        sys.exit(1)

    # -- Mode echantillons : juste verifier l'orientation --
    if "--samples" in sys.argv:
        loader = KMNISTLoader()
        (x_train, y_train), _ = loader.load(data_dir)
        print(f"[OK] KMNIST charge : {x_train.shape}, labels {y_train.min()}-{y_train.max()}")
        show_samples(x_train, y_train, title="KMNIST - kana verifies")
        sys.exit(0)

    # -- Chargement --
    print(" Chargement KMNIST...")
    loader = KMNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load(data_dir)

    print(f"[OK] KMNIST charge")
    print(f"  x_train : {x_train.shape}  ({x_train.dtype})")
    print(f"  y_train : {y_train.shape}  labels {y_train.min()}-{y_train.max()} ({NUM_CLASSES} classes)")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")

    # Verif repartition des classes
    counts = np.bincount(y_train.astype(np.int64), minlength=NUM_CLASSES)
    print(f"  Classes equilibrees : {counts.min()}-{counts.max()} par classe")

    show_samples(x_train, y_train, title="KMNIST (train)")

    # -- Flags --
    full = "--full" in sys.argv
    epochs_override = get_option("--epochs")
    limit_override = get_option("--limit")

    # -- Modele --
    model = build_model()
    print(f"\n {model}")

    if full:
        print(f"\n{'=' * 50}")
        print(f"  ENTRAINEMENT COMPLET  ({len(x_train)} images, {NUM_CLASSES} classes)")
        print(f"{'=' * 50}")

        batch_size = 128
        n_epochs = epochs_override or 10

        train_loader = preprocess_pipeline(x_train, y_train, batch_size=batch_size,
                                           shuffle=True, num_classes=NUM_CLASSES)
        test_loader = preprocess_pipeline(x_test, y_test, batch_size=batch_size,
                                          shuffle=False, num_classes=NUM_CLASSES)

        print(f"\n {len(x_train)} train / {len(x_test)} test")
        print(f"   batch={batch_size}, epochs={n_epochs}, lr=0.01")
        print(f"   ~{len(train_loader)} batches/epoch")

        t_start = time.time()
        history = model.train(train_loader, epochs=n_epochs, lr=0.01, verbose=True)
        t_elapsed = time.time() - t_start

        h, m, s = int(t_elapsed // 3600), int((t_elapsed % 3600) // 60), int(t_elapsed % 60)
        print(f"\n[time]  {h}h {m}m {s}s" if h else f"[time]  {m}m {s}s")

        print(f"\n Evaluation sur {len(x_test)} images de test...")
        test_acc = model.evaluate(test_loader)
        print(f"\n Accuracy test : {test_acc:.4f}  ({test_acc * 100:.1f}%)")

        save_path = join(ROOT_DIR, "models", "kmnist_weights_full.npz")
        model.save_weights(save_path)

    else:
        n_sub = limit_override or 5000
        n_sub = min(n_sub, len(x_train))

        print(f"\n{'=' * 50}")
        print(f"  Entrainement rapide (sous-ensemble {n_sub} images)")
        print(f"{'=' * 50}")

        x_sub, y_sub = x_train[:n_sub], y_train[:n_sub]
        n_epochs = epochs_override or 3
        batch_size = 64

        # KMNIST est deja melange (contrairement a EMNIST, trie par classe) :
        # un simple prefixe contient donc toutes les classes. On tire quand
        # meme le test au hasard pour garder une evaluation equilibree.
        rng = np.random.RandomState(42)
        test_idx = rng.choice(len(x_test), size=1000, replace=False)

        train_sub = preprocess_pipeline(x_sub, y_sub, batch_size=batch_size,
                                        shuffle=True, num_classes=NUM_CLASSES)
        test_sub = preprocess_pipeline(x_test[test_idx], y_test[test_idx], batch_size=batch_size,
                                       shuffle=False, num_classes=NUM_CLASSES)

        print(f"\n {x_sub.shape[0]} train, batch={batch_size}, epochs={n_epochs}, lr=0.01")

        t_start = time.time()
        history = model.train(train_sub, epochs=n_epochs, lr=0.01, verbose=True)
        t_elapsed = time.time() - t_start

        m, s = divmod(t_elapsed, 60)
        print(f"[time]  {int(m)}m {int(s)}s")

        test_acc = model.evaluate(test_sub)
        print(f"\n Accuracy test (1000 images) : {test_acc:.4f}  ({test_acc * 100:.1f}%)")

        save_path = join(ROOT_DIR, "models", "kmnist_weights.npz")
        model.save_weights(save_path)

    # -- Graphique --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"], marker="o", linewidth=2, markersize=6)
    ax1.set_title(f"Loss ({'full' if full else 'rapide'}, {len(history['loss'])} epochs)", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(history["accuracy"], marker="s", linewidth=2, markersize=6, color="green")
    ax2.set_title(f"Accuracy train\nTest final : {test_acc:.1%}", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    graph_path = join(ROOT_DIR, "kmnist_training_result.png")
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    print(f" Graphique -> {graph_path}")

    # -- Predictions d'exemple --
    print(f"\n Predictions d'exemple :")
    from data import normalize, add_channel_dim
    np.random.seed(0)
    idxs = np.random.choice(len(x_test), size=10, replace=False)
    correct = 0
    for idx in idxs:
        img = x_test[idx]
        x_proc = normalize(img)
        x_proc = add_channel_dim(x_proc)       # (28, 28, 1)
        x_proc = x_proc.transpose(2, 0, 1)     # (1, 28, 28)
        pred = model.predict(x_proc)
        ok = pred == y_test[idx]
        correct += ok
        print(f"  Vrai: '{CLASSES[y_test[idx]]}'  ->  Predit: '{CLASSES[pred]}'  {'OK' if ok else 'FAIL'}")
    print(f"\n {correct}/10 correctes")
