#!/usr/bin/env python3
"""
train_emnist.py — Entraînement du CNN sur EMNIST Letters (a-z)

Même réseau from-scratch que MNIST, mais avec 26 classes au lieu de 10.

Usage :
    python3 train_emnist.py                # Rapide : 5000 images, 3 epochs
    python3 train_emnist.py --full         # Complet : 124800 images
    python3 train_emnist.py --full --epochs 15
    python3 train_emnist.py --samples      # Affiche des lettres EMNIST (vérif orientation)

Données : data/emnist/emnist-letters-*.idx*  (téléchargées depuis le site NIST)
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import EMNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN
from download_emnist import ensure_data

NUM_CLASSES = 26
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def build_model():
    """Même architecture que MNIST, sortie 26 au lieu de 10."""
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


def show_samples(x, y, n=16, title="Lettres EMNIST", out="emnist_samples.png"):
    """Affiche n lettres avec leur label pour vérifier l'orientation."""
    n_cols, n_rows = 4, n // 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
    indices = np.random.choice(len(x), size=n, replace=False)
    for i, idx in enumerate(indices):
        ax = axes[i // n_cols][i % n_cols]
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"'{LETTERS[y[idx]]}'", fontsize=12)
        ax.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = join(ROOT_DIR, out)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" Échantillons -> {path}")


if __name__ == "__main__":
    data_dir = join(ROOT_DIR, "data", "emnist")

    # -- S'assurer que les données existent (télécharge si besoin) --
    if not ensure_data():
        sys.exit(1)

    # -- Mode échantillons : juste vérifier l'orientation --
    if "--samples" in sys.argv:
        loader = EMNISTLoader("letters")
        (x_train, y_train), _ = loader.load(data_dir)
        print(f"[OK] EMNIST letters chargé : {x_train.shape}, labels {y_train.min()}–{y_train.max()}")
        show_samples(x_train, y_train, title="EMNIST Letters — orientation vérifiée")
        sys.exit(0)

    # -- Chargement --
    print(" Chargement EMNIST letters…")
    loader = EMNISTLoader("letters")
    (x_train, y_train), (x_test, y_test) = loader.load(data_dir)

    print(f"[OK] EMNIST letters chargé")
    print(f"  x_train : {x_train.shape}  ({x_train.dtype})")
    print(f"  y_train : {y_train.shape}  labels {y_train.min()}–{y_train.max()} ({NUM_CLASSES} classes)")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")

    # Vérif répartition des classes
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    print(f"  Classes équilibrées : {counts.min()}–{counts.max()} par classe")

    show_samples(x_train, y_train, title="EMNIST Letters (train)")

    # -- Flags --
    full = "--full" in sys.argv
    epochs_override = None
    for i, arg in enumerate(sys.argv):
        if arg == "--epochs" and i + 1 < len(sys.argv):
            epochs_override = int(sys.argv[i + 1])

    # -- Modèle --
    model = build_model()
    print(f"\n {model}")

    if full:
        print(f"\n{'=' * 50}")
        print(f"  ENTRAÎNEMENT COMPLET  ({len(x_train)} images, 26 classes)")
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

        print(f"\n Évaluation sur {len(x_test)} images de test…")
        test_acc = model.evaluate(test_loader)
        print(f"\n Accuracy test : {test_acc:.4f}  ({test_acc * 100:.1f}%)")

        save_path = join(ROOT_DIR, "models", "emnist_letters_weights_full.npz")
        model.save_weights(save_path)

    else:
        print(f"\n{'=' * 50}")
        print(f"  Entraînement rapide (sous-ensemble 5000 images)")
        print(f"{'=' * 50}")

        x_sub, y_sub = x_train[:5000], y_train[:5000]
        n_epochs = epochs_override or 3
        batch_size = 64

        # [warn] EMNIST est trié par classe : il faut échantillonner le test
        # aléatoirement, sinon on évalue sur 1-2 classes seulement.
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

        save_path = join(ROOT_DIR, "models", "emnist_letters_weights.npz")
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
    graph_path = join(ROOT_DIR, "emnist_training_result.png")
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    print(f" Graphique -> {graph_path}")

    # -- Prédictions d'exemple --
    print(f"\n Prédictions d'exemple :")
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
        print(f"  Vrai: '{LETTERS[y_test[idx]]}'  ->  Prédit: '{LETTERS[pred]}'  {'OK' if ok else 'FAIL'}")
    print(f"\n {correct}/10 correctes")
