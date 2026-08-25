#!/usr/bin/env python3
"""
--------------------------------------------------------------------------
   COMPARE ALL — Comparaison de tous les optimiseurs / techniques
--------------------------------------------------------------------------

  Lance TOUS les optimiseurs et techniques à la suite, puis génère
  un graphique comparatif unique. Parfait pour voir d'un coup d'œil
  ce qui marche le mieux.

  Lancement : python3 experiments/compare_all.py

  Résultats :
    results/comparison.png        — graphique combiné (loss + accuracy)
    results/comparison_results.csv — tableau des résultats (optionnel, version push)
--------------------------------------------------------------------------
"""

import sys, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(ROOT, "src"))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense, Dropout
from model import CNN
from optimizers import SGD, Momentum, Adam


# +==========================================================================+
# |   RÉGLAGES COMMUNS                                                    |
# +==========================================================================+

BATCH_SIZE = 64
EPOCHS     = 5
DATA_LIMIT = 2000

# -- Configuration des expériences --
#   Chaque entrée = une expérience. Ajoute ou modifie ici.
#   Format : (nom, optimiseur, ajouter_dropout)
EXPERIMENTS = [
    ("Baseline SGD",    SGD(lr=0.01),                        False),
    ("Momentum",        Momentum(lr=0.01, momentum=0.9),      False),
    ("Adam",            Adam(lr=0.001),                       False),
    ("Dropout (SGD)",   SGD(lr=0.01),                         True),
    ("L2 Weight Decay", SGD(lr=0.01, weight_decay=0.001),     False),
]

# ===========================================================================

OUT_DIR = dirname(abspath(__file__))
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def build_model(optimizer, use_dropout=False):
    """Construit l'architecture standard, avec Dropout optionnel."""
    model = CNN(optimizer=optimizer)
    model.add(Conv2D(1, 32, kernel_size=3, stride=1, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, 64, kernel_size=3, stride=1, pad=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(3136, 128))
    model.add(ReLU())
    if use_dropout:
        model.add(Dropout(p=0.5))
    model.add(Dense(128, 10))
    return model


def get_data():
    """Charge et prépare les données (une seule fois)."""
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load(join(ROOT, "data"))
    if DATA_LIMIT:
        x_train = x_train[:DATA_LIMIT]
        y_train = y_train[:DATA_LIMIT]
    train_loader = preprocess_pipeline(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = preprocess_pipeline(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, x_train.shape[0], x_test.shape[0]


# ===========================================================================

def main():
    train_loader, test_loader, n_train, n_test = get_data()
    print(f" {n_train} train / {n_test} test — Batch {BATCH_SIZE} — {EPOCHS} epochs\n")

    results = []  # [(name, history, test_acc, elapsed, opt_str)]

    for idx, (name, optimizer, use_dropout) in enumerate(EXPERIMENTS):
        opt_str = str(optimizer)
        dropout_str = " + Dropout" if use_dropout else ""
        label = f"{name}{dropout_str}"
        print(f"{'-' * 55}")
        print(f"   {idx + 1}/{len(EXPERIMENTS)}  {label}")
        print(f"     {opt_str}")
        print(f"{'-' * 55}")

        model = build_model(optimizer, use_dropout)

        t0 = time.time()
        history = model.train(train_loader, epochs=EPOCHS, verbose=True)
        elapsed = time.time() - t0

        acc = model.evaluate(test_loader)
        m, s = divmod(elapsed, 60)
        print(f"     [time]  {int(m)}m {int(s)}s  -   Test accuracy: {acc:.4f} ({acc*100:.1f}%)\n")

        results.append((name, optimizer, opt_str, use_dropout, history, acc, elapsed))

    # -- Graphique comparatif --
    save_comparison_plot(results)

    # -- Tableau récapitulatif --
    save_summary(results)

    # -- Synthèse --
    print_summary(results)


# ===========================================================================

def save_comparison_plot(results):
    """Génère le graphique avec toutes les courbes superposées."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, _, _, use_dropout, history, test_acc, _) in enumerate(results):
        color = COLORS[idx % len(COLORS)]
        dropout_label = " + Dropout" if use_dropout else ""
        label = f"{name}{dropout_label}"

        epochs = range(1, len(history["loss"]) + 1)

        ax1.plot(epochs, history["loss"], marker="o", linewidth=2,
                 markersize=6, color=color, label=label)

        ax2.plot(epochs, history["accuracy"], marker="s", linewidth=2,
                 markersize=6, color=color,
                 label=f"{label}  ({test_acc*100:.1f}%)")

    ax1.set_title("Loss (entraînement)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Accuracy (test final entre parenthèses)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.suptitle(f"Comparaison des optimiseurs — {EPOCHS} epochs, batch={BATCH_SIZE}, data={DATA_LIMIT}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = join(ROOT, "results", "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n Graphique comparatif -> {path}")


def save_summary(results):
    """Sauvegarde un CSV récapitulatif."""
    lines = [
        "technique,optimizer,epochs,batch_size,data_limit,"
        "test_accuracy,training_time_s,final_train_loss,final_train_accuracy"
    ]

    for name, optimizer, opt_str, use_dropout, history, test_acc, elapsed in results:
        dropout_tag = "_dropout" if use_dropout else ""
        tech_name = f"{name.lower().replace(' ', '_')}{dropout_tag}"
        final_loss = history["loss"][-1]
        final_acc = history["accuracy"][-1]
        lines.append(
            f"{tech_name},\"{opt_str}\",{EPOCHS},{BATCH_SIZE},{DATA_LIMIT},"
            f"{test_acc:.6f},{elapsed:.1f},{final_loss:.6f},{final_acc:.6f}"
        )

    csv_path = join(ROOT, "results", "comparison_results.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f" Résultats CSV -> {csv_path}")


def print_summary(results):
    """Affiche un tableau récapitulatif dans le terminal."""
    print(f"\n{'=' * 70}")
    print(f"   RÉCAPITULATIF")
    print(f"{'=' * 70}")
    print(f"  {'Technique':<22} {'Optimiseur':<30} {'Test Acc':>8} {'Temps':>8}")
    print(f"  {'-' * 22} {'-' * 30} {'-' * 8} {'-' * 8}")

    # Trier par accuracy décroissante
    sorted_results = sorted(results, key=lambda r: r[5], reverse=True)
    for name, _, opt_str, use_dropout, history, test_acc, elapsed in sorted_results:
        dropout_label = " + Dropout" if use_dropout else ""
        label = f"{name}{dropout_label}"
        m, s = divmod(elapsed, 60)
        time_str = f"{int(m)}m{int(s)}s" if m > 0 else f"{int(s)}s"
        print(f"  {label:<22} {opt_str:<30} {test_acc*100:>7.1f}% {time_str:>8}")

    print(f"{'=' * 70}")


# ===========================================================================

if __name__ == "__main__":
    main()
