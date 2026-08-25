#!/usr/bin/env python3
"""
predict.py — Charge le modèle entraîné et classifie des chiffres MNIST

Usage :
    python3 predict.py                                    # Rapide (model_weights.npz)
    python3 predict.py --weights model_weights_full.npz   # Complet
    python3 predict.py --all                              # Accuracy complète (10000 test)
    python3 predict.py --interactive                      # Affiche image par image

Poids disponibles :
    model_weights.npz       → entraînement rapide (2000 images, 3 epochs)
    model_weights_full.npz  → entraînement complet  (60000 images)
"""

import sys
import numpy as np
from os.path import join, dirname, abspath, exists

ROOT_DIR = dirname(dirname(abspath(__file__)))  # → CNN-Handmade/
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN

import sys
import numpy as np
from os.path import join, dirname, abspath, exists

ROOT_DIR = dirname(dirname(abspath(__file__)))  # → CNN-Handmade/
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import MNISTLoader, preprocess_pipeline
from layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense
from model import CNN


# ── Architecture (DOIT correspondre à celle utilisée pour l'entraînement) ──

def build_model():
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
    model.add(Dense(128, 10))
    return model


# ── Chargement des données ──

def load_data():
    loader = MNISTLoader()
    (_, _), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))
    return x_test, y_test


# ── Chargement du modèle ──

def load_model(weights_path=None):
    if weights_path is None:
        # D'abord essayer le rapide, sinon le full
        quick = join(ROOT_DIR, "models", "model_weights.npz")
        full  = join(ROOT_DIR, "models", "model_weights_full.npz")
        if exists(quick):
            weights_path = quick
        elif exists(full):
            weights_path = full
        else:
            print(f"❌ Aucun fichier de poids trouvé !")
            print(f"   Lance d'abord : python3 src/cnn.py  (rapide)")
            print(f"   Ou bien :       python3 src/cnn.py --full  (complet)")
            sys.exit(1)

    if not exists(weights_path):
        print(f"❌ Fichier introuvable : {weights_path}")
        sys.exit(1)

    model = build_model()
    model.load_weights(weights_path)
    tag = "rapide" if "model_weights.npz" in weights_path else "complet"
    print(f"   ({tag})")
    return model


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse --weights option
    weights_path = None
    i = 0
    while i < len(args):
        if args[i] == "--weights" and i + 1 < len(args):
            weights_path = join(ROOT_DIR, args[i + 1])
            # Remove these from args
            args.pop(i + 1)
            args.pop(i)
        else:
            i += 1

    mode = args[0] if args else ""

    model = load_model(weights_path)
    x_test, y_test = load_data()

    if mode == "--all":
        # ── Accuracy complète ──
        from data import normalize, add_channel_dim, to_one_hot

        x_proc = normalize(x_test)
        x_proc = add_channel_dim(x_proc)
        x_proc = x_proc.transpose(0, 3, 1, 2)

        correct = 0
        total = len(x_test)
        batch_size = 128

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = x_proc[start:end]
            preds = model.predict(batch)
            correct += (preds == y_test[start:end]).sum()

        acc = correct / total
        print(f"\n🎯 Accuracy sur {total} images de test : {acc:.4f}  ({acc * 100:.1f}%)")

    elif mode == "--interactive":
        # ── Mode interactif : affiche et demande à chaque fois ──
        import matplotlib.pyplot as plt

        indices = np.random.choice(len(x_test), size=20, replace=False)
        for idx in indices:
            img = x_test[idx]
            true_label = y_test[idx]

            # Prétraitement pour le modèle
            from data import normalize, add_channel_dim
            x_proc = normalize(img)
            x_proc = add_channel_dim(x_proc)          # (28, 28, 1)
            x_proc = x_proc.transpose(2, 0, 1)        # (1, 28, 28)

            pred = model.predict(x_proc)

            plt.imshow(img, cmap="gray")
            plt.title(f"Vrai: {true_label}  →  Prédit: {pred}" +
                      (" ✅" if pred == true_label else " ❌"), fontsize=14)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            rep = input("  → Entrée pour continuer, 'q' pour quitter : ")
            if rep.lower() == "q":
                break

    else:
        # ── Mode par défaut : 10 images aléatoires, résumé ──
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from data import normalize, add_channel_dim

        np.random.seed(42)
        indices = np.random.choice(len(x_test), size=10, replace=False)
        n_cols, n_rows = 5, 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4))
        correct = 0

        for i, idx in enumerate(indices):
            img = x_test[idx]
            true_label = y_test[idx]

            x_proc = normalize(img)
            x_proc = add_channel_dim(x_proc)
            x_proc = x_proc.transpose(2, 0, 1)

            pred = model.predict(x_proc)
            ok = pred == true_label
            correct += ok

            ax = axes[i // n_cols][i % n_cols]
            ax.imshow(img, cmap="gray")
            color = "green" if ok else "red"
            ax.set_title(f"V:{true_label} P:{pred}", fontsize=10, color=color)
            ax.axis("off")

        graph_path = join(ROOT_DIR, "results", "predictions.png")
        plt.suptitle(f"{correct}/10 correctes", fontsize=14)
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches="tight")
        print(f"📊 Prédictions → {graph_path}")
        print(f"🎯 {correct}/10 correctes")

    print("\n💡 Pour classifier tes propres images :")
    print("   from scripts.predict import load_model")
    print("   model = load_model()")
    print("   pred = model.predict(mon_image_normalisée)")
