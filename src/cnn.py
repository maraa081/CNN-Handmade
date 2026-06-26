"""
CNN-Handmade — Réseau de neurones convolutionnel pour MNIST
Fait à la main, de A à Z. Pas de TensorFlow, pas de PyTorch.

┌─────────────────────────────────────────────────────────────────────────────┐
│  TODO : Les sections marquées █ seront remplies dans le futur              │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1️⃣  IMPORTS                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import struct
from array import array
from os.path import join
import matplotlib.pyplot as plt
import random


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2️⃣  CHARGEMENT DES DONNÉES (DATASET)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MNISTLoader:
    """
    Lit les fichiers IDX bruts du dataset MNIST.
    Format officiel : https://yann.lecun.com/exdb/mnist/
    """

    def read_images(self, filepath):
        """Lit un fichier IDX d'images → retourne un numpy array (N, 28, 28)"""
        with open(filepath, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Magic number invalide pour les images : {magic} (attendu 2051)"
                )
            data = array("B", f.read())
        images = np.array(data, dtype=np.float32).reshape(size, rows, cols)
        return images

    def read_labels(self, filepath):
        """Lit un fichier IDX de labels → retourne un numpy array (N,)"""
        with open(filepath, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(
                    f"Magic number invalide pour les labels : {magic} (attendu 2049)"
                )
            labels = array("B", f.read())
        return np.array(labels, dtype=np.uint8)

    def load(self, data_dir="."):
        """
        Charge les 4 fichiers MNIST depuis un dossier.

        Retourne :
            (x_train, y_train), (x_test, y_test)
        """
        x_train = self.read_images(join(data_dir, "train-images-idx3-ubyte"))
        y_train = self.read_labels(join(data_dir, "train-labels-idx1-ubyte"))
        x_test = self.read_images(join(data_dir, "t10k-images-idx3-ubyte"))
        y_test = self.read_labels(join(data_dir, "t10k-labels-idx1-ubyte"))
        return (x_train, y_train), (x_test, y_test)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3️⃣  PRÉPARATION DES DONNÉES (PREPROCESSING)                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def normalize(images):
    """
    Normalise les pixels de [0, 255] → [0.0, 1.0].
    """
    return images / 255.0


def add_channel_dim(images):
    """
    Ajoute la dimension du canal : (N, 28, 28) → (N, 28, 28, 1)

    Format "channels_last" (comme TensorFlow).
    """
    return images[..., np.newaxis]


def to_one_hot(labels, num_classes=10):
    """
    Convertit des labels entiers (ex: 3) en vecteur one-hot (ex: [0,0,0,1,0,0,0,0,0,0]).
    """
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot


class DataLoader:
    """
    Découpe les données en mini-batches.

    Exemple :
        loader = DataLoader(x_train, y_train, batch_size=32, shuffle=True)
        for batch_x, batch_y in loader:
            # batch_x.shape = (32, 28, 28, 1)
            # batch_y.shape = (32, 10)  → one-hot
            ...
    """

    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(images)
        self.indices = np.arange(self.n_samples)

    def __iter__(self):
        """Itérateur : yield (batch_images, batch_labels) à chaque pas."""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = self.indices[start:end]
            yield self.images[batch_idx], self.labels[batch_idx]

    def __len__(self):
        """Nombre de batches par epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def preprocess_pipeline(images, labels, batch_size=32, shuffle=True):
    """
    Enchaîne normalisation + reshape one-hot et retourne un DataLoader.

    Args :
        images   → numpy array (N, 28, 28)  brut
        labels   → numpy array (N,)         brut
        batch_size → taille des batches
        shuffle    → mélanger ?

    Retourne :
        DataLoader prêt à itérer
    """
    images = normalize(images)
    images = add_channel_dim(images)
    labels = to_one_hot(labels)
    return DataLoader(images, labels, batch_size, shuffle)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ██  À VENIR : ARCHITECTURE DU RÉSEAU                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Chaque couche implémentera :
#   - forward(inputs)  → outputs
#   - backward(grad)   → grad_inputs
#   - update(lr)       → ajuste les poids

# ─── Couches à venir ───
# class Dense:       ...
# class Conv2D:      ...
# class MaxPool2D:   ...
# class Flatten:     ...
# class ReLU:        ...
# class Softmax:     ...
# class Sequential:  ...

# ─── Loss, Optimizer, Entraînement ───
# class CrossEntropy:  ...
# class SGD:           ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ██  TEST / DÉMONSTRATION                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # --- Chargement ---
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load("data")

    print(f"[OK] MNIST chargé")
    print(f"  x_train : {x_train.shape}  ({x_train.dtype})")
    print(f"  y_train : {y_train.shape}  ({y_train.dtype})")
    print(f"  x_test  : {x_test.shape}   ({x_test.dtype})")
    print(f"  y_test  : {y_test.shape}   ({y_test.dtype})")

    # --- Preprocessing ---
    train_loader = preprocess_pipeline(x_train, y_train, batch_size=32, shuffle=True)
    test_loader  = preprocess_pipeline(x_test, y_test, batch_size=32, shuffle=False)

    # Vérification d'un batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\n[OK] Premier batch :")
    print(f"  batch_x : {batch_x.shape}  (min={batch_x.min():.2f}, max={batch_x.max():.2f})")
    print(f"  batch_y : {batch_y.shape}  (exemple one-hot: {batch_y[0]})")
    print(f"  Nombre de batches par epoch : {len(train_loader)}")

    # Vérification test
    batch_x_t, batch_y_t = next(iter(test_loader))
    print(f"\n[OK] Batch test : {batch_x_t.shape}, {batch_y_t.shape}")
