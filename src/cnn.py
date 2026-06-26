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
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import random

# Chemin racine du projet (dossier parent de src/)
ROOT_DIR = dirname(dirname(abspath(__file__)))


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
        # Appel avec ROOT_DIR pour être sûr du chemin
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
# ║  4️⃣  COUCHE DE CONVOLUTION (Conv2D)                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def im2col(images, kernel_h, kernel_w, stride=1, pad=0):
    """
    Transforme un batch d'images en colonnes pour faire la convolution
    comme un produit matriciel.

    Principe :
      Au lieu de faire des boucles pour chaque position du kernel,
      on "déroule" tous les patchs de l'image en colonnes d'une grande matrice.
      Puis la convolution = cette matrice × le kernel aplati.

    Entrée : images (N, C, H, W)
    Sortie : cols (N * H_out * W_out, C * kH * kW)
    """
    N, C, H, W = images.shape

    # Hauteur et largeur de la sortie après convolution
    H_out = (H + 2 * pad - kernel_h) // stride + 1
    W_out = (W + 2 * pad - kernel_h) // stride + 1

    # Padding
    if pad > 0:
        images = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    # Pour chaque position du kernel, on extrait un patch et on l'aplatit
    cols = np.zeros((N * H_out * W_out, C * kernel_h * kernel_w))
    idx = 0
    for y in range(H_out):
        for x in range(W_out):
            # Coordonnées du patch dans l'image (avec stride)
            y_start = y * stride
            x_start = x * stride
            # Patch : (N, C, kH, kW) → (N, C*kH*kW)
            patch = images[:, :, y_start:y_start + kernel_h, x_start:x_start + kernel_w]
            cols[idx::H_out * W_out] = patch.reshape(N, -1)
            idx += 1

    return cols, H_out, W_out


class Conv2D:
    """
    Couche de convolution 2D.

    Transforme une entrée (N, C_in, H, W) en sortie (N, C_out, H_out, W_out)
    en faisant glisser C_out kernels sur l'image.

    Le calcul utilise im2col pour transformer la convolution en produit matriciel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        """
        in_channels  : nombre de canaux d'entrée (1 pour MNIST en niveaux de gris)
        out_channels : nombre de filtres / canaux de sortie
        kernel_size  : taille du kernel (3 = 3x3, 5 = 5x5)
        stride       : pas de déplacement du kernel
        pad          : padding (0 = pas de padding)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # Initialisation des poids (He init : / sqrt(fan_in / 2))
        fan_in = in_channels * kernel_size * kernel_size
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros((out_channels, 1))

        # Pour la backprop (sera rempli par forward)
        self.input = None
        self.cols = None

    def forward(self, x):
        """
        x : entrée (N, C_in, H, W)
        retourne : (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        assert C_in == self.in_channels, \
            f"Canaux d'entrée : {C_in}, attendu {self.in_channels}"

        self.input = x

        # im2col : on transforme l'image en matrice de colonnes
        cols, H_out, W_out = im2col(x, self.kernel_size, self.kernel_size,
                                     self.stride, self.pad)
        self.cols = cols  # (N*H_out*W_out, C_in*k*k)

        # On aplatit les kernels en une matrice (C_out, C_in*k*k)
        kernels_flat = self.kernels.reshape(self.out_channels, -1)

        # Produit matriciel : (N*H_out*W_out, C_in*k*k) × (C_in*k*k, C_out)
        # Puis on ajoute le biais
        out = cols @ kernels_flat.T  # (N*H_out*W_out, C_out)
        out += self.bias.T

        # Reshape en sortie 4D
        out = out.reshape(N, H_out, W_out, self.out_channels)
        # On permute pour avoir (N, C_out, H_out, W_out) — format standard
        out = out.transpose(0, 3, 1, 2)

        return out

    def backward(self, grad_output):
        """
        grad_output : gradient de la perte par rapport à la sortie (N, C_out, H_out, W_out)
        → retourne le gradient par rapport à l'entrée

        TODO : sera implémenté après le forward
        """
        raise NotImplementedError("Backward Conv2D — à venir !")

    def update(self, lr):
        """
        Met à jour les poids avec le learning rate.

        TODO : sera implémenté après le forward
        """
        raise NotImplementedError("Update Conv2D — à venir !")

    def __repr__(self):
        return (f"Conv2D({self.in_channels}→{self.out_channels}, "
                f"kernel={self.kernel_size}x{self.kernel_size}, "
                f"stride={self.stride}, pad={self.pad})")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ██  PROCHAINE COUCHE : MAXPOOLING / RELU / DENSE / ...                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ██  TEST / DÉMONSTRATION                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # --- Chargement ---
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))

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

    # --- Affichage de 10 images du train avec leur label ---
    n_cols = 5
    n_rows = 2
    plt.figure(figsize=(10, 4))
    indices = np.random.choice(len(x_train), size=n_cols * n_rows, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(x_train[idx], cmap="gray")
        plt.title(f"Label: {y_train[idx]}", fontsize=10)
        plt.axis("off")

    plt.suptitle("10 chiffres MNIST (train)", fontsize=14)
    plt.tight_layout()
    plt.show()  # ← ouvre une fenêtre avec les images

    # --- Test rapide de Conv2D ---
    print("\n" + "═" * 50)
    print("🧪 Test Conv2D forward")
    print("═" * 50)

    # On crée une image de test (batch=1, canal=1, 28x28)
    test_img = x_train[:4]  # 4 images
    test_img = normalize(test_img)
    test_img = add_channel_dim(test_img)  # (4, 28, 28, 1)
    # On permute en (N, C, H, W) pour la convolution
    test_img = test_img.transpose(0, 3, 1, 2)
    print(f"Entrée Conv2D : {test_img.shape}")

    # 1 filtre 3x3, stride=1, pad=0
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, pad=1)
    print(f"Couche créée : {conv}")

    out = conv.forward(test_img)
    print(f"Sortie Conv2D  : {out.shape}")
    print(f"  (devrait être 4 × 4 × 28 × 28 avec padding)")

    # Test sans padding
    conv2 = Conv2D(in_channels=1, out_channels=8, kernel_size=5, stride=2, pad=0)
    out2 = conv2.forward(test_img)
    print(f"\nConv2D(1→8, kernel=5, stride=2, pad=0)")
    print(f"  Entrée : {test_img.shape}  →  Sortie : {out2.shape}")
    print(f"  (devrait être 4 × 8 × 12 × 12)")
