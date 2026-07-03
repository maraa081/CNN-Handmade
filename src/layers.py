"""
layers.py — Couches du réseau CNN-Handmade

Fonctions utilitaires :
    im2col, col2im     — transformation image ↔ colonnes pour convolution

Couches :
    Conv2D    — convolution 2D (forward, backward, update)
    MaxPool2D — max pooling 2D (forward, backward, update)
    ReLU      — activation ReLU (forward, backward, update)
    Flatten   — aplatissement (forward, backward, update)
    Dense     — Fully Connected    [TODO]
"""

import numpy as np


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  UTILITAIRES CONVOLUTION                                                  ║
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
    W_out = (W + 2 * pad - kernel_w) // stride + 1

    # Padding
    if pad > 0:
        images = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    # Pour chaque position du kernel, on extrait un patch et on l'aplatit
    cols = np.zeros((N * H_out * W_out, C * kernel_h * kernel_w))
    idx = 0
    for y in range(H_out):
        for x in range(W_out):
            y_start = y * stride
            x_start = x * stride
            patch = images[:, :, y_start:y_start + kernel_h, x_start:x_start + kernel_w]
            cols[idx::H_out * W_out] = patch.reshape(N, -1)
            idx += 1

    return cols, H_out, W_out


def col2im(cols, input_shape, kernel_h, kernel_w, stride=1, pad=0):
    """
    Opération inverse de im2col.

    Prend les gradients sous forme de colonnes et les redistribue
    dans l'image d'origine, en accumulant aux positions qui se chevauchent
    (un pixel peut contribuer à plusieurs positions de sortie).

    Entrée :
        cols         : (N * H_out * W_out, C_in * kH * kW)
        input_shape  : (N, C, H, W)
        kernel_h, kernel_w : taille du noyau
        stride, pad

    Sortie :
        grad : (N, C, H, W)
    """
    N, C, H, W = input_shape
    H_out = (H + 2 * pad - kernel_h) // stride + 1
    W_out = (W + 2 * pad - kernel_w) // stride + 1

    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    grad_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    for y in range(H_out):
        for x in range(W_out):
            y_start = y * stride
            x_start = x * stride

            row_indices = np.arange(N) * H_out * W_out + y * W_out + x
            grads = cols[row_indices]  # (N, C * kH * kW)
            grads_reshaped = grads.reshape(N, C, kernel_h, kernel_w)

            grad_padded[:, :, y_start:y_start + kernel_h, x_start:x_start + kernel_w] += grads_reshaped

    if pad > 0:
        return grad_padded[:, :, pad:-pad, pad:-pad]
    return grad_padded


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONV2D                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


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
        self.d_kernels = None
        self.d_bias = None

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

        # Produit matriciel
        out = cols @ kernels_flat.T  # (N*H_out*W_out, C_out)
        out += self.bias.T

        # Reshape en sortie 4D
        out = out.reshape(N, H_out, W_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)

        return out

    def backward(self, grad_output):
        """
        grad_output : gradient de la perte par rapport à la sortie (N, C_out, H_out, W_out)

        Calcule :
          - d_kernels : gradient pour les poids de la convolution
          - d_bias    : gradient pour le biais
          - d_input   : gradient à rétropropager à la couche précédente

        Retourne :
            d_input : (N, C_in, H, W)
        """
        N, C_out, H_out, W_out = grad_output.shape
        assert C_out == self.out_channels, \
            f"Canaux du gradient : {C_out}, attendu {self.out_channels}"

        # ── 1. Remettre grad_output en forme matricielle ──
        dout = grad_output.transpose(0, 2, 3, 1)  # (N, H_out, W_out, C_out)
        dout_flat = dout.reshape(-1, C_out)        # (N*H_out*W_out, C_out)

        # ── 2. Gradient des kernels ──
        self.d_kernels_flat = dout_flat.T @ self.cols  # (C_out, C_in*k*k)
        self.d_kernels = self.d_kernels_flat.reshape(
            self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size
        )

        # ── 3. Gradient du biais ──
        self.d_bias = dout_flat.sum(axis=0, keepdims=True).T  # (C_out, 1)

        # ── 4. Gradient de l'entrée (col2im) ──
        kernels_flat = self.kernels.reshape(self.out_channels, -1)
        d_cols = dout_flat @ kernels_flat  # (N*H_out*W_out, C_in*k*k)

        N_in, C_in, H_in, W_in = self.input.shape
        d_input = col2im(d_cols, (N_in, C_in, H_in, W_in),
                         self.kernel_size, self.kernel_size,
                         self.stride, self.pad)

        return d_input

    def update(self, lr):
        """
        Met à jour les poids avec la descente de gradient :
            W ← W - lr * dW
        """
        if self.d_kernels is None:
            raise RuntimeError("update() appelé avant backward()")

        self.kernels -= lr * self.d_kernels
        self.bias -= lr * self.d_bias

    def __repr__(self):
        return (f"Conv2D({self.in_channels}→{self.out_channels}, "
                f"kernel={self.kernel_size}x{self.kernel_size}, "
                f"stride={self.stride}, pad={self.pad})")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAX POOLING 2D                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class MaxPool2D:
    """
    Couche de Max Pooling 2D.

    Réduit la dimension spatiale en prenant la valeur maximale
    dans chaque fenêtre (pool_h × pool_w).

    Entrée : (N, C, H, W)
    Sortie : (N, C, H_out, W_out)

    Pas de paramètres à apprendre.
    """

    def __init__(self, pool_size=2, stride=None):
        """
        pool_size : taille de la fenêtre (2 = 2×2, ou tuple (h, w))
        stride    : pas de déplacement (défaut = pool_size)
        """
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size[0]
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        self.input = None
        self.max_indices = None

    def forward(self, x):
        """
        x : (N, C, H, W)
        retourne : (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride

        H_out = (H - pool_h) // stride_h + 1
        W_out = (W - pool_w) // stride_w + 1

        self.input = x

        windows = np.zeros((N, C, H_out, W_out, pool_h, pool_w))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride_h
                w_start = j * stride_w
                windows[:, :, i, j, :, :] = x[:, :,
                                                h_start:h_start + pool_h,
                                                w_start:w_start + pool_w]

        out = np.max(windows, axis=(4, 5))  # (N, C, H_out, W_out)

        windows_flat = windows.reshape(N, C, H_out, W_out, -1)
        self.max_indices = np.argmax(windows_flat, axis=4)

        return out

    def backward(self, grad_output):
        """
        grad_output : (N, C, H_out, W_out)
        retourne    : (N, C, H, W)
        """
        N, C, H, W = self.input.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]

        grad_input = np.zeros_like(self.input)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride_h
                w_start = j * stride_w

                max_idx = self.max_indices[:, :, i, j]
                grad_val = grad_output[:, :, i, j]

                ph = max_idx // pool_w
                pw = max_idx % pool_w

                h_abs = h_start + ph
                w_abs = w_start + pw

                n_idx = np.arange(N)[:, None]
                c_idx = np.arange(C)[None, :]
                grad_input[n_idx, c_idx, h_abs, w_abs] += grad_val

        return grad_input

    def update(self, lr):
        """MaxPooling : pas de paramètres à apprendre."""
        pass

    def __repr__(self):
        return (f"MaxPool2D(pool={self.pool_size[0]}×{self.pool_size[1]}, "
                f"stride={self.stride[0]}×{self.stride[1]})")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RELU                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ReLU:
    """
    Fonction d'activation ReLU (Rectified Linear Unit).

    Forward : f(x) = max(0, x)
    Backward : f'(x) = 1 si x > 0, sinon 0
    """

    def __init__(self):
        self.input = None
        self.mask = None

    def forward(self, x):
        """
        x : entrée (n'importe quelle forme)
        retourne : max(0, x) (même forme)
        """
        self.input = x
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        grad_output : gradient de la perte par rapport à la sortie
        retourne : gradient par rapport à l'entrée
        """
        return grad_output * self.mask

    def update(self, lr):
        """ReLU n'a pas de paramètres."""
        pass

    def __repr__(self):
        return "ReLU()"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FLATTEN                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class Flatten:
    """
    Aplatit les dimensions spatiales en un seul vecteur par échantillon.

    Forward : (N, C, H, W) → (N, C × H × W)
    Backward : (N, C × H × W) → (N, C, H, W)  [reshape inverse]

    Pas de paramètres à apprendre.
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        """
        x : (N, ...)
        retourne : (N, produit des dimensions)
        """
        self.input_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, grad_output):
        """
        grad_output : (N, C*H*W)
        retourne : (N, C, H, W)
        """
        return grad_output.reshape(self.input_shape)

    def update(self, lr):
        """Flatten n'a pas de paramètres."""
        pass

    def __repr__(self):
        return "Flatten()"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DENSE (FULLY CONNECTED) — [TODO]                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class Dense:
    """
    Couche Fully Connected (linéaire).

    y = x @ W.T + b

    TODO : forward, backward, update à implémenter.
    """

    def __init__(self, in_features, out_features):
        """
        in_features  : taille du vecteur d'entrée
        out_features : taille du vecteur de sortie
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialisation He
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features, 1))

        self.input = None

    def forward(self, x):
        """
        x : (N, in_features)
        retourne : (N, out_features)
        """
        raise NotImplementedError("Dense.forward — à implémenter")

    def backward(self, grad_output):
        """
        grad_output : (N, out_features)
        retourne : (N, in_features)
        """
        raise NotImplementedError("Dense.backward — à implémenter")

    def update(self, lr):
        """W ← W - lr * dW, b ← b - lr * db"""
        raise NotImplementedError("Dense.update — à implémenter")

    def __repr__(self):
        return f"Dense({self.in_features}→{self.out_features})"
