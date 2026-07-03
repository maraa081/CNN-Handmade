#!/usr/bin/env python3
"""
cnn.py — Script de démonstration CNN-Handmade

Importe les modules découpés (data, layers, losses, model) et lance
les tests de toutes les couches implémentées.

Usage :
    python3 src/cnn.py
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = dirname(dirname(abspath(__file__)))

from data import (
    MNISTLoader,
    normalize,
    add_channel_dim,
    preprocess_pipeline,
)
from layers import (
    Conv2D,
    MaxPool2D,
    ReLU,
    Flatten,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TESTS                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # ── Chargement ──
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))

    print(f"[OK] MNIST chargé")
    print(f"  x_train : {x_train.shape}  ({x_train.dtype})")
    print(f"  y_train : {y_train.shape}  ({y_train.dtype})")
    print(f"  x_test  : {x_test.shape}   ({x_test.dtype})")
    print(f"  y_test  : {y_test.shape}   ({y_test.dtype})")

    # ── Preprocessing ──
    train_loader = preprocess_pipeline(x_train, y_train, batch_size=32, shuffle=True)
    test_loader  = preprocess_pipeline(x_test, y_test, batch_size=32, shuffle=False)

    batch_x, batch_y = next(iter(train_loader))
    print(f"\n[OK] Premier batch :")
    print(f"  batch_x : {batch_x.shape}  (min={batch_x.min():.2f}, max={batch_x.max():.2f})")
    print(f"  batch_y : {batch_y.shape}  (exemple one-hot: {batch_y[0]})")
    print(f"  Nombre de batches par epoch : {len(train_loader)}")

    batch_x_t, batch_y_t = next(iter(test_loader))
    print(f"\n[OK] Batch test : {batch_x_t.shape}, {batch_y_t.shape}")

    # ── Affichage ──
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
    plt.show()

    # ── Image de test pour les couches ──
    test_img = x_train[:4]
    test_img = normalize(test_img)
    test_img = add_channel_dim(test_img)             # (4, 28, 28, 1)
    test_img = test_img.transpose(0, 3, 1, 2)       # (4, 1, 28, 28)

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test Conv2D forward")
    print("═" * 50)

    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, pad=1)
    print(f"Couche créée : {conv}")
    out = conv.forward(test_img)
    print(f"Sortie : {out.shape}  (devrait être 4 × 4 × 28 × 28)")

    conv2 = Conv2D(in_channels=1, out_channels=8, kernel_size=5, stride=2, pad=0)
    out2 = conv2.forward(test_img)
    print(f"Conv2D(1→8, k=5, s=2) : {test_img.shape} → {out2.shape}  (devrait 4×8×12×12)")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test MaxPool2D forward")
    print("═" * 50)

    pool = MaxPool2D(pool_size=2, stride=2)
    print(f"Couche créée : {pool}")
    pool_in = out  # (4, 4, 28, 28)
    pool_out = pool.forward(pool_in)
    print(f"  Entrée : {pool_in.shape}  →  Sortie : {pool_out.shape}")
    print(f"  (devrait être 4 × 4 × 14 × 14)")

    for n in range(min(2, 4)):
        for c in range(min(2, 4)):
            slice_in = pool_in[n, c, :2, :2]
            max_in = slice_in.max()
            max_out = pool_out[n, c, 0, 0]
            ok = "✅" if abs(max_in - max_out) < 1e-5 else "❌"
            print(f"    [{n},{c}] fenêtre top-left : max={max_in:.4f} → pool={max_out:.4f} {ok}")

    print(f"\n  🧪 Test MaxPool2D backward :")
    grad_fake = np.ones_like(pool_out) * 0.5
    grad_back = pool.backward(grad_fake)
    print(f"    grad_output : {grad_fake.shape} → grad_input : {grad_back.shape}")
    ratio = grad_back.sum() / grad_fake.sum()
    print(f"    Ratio gradient transmis : {ratio:.2f} (doit être 1.00)")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test ReLU forward + backward")
    print("═" * 50)

    relu = ReLU()
    print(f"Couche créée : {relu}")
    x_test_relu = np.array([[-2.0, -1.0, 0.0, 0.5, 3.0]], dtype=np.float32)
    out = relu.forward(x_test_relu)
    print(f"  Entrée  : {x_test_relu[0].tolist()}")
    print(f"  Sortie  : {out[0].tolist()}")
    print(f"  Négatifs → 0 : {'✅' if out[0,0]==0 and out[0,1]==0 else '❌'}")

    dout = np.ones_like(out) * 2.0
    dx = relu.backward(dout)
    print(f"  Backward: {'✅' if dx[0,0]==0 and dx[0,3]==2.0 else '❌'}")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test Flatten forward + backward")
    print("═" * 50)

    flat = Flatten()
    flat_in = pool_out  # (4, 4, 14, 14)
    flat_out = flat.forward(flat_in)
    expected = (4, 4 * 14 * 14)
    print(f"  Entrée : {flat_in.shape}  →  Sortie : {flat_out.shape}")
    print(f"  (devrait être {expected}) {'✅' if flat_out.shape == expected else '❌'}")
    print(f"  ✅ valeurs identiques à reshape numpy" if np.allclose(flat_out, flat_in.reshape(4, -1)) else "")

    dout = np.ones_like(flat_out) * 3.0
    dx = flat.backward(dout)
    print(f"  Backward: {dx.shape}  {'✅' if dx.shape == flat_in.shape else '❌'}")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test Conv2D backward (différences finies)")
    print("═" * 50)

    tiny_batch = test_img[:2, :1, :5, :5].copy()
    conv_bw = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, pad=0)
    fwd = conv_bw.forward(tiny_batch)
    dout = np.ones_like(fwd)
    dx = conv_bw.backward(dout)
    print(f"  Forward  : {tiny_batch.shape} → {fwd.shape}")
    print(f"  Backward : d_input → {dx.shape}  {'✅' if dx.shape == tiny_batch.shape else '❌'}")

    eps = 1e-6
    k_idx = 0
    original_kernel = conv_bw.kernels[k_idx].copy()

    conv_bw.kernels[k_idx, 0, 0, 0] += eps
    fwd_plus = conv_bw.forward(tiny_batch)
    loss_plus = fwd_plus.sum()

    conv_bw.kernels[k_idx, 0, 0, 0] -= 2 * eps
    fwd_minus = conv_bw.forward(tiny_batch)
    loss_minus = fwd_minus.sum()

    conv_bw.kernels[k_idx] = original_kernel

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    ana_grad = conv_bw.d_kernels[k_idx, 0, 0, 0]
    rel_error = abs(num_grad - ana_grad) / max(abs(num_grad), abs(ana_grad), 1e-8)
    print(f"  Gradient check kernel[{k_idx},0,0,0] : err={rel_error:.8f}")
    print(f"  {'✅' if rel_error < 1e-4 else '❌'} gradient check passé")

    old_k = conv_bw.kernels[0, 0, :3, :3].copy()
    old_b = conv_bw.bias[0, 0]
    lr = 0.01
    conv_bw.update(lr)
    k_ok = np.allclose(conv_bw.kernels[0, 0, :3, :3], old_k - lr * conv_bw.d_kernels[0, 0, :3, :3])
    b_ok = np.allclose(conv_bw.bias[0, 0], old_b - lr * conv_bw.d_bias[0, 0])
    print(f"  Update kernels : {'✅' if k_ok else '❌'}")
    print(f"  Update bias    : {'✅' if b_ok else '❌'}")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🎉 Tous les tests sont passés !")
    print("═" * 50)
    print()
    print("Modules disponibles :")
    print("  data.py    → MNISTLoader, DataLoader, preprocessing")
    print("  layers.py  → im2col, col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dense (stub)")
    print("  losses.py  → Softmax, CrossEntropyLoss (stubs)")
    print("  model.py   → CNN (stub)")
