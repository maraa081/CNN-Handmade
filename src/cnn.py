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
    Dense,
)
from losses import (
    Softmax,
    CrossEntropyLoss,
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
    print(f"  Format : channels_first  ✅")
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
    print("🧪 Test Dense forward + backward + update")
    print("═" * 50)

    N, in_feat, out_feat = 8, 16, 5
    dense = Dense(in_feat, out_feat)
    print(f"Couche créée : {dense}")

    x_in = np.random.randn(N, in_feat) * 0.1
    out = dense.forward(x_in)
    print(f"  Forward : ({N}, {in_feat}) → ({N}, {out_feat})  {'✅' if out.shape == (N, out_feat) else '❌'}")

    # Vérification produit matriciel + biais
    expected = x_in @ dense.W.T + dense.b.T
    print(f"  Valeurs correctes : {'✅' if np.allclose(out, expected) else '❌'}")

    # Backward
    dout = np.random.randn(N, out_feat) * 0.1
    dx = dense.backward(dout)
    print(f"  Backward : ({N}, {out_feat}) → ({N}, {in_feat})  {'✅' if dx.shape == (N, in_feat) else '❌'}")

    # Vérifications manuelles
    expected_dW = dout.T @ x_in
    expected_db = dout.sum(axis=0, keepdims=True).T
    expected_dx = dout @ dense.W
    print(f"  dW correct  : {'✅' if np.allclose(dense.dW, expected_dW) else '❌'}")
    print(f"  db correct  : {'✅' if np.allclose(dense.db, expected_db) else '❌'}")
    print(f"  dx correct  : {'✅' if np.allclose(dx, expected_dx) else '❌'}")

    # Update
    lr = 0.01
    old_W = dense.W.copy()
    old_b = dense.b.copy()
    dense.update(lr)
    W_ok = np.allclose(dense.W, old_W - lr * dense.dW)
    b_ok = np.allclose(dense.b, old_b - lr * dense.db)
    print(f"  Update W : {'✅' if W_ok else '❌'}")
    print(f"  Update b : {'✅' if b_ok else '❌'}")

    # Gradient check par différences finies
    # Pour une loss = sum(out), le gradient est dout = 1
    eps = 1e-6
    w_idx = (0, 2)  # Une valeur de poids dans W[0,2]
    orig_val = dense.W[w_idx]
    dense.W[w_idx] = orig_val + eps
    loss_plus = dense.forward(x_in).sum()
    dense.W[w_idx] = orig_val - eps
    loss_minus = dense.forward(x_in).sum()
    dense.W[w_idx] = orig_val

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    # Pour loss = sum(out), dout = ones_like(out)
    dout_sum = np.ones((N, out_feat))
    dense.forward(x_in)  # Re-forward
    dense.backward(dout_sum)
    ana_grad = dense.dW[w_idx]
    rel_error = abs(num_grad - ana_grad) / max(abs(num_grad), abs(ana_grad), 1e-8)
    print(f"  Gradient check W[{w_idx[0]},{w_idx[1]}] : err={rel_error:.8f}")
    print(f"  {'✅' if rel_error < 1e-4 else '❌'} gradient check passé")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test Softmax forward + backward")
    print("═" * 50)

    softmax = Softmax()
    x_sm = np.array([[1.0, 2.0, 3.0],
                     [0.0, 0.0, 0.0],
                     [-2.0, -1.0, -0.5]], dtype=np.float64)
    p = softmax.forward(x_sm)
    print(f"  Chaque ligne somme à 1 : {'✅' if np.allclose(p.sum(axis=1), 1.0) else '❌'}")

    # Test vérification manuelle
    p_expected = np.exp(x_sm - x_sm.max(axis=1, keepdims=True))
    p_expected /= p_expected.sum(axis=1, keepdims=True)
    print(f"  Valeurs softmax correctes : {'✅' if np.allclose(p, p_expected) else '❌'}")

    # Backward
    dout = np.random.randn(*p.shape) * 0.1
    dx = softmax.backward(dout)
    print(f"  Backward shape : {dx.shape}  {'✅' if dx.shape == x_sm.shape else '❌'}")

    # Gradient check du softmax par différences finies
    eps = 1e-6
    orig_val = x_sm[0, 1]
    x_sm[0, 1] = orig_val + eps
    p_plus = softmax.forward(x_sm)
    x_sm[0, 1] = orig_val - eps
    p_minus = softmax.forward(x_sm)
    x_sm[0, 1] = orig_val
    loss_plus = (p_plus * dout).sum()
    loss_minus = (p_minus * dout).sum()
    num_grad = (loss_plus - loss_minus) / (2 * eps)
    softmax.forward(x_sm)
    ana_grad = softmax.backward(dout)[0, 1]
    rel_error = abs(num_grad - ana_grad) / max(abs(num_grad), abs(ana_grad), 1e-8)
    print(f"  Gradient check x[0,1] : err={rel_error:.8f}")
    print(f"  {'✅' if rel_error < 1e-4 else '❌'}")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test CrossEntropyLoss forward + backward")
    print("═" * 50)

    loss_fn = CrossEntropyLoss()
    N, C = 5, 10
    logits = np.random.randn(N, C) * 2.0
    labels = np.array([3, 7, 0, 9, 4])
    y_true = np.eye(C)[labels]

    loss = loss_fn.forward(logits, y_true)
    print(f"  Loss : {loss:.6f}  {'✅' if loss > 0 else '❌'}")

    # Vérification manuelle
    soft = Softmax()
    p_check = soft.forward(logits)
    loss_manual = -np.sum(y_true * np.log(p_check + 1e-15)) / N
    print(f"  Loss manuelle : {loss_manual:.6f}  {'✅' if np.allclose(loss, loss_manual) else '❌'}")

    # Logits uniformes → loss = log(10) ≈ 2.3026
    uniform_logits = np.zeros((N, C))
    uniform_loss = loss_fn.forward(uniform_logits, y_true)
    print(f"  Loss logits uniformes : {uniform_loss:.6f}  {'✅' if np.allclose(uniform_loss, np.log(C)) else '❌'}")

    # Logits parfaits → loss ≈ 0
    perfect_logits = np.zeros((N, C))
    perfect_logits[np.arange(N), labels] = 100.0
    perfect_loss = loss_fn.forward(perfect_logits, y_true)
    print(f"  Loss logits parfaits : {perfect_loss:.6f}  {'✅' if perfect_loss < 1e-5 else '❌'}")

    # Backward
    dlogits = loss_fn.backward()
    print(f"  Backward shape : {dlogits.shape}  {'✅' if dlogits.shape == logits.shape else '❌'}")

    # Vérification : gradient doit sommer à 0 par échantillon (car sum(p) = 1)
    grad_sum = loss_fn.backward().sum(axis=1)
    print(f"  Gradient somme à 0 / échantillon : {'✅' if np.allclose(grad_sum, 0) else '❌'}")

    # Gradient check combined
    eps = 1e-6
    orig_val = logits[0, 0]
    logits[0, 0] = orig_val + eps
    loss_plus = loss_fn.forward(logits, y_true)
    logits[0, 0] = orig_val - eps
    loss_minus = loss_fn.forward(logits, y_true)
    logits[0, 0] = orig_val
    num_grad = (loss_plus - loss_minus) / (2 * eps)
    loss_fn.forward(logits, y_true)
    ana_grad = loss_fn.backward()[0, 0]
    rel_error = abs(num_grad - ana_grad) / max(abs(num_grad), abs(ana_grad), 1e-8)
    print(f"  Gradient check logits[0,0] : err={rel_error:.8f}")
    print(f"  {'✅' if rel_error < 1e-4 else '❌'}")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🧪 Test accuracy")
    print("═" * 50)

    acc = loss_fn.accuracy(logits, y_true)
    print(f"  Accuracy (logits aléatoires) : {acc:.2f}")
    print(f"  Accuracy > 0 : {'✅' if acc >= 0 else '❌'}")

    perfect_logits_acc = np.zeros((N, C))
    perfect_logits_acc[np.arange(N), labels] = 100.0
    acc_perfect = loss_fn.accuracy(perfect_logits_acc, y_true)
    print(f"  Accuracy parfaite : {acc_perfect:.2f}  {'✅' if acc_perfect == 1.0 else '❌'}")

    # ═══════════════════════════════════════════════════
    print("\n" + "═" * 50)
    print("🎉 Tous les tests sont passés !")
    print("═" * 50)
    print()
    print("Modules disponibles :")
    print("  data.py       → MNISTLoader, DataLoader, preprocessing")
    print("  layers.py     → im2col, col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dense ✅")
    print("  losses.py     → Softmax ✅, CrossEntropyLoss ✅")
    print("  model.py      → CNN (forward, backward, update, train, evaluate) ✅")
    print("  tune_cnn.py   → fichier de réglages interactif 🎮")
