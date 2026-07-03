# 🧠 CNN Handmade

**Un réseau de neurones convolutionnel pour reconnaître les chiffres manuscrits (MNIST), fait à la main, de A à Z.**

Pas de TensorFlow, pas de PyTorch, pas de Keras. Juste Python, NumPy, et moi. 🔧

## 🎯 Objectif

Implémenter un CNN capable de classifier le dataset MNIST (0-9) **sans framework de deep learning**. Comprendre chaque brique en la codant soi-même plutôt que d'appeler une API.

## ✅ Avancement actuel

| Partie | Statut | Notes |
|---|---|---|
| Chargement MNIST (IDX) | ✅ | `MNISTLoader` — lecture des fichiers binaires |
| Preprocessing | ✅ | Normalisation [0,1], one-hot encoding, DataLoader par batches |
| Partie | Statut | Notes |
|---|---|---|
| Chargement MNIST (IDX) | ✅ | `MNISTLoader` — lecture des fichiers binaires |
| Preprocessing | ✅ | Normalisation [0,1], one-hot encoding, DataLoader par batches |
| **Conv2D forward** | ✅ | Implémentation par **im2col + produit matriciel** |
| **Conv2D backward** | ✅ | rétropropagation complète (gradients kernels, bias, col2im) |
| **MaxPool2D** | ✅ | Forward + backward (indices des max stockés) |
| **ReLU** | ✅ | Forward + backward (masque x > 0) |
| **Flatten** | ✅ | Forward + backward (reshape inverse) |
| **Découpage en modules** | ✅ | `data.py`, `layers.py`, `losses.py`, `model.py` |
| Dense / Fully Connected | ❌ | Stub dans layers.py — à implémenter |
| Softmax + Cross-Entropy | ❌ | Stubs dans losses.py |
| Training loop | ❌ | Stub dans model.py |
| Évaluation / Accuracy | ❌ | |

## 🧱 Architecture prévue

```
Entrée : (N, 1, 28, 28)
    │
    ├── Conv2D  (1 → 32, kernel=3, pad=1)     →  (N, 32, 28, 28)
    ├── ReLU
    ├── MaxPool2D  (2×2, stride=2)             →  (N, 32, 14, 14)
    │
    ├── Conv2D  (32 → 64, kernel=3, pad=1)     →  (N, 64, 14, 14)
    ├── ReLU
    ├── MaxPool2D  (2×2, stride=2)             →  (N, 64, 7, 7)
    │
    ├── Flatten                                →  (N, 3136)
    ├── Dense (3136 → 128)
    ├── ReLU
    ├── Dense (128 → 10)
    └── Softmax                                →  (N, 10)
```

## 📦 Structure du projet

```
CNN-Handmade/
├── README.md
├── requirements.txt          (numpy + matplotlib)
├── docs/
│   ├── data-flow.md          — trace complète d'une image dans le réseau
│   └── memoire-projet.md     — carnet de bord, concepts à retenir
├── data/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── traces/
│   └── forward_trace.py
└── src/
    ├── __init__.py
    ├── data.py      — MNISTLoader, preprocessing, DataLoader
    ├── layers.py    — im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dense (stub)
    ├── losses.py    — Softmax, CrossEntropyLoss (stubs)
    ├── model.py     — CNN (stub)
    └── cnn.py       — script de démonstration / tests
```

## 📈 Prochaines étapes (ordre)

1. **ReLU** — forward `max(0, x)` et backward
2. **Flatten** — transformation (N, C, H, W) → (N, C×H×W)
3. **Dense (Fully Connected)** — couche linéaire + backward
4. **Softmax + Cross-Entropy Loss** — fonction de perte
5. **Training loop** — descente de gradient, epochs, évaluation
6. **Tests & tuning** — faire converger le modèle sur MNIST (> 95%)

## 🚀 Lancer le test

```bash
cd src/
python3 cnn.py
```

Affiche le chargement MNIST, un test Conv2D forward, et un test MaxPool2D (forward + backward).

---

**#NoFrameworks #FromScratch #MNIST**
