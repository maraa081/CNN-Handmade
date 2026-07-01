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
| **Conv2D forward** | ✅ | Implémentation par **im2col + produit matriciel** (kernel 3×3, stride, padding) |
| **MaxPool2D** | ✅ | Forward + backward fonctionnels (stockage des indices pour rétropropagation) |
| Conv2D backward | ⏳ | Stub — `NotImplementedError` |
| ReLU | ❌ | |
| Dense / Fully Connected | ❌ | |
| Softmax + Cross-Entropy | ❌ | |
| Training loop | ❌ | |
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

## 📦 Fichier unique

Pour l'instant tout est dans `src/cnn.py` — le code évoluera vers une séparation en modules propres une fois les bases posées.

```
CNN-Handmade/
├── README.md
├── requirements.txt      (numpy + matplotlib)
├── data/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── src/
    └── cnn.py
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
