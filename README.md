# 🧠 CNN Handmade

**Un réseau de neurones convolutionnel pour reconnaître les chiffres manuscrits (MNIST), fait à la main, de A à Z.**

Pas de TensorFlow, pas de PyTorch, pas de Keras. Juste Python, NumPy, et moi.

## 🎯 Objectif

Implémenter un CNN capable de classifier les chiffres du dataset MNIST (0-9) **sans framework de deep learning**. Comprendre chaque brique du réseau en la codant soi-même :

- Convolution
- Pooling (Max / Average)
- Fonctions d'activation (ReLU, Softmax)
- Forward propagation
- Backpropagation
- Descente de gradient

## 📦 Dataset

Le dataset [MNIST](http://yann.lecun.com/exdb/mnist/) — 70 000 images 28×28 de chiffres manuscrits.

## 🧱 Architecture (à définir)

*À remplir au fil du développement.*

## 📂 Structure

```
CNN-Handmade/
├── README.md
├── requirements.txt
├── src/
│   ├── cnn.py          # Le réseau
│   ├── layers.py       # Couches (Conv, Pool, Dense, etc.)
│   ├── activations.py  # ReLU, Softmax, Sigmoid...
│   ├── loss.py         # Fonctions de perte
│   ├── optimizer.py    # Descente de gradient
│   ├── utils.py        # Chargement MNIST, helpers
│   └── train.py        # Entraînement
├── notebooks/
│   └── exploration.ipynb
└── tests/
    └── test_layers.py
```

*Structure évolutive.*

---

**#NoFrameworks #FromScratch #MNIST**
