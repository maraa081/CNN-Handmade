# 🧠 CNN Handmade

**Un réseau de neurones convolutionnel pour reconnaître les chiffres manuscrits (MNIST), fait à la main, de A à Z.**

Pas de TensorFlow, pas de PyTorch, pas de Keras. Juste Python, NumPy, et moi. 🔧

## 🎯 Pourquoi ?

Comprendre chaque brique du deep learning en la codant soi-même — im2col, rétropropagation, descente de gradient… plutôt que d'appeler une API magique.

## ✅ Ce qui est implémenté

| Module | Statut |
|---|---|
| **MNISTLoader** (fichiers IDX bruts) | ✅ |
| **Preprocessing** (normalisation, one-hot, DataLoader) | ✅ |
| **Conv2D** (forward par im2col, backward, update) | ✅ |
| **MaxPool2D** (forward + backward avec indices) | ✅ |
| **ReLU** (forward + backward) | ✅ |
| **Flatten** (forward + backward) | ✅ |
| **Dense / Fully Connected** (forward + backward + update) | ✅ |
| **Softmax** (forward + backward) | ✅ |
| **CrossEntropyLoss** (forward + backward + accuracy) | ✅ |
| **Training loop** (CNN.train avec historique) | ✅ |
| **Évaluation** (accuracy sur test set) | ✅ |
| **Sauvegarde / Chargement des poids** | ✅ |
| **Predict** (classifier une image chargée) | ✅ |
| **Graphiques d'entraînement** (loss + accuracy) | ✅ |

## 🧱 Architecture

```
Entrée : (N, 1, 28, 28)
    │
    ├── Conv2D   (1 → 32,  kernel=3, pad=1)     →  (N, 32, 28, 28)
    ├── ReLU
    ├── MaxPool2D  (2×2, stride=2)               →  (N, 32, 14, 14)
    │
    ├── Conv2D   (32 → 64, kernel=3, pad=1)      →  (N, 64, 14, 14)
    ├── ReLU
    ├── MaxPool2D  (2×2, stride=2)               →  (N, 64, 7, 7)
    │
    ├── Flatten                                  →  (N, 3136)
    ├── Dense   (3136 → 128)
    ├── ReLU
    ├── Dense   (128 → 10)
    └── Softmax                                  →  (N, 10)
```

## 📦 Structure du projet

```
CNN-Handmade/
├── README.md
├── requirements.txt
├── predict.py              ← charge le modèle et classifie
├── docs/
│   ├── data-flow.md
│   └── memoire-projet.md
├── data/
│   └── (fichiers MNIST .ubyte)
├── traces/
│   └── forward_trace.py
└── src/
    ├── __init__.py
    ├── data.py      — MNISTLoader, preprocessing, DataLoader
    ├── layers.py    — im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dense
    ├── losses.py    — Softmax, CrossEntropyLoss
    ├── model.py     — CNN (train, evaluate, save_weights, load_weights, predict)
    ├── tune_cnn.py  — réglages interactifs des hyperparamètres
    └── cnn.py       — script principal de démo
```

## 🚀 Utilisation

### 1. Installer les dépendances

```bash
pip install numpy matplotlib
```

### 2. Lancer les tests + entraînement rapide

```bash
python src/cnn.py
```

- Teste toutes les couches une par une (forward, backward, gradient check)
- Entraîne sur **2000 images** (3 epochs)
- Sauvegarde les poids dans `model_weights.npz`
- Génère le graphique `training_result.png`

### 3. Entraînement complet (recommandé)

```bash
python src/cnn.py --full
```

- Entraîne sur **les 60000 images** MNIST (10 epochs, ~15-20 min)
- Sauvegarde les poids dans `model_weights_full.npz`
- Génère `training_result_full.png`

Options supplémentaires :
```bash
python src/cnn.py --full --epochs 15      # 15 epochs au lieu de 10
python src/cnn.py --train-only            # saute les tests, entraîne direct
```

### 4. Prédire sans réentraîner

```bash
python predict.py                          # 10 prédictions → predictions.png
python predict.py --all                    # accuracy sur les 10000 images de test
python predict.py --weights model_weights_full.npz   # choisir les poids
python predict.py --interactive            # mode pas à pas avec affichage
```

## ⚙️ Tuning interactif

```bash
python src/tune_cnn.py
```

Ouvre un fichier de réglages où tu peux modifier :
- `LEARNING_RATE` (0.1, 0.01, 0.001…)
- `BATCH_SIZE` (32, 64, 128…)
- `EPOCHS` (5, 10, 20…)
- `DATA_LIMIT` (2000, 5000, None pour tout)
- Architecture du réseau

Résultat sauvegardé dans `tune_result.png`.

## 💡 Exemple rapide

```python
from predict import load_model

model = load_model("model_weights_full.npz")
pred = model.predict(mon_image)   # mon_image: (1, 28, 28) normalisée
print(f"Prédiction : {pred}")
```

---

**#NoFrameworks #FromScratch #MNIST**
