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
| **Optimiseur SGD** (vanilla) | ✅ |
| **Optimiseur Momentum** (SGD + élan) | ✅ |
| **Optimiseur Adam** (lr adaptatif + momentum) | ✅ |
| **Framework d'expérimentations** (comparaison d'optimiseurs) | ✅ |

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
├── predict.py                     ← charge le modèle et classifie
├── push.sh                        ← git push rapide
├── docs/
│   ├── data-flow.md
│   └── memoire-projet.md          ← carnet de bord du projet
├── data/
│   └── (fichiers MNIST .ubyte)
├── traces/
│   └── forward_trace.py
├── src/
│   ├── __init__.py
│   ├── data.py        — MNISTLoader, preprocessing, DataLoader
│   ├── layers.py      — im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dense
│   ├── losses.py      — Softmax, CrossEntropyLoss
│   ├── optimizers.py  — SGD, Momentum, Adam              ← NOUVEAU
│   ├── model.py       — CNN (optimiseur interchangeable) ← MODIFIÉ
│   ├── tune_cnn.py    — réglages interactifs (baseline)
│   └── cnn.py         — script principal de démo
│
└── experiments/                    ← NOUVEAU : comparatifs d'optimiseurs
    ├── baseline/
    │   ├── tune_cnn.py            — SGD vanilla
    │   ├── tune_result.png        — graphiques
    │   └── model_weights.npz      — poids entraînés
    │
    ├── momentum/
    │   ├── tune_cnn.py            — SGD + élan
    │   ├── tune_result.png
    │   └── model_weights.npz
    │
    ├── adam/
    │   ├── tune_cnn.py            — Adaptive Moment Estimation
    │   ├── tune_result.png
    │   └── model_weights.npz
    │
    └── (prochains : dropout, lr_scheduler, data_augmentation…)
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

Paramètres réglables :
- `LEARNING_RATE` (0.1, 0.01, 0.001…)
- `BATCH_SIZE` (32, 64, 128…)
- `EPOCHS` (5, 10, 20…)
- `DATA_LIMIT` (2000, 5000, None pour tout)
- Architecture du réseau

Résultat sauvegardé dans `tune_result.png`.

## 🔬 Expérimentations — Comparer les optimiseurs

Chaque dossier dans `experiments/` est un test indépendant avec un optimiseur différent. **Même architecture, mêmes données, seul l'optimiseur change.**

### Lancer un test

```bash
# Baseline (SGD vanilla)
python experiments/baseline/tune_cnn.py

# Momentum (SGD + élan)
python experiments/momentum/tune_cnn.py

# Adam (Adaptive Moment Estimation)
python experiments/adam/tune_cnn.py
```

Chaque script produit :
- Un graphique `tune_result.png` (loss + accuracy)
- Un fichier `model_weights.npz` (poids du modèle)

### Résultats attendus (indicatifs, 5 epochs, 2000 images)

| Optimiseur | lr | Loss finale | Accuracy | Convergence |
|---|---|---|---|---|
| **SGD** | 0.01 | ~0.60 | ~80% | Lente mais stable |
| **Momentum** | 0.01 | ~0.35 | ~88% | Plus rapide que SGD |
| **Adam** | 0.001 | ~0.15 | ~95% | Rapide et robuste |

> ℹ️ Résultats variables selon les initialisations aléatoires — l'ordre de grandeur est là.

### Prochaines expériences prévues

| Expérience | Statut |
|---|---|
| Baseline (SGD) | ✅ |
| Momentum | ✅ |
| Adam | ✅ |
| Learning Rate Scheduler | ⏳ |
| Dropout (régularisation) | ⏳ |
| Weight Decay (L2) | ⏳ |
| Grid Search automatique | ⏳ |
| Data Augmentation | ⏳ |

### Modifier le code d'un test

Ouvre le fichier `tune_cnn.py` dans le dossier de l'expérience. Les réglages sont en haut du fichier. Tu peux aussi changer l'architecture du réseau plus bas.

### Ajouter une nouvelle expérience

1. Crée `experiments/mon_opti/tune_cnn.py`
2. Importe ton optimiseur depuis `src/optimizers.py` (ou crée-le là)
3. Passe-le au modèle : `model = CNN(optimizer=MonOpti(lr=...))`
4. Lance et compare les graphiques !

## 🔧 Les optimiseurs — explications

Les optimiseurs sont dans `src/optimizers.py`. Chacun implémente une classe avec une méthode `update(layers, lr=None)`.

### SGD — La base
```python
θ ← θ - lr · ∇θ
```
Chaque paramètre est mis à jour dans la direction opposée au gradient. Simple, stable, mais peut être lent.

### Momentum — Avec élan
```python
v ← α · v - lr · ∇θ
θ ← θ + v
```
On accumule une "vitesse" qui lisse les oscillations et accélère la convergence. Le coefficient `α` (typiquement 0.9) contrôle l'inertie.

### Adam — Le champion
```python
m ← β1 · m + (1 - β1) · g         (moyenne des gradients)
v ← β2 · v + (1 - β2) · g²        (variance des gradients)
θ ← θ - lr · m̂ / (√v̂ + ε)
```
Combine le momentum avec un learning rate adaptatif par paramètre. Le plus robuste — moins besoin de tuner le lr.

## 💡 Exemple rapide

```python
from predict import load_model

model = load_model("model_weights_full.npz")
pred = model.predict(mon_image)   # mon_image: (1, 28, 28) normalisée
print(f"Prédiction : {pred}")
```

---

**#NoFrameworks #FromScratch #MNIST**
