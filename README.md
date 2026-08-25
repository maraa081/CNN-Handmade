#  CNN Handmade

**Un réseau de neurones convolutionnel pour reconnaître les chiffres manuscrits (MNIST), fait à la main, de A à Z.**

Pas de TensorFlow, pas de PyTorch, pas de Keras. Juste Python, NumPy, et moi. 

##  Pourquoi ?

Comprendre chaque brique du deep learning en la codant soi-même — im2col, rétropropagation, descente de gradient… plutôt que d'appeler une API magique.

## OK Ce qui est implémenté

| Module | Statut |
|---|---|
| **MNISTLoader** (fichiers IDX bruts) | OK |
| **Preprocessing** (normalisation, one-hot, DataLoader) | OK |
| **Conv2D** (forward par im2col, backward, update) | OK |
| **MaxPool2D** (forward + backward avec indices) | OK |
| **ReLU** (forward + backward) | OK |
| **Flatten** (forward + backward) | OK |
| **Dropout** (régularisation par désactivation aléatoire) | OK |
| **Dense / Fully Connected** (forward + backward + update) | OK |
| **Softmax** (forward + backward) | OK |
| **CrossEntropyLoss** (forward + backward + accuracy) | OK |
| **Training loop** (CNN.train + train/eval mode) | OK |
| **Évaluation** (accuracy sur test set) | OK |
| **Sauvegarde / Chargement des poids** | OK |
| **Predict** (classifier une image chargée) | OK |
| **Graphiques d'entraînement** (loss + accuracy) | OK |
| **Optimiseur SGD** (vanilla + weight_decay) | OK |
| **Optimiseur Momentum** (SGD + élan + weight_decay) | OK |
| **Optimiseur Adam** (lr adaptatif + momentum + weight_decay) | OK |
| **Framework d'expérimentations** (comparaison d'optimiseurs) | OK |

##  Architecture

```
Entrée : (N, 1, 28, 28)
    |
    |-- Conv2D   (1 -> 32,  kernel=3, pad=1)     ->  (N, 32, 28, 28)
    |-- ReLU
    |-- MaxPool2D  (2×2, stride=2)               ->  (N, 32, 14, 14)
    |
    |-- Conv2D   (32 -> 64, kernel=3, pad=1)      ->  (N, 64, 14, 14)
    |-- ReLU
    |-- MaxPool2D  (2×2, stride=2)               ->  (N, 64, 7, 7)
    |
    |-- Flatten                                  ->  (N, 3136)
    |-- Dense   (3136 -> 128)
    |-- ReLU
    |-- Dropout(p=0.5)          (optionnel)
    |-- Dense   (128 -> 10)
    `-- Softmax                                  ->  (N, 10)
```

##  Structure du projet

```
CNN-Handmade/
|-- README.md
|-- requirements.txt
|-- push.sh                        <- git push rapide
|-- src/                           <- cœur du code (CNN from scratch)
|   |-- __init__.py
|   |-- data.py        — MNISTLoader, EMNISTLoader, preprocessing, DataLoader
|   |-- layers.py      — im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dropout, Dense
|   |-- losses.py      — Softmax, CrossEntropyLoss
|   |-- optimizers.py  — SGD, Momentum, Adam
|   |-- model.py       — CNN (optimiseur interchangeable, train/eval mode)
|   |-- tune_cnn.py    — réglages interactifs (baseline)
|   `-- cnn.py         — script principal de démo + entraînement
|-- scripts/                      <- scripts utilisateur
|   |-- predict.py        — charger le modèle et classifier
|   |-- train_emnist.py   — entraîner sur EMNIST letters
|   |-- download_emnist.py— installer les données EMNIST
|   `-- voir_emnist.py    — visualiser les lettres (Spyder/IPython)
|-- models/                      <- poids entraînés (.npz)
|   |-- model_weights.npz          — rapide (2000 img, 3 epochs)
|   |-- model_weights_full.npz     — complet (60000 img)
|   `-- max_config_weights.npz     — Adam + Dropout + L2
|-- results/                     <- graphiques générés (PNG/CSV)
|-- adversarial/                 <- sécurité IA : attaques & défense
|   |-- README.md / memoire.md
|   `-- scripts/ (fgsm.py, pgd.py, transfer.py, defend.py)
|-- docs/
|   |-- data-flow.md
|   `-- memoire-projet.md          <- carnet de bord du projet
|-- data/
|   `-- (fichiers MNIST/EMNIST .ubyte)
|-- traces/
|   `-- forward_trace.py
`-- experiments/                    <- comparatifs d'optimiseurs
    |-- baseline/  momentum/  adam/  dropout/  l2/
    |-- max_config.py          —  Adam + Dropout + L2 combinés
    `-- compare_all.py         — lancer tous les optimiseurs d'un coup
```

##  Utilisation

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
- Sauvegarde les poids dans `models/model_weights.npz`
- Génère le graphique `training_result.png`

### 3. Entraînement complet (recommandé)

```bash
python src/cnn.py --full
```

- Entraîne sur **les 60000 images** MNIST (10 epochs, ~15-20 min)
- Sauvegarde les poids dans `models/model_weights_full.npz`
- Génère `training_result_full.png`

Options supplémentaires :
```bash
python src/cnn.py --full --epochs 15      # 15 epochs au lieu de 10
python src/cnn.py --train-only            # saute les tests, entraîne direct
```

### 4. Prédire sans réentraîner

```bash
python scripts/predict.py                          # 10 prédictions -> results/predictions.png
python scripts/predict.py --all                    # accuracy sur les 10000 images de test
python scripts/predict.py --weights models/model_weights_full.npz   # choisir les poids
python scripts/predict.py --interactive            # mode pas à pas avec affichage
```

##  Tuning interactif

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

##  EMNIST — Les lettres (26 classes)

Le même CNN from-scratch, mais pour reconnaître les **lettres manuscrites a-z** au lieu des chiffres.

```bash
# Vérifier l'orientation des images (échantillons -> results/emnist_samples.png)
python3 scripts/train_emnist.py --samples

# Entraînement rapide (5000 images, 3 epochs, ~4 min)
python3 scripts/train_emnist.py

# Entraînement complet (124800 images, 10 epochs)
python3 scripts/train_emnist.py --full
```

**Données :** [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) — même format IDX que MNIST. Un zip léger des lettres (~36 Mo) est inclus dans le repo ; le script les installe tout seul :

```bash
python3 scripts/download_emnist.py
```

Si le zip local n'est pas là, il télécharge automatiquement depuis le site NIST (~561 Mo, plus lent).

**Ce qui change vs MNIST :**
- `EMNISTLoader` dans `src/data.py` (labels 1-26 -> 0-25, images pivotées remises à l'endroit)
- `Dense(128 -> 26)` au lieu de `Dense(128 -> 10)`
- `preprocess_pipeline(..., num_classes=26)` pour le one-hot

**Résultat rapide** (5000 images, 3 epochs) : **43.2% test** (hasard = 3.8%). L'entraînement complet vise ~90%.

> [warn] **Piège** : le test set EMNIST est trié par classe — échantillonner aléatoirement pour évaluer, jamais `x_test[:N]`.

##  Expérimentations — Comparer les techniques

Chaque dossier dans `experiments/` est un test indépendant. **Même architecture, mêmes données, seule la technique change.**

### Lancer un test

```bash
# Optimiseurs
python experiments/baseline/baseline_sgd.py
python experiments/momentum/momentum.py
python experiments/adam/adam.py

# Régularisation
python experiments/dropout/dropout_sgd.py     # Dropout(p=0.5)
python experiments/l2/l2_sgd.py              # Weight decay L2(0.001)
```

### Prochaines expériences prévues

| Expérience | Statut |
|---|---|
| Baseline (SGD) | OK |
| Momentum | OK |
| Adam | OK |
| Dropout (régularisation) | OK |
| Weight Decay (L2) | OK |
| Learning Rate Scheduler | [wait] |
| Grid Search automatique | [wait] |
| Max Config (Adam + Dropout + L2) | OK |
| Data Augmentation | [wait] |

### Ajouter une nouvelle expérience

1. Crée `experiments/mon_opti/mon_opti.py`
2. Importe ton optimiseur depuis `src/optimizers.py` (ou crée-le là)
3. Si tu ajoutes des couches (Dropout, etc.), importe-les depuis `src/layers.py`
4. Passe l'optimiseur au modèle : `model = CNN(optimizer=MonOpti(lr=...))`
5. Lance et compare les graphiques !

##  Les optimiseurs — explications

Les optimiseurs sont dans `src/optimizers.py`. Chacun implémente une méthode `update(layers, lr=None)`.

### SGD — La base
```python
θ <- θ - lr · ∇θ
```
Chaque paramètre est mis à jour dans la direction opposée au gradient. Simple, stable, mais peut être lent.

### Momentum — Avec élan
```python
v <- α · v - lr · ∇θ
θ <- θ + v
```
On accumule une "vitesse" qui lisse les oscillations et accélère la convergence. Le coefficient `α` (typiquement 0.9) contrôle l'inertie.

### Dropout — Désactivation aléatoire
```python
# Entraînement : masque binaire × scaling
masque ~ Bernoulli(1-p)
sortie = entrée × masque / (1-p)

# Évaluation : passe-through
sortie = entrée
```
Empêche la co-adaptation des neurones. Force le réseau à apprendre des représentations redondantes. Agit comme un ensemble de sous-réseaux. `p=0.5` pour les Dense, `p=0.2-0.3` pour les Conv.

### L2 Weight Decay — Pénalise les gros poids
```python
∇θ_effectif = ∇θ + λ · θ
θ <- θ - lr · ∇θ_effectif
```
Ajoute une pénalité quadratique sur les poids. Les poids trop grands sont tirés vers zéro. Revient à chercher des solutions plus simples. `λ` typique : 0.0001 ~ 0.001.

### Adam — Le champion
```python
m <- β1 · m + (1 - β1) · g         (moyenne des gradients)
v <- β2 · v + (1 - β2) · g²        (variance des gradients)
θ <- θ - lr · m / (√v + ε)
```
Combine le momentum avec un learning rate adaptatif par paramètre. Le plus robuste — moins besoin de tuner le lr. Supporte aussi le weight_decay.

##  Exemple rapide

```python
from scripts.predict import load_model

model = load_model("models/model_weights_full.npz")
pred = model.predict(mon_image)   # mon_image: (1, 28, 28) normalisée
print(f"Prédiction : {pred}")
```

##  Ce qu'on a appris (pour la suite)

### Résultats clés (5 epochs, 2000 images)

| Technique | Test Acc | Temps | Verdict |
|---|---|---|---|
| **SGD** (baseline) | 88.4% | 20s | Référence |
| **Momentum** | 94.2% | 21s | +6% pour presque rien |
| **Adam** | 95.5% | 24s |  Meilleur seul |
| SGD + Dropout | 83.3% | 23s | À réserver aux gros datasets |
| SGD + L2 | 86.4% | 20s | Idem, peu utile sur MNIST |
| **Max Config** (10 epochs) | **96.0%** | 49s | Tout cumulé, meilleur long terme |

### Ce qu'il faut retenir

1. **Adam suffit sur MNIST** — 99% sur les 60000 images, pas besoin de régularisation
2. **Dropout + L2** -> utiles sur des vrais problèmes (overfitting), pas sur MNIST propre
3. **Le meilleur rapport perf/simplicité** : juste Adam
4. **Max Config** utile quand on monte en epochs ou en données

### Prochaines pistes

- **LR Scheduler** — réduire le lr en cours d'entraînement (step decay, cosine annealing)
- **Grid Search** — trouver automatiquement les meilleurs hyperparamètres
- **Data Augmentation** — rotations, décalages pour généraliser
- **Entraînement complet** (60000 images, 20 epochs) -> viser 99%+

---

**#NoFrameworks #FromScratch #MNIST**
