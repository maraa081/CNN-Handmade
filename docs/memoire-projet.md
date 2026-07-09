# 📓 Mémoire de projet — CNN-Handmade

_Carnet de bord du projet, mis à jour à chaque étape significative._

---

## 🧠 Points à revoir / consolider

> Concepts vus une fois, mais qui méritent d'être retravaillés pour être bien ancrés.

### im2col / col2im

- **im2col** : transforme un batch d'images `(N, C, H, W)` en une matrice de colonnes `(N·H_out·W_out, C·k·k)`. Chaque patch (fenêtre de convolution) est aplati en une ligne de la matrice. Permet de faire la convolution comme un simple produit matriciel.
- **col2im** : opération inverse. Chaque ligne de `cols` est remise à sa position spatiale dans l'image. **Les chevauchements sont importants** : un pixel peut contribuer à plusieurs positions de sortie (surtout quand stride < kernel_size). Donc col2im SOMME les gradients aux positions qui se chevauchent — c'est normal et nécessaire.

### Partage de poids en convolution

- Un même filtre (kernel) est réutilisé à **toutes les positions spatiales** de l'image.
- En backprop, ça signifie que le gradient du filtre est la **somme** des gradients sur toutes les positions où il a été appliqué.
- Même principe pour le biais : ajouté à chaque position → gradient = somme sur toutes les positions.
- Pour col2im : un pixel apparaît dans plusieurs patchs → son gradient cumule les contributions de toutes les positions où il a été utilisé.

### Routage de gradient pour MaxPool

- Pendant le **forward**, on stocke `max_indices` : pour chaque fenêtre, l'index pixel qui a la valeur maximale.
- Pendant le **backward**, seul ce pixel reçoit le gradient (les autres reçoivent 0).
- Logique : seul le max a contribué à la sortie, donc seul lui mérite le gradient. C'est ce qui rend MaxPool non-linéaire.
- Les indices stockés permettent d'éviter de refaire le calcul du max au backward.

---

## 🏁 Expérimentations — Comparaison d'optimiseurs

> Structure en dossiers séparés pour tester et comparer chaque optimisation.

### Principe

Chaque dossier `experiments/<nom>/` contient :
- `tune_cnn.py` — script d'entraînement indépendant
- `tune_result.png` — graphiques loss + accuracy
- `model_weights.npz` — poids sauvegardés

Tous utilisent la **même architecture**, les **mêmes données**, seul l'optimiseur change.

### Optimiseurs disponibles

| Optimiseur | Fichier | Paramètres | Statut |
|---|---|---|---|
| **SGD** (vanilla) | `experiments/baseline/` | `lr` | ✅ |
| **Momentum** (SGD + élan) | `experiments/momentum/` | `lr, momentum` | ✅ |
| **Adam** (Adaptive Moment Estimation) | `experiments/adam/` | `lr, beta1, beta2, eps` | ✅ |

### Prochaines expériences

| Expérience | Statut |
|---|---|
| Learning Rate Scheduler | ⏳ |
| Dropout (régularisation) | ✅ |
| Weight Decay (L2) | ✅ |
| Grid Search automatique | ⏳ |
| Data Augmentation | ⏳ |

### Architecture du framework

- `src/optimizers.py` : définition des optimiseurs (SGD, Momentum, Adam)
- `model.py` : `CNN(optimizer=...)` — l'optimiseur est passé au constructeur
- Chaque optimiseur implémente `update(layers, lr=None)`

---

## ⏭️ Prochaines étapes (code)

> Partie plus mécanique, moins de théorie, avancer sur l'implémentation.

| Étape | Statut |
|---|---|
| **Refactoring modules** | ✅ |
| Dense (forward, backward, update) | ✅ |
| Softmax | ✅ |
| CrossEntropyLoss | ✅ |
| Boucle d'entraînement (forward → loss → backward → update) | ✅ |
| Évaluation / accuracy | ✅ |
| **Framework d'expérimentations** (optimiseurs) | ✅ |
| SGD (baseline) | ✅ |
| Momentum | ✅ |
| Adam | ✅ |

---

## 🔧 Refactoring — fait ✅

`cnn.py` découpé en modules propres :

| Module | Contenu |
|---|---|
| **`data.py`** | MNISTLoader, preprocessing, DataLoader |
| **`layers.py`** | im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dense ✅ |
| **`losses.py`** | Softmax ✅, CrossEntropyLoss ✅ |
| **`model.py`** | CNN (forward, backward, update, train, evaluate) ✅ |
| **`tune_cnn.py`** | fichier de tuning interactif (paramètres en haut) ✅ |

`cnn.py` est devenu un simple script de démonstration qui importe les modules.

---

## 📜 Historique

### 2026-07-05 — Entraînement + évaluation + fichier tuning 🎉

- **model.py** : CNN.forward, backward, update, train (avec historique), evaluate.
- **tune_cnn.py** : fichier de tuning ultra-simple avec tous les paramètres en haut.
- **Optimisation :** im2col/col2im vectorisés avec `numpy.lib.stride_tricks.as_strided` (fini les boucles Python).
- **Données :** format channels_first dans le DataLoader (transparent pour l'utilisateur).
- Tests : loss décroissante, accuracy croissante ✅ sur un mini-entraînement.

### 2026-07-05 — Softmax + CrossEntropyLoss implémentés ✅

- **Softmax :** forward stable (shift max), backward avec Jacobienne complète.
- **CrossEntropyLoss :** prend des **logits** (pas des probas), softmax intégré en interne.
- **Gradient combiné magique :** `(softmax(logits) - y_true) / N` — pas besoin de multiplier les Jacobiennes.
- Tests : loss manuelle, logits uniformes → log(C), logits parfaits → 0, gradient check ✅
- Ajouté `accuracy()` directement dans CrossEntropyLoss.

### 2026-07-05 — Dense implémenté ✅

- Implémentation de `Dense` (forward, backward, update).
- Forward : `y = x @ W.T + b`
- Backward : `dW = d_out.T @ x`, `db = d_out.sum(axis=0)`, `dx = d_out @ W`
- Gradient check par différences finies ✅ (erreur relative < 1e-4).
- Tests unitaires dans cnn.py.

### 2026-07-04 — Refactoring en modules

- Découpage de `cnn.py` en `data.py`, `layers.py`, `losses.py`, `model.py`
- `cnn.py` réécrit comme script de démo/test
- Ajout du stub `Dense` dans layers (prêt à implémenter)
- README mis à jour avec la structure

### 2026-07-04 — Conv2D backward + col2im

### 2026-07-04 — Conv2D backward + col2im

- Implémentation de `col2im()` : inverse d'`im2col`, accumulation des gradients aux positions qui se chevauchent.
- `Conv2D.backward()` : calcule `d_kernels`, `d_bias`, `d_input`.
- `Conv2D.update(lr)` : mise à jour des poids par descente de gradient.
- Gradient check par différences finies ✅ (erreur relative < 1e-4).
- README mis à jour (ReLU, Flatten, Conv2D backward marqués ✅).
- Création de ce fichier mémoire.

### 2026-07-03 — ReLU + Flatten

- Implémentation de `ReLU` (forward + backward avec masque).
- Implémentation de `Flatten` (forward + backward par simple reshape).
- Tests unitaires dans le `__main__`.

### Dates antérieures — Fondations

- Chargement MNIST (IDX) + preprocessing pipeline.
- Conv2D forward (im2col + produit matriciel).
- MaxPool2D forward + backward.
