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

## ⏭️ Prochaines étapes (code)

> Partie plus mécanique, moins de théorie, avancer sur l'implémentation.

| Étape | Statut |
|---|---|
| Dense (forward, backward, update) | ❌ |
| Softmax | ❌ |
| CrossEntropyLoss | ❌ |
| Boucle d'entraînement (forward → loss → backward → update) | ❌ |
| Évaluation / accuracy | ❌ |

---

## 🔧 Refactoring à prévoir

`cnn.py` va devenir trop gros. Il faudra découper en modules :

- **`layers.py`** — Conv2D, ReLU, MaxPool2D, Flatten, Dense
- **`losses.py`** — Softmax, CrossEntropyLoss
- **`data.py`** — MNISTLoader, preprocessing, DataLoader
- **`model.py`** — classe CNN qui assemble tout

---

## 📜 Historique

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
