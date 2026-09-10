#  Mémoire de projet — CNN-Handmade

_Carnet de bord du projet, mis à jour à chaque étape significative._

---

##  Points à revoir / consolider

> Concepts vus une fois, mais qui méritent d'être retravaillés pour être bien ancrés.

### im2col / col2im

- **im2col** : transforme un batch d'images `(N, C, H, W)` en une matrice de colonnes `(N·H_out·W_out, C·k·k)`. Chaque patch (fenêtre de convolution) est aplati en une ligne de la matrice. Permet de faire la convolution comme un simple produit matriciel.
- **col2im** : opération inverse. Chaque ligne de `cols` est remise à sa position spatiale dans l'image. **Les chevauchements sont importants** : un pixel peut contribuer à plusieurs positions de sortie (surtout quand stride < kernel_size). Donc col2im SOMME les gradients aux positions qui se chevauchent — c'est normal et nécessaire.

### Partage de poids en convolution

- Un même filtre (kernel) est réutilisé à **toutes les positions spatiales** de l'image.
- En backprop, ça signifie que le gradient du filtre est la **somme** des gradients sur toutes les positions où il a été appliqué.
- Même principe pour le biais : ajouté à chaque position -> gradient = somme sur toutes les positions.
- Pour col2im : un pixel apparaît dans plusieurs patchs -> son gradient cumule les contributions de toutes les positions où il a été utilisé.

### Routage de gradient pour MaxPool

- Pendant le **forward**, on stocke `max_indices` : pour chaque fenêtre, l'index pixel qui a la valeur maximale.
- Pendant le **backward**, seul ce pixel reçoit le gradient (les autres reçoivent 0).
- Logique : seul le max a contribué à la sortie, donc seul lui mérite le gradient. C'est ce qui rend MaxPool non-linéaire.
- Les indices stockés permettent d'éviter de refaire le calcul du max au backward.

---

##  Expérimentations — Comparaison d'optimiseurs

> Structure en dossiers séparés pour tester et comparer chaque optimisation.

### Principe

Chaque dossier `experiments/<nom>/` contient :
- `<technique>.py` — script d'entraînement (ex: `baseline_sgd.py`, `adam.py`)
- `<technique>.png` — graphiques loss + accuracy (ex: `adam.png`)
- `<technique>_weights.npz` — poids sauvegardés (ex: `adam_weights.npz`)

Tous utilisent la **même architecture**, les **mêmes données**, seul l'optimiseur change.

### Optimiseurs disponibles

| Optimiseur | Fichier | Paramètres | Statut |
|---|---|---|---|
| **SGD** (vanilla) | `experiments/baseline/` | `lr` | OK |
| **Momentum** (SGD + élan) | `experiments/momentum/` | `lr, momentum` | OK |
| **Adam** (Adaptive Moment Estimation) | `experiments/adam/` | `lr, beta1, beta2, eps` | OK |

### Prochaines expériences

| Expérience | Statut |
|---|---|
| Learning Rate Scheduler | [wait] |
| Dropout (régularisation) | OK |
| Weight Decay (L2) | OK |
| Grid Search automatique | [wait] |
| Data Augmentation | [wait] |

### Architecture du framework

- `src/optimizers.py` : définition des optimiseurs (SGD, Momentum, Adam)
- `model.py` : `CNN(optimizer=...)` — l'optimiseur est passé au constructeur
- Chaque optimiseur implémente `update(layers, lr=None)`

---

## >> Prochaines étapes (code)

> Partie plus mécanique, moins de théorie, avancer sur l'implémentation.

| Étape | Statut |
|---|---|
| **Refactoring modules** | OK |
| Dense (forward, backward, update) | OK |
| Softmax | OK |
| CrossEntropyLoss | OK |
| Boucle d'entraînement (forward -> loss -> backward -> update) | OK |
| Évaluation / accuracy | OK |
| **Framework d'expérimentations** (optimiseurs) | OK |
| SGD (baseline) | OK |
| Momentum | OK |
| Adam | OK |

---

##  Refactoring — fait OK

`cnn.py` découpé en modules propres :

| Module | Contenu |
|---|---|
| **`data.py`** | MNISTLoader, preprocessing, DataLoader |
| **`layers.py`** | im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dropout, Dense OK |
| **`optimizers.py`** | SGD, Momentum, Adam (tous avec L2 weight_decay) OK |
| **`losses.py`** | Softmax OK, CrossEntropyLoss OK |
| **`model.py`** | CNN (forward, backward, update, train, evaluate) OK |
| **`tune_cnn.py`** | fichier de tuning interactif (paramètres en haut) OK |

`cnn.py` est devenu un simple script de démonstration qui importe les modules.

---

##  Historique

### 2026-09-10 - Optimisation du moteur NumPy (x2.4, gratuit)

Le GPU n'est pour rien dans cette histoire : le code etait juste lent a cause de
trois erreurs classiques. Mesures sur le i5-6300U (4 coeurs) :

| Etape | Debit | Gain |
|---|---|---|
| depart | 73 img/s | - |
| + produits matriciels sur tableaux contigus | 101 img/s | x1.38 |
| + col2im sans `np.add.at` | 102 img/s | ~ |
| + float32 partout (au lieu de float64) | 172 img/s | x1.69 |
| **total** | **172 img/s** | **x2.4** |

1. **Produits matriciels sur vues transposees.** `A @ B.T` ou `B.T` est une vue
   (F-contigue, sans copie) tombe dans un chemin lent de NumPy. Mesure sur la
   conv2 : **271 ms au lieu de 50 ms**, soit x5.4. Correction :
   `A @ np.ascontiguousarray(B.T)`. Resultat **bit-identique** (ecart 0.0).
2. **`col2im` sans `np.add.at`.** `np.add.at` est un chemin generique tres lent
   (**43 ms par appel**, 13 % du temps total). Remplace par une boucle sur les
   positions du noyau avec des `+=` vectorises (la destination d'une position
   donnee ne se chevauche pas).
3. **`float32` partout.** Les poids etaient crees en float64 (`np.random.randn`
   sans `astype`) alors que les donnees sont en float32 -> **tous** les produits
   matriciels tournaient en double precision. Les fichiers de poids passent de
   3296 Ko a 1650 Ko.

Consequences : epoch propre sur 60000 images de 13.7 min -> **5.8 min** ; epoch
PGD-5 de ~90 min -> **~41 min**.

Verifications : logits bit-identiques apres (1) et (2) ; accuracy propre sur
l'echantillon de 500 images **98.60 %**, exactement la valeur documentee ;
gradient check conv2d ecart 8.5e-11 ; FGSM bout-en-bout inchange (1.5 % a
eps=0.3).

> [warn] Piege : lancer `python3 src/cnn.py` (le script de demo) **ecrase**
> `models/model_weights.npz`, qui est le modele "classic" utilise par les
> experiences de transfert. Toujours travailler sur une copie.

Ce qui reste : le moteur est desormais proche de la limite du BLAS de cette
machine. Au-dela, il faut passer a autre chose qu'un tableau NumPy sur CPU
(PyTorch / GPU), pas micro-optimiser.

### 2026-08-26 — Docs adversariales + résultats EMNIST full

- `adversarial/attacks.md` (théorie FGSM / PGD / ciblées / transfert, threat model)
  et `adversarial/defenses.md` (min-max de Madry, limites, méthodologie) écrits.
- PGD vs FGSM sur EMNIST full (26 lettres, 92.0% test) : PGD tombe à **0.0% dès
  eps=0.20** (FGSM 10.0%). EMNIST est plus fragile que MNIST (44.4% à eps=0.10).
- 7 courbes versionnées dans `adversarial/results/`.
- Nettoyage : flèches unicode remplacées par `->` dans les deux nouveaux docs.

### 2026-08-25 — Défenses : adversarial training puis version durcie

- `defend.py` : adversarial training FGSM (eps 0.15), avec comparaison ÉQUITABLE
  à données égales (5000 images identiques) -> gain de +5 à +18.4 pts partout.
  Limite : ne tient pas face à PGD (à eps=0.05 le défendu est même pire que le
  standard). FGSM training est une base, pas une fin.
- `harden.py` : version durcie en 3 couches (adversarial training PGD 7 steps
  eps 0.3, feature squeezing, évaluation multi-attaques).
  Résultats : clean 67.2% (vs 98.6%), FGSM eps=0.30 -> 23.8% (vs 1.8%),
  PGD eps=0.20 -> 9.6% (vs 0.0%), attaques ciblées tenues à 49-64%.
- Leçon : une défense se prouve contre l'attaquant le plus fort, pas contre la
  version la plus simple de l'attaque. Prix payé : l'accuracy propre.

### 2026-08-24 — Attaques adversariales (FGSM, PGD, transfert)

- `adversarial/fgsm.py` opérationnel : MNIST full 98.5% -> **2.1%** à eps=0.30.
- `adversarial/pgd.py` : **0.0% dès eps=0.20** — PGD écrase FGSM (21.6% à eps=0.20) :
  FGSM sous-estime la vulnérabilité réelle.
- `adversarial/transfer.py` : transfert full -> classic jusqu'à **72.3%** à eps=0.30
  (attaque boîte noire viable) ; la régularisation (Dropout + L2) freine le transfert.
- EMNIST full entraîné (124 800 images, 10 epochs, 92.0% test) sur la machine de Maraa.
- Leçon : le CNN from scratch est vulnérable, comme tout modèle linéaire en grande
  dimension (Goodfellow 2014).

### 2026-08-24 — EMNIST Letters (26 classes) 

- **Objectif** : passer de MNIST (10 chiffres) aux lettres manuscrites a-z.
- **EMNISTLoader** ajouté dans `src/data.py` (hérite de MNISTLoader, même format IDX) :
  - Labels 1-26 -> 0-25 (mapping vérifié avec `emnist-letters-mapping.txt`)
  - Orientation corrigée (transpose + flip = rotation 90° — les images EMNIST sont stockées pivotées)
  - Orientation validée visuellement en ASCII art avant entraînement
- **`preprocess_pipeline()`** : paramètre `num_classes` ajouté (one-hot 26 au lieu de 10)
- **`scripts/train_emnist.py`** : script dédié (rapide 5000 images / `--full` 124800 images, `--samples` pour vérifier l'orientation)
- **Résultat rapide** (5000 images, 3 epochs, ~3m40) : train 63.5% -> test **43.2%**
- [warn] **Piège découvert** : le test set EMNIST est TRIÉ PAR CLASSE. `x_test[:1000]` ne contenait que des 'a'/'b' -> accuracy 5.8% trompeuse. Il faut échantillonner aléatoirement (rng.choice).
- Données dans `data/emnist/` (gitignorées, zip 561 Mo à télécharger depuis biometrics.nist.gov)

### 2026-07-10 — Fiches optimiseurs

- `docs/optimizers/` : fiches détaillées SGD, Momentum, Adam + TEMPLATE pour les nouveaux.

### 2026-07-05 — Entraînement + évaluation + fichier tuning 

- **model.py** : CNN.forward, backward, update, train (avec historique), evaluate.
- **tune_cnn.py** : fichier de tuning ultra-simple avec tous les paramètres en haut.
- **Optimisation :** im2col/col2im vectorisés avec `numpy.lib.stride_tricks.as_strided` (fini les boucles Python).
- **Données :** format channels_first dans le DataLoader (transparent pour l'utilisateur).
- Tests : loss décroissante, accuracy croissante OK sur un mini-entraînement.

### 2026-07-05 — Softmax + CrossEntropyLoss implémentés OK

- **Softmax :** forward stable (shift max), backward avec Jacobienne complète.
- **CrossEntropyLoss :** prend des **logits** (pas des probas), softmax intégré en interne.
- **Gradient combiné magique :** `(softmax(logits) - y_true) / N` — pas besoin de multiplier les Jacobiennes.
- Tests : loss manuelle, logits uniformes -> log(C), logits parfaits -> 0, gradient check OK
- Ajouté `accuracy()` directement dans CrossEntropyLoss.

### 2026-07-05 — Dense implémenté OK

- Implémentation de `Dense` (forward, backward, update).
- Forward : `y = x @ W.T + b`
- Backward : `dW = d_out.T @ x`, `db = d_out.sum(axis=0)`, `dx = d_out @ W`
- Gradient check par différences finies OK (erreur relative < 1e-4).
- Tests unitaires dans cnn.py.

### 2026-07-04 — Refactoring en modules

- Découpage de `cnn.py` en `data.py`, `layers.py`, `losses.py`, `model.py`
- `cnn.py` réécrit comme script de démo/test
- Ajout du stub `Dense` dans layers (prêt à implémenter)
- README mis à jour avec la structure

### 2026-07-04 — Conv2D backward + col2im

- Implémentation de `col2im()` : inverse d'`im2col`, accumulation des gradients aux positions qui se chevauchent.
- `Conv2D.backward()` : calcule `d_kernels`, `d_bias`, `d_input`.
- `Conv2D.update(lr)` : mise à jour des poids par descente de gradient.
- Gradient check par différences finies OK (erreur relative < 1e-4).
- README mis à jour (ReLU, Flatten, Conv2D backward marqués OK).
- Création de ce fichier mémoire.

### 2026-07-03 — ReLU + Flatten

- Implémentation de `ReLU` (forward + backward avec masque).
- Implémentation de `Flatten` (forward + backward par simple reshape).
- Tests unitaires dans le `__main__`.

### Dates antérieures — Fondations

- Chargement MNIST (IDX) + preprocessing pipeline.
- Conv2D forward (im2col + produit matriciel).
- MaxPool2D forward + backward.
