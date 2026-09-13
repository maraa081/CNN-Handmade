# CNN Handmade

**Français** | [English](README.en.md)

> **En bref**
> - Pas de TensorFlow, pas de PyTorch, pas de Keras : Python et NumPy seulement. Chaque couche est écrite à la main (forward im2col, backward, update).
> - MNIST propre : **98.6%** d'accuracy test ; les optimiseurs (SGD, Momentum, Adam) sont faits main eux aussi.
> - Attaques adversariales, faites main aussi : FGSM fait tomber le modèle full à **1.8%**, PGD à **0.0%** à eps=0.30.
> - Meilleur modèle durci : **91.25%** de pire cas **officiel** (AutoAttack, 10 000 images, eps=0.30) pour ~99% propre, avec un budget d'attaque interne **adaptatif par batch**.
> - **Le résultat central : ce n'est pas la force de l'attaque qui durcit le réseau, c'est son ORDONNANCEMENT.** À budgets internes identiques et à coût identique, le plan croissant `0.2 -> 2 eps` donne **85.8%** et le plan décroissant `2 -> 0.2 eps` **63.8%** (suite maison) : **22 points d'écart dus au seul ordre**.
> - Réplication sur **KMNIST** (kana japonais, l'ancre Japon du repo) : la loi tient et s'amplifie (**+41 points** contre +22 sur MNIST), mais le niveau ne se transpose pas (**-23.9 points** à recette identique, chiffres officiels des deux côtés).
> - Tous les chiffres annoncés sont officiels (AutoAttack) : notre suite maison est plus rapide mais **optimiste de 3 à 13 points** — chiffré et publié, comme les six règles de mesure.
> - Deux moteurs, mêmes maths : NumPy fait main et PyTorch, ~9x plus rapide sur CPU, poids `.npz` interchangeables.
> - Suite immédiate : smoothing certifié, variante TRADES, puis publication (model card Hugging Face + Space de démo) — la trame du write-up est dans `adversarial/article/TRAME-fr.md`.

**Un réseau de neurones convolutionnel pour reconnaître les chiffres manuscrits (MNIST), fait à la main, de A à Z.**

Pas de TensorFlow, pas de PyTorch, pas de Keras. Juste Python, NumPy, et moi. 

## Pourquoi ?

Comprendre chaque brique du deep learning en la codant soi-même — im2col, rétropropagation, descente de gradient… plutôt que d'appeler une API magique.

## Ce qui est implémenté

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
| **Attaques adversariales** (FGSM, PGD, ciblées, transfert) | OK |
| **Défenses** (adversarial training FGSM, version durcie PGD, feature squeezing) | OK |

## Architecture

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

## Structure du projet

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
|   |-- voir_emnist.py    — visualiser les lettres (Spyder/IPython)
|   |-- train_kmnist.py   — entraîner sur KMNIST (kana japonais)
|   `-- download_kmnist.py— installer les données KMNIST
|-- models/                      <- poids entraînés (.npz)
|   |-- model_weights.npz          — rapide (2000 img, 3 epochs)
|   |-- model_weights_full.npz     — complet (60000 img)
|   `-- max_config_weights.npz     — Adam + Dropout + L2
|-- results/                     <- graphiques générés (PNG/CSV)
|-- adversarial/                 <- sécurité IA : attaques & défense
|   |-- README.md                     - mode d'emploi + résultats clés
|   |-- memoire.md                    - carnet de bord des expériences
|   |-- attacks.md                    - théorie des attaques (FGSM, PGD, transfert)
|   |-- defenses.md                   - théorie des défenses + méthodologie
|   |-- results/                      - courbes et images (PNG) + logs/ (gitignoré)
|   |-- scripts/                      - implémentation faite main (NumPy)
|   |   |-- fgsm.py / pgd.py / transfer.py     - attaques
|   |   |-- defend.py / harden.py / harden2.py - défenses
|   |   |-- augment.py                         - augmentation de données
|   |   |-- eval_defended.py                   - évaluation sans ré-entraîner
|   |   |-- bpda_eot.py                        - attaques adaptatives (BPDA + EOT)
|   |   `-- campagne.sh                        - les 3 recettes durcies en série
|   `-- torch/                        - piste PyTorch (autograd, GPU)
|       |-- modele.py / attaques.py / entrainement.py  - mêmes maths, autre moteur
|       |-- attaques_avancees.py                       - CW, APGD, Square, NES, Boundary
|       |-- eval_suite.py                              - suite d'attaques multi-familles
|       |-- smoothing.py                               - robustesse certifiée (rayon L2)
|       |-- lire-le-code.md                            - visite guidée du code
|       `-- harden_torch.py                            - point d'entrée (mêmes options)
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

## Utilisation

### 1. Installer les dépendances

```bash
pip install numpy matplotlib
```

Optionnel, pour la piste PyTorch (entraînement plus rapide + GPU) :

```bash
pip install torch          # CPU
# ou, pour un GPU AMD : voir adversarial/torch/README.md
```

> [warn] PyTorch ne supporte pas toutes les versions de Python. Python 3.10 à
> 3.12 est le plus sûr (3.14 n'est pas supporté).

Recette complète si ton `python3` est trop récent (exemple Windows/Git Bash) :

```bash
py install 3.12
py -3.12 -m venv .venv
source .venv/Scripts/activate        # Linux/macOS : source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu numpy
```

> [warn] **Sous Windows/Git Bash, lancer ensuite `python` et NON `python3`.** Le
> venv n'expose pas de `python3` : la commande tombe sur l'alias Windows du
> Python Manager (le Python systeme), qui n'a pas PyTorch et peut echouer a
> s'initialiser. En cas de doute, utiliser le chemin explicite
> `.venv/Scripts/python.exe`.

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

## Tuning interactif

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

## EMNIST — Les lettres (26 classes)

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

## KMNIST — Les kana japonais (10 classes)

Le même CNN, cette fois sur **KMNIST (Kuzushiji-MNIST)** : 70000 images de
**kana japonais manuscrits** (hiragana), 28x28, 10 classes.

```bash
# Télécharger les données (~21 Mo)
python3 scripts/download_kmnist.py

# Planche d'échantillons
python3 scripts/train_kmnist.py --samples

# Entraînement rapide (5000 images, 3 epochs)
python3 scripts/train_kmnist.py

# Entraînement complet (60000 images)
python3 scripts/train_kmnist.py --full
```

**Données :** [KMNIST](https://codh.rois.ac.jp/kmnist/) — publié par le CODH
(université de Tohoku). La distribution officielle est en 4 fichiers `.npz`
(et non en fichiers IDX comme MNIST/EMNIST).

**Ce qui change vs MNIST :**
- `KMNISTLoader` dans `src/data.py` lit directement les `.npz`
- aucune correction d'orientation ni de décalage de labels (contrairement à EMNIST)
- `num_classes=10`, comme MNIST

**Résultat rapide** (5000 images, 3 epochs) : **70.1% test**.

**Classes (ordre officiel des labels) :** o, ki, su, tsu, na, ha, ma, ya, re, wo.

## Sécurité IA — attaques et défense (adversarial)

Un modèle à 98.6% peut tomber à 0% à cause d'un bruit invisible à l'oeil.
C'est la partie "sécurité IA" du projet : attaquer mon propre CNN, puis
chercher à le défendre. Tout est fait à la main, sans bibliothèque d'attaque.

```bash
# Attaque FGSM sur le modèle MNIST complet
python3 adversarial/scripts/fgsm.py --weights models/model_weights_full.npz --n 1000

# Attaque PGD (20 itérations, démarrage aléatoire) + comparaison avec FGSM
python3 adversarial/scripts/pgd.py --weights models/model_weights_full.npz --n 500 --compare

# Transfert d'attaque entre deux modèles (simule une attaque boîte noire)
python3 adversarial/scripts/transfer.py --src full --dst max_config --attack pgd

# Défense : adversarial training PGD + feature squeezing, puis évaluation multi-attaques
python3 adversarial/scripts/harden.py --n-train 5000 --epochs 3

# Attaques adaptatives : casser la défense (BPDA + EOT)
python3 adversarial/scripts/bpda_eot.py
```

**Résultats clés**

| Expérience | Résultat |
|---|---|
| FGSM, MNIST full (98.6% propre) | 1.8% à eps=0.30 |
| PGD, MNIST full (98.6% propre) | 0.0% dès eps=0.20 |
| PGD, EMNIST full (92.0% propre) | 0.0% dès eps=0.20 |
| Transfert full -> classic | 72.3% de transfert à eps=0.30 |
| v1 durcie (`harden.py`, 5000 img) | clean 67.2%, 23.8% sous FGSM eps=0.30 |
| **campagne A (torch, 60k img, PGD-5)** | **clean 98.8%, 65.4% sous PGD eps=0.30** |
| campagne B (A + augmentation) | clean 99.5%, 29.8% sous PGD eps=0.30 |
| campagne C (B + TRADES) | clean 96.9%, 2.2% sous PGD eps=0.30 |
| **campagne B, 120 epochs (augmentation)** | **clean 99.6%, 91.0% sous PGD eps=0.30** |
| **suite d'attaques sur ce même modèle** | **pire cas 42.0%** (Square, sans gradient) contre 91.0% sous PGD-20 |
| Attaques adaptatives BPDA+EOT (défense v1) | gradient masking : l'attaquant naïf laisse 62.5%, BPDA la casse à 1.5% (eps=0.30) |

> **Résultat de référence (2026-09-11).** Le modèle durci entraîné en PyTorch sur
> 60000 images (**120 epochs**, PGD-5, **avec augmentation**) atteint **99.8% de
> précision propre**. Sa robustesse dépend fortement de l'attaque employée :
> **91.0% sous PGD-20**, mais **42.0% sous Square Attack** (sans gradient,
> 3000 requêtes). C'est donc **42.0% qui est le chiffre à retenir**.
>
> **Leçon centrale : le chiffre d'entraînement n'est pas le chiffre de
> robustesse.** Le modèle avait été entraîné contre PGD et évalué contre PGD :
> 91%. Donné à une attaque d'une autre famille, il tombe à 42%. Un modèle se
> juge contre **l'attaque la plus forte qu'on sait construire**, pas contre
> celle qu'on a utilisée pour l'entraîner. Le facteur limitant était le **budget
> d'epochs** pour la précision, et le **budget d'attaque** pour la robustesse.

### La campagne durcie : 3 recettes, une seule variable qui change

`adversarial/scripts/campagne.sh` enchaîne 3 runs identiques sauf sur un point,
pour répondre à une question précise : *est-ce le dataset ou la méthode qui
compte ?*

| | Recette | Propre | FGSM eps=0.30 | PGD eps=0.30 |
|---|---|---|---|---|
| **A** | référence (60000 img, PGD-5, sans augmentation) | **98.8%** | **88.2%** | **65.4%** |
| **B** | A + augmentation de données | 99.5% | 54.8% | 29.8% |
| **C** | B + TRADES (beta=2) | 96.9% | 23.4% | 2.2% |

| **B (120 epochs)** | B poussé à 120 epochs | **99.6%** | **96.0%** | **91.0%** |

```bash
# La campagne complète (3 runs, reprise automatique), en NumPy ou en PyTorch
./adversarial/scripts/campagne.sh
./adversarial/scripts/campagne.sh --torch

# Version courte pour vérifier la chaîne avant un long run
./adversarial/scripts/campagne.sh --rapide
```

> **L'augmentation a NUI à 10 epochs — et GAGNE à 120.** À budget égal, la loss
> d'entraînement du run B reste bloquée à ~0.82 quand celle du run A descend à
> 0.24 : le modèle augmenté est **sous-entraîné** (chaque epoch est plus
> difficile). C'était une leçon de **budget**, pas de méthode. Vérifié en
> poussant le run B à **120 epochs** : loss **0.31** et **91.0%** sous PGD —
> **+26 pts devant le run A** (65.4%). L'augmentation paie, mais seulement avec
> 2-3x plus d'epochs, et elle ne remplace pas l'adversarial training.
>
> **TRADES reste à reprendre (run C)** : learning rate trop bas (loss figée à
> ~1.68) et décroissance trop agressive. C'est un problème de réglage, pas de
> méthode.

> **Leçon centrale** : FGSM sous-estime la vulnérabilité réelle. Un modèle se
> juge contre l'attaque la plus forte (PGD), pas contre la plus simple.

**Deux moteurs, mêmes maths.** Le dossier `adversarial/scripts/` contient
l'implémentation faite main (chaque gradient écrit à la main : la référence
pédagogique). `adversarial/torch/` rejoue **exactement les mêmes expériences**
avec PyTorch et l'autograd : même architecture, mêmes attaques, mêmes recettes,
mais ~9x plus rapide sur CPU et utilisable sur GPU. Les poids sont
interchangeables entre les deux (format `.npz`), et le mode `--parite` vérifie
que les deux donnent le même résultat.

Détail et setup GPU : [`adversarial/torch/README.md`](adversarial/torch/README.md).

### La suite prévue

Le modèle durci tient 91.0% sous PGD — mais PGD est justement l'attaque contre
laquelle il a été entraîné. Pour ne pas se raconter d'histoires, la démarche est
de passer **plusieurs familles d'attaques** et de garder le pire cas.

| Étape | Contenu | État |
|---|---|---|
| 1 | **Attaques adaptatives** : BPDA + EOT (casser une défense à gradient obfusqué) | fait le 2026-09-11 |
| 2 | **Carlini-Wagner (CW)** : attaque white-box L2 de référence | fait le 2026-09-11 |
| 3 | **Black-box** : score-based (Square, NES) puis decision-based (Boundary) | fait le 2026-09-11 |
| 4 | **Robustesse certifiée** : randomized smoothing (borne L2 garantie) | fait le 2026-09-11 |
| 5 | **APGD-CE / APGD-DLR** : le coeur de l'AutoAttack (Croce & Hein 2020) | fait le 2026-09-11 |

```bash
# Suite complete : attaques a gradient, sans gradient, et metriques L2
python3 adversarial/torch/eval_suite.py --weights models/....pt
python3 adversarial/torch/eval_suite.py --weights models/....pt --quick

# Robustesse certifiee (borne garantie, pas une observation)
python3 adversarial/torch/smoothing.py --entrainer --sigma 0.5 --epochs 90
python3 adversarial/torch/smoothing.py --certifier --sigma 0.5
```

> **Pourquoi plusieurs familles ?** Une défense qui ne tient que contre
> l'attaque sur laquelle elle a été entraînée n'est pas une défense. Le test
> BPDA/EOT l'a montré pour la v1 : la couche de feature squeezing était du
> **gradient masking**, caduque dès qu'on change d'attaquant. Le scan complet
> (gradient, sans gradient, label seul) est la seule réponse honnête.

Pistes complémentaires déjà notées : transfert avec PGD en source, transfert
cross-dataset (MNIST -> EMNIST), attaque d'ensemble, FAB, RobustBench.

Le détail complet est dans [`adversarial/README.md`](adversarial/README.md) :
théorie et algorithmes dans [`attacks.md`](adversarial/attacks.md) et
[`defenses.md`](adversarial/defenses.md), historique des runs dans
[`adversarial/memoire.md`](adversarial/memoire.md).

## Expérimentations — Comparer les techniques

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
| Data Augmentation | OK (dans `adversarial/scripts/augment.py`) |

### Ajouter une nouvelle expérience

1. Crée `experiments/mon_opti/mon_opti.py`
2. Importe ton optimiseur depuis `src/optimizers.py` (ou crée-le là)
3. Si tu ajoutes des couches (Dropout, etc.), importe-les depuis `src/layers.py`
4. Passe l'optimiseur au modèle : `model = CNN(optimizer=MonOpti(lr=...))`
5. Lance et compare les graphiques !

## Les optimiseurs — explications

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

## Exemple rapide

```python
from scripts.predict import load_model

model = load_model("models/model_weights_full.npz")
pred = model.predict(mon_image)   # mon_image: (1, 28, 28) normalisée
print(f"Prédiction : {pred}")
```

## Ce qu'on a appris (pour la suite)

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
- **Data Augmentation** — implémentée côté adversarial : `adversarial/scripts/augment.py`
  (rotation, zoom, translation, bruit impulsionnel, cutout, épaisseur du trait)
- **Entraînement complet** (60000 images, 20 epochs) -> viser 99%+

---

**#NoFrameworks #FromScratch #MNIST**
