# Adversarial Attacks — Attaquer (et défendre) mon CNN from scratch

> **Objectif :** apprendre la sécurité des modèles en attaquant mon propre CNN.
> Je contrôle le gradient de A à Z (aucun framework) -> je peux implémenter les attaques moi-même.
>
> **Documentation soignée** — chaque script est commenté, chaque résultat est expliqué,
> les expériences sont tracées dans `memoire.md`. Pas de brouillon. 

---

## Organisation

```
adversarial/
|-- README.md          <- vue d'ensemble : concepts, commandes, résultats clés
|-- attacks.md         <- les ATTAQUES en détail : théorie, algorithmes, implémentation
|-- defenses.md        <- les DÉFENSES en détail : min-max, adversarial training, limites
|-- memoire.md         <- carnet de bord : chaque expérience tracée (date, paramètres, résultat)
|-- scripts/
|   |-- fgsm.py        <- attaque FGSM (1 étape de gradient)        OK opérationnel
|   |-- pgd.py         <- attaque PGD (itérative, plus forte)       OK opérationnel
|   |-- transfer.py    <- transfert d'attaque entre modèles         OK opérationnel
|   |-- defend.py      <- adversarial training (défense)            OK opérationnel
|   |-- eval_defended.py <- éval défendu sans ré-entraîner (FGSM+PGD) OK opérationnel
|   |-- harden.py      <- VERSION DURCIE : défenses combinées       OK opérationnel
|   |-- harden2.py     <- v2 : warm start, 60k images, TRADES, clipping, sélection robuste  OK opérationnel
|   |-- augment.py     <- augmentation de données (rotation, zoom, bruit, cutout)  OK opérationnel
|   `-- campagne.sh    <- lance les 3 recettes durcies en série (reprise auto)  OK opérationnel
|-- torch/             <- piste PyTorch : mêmes maths, autograd, ~9x plus rapide, GPU
|   |-- modele.py      <- même architecture en nn.Module + conversion .npz
|   |-- attaques.py    <- FGSM et PGD (mêmes formules)
|   |-- entrainement.py<- pgdat / trades, augmentation, validation robuste
|   `-- harden_torch.py<- point d'entrée (mêmes options que harden2.py) + --parite
`-- results/           <- images + chiffres générés par les scripts (versionnés)
```

Pour la théorie complète : **`attacks.md`** (pourquoi les attaques marchent,
modèle de menace, algorithmes pas à pas) et **`defenses.md`** (le min-max de
Madry, pourquoi FGSM training échoue face à PGD, feature squeezing, la
méthodologie de preuve, le compromis robustesse/accuracy).

## Les concepts (à maîtriser)

### Évasion adversarial (adversarial examples)

Un **exemple adversarial** est une entrée modifiée de façon **imperceptible**
(bruit de quelques millièmes) qui fait se tromper le modèle avec haute confiance.

```
image originale          bruit (×10 grossi)          image attaquée
    +-----+                   +-----+                   +-----+
    | 'a' |       +    ε·sign(∇)  |     |       =        | 'h' |  (identique à l'œil)
    `-----+                   `-----+                   `-----+
```

### FGSM — Fast Gradient Sign Method (Goodfellow, 2014)

L'attaque fondatrice. Une seule étape :

```
x_adv = x + ε · sign(∇_x L(f(x), y))
```

- `∇_x L` : gradient de la loss **par rapport à l'entrée** (ce que mon backward retourne)
- `sign()` : on ne garde que la direction (+1/-1 par pixel)
- `ε` (epsilon) : l'amplitude du bruit — plus c'est grand, plus l'attaque est forte (et visible)
- `x_adv = clip(x_adv, 0, 1)` : on reste dans l'espace image valide

**Pourquoi ça marche :** le modèle est linéaire par morceaux ; une toute petite poussée
dans la direction du gradient cumule des effets sur toutes les dimensions et fait
basculer la sortie. C'est le "high-dimensional linearity" de Goodfellow.

### PGD — Projected Gradient Descent (Madry, 2018) OK implémenté

Version itérative de FGSM : plusieurs petites étapes avec projection dans la boule
L∞ de rayon ε. Attaque plus forte (le "gold standard" des attaques).

```
x_0 = x + U(-ε, ε)                       # démarrage aléatoire
x_{t+1} = clip(x_t + α·sign(∇L), x-ε, x+ε)  # pas de gradient projeté
```

### Transfert d'attaque OK implémenté

Un exemple adversarial généré contre MON modèle trompe aussi d'autres modèles.
C'est ce qui rend les attaques dangereuses en pratique (attaques boîte noire).
Mesuré sur 3 modèles MNIST entraînés différemment (full, classic, max_config).

### La défense : adversarial training OK implémenté

Réentraîner le modèle **avec** des exemples adverses -> il devient robuste.
C'est le pendant défensif — indispensable pour raconter les deux côtés.

---

## Lancer une attaque

```bash
# Attaque FGSM sur le modèle MNIST complet
python3 adversarial/scripts/fgsm.py --dataset mnist --weights models/model_weights_full.npz

# Attaque PGD sur MNIST (20 itérations, démarrage aléatoire)
python3 adversarial/scripts/pgd.py --dataset mnist --weights models/model_weights_full.npz

# Comparaison directe FGSM vs PGD sur les mêmes images
python3 adversarial/scripts/pgd.py --compare

# Sur le modèle EMNIST letters
python3 adversarial/scripts/fgsm.py --dataset emnist --weights models/emnist_letters_weights.npz
python3 adversarial/scripts/pgd.py --dataset emnist --weights models/emnist_letters_weights.npz

# Attaque ciblée (forcer la prédiction vers une classe précise)
python3 adversarial/scripts/fgsm.py --targeted --target 3
python3 adversarial/scripts/pgd.py --targeted --target 3 --steps 40
```

Résultats dans `adversarial/results/` : images comparatives + résumé chiffré.

---

## Résultats clés (mis à jour à chaque expérience)

### FGSM — MNIST (models/model_weights_full.npz, 1000 images)

| ε | Acc attaqué | Flip | Observation |
|---|---|---|---|
| 0.05 | 94.0% | 4.6% | bruit invisible, déjà -4.5 pts |
| 0.10 | 76.1% | 22.6% | 1 image sur 4 change de prédiction |
| 0.20 | 21.9% | 76.9% | effondrement |
| 0.30 | 2.1% | 96.7% | le modèle ne reconnaît presque plus rien |

> **Propre : 98.5%** — une seule étape de gradient suffit à détruire le modèle.

### PGD vs FGSM — EMNIST full (models/emnist_letters_weights_full.npz, 500 images)

> **Modèle full** (124 800 images, **92.0% test** — entraîné sur la machine de
> Maraa le 2026-08-25, ~37 min). Attaques lancées le 2026-08-26.

| ε | PGD (acc) | FGSM (acc) | Observation |
|---|---|---|---|
| 0.05 | 61.6% | 74.8% | bruit invisible, déjà -29 pts |
| 0.10 | 10.0% | 48.2% | effondrement PGD |
| 0.20 | **0.0%** | 10.0% | PGD détruit tout |
| 0.30 | **0.0%** | 4.0% | 0.0% exact |

> **Propre : 91.0%** sur l'échantillon. EMNIST (26 classes) est **beaucoup plus
> fragile que MNIST (10 classes)** : à ε=0.10 PGD tombe à 10.0% quand MNIST
> tenait encore à 44.4%. Plus de classes = frontières de décision plus denses
> = plus facile à tromper. PGD ≫ FGSM confirmé.

### FGSM — EMNIST letters (modèle rapide, 500 images)

> Modèle rapide (5000 images, 44.8% propre). Le full (92.0%) est documenté
> ci-dessus — les résultats du rapide sont gardés pour comparaison historique.

| ε | Acc attaqué | Flip | Observation |
|---|---|---|---|
| 0.05 | 28.6% | 19.2% | |
| 0.10 | 16.2% | 34.8% | |
| 0.20 | 6.2% | 49.8% | |
| 0.30 | 2.8% | 55.8% | |

> **Propre : 44.8%** (modèle rapide 5000 images).

### PGD vs FGSM — MNIST (500 images, même échantillon, modèle full 98.6%)

| Attaque | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|---|
| FGSM | 95.4% | 76.2% | 21.6% | 1.8% |
| PGD (20 steps) | 90.0% | 44.4% | **0.0%** | **0.0%** |

> **PGD est bien plus fort que FGSM** : à ε=0.2, FGSM laisse 21.6% d'accuracy,
> PGD détruit tout (0.0%). Leçon : pour évaluer la robustesse d'un modèle,
> FGSM seul ne suffit pas — il faut une attaque itérative (Madry et al. 2018).

### Transfert d'attaque — MNIST (FGSM, 500 images)

Taux de transfert = images où la CIBLE change de prédiction, parmi celles que la
source a trompées et que la cible prédisait correctement.

| Source -> Cible | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|
| full -> classic (même archi) | 25.7% | 50.0% | 72.3% |
| full -> max_config (Dropout+L2) | 8.2% | 15.1% | 50.2% |
| max_config -> full | 8.0% | 31.8% | 60.7% |

> **La transferabilité dépend de la similarité des modèles** : deux SGD
> entraînés pareil -> les exemples adverses traversent (72% à ε=0.3). C'est ce
> qui rend les attaques **boîte noire** possibles. La régularisation
> (Dropout + L2) casse une partie du transfert.
>
> [warn] **Attention à la lecture du taux** : il est mesuré *parmi les images où
> la SOURCE a été trompée*. max_config étant plus dur à tromper, son taux brut
> paraît plus élevé (31.8% contre 15.1% à ε=0.20) sans que le modèle simple soit
> mieux protégé. Le chiffre qui compte côté défense, c'est l'accuracy **absolue**
> de la cible : `full` tombe à **40.2%** à ε=0.30 sous une attaque transférée
> depuis max_config.

### La défense : adversarial training — MNIST (5000 img, 3 epochs, eps train 0.15)

| ε | standard (full) | défendu | gain |
|---|---|---|---|
| clean | 98.6% | 88.8% | -9.8% |
| 0.05 | 95.4% | 78.4% | -17.0% |
| 0.10 | 76.2% | 60.4% | -15.8% |
| 0.20 | 21.6% | 37.6% | +16.0% |
| 0.30 | 1.8% | 18.0% | +16.2% |

> **Le compromis robustesse/accuracy** : le modèle défendu perd ~10 pts en
> accuracy propre mais résiste 10× mieux à ε=0.3 (18% vs 1.8%). À faible ε il
> est moins bon que le standard (entraîné à ε=0.15, il n'est pas optimisé
> pour les petits bruits).

### Comparaison ÉQUITABLE — mêmes 5000 images d'entraînement (FGSM)

Le baseline standard est cette fois entraîné sur **exactement les mêmes 5000
images** que le défendu (`standard_same_data.npz`), pour isoler l'effet de la
défense de l'effet de la quantité de données :

| ε | standard (mêmes données) | défendu | gain |
|---|---|---|---|
| clean | 81.0% | 86.0% | +5.0% |
| 0.05 | 59.6% | 72.8% | +13.2% |
| 0.10 | 43.0% | 59.6% | +16.6% |
| 0.20 | 16.2% | 34.6% | +18.4% |
| 0.30 | 8.4% | 17.4% | +9.0% |

> **À données égales, le défendu gagne partout — même en accuracy propre (+5 pts).**
> L'adversarial training double la taille effective du jeu (chaque batch + sa
> version attaquée) : c'est une forme d'augmentation de données. La comparaison
> vs `model_weights_full` (60k images) pénalisait le défendu en clean : ce n'était
> pas l'effet de la défense mais l'effet de la quantité de données.
> Reproduire : `python3 adversarial/scripts/eval_defended.py`

### Éval PGD du défendu — la limite du FGSM training (2026-08-25)

| ε | standard (full) | défendu (FGSM train) | gain |
|---|---|---|---|
| 0.05 | 90.0% | 64.8% | -25.2% |
| 0.10 | 44.4% | 36.8% | -7.6% |
| 0.20 | 0.0% | 3.6% | +3.6% |
| 0.30 | 0.0% | 0.0% | +0.0% |

> **Leçon importante : l'adversarial training FGSM ne suffit PAS contre PGD.**
> À faible ε le défendu est même pire que le standard (il n'a été entraîné qu'à
> ε=0.15 avec des exemples FGSM à 1 étape — pas assez fort). PGD trouve les
> faiblesses résiduelles. **C'est pour ça qu'il faut la version durcie** :
> adversarial training PGD (Madry et al. 2018) + feature squeezing.

---

## La version durcie — se défendre contre TOUTES les attaques

Les attaques précédentes (FGSM, PGD, ciblées, transfert) exploitent toutes la
même faiblesse : le modèle est trop linéaire dans les petites directions du
gradient. `harden.py` implémente **3 couches de défense** combinées :

### Couche 1 — Adversarial training PGD (la défense de référence)

Au lieu d'attaquer chaque batch avec FGSM (1 étape, faible), on l'attaque avec
**PGD (7 itérations)** pendant l'entraînement. Le modèle apprend donc à résister
à l'attaque itérative la plus forte — pas seulement à sa version simplifiée.
C'est le résultat de Madry et al. 2018 : entraîner contre l'attaquant le plus
fort possible donne la robustesse la plus élevée possible.

```
x_adv = PGD(modèle_courant, batch, eps=0.3, steps=7)   # attaque forte à la volée
entraîne sur [batch ; x_adv]                            # propre + adverses
```

### Couche 2 — Feature squeezing (défense d'entrée, Xu et al. 2018)

Avant l'inférence, on **réduit la profondeur de bits** des pixels (8 bits -> 3-4
bits). Une perturbation adversarial est un écart minuscule sur chaque pixel :
la quantification l'écrase. L'image utile reste lisible (le modèle n'a pas
besoin de 256 niveaux de gris), mais le bruit adversarial disparaît.

```
x_entrée = round(x * 7) / 7   # 3 bits : 8 niveaux de gris
```

C'est une défense **sans ré-entraînement**, complémentaire : elle protège aussi
les modèles déjà déployés.

### Couche 3 — Évaluation multi-attaques (la preuve)

Une défense qui ne tient que contre FGSM n'est pas une défense. `harden.py`
évalue le modèle durci contre **toutes** les attaques du dossier :

- FGSM non ciblée
- PGD non ciblée (20 steps, random start)
- FGSM ciblée (force la classe 3)
- PGD ciblée (force la classe 3)
- Transfert : attaque générée sur le modèle standard -> testée sur le durci

### Lancer la version durcie

```bash
# Entraînement complet + évaluation (PGD adversarial training, ~30-60 min)
python3 adversarial/scripts/harden.py --n-train 5000 --epochs 3

# Adversarial training FGSM (plus rapide, moins robuste)
python3 adversarial/scripts/harden.py --attack fgsm --n-train 5000 --epochs 3

# Évaluer un modèle déjà durci sans ré-entraîner
python3 adversarial/scripts/harden.py --load models/defend_pgd_mnist_weights.npz

# Ajouter le feature squeezing à l'évaluation
python3 adversarial/scripts/harden.py --load models/defend_pgd_mnist_weights.npz --squeeze 3
```

Poids du modèle durci : `models/defend_pgd_mnist_weights.npz`.

### Version durcie v2 (`harden2.py`) - pousser la frontière

La v1 plafonne à 67% propre / 24% sous FGSM ε=0.3 : c'est un problème de budget
(5000 images, 3 epochs, départ aléatoire), pas de méthode. La v2 part du modèle
propre à 98.6%, utilise les 60000 images, décroît le learning rate, écrête les
gradients et sélectionne le modèle sur sa robustesse réelle.

```bash
# Recette recommandée (warm start automatique sur le modèle propre)
python3 adversarial/scripts/harden2.py --n-train 60000 --epochs 15 --pgd-steps 5

# TRADES : meilleure frontière précision/robustesse (Zhang et al. 2019)
python3 adversarial/scripts/harden2.py --n-train 60000 --epochs 15 --loss trades

# Fast-AT : beaucoup plus rapide, robustesse moindre (Wong et al. 2020)
python3 adversarial/scripts/harden2.py --n-train 60000 --epochs 15 --attack fgsm-rs

# Évaluation honnête d'un modèle sauvegardé (PGD 20 pas, 3 restarts, pire cas)
python3 adversarial/scripts/harden2.py --report models/harden2_best.npz --restarts 3

# Vérifier que tout tourne avant de lancer un long run
python3 adversarial/scripts/harden2.py --quick
```

Chaque epoch affiche l'accuracy propre **et** la robustesse sur un jeu de
validation, avec le temps restant estimé. Les leviers sont détaillés dans
[`defenses.md`](defenses.md) section 6.

### Résultats de la version durcie (2026-08-25)

**Modèle durci : adversarial training PGD (7 steps, eps 0.3, 5000 img, 3 epochs)**

| ε | standard (full) | durci (PGD train) | gain |
|---|---|---|---|
| clean | 98.6% | 67.2% | -31.4% |
| 0.05 (FGSM) | 95.4% | 58.8% | -36.6% |
| 0.10 (FGSM) | 76.2% | 49.2% | -27.0% |
| 0.20 (FGSM) | 21.6% | 34.8% | +13.2% |
| 0.30 (FGSM) | 1.8% | 23.8% | +22.0% |
| 0.05 (PGD) | 90.0% | 53.4% | -36.6% |
| 0.10 (PGD) | 44.4% | 37.2% | -7.2% |
| 0.20 (PGD) | 0.0% | 9.6% | +9.6% |
| 0.30 (PGD) | 0.0% | 1.2% | +1.2% |

> **Le durci domine dès que l'attaque devient forte** : à ε=0.3 FGSM il garde
> 23.8% quand le standard tombe à 1.8% (+22 pts) ; à ε=0.2 PGD il garde 9.6%
> quand le standard est à 0.0%. **Le prix : -31 pts en accuracy propre** —
> c'est le compromis robustesse/accuracy, d'autant plus marqué ici que
> l'entraînement s'est fait à eps 0.3 (très agressif).

**Attaques ciblées (forcer la classe 3) — le durci résiste très bien :**

| ε | standard (FGSM/PGD cibl.) | durci (FGSM/PGD cibl.) |
|---|---|---|
| 0.05 | 98.4% / 97.8% | 64.2% / 62.4% |
| 0.10 | 96.8% / 94.4% | 61.0% / 58.0% |
| 0.20 | 91.8% / 87.6% | 57.8% / 55.2% |
| 0.30 | 87.6% / 85.2% | 53.4% / 49.4% |

> Les attaques ciblées sont intrinsèquement plus dures à réussir (il faut
> pousser vers UNE classe précise, pas juste ailleurs) : les deux modèles y
> résistent mieux qu'aux non ciblées.

**Transfert (attaque générée sur le standard -> testée sur le durci) :**

| ε | acc du durci sous transfert |
|---|---|
| 0.05 | 65.8% |
| 0.10 | 62.6% |
| 0.20 | 51.6% |
| 0.30 | 40.4% |

> Le durci garde 40-66% d'accuracy face à des attaques générées sur un AUTRE
> modèle : la robustesse se transfère aussi contre le transfert d'attaque.

Reproduire : `python3 adversarial/scripts/harden.py --n-train 5000 --epochs 3`

> [warn] Ces deux courbes ne sont pas encore versionnées dans le dépôt : le
> script les écrit lui-même dans `adversarial/results/harden_curve_pgd.png` et
> `adversarial/results/harden_history_pgd.png` au moment du run (elles ont été
> générées sur la machine de Maraa, pas poussées).

---

## La campagne complète : 3 recettes, une seule variable (2026-09-10)

`campagne.sh` répond à une question précise : **est-ce le dataset ou la méthode
qui compte ?** Trois runs identiques, sauf un point chacun.

| | Recette | Propre | FGSM ε=0.30 | PGD ε=0.30 |
|---|---|---|---|---|
| **A** | référence (60000 img, PGD-5, sans augmentation) | **98.8%** | **88.2%** | **65.4%** |
| **B** | A + augmentation de données | 99.5% | 54.8% | 29.8% |
| **C** | B + TRADES (β=2) | 96.9% | 23.4% | 2.2% |

```bash
./campagne.sh              # NumPy (harden2.py)
./campagne.sh --torch      # PyTorch (harden_torch.py)
./campagne.sh --rapide     # version courte, pour vérifier la chaîne
./campagne.sh --liste      # affiche seulement les runs prévus
```

**Le run A gagne partout, et de loin.** Contre la v1 : **+31.6 pts de précision
propre** et de +64.4 pts (FGSM) à +76.2 pts (PGD) à ε=0.30.

**L'augmentation a NUI (run B).** Contre-intuitif, mais expliqué par la loss :
elle reste bloquée à **~0.82** alors que celle du run A descend à **0.24**. Le
modèle augmenté est **sous-entraîné** — chaque epoch est plus difficile (images
déjà déformées), donc à budget d'epochs égal il prend du retard. Il gagne en
précision propre (99.5% contre 98.8%) mais perd **35 pts** de robustesse à
ε=0.3. Conclusion : l'augmentation paie seulement avec **2-3x plus d'epochs**,
et elle **ne remplace pas** l'adversarial training.

**TRADES reste à reprendre (run C)** : `--lr 0.002` était beaucoup trop bas (la
loss reste figée à ~1.68) et la décroissance du lr aux epochs 5 et 8 achevait de
figer le modèle. À reprendre avec `--lr 0.01` et sans décroissance agressive.

**Un plafond identifié** : la robustesse de validation du run A plafonne à ~70%
dès l'epoch 5 (26.3 -> 36.7 -> 55.3 -> 64.5 -> 68.5 -> 69.2 -> 70.1 -> 70.1 ->
70.7 -> 70.3). Le facteur limitant n'est plus la recette, c'est le **budget
d'épochs**.

---

## La piste PyTorch — quand le modèle grossit

Le dossier `adversarial/scripts/` contient l'implémentation **faite main** :
chaque gradient est écrit à la main (`Conv2D.backward`, `col2im`, le routage du
gradient dans le MaxPool...). C'est la référence pédagogique du projet, et elle
ne bouge pas.

Mais elle est **au plafond de son BLAS** : mesuré à 172 img/s sur un i5-6300U,
avec un temps dominé par les produits matriciels eux-mêmes. Doubler la taille du
modèle double la durée d'entraînement, et le GPU n'est pas accessible depuis
NumPy.

`adversarial/torch/` rejoue **exactement les mêmes expériences** avec PyTorch :

- même architecture (421 642 paramètres), mêmes formules d'attaque,
  mêmes recettes (`pgdat`, `trades`, augmentation, warm start, écrasement des
  gradients, sélection sur la robustesse) ;
- les poids sont **interchangeables** (format `.npz` dans les deux sens) ;
- environ **9x plus rapide sur CPU**, et utilisable sur GPU.

```bash
# Vérifier l'équivalence avec la version faite main (à lancer en premier)
python3 adversarial/torch/harden_torch.py --parite

# Même campagne qu'en NumPy, en plus rapide
python3 adversarial/torch/harden_torch.py --n-train 60000 --epochs 15 --pgd-steps 5 --augment
```

Résultat du test de parité (200 images, mêmes poids) : accuracy propre
**98.50 %** dans les deux implémentations, FGSM ε=0.3 **1.50 %** dans les deux,
PGD ε=0.2 **0.00 %** dans les deux.

La campagne complète a été lancée avec ce moteur (60000 images, 10 epochs,
PGD-5, ~1 min par epoch sur la machine de Maraa). C'est le run A du tableau
ci-dessus qui a produit le résultat de référence (98.8% / 65.4%).

> [warn] **Python 3.10 à 3.12** uniquement — PyTorch ne supporte pas 3.14. Sur
> une machine récente, créer un venv dédié :
>
> ```bash
> py install 3.12                  # Windows ; sinon : voir la doc de ta distro
> py -3.12 -m venv .venv
> source .venv/Scripts/activate    # Linux/macOS : source .venv/bin/activate
> pip install torch --index-url https://download.pytorch.org/whl/cpu numpy
> ```

Tout le détail (correspondance terme à terme, ce qu'on perd, setup ROCm pour
cartes AMD, réglage du batch) : [`torch/README.md`](torch/README.md).

---

## La suite prévue

Le modèle durci tient 65.4% sous PGD-20 — mais PGD est exactement l'attaque
contre laquelle il a été entraîné. Tant qu'on n'a pas testé plus fort, on ne
sait pas si c'est une **vraie défense** ou du **gradient masking** (une défense
qui ne fait que déplacer la vulnérabilité vers des directions non visitées).

| Étape | Contenu | Référence |
|---|---|---|
| 1 | **Attaques adaptatives** : BPDA + EOT sur le modèle durci (casser la défense) | Athalye et al. 2018 ; Tramèr et al. 2020 |
| 2 | **Carlini-Wagner (CW)** : l'attaque white-box de référence qui manque | Carlini & Wagner 2017 |
| 3 | **Black-box** : score-based (ZOO/NES) puis decision-based (Boundary/HSJA) | Chen et al. 2017 ; Brendel & Bethge 2019 |
| 4 | **Robustesse certifiée** : randomized smoothing (borne L2 garantie) | Cohen et al. 2019 |

Pistes complémentaires notées : transfert avec PGD en source, transfert
cross-dataset MNIST -> EMNIST, attaque d'ensemble, AutoAttack / RobustBench,
patches physiques, cartographie MITRE ATLAS.

---

## Références

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014)
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018)
- Zhang et al., *Theoretically Principled Trade-off between Robustness and Accuracy* (TRADES, 2019)
- Wong et al., *Fast is better than free: revisiting adversarial training* (2020)
- Carlini & Wagner, *Towards Evaluating the Robustness of Neural Networks* (2017)
- Athalye et al., *Obfuscated Gradients Give a False Sense of Security* (2018)
- Chen et al., *ZOO: Zeroth Order Optimization based Black-box Attacks* (2017)
- Brendel & Bethge, *Decision-Based Adversarial Attacks* (2019)
- Cohen et al., *Certified Adversarial Robustness via Randomized Smoothing* (2019)
- MITRE ATLAS : atlas.mitre.org (les attaques IA côté défenseur)
