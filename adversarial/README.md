#  Adversarial Attacks — Attaquer (et défendre) mon CNN from scratch

> **Objectif :** apprendre la sécurité des modèles en attaquant mon propre CNN.
> Je contrôle le gradient de A à Z (aucun framework) -> je peux implémenter les attaques moi-même.
>
> **Documentation soignée** — chaque script est commenté, chaque résultat est expliqué,
> les expériences sont tracées dans `memoire.md`. Pas de brouillon. 

---

##  Organisation

```
adversarial/
|-- README.md          <- ce fichier : vue d'ensemble, concepts, résultats clés
|-- memoire.md         <- carnet de bord : chaque expérience tracée (date, paramètres, résultat)
|-- scripts/
|   |-- fgsm.py        <- attaque FGSM (1 étape de gradient)        OK opérationnel
|   |-- pgd.py         <- attaque PGD (itérative, plus forte)       OK opérationnel
|   |-- transfer.py    <- transfert d'attaque entre modèles         OK opérationnel
|   |-- defend.py      <- adversarial training (défense)            OK opérationnel
|   |-- eval_defended.py <- éval défendu sans ré-entraîner (FGSM+PGD) OK opérationnel
|   `-- harden.py      <- VERSION DURCIE : défenses combinées       OK opérationnel
`-- results/           <- images + chiffres générés par les scripts (gitignoré)
```

##  Les concepts (à maîtriser)

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

##  Lancer une attaque

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

##  Résultats clés (mis à jour à chaque expérience)

### FGSM — MNIST (models/model_weights_full.npz, 1000 images)

| ε | Acc attaqué | Flip | Observation |
|---|---|---|---|
| 0.05 | 94.0% | 4.6% | bruit invisible, déjà -4.5 pts |
| 0.10 | 76.1% | 22.6% | 1 image sur 4 change de prédiction |
| 0.20 | 21.9% | 76.9% | effondrement |
| 0.30 | 2.1% | 96.7% | le modèle ne reconnaît presque plus rien |

> **Propre : 98.5%** — une seule étape de gradient suffit à détruire le modèle.

### FGSM — EMNIST letters (models/emnist_letters_weights.npz, 500 images)

> **Modèle rapide** (5000 images, 44.8% propre). Le modèle **full** (124 800
> images, **92.0% propre**) est prêt : `models/emnist_letters_weights_full.npz`
> (entraîné sur la machine de Maraa le 2026-08-25, ~37 min).
> Prochaine étape : FGSM/PGD sur les 26 lettres avec le modèle full.

| ε | Acc attaqué | Flip | Observation |
|---|---|---|---|
| 0.05 | 28.6% | 19.2% | |
| 0.10 | 16.2% | 34.8% | |
| 0.20 | 6.2% | 49.8% | |
| 0.30 | 2.8% | 55.8% | |

> **Propre : 44.8%** (modèle rapide 5000 images — le full n'a pas abouti, relancé).

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

##  La version durcie — se défendre contre TOUTES les attaques

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
(courbes : `adversarial/results/harden_curve_pgd.png`, `harden_history_pgd.png`)

---

##  Références

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014)
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018)
- MITRE ATLAS : atlas.mitre.org (les attaques IA côté défenseur)
