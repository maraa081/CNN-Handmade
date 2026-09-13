# Adversarial Attacks — Attaquer (et défendre) mon CNN from scratch

**Français** | [English](README.en.md)

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
|   |-- bpda_eot.py    <- attaques ADAPTATIVES : BPDA + EOT (casser une défense à gradient obfusqué)  OK opérationnel
|   `-- campagne.sh    <- lance les 3 recettes durcies en série (reprise auto)  OK opérationnel
|-- torch/             <- piste PyTorch : mêmes maths, autograd, ~9x plus rapide, GPU
|   |-- modele.py      <- même architecture en nn.Module + conversion .npz
|   |-- attaques.py    <- FGSM et PGD (mêmes formules)
|   |-- entrainement.py<- pgdat / trades, augmentation, validation robuste
|   |-- attaques_avancees.py <- CW, APGD (CE/DLR), Square, NES, Boundary
|   |-- eval_suite.py  <- suite d'attaques multi-familles (le juge)
|   |-- tableau_recap.py <- tableau final Markdown depuis les JSON d'evaluation
|   |-- smoothing.py   <- robustesse CERTIFIEE : randomized smoothing (rayon L2)
|   |-- lire-le-code.md<- visite guidee du code (pour le modifier)
|   `-- harden_torch.py<- point d'entree (memes options que harden2.py) + --parite
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

## Les attaques du repo, en deux familles (2026-09-12)

Le repo contient beaucoup d'attaques, et elles ne servent pas toutes a la meme
chose. La distinction qui compte :

- **attaques d'ENTRAINEMENT** : differentiables, utilisees dans la boucle pour
  fabriquer des exemples adverses a la volee. Il n'y en a que **trois**.
- **attaques d'EVALUATION** : le juge. Elles mesurent la robustesse d'un modele
  deja entraine. Elles ne servent JAMAIS a entrainer (non differentiables, ou
  trop couteuses : Square va jusqu'a 3000 requetes par image).

### Attaques d'entrainement (3)

| Attaque | Principe | Option | Moteur |
|---|---|---|---|
| **FGSM** (Goodfellow 2014) | Un seul pas : `x_adv = clip(x + eps * sign(grad_x L))`. On lit le gradient de la perte par rapport a l'IMAGE, on ne garde que son signe (+1/-1 par pixel) et on avance de eps. La plus grossiere des trois. | `--attack fgsm` | NumPy + torch |
| **FGSM-RS** (random start) | FGSM, mais on part d'un point ALEATOIRE dans la boule eps au lieu de x : evite de lire le gradient a un endroit ou le signe ne varie plus, ce qui produit une attaque plus forte. | `--attack fgsm-rs` | torch |
| **PGD** (Madry 2018) | FGSM itere : N petits pas de taille alpha, et apres chaque pas on reprojette dans la boule L-inf autour de x, puis dans [0,1]. Depart aleatoire par defaut. C'est LA reference de l'adversarial training, et elle est differentiable, donc utilisable dans la boucle. | `--attack pgd --pgd-steps N --pgd-alpha a` | NumPy + torch |
| **APGD** (Croce & Hein 2020) | PGD a **pas adaptatif** (le pas est divise quand l'objectif stagne) et objectif **DLR** (normalise, reste informatif quand le modele est deja confiant). Plus couteux, mais c'est l'attaque a gradient la plus forte qu'on ait -- donc celle contre laquelle on veut s'entrainer. | `--attack apgd-dlr` (ou `apgd-ce`) `--pgd-steps N` | torch |

alpha vaut eps/4 par defaut. Le run v4 utilise un pas plus fin (eps/10) avec
plus de pas (20) : l'attaque d'entrainement est plus precise, et la robustesse
apprise moins specifique a une trajectoire d'attaque grossiere. Pour APGD,
`alpha` est ignore : le pas est adaptatif, les paliers sont internes.

> **TRADES n'est PAS une 4e attaque.** Piege classique : l'attaque reste PGD,
ce qui change c'est la **perte**. Detail complet dans `defenses.md` (section 6.5).

| Perte | Formule | Effet |
|---|---|---|
| `pgdat` (Madry) | `CE(propre + adverse)` | "sois juste sur les exemples adverses" |
| `trades` (Zhang 2019) | `CE(propre) + beta * KL(p_adverse \|\| p_propre)` | "garde la MEME prediction dans un voisinage" (robustesse locale, pas seulement justesse) |

L'**augmentation** (rotation, zoom, translation, sel-poivre, cutout) n'est pas
une attaque : c'est de la diversite de donnees. Elle apparait dans la meme
boucle, d'ou la confusion frequente.

### Attaques d'evaluation (le juge)

| Attaque | Famille | Principe |
|---|---|---|
| FGSM, PGD (+ restarts) | gradient | le socle ; PGD-20 avec 3 restarts donne le chiffre "officiel" |
| APGD-CE, APGD-DLR | gradient | le coeur d'AutoAttack : pas adaptatifs, deux pertes (CE et DLR) |
| CW-L2 (Carlini & Wagner 2017) | gradient | minimise la distance de perturbation : trouve l'exemple adverse le plus PROCHE |
| Square (Andriushchenko 2020) | SANS gradient | cherche par scores ; c'est elle qui a casse le 91% (42.0% a 3000 pas) |
| NES | SANS gradient | gradient estime par differences finies (estimateur antithetique) |
| Boundary (Brendel 2019) | decision | n'utilise QUE l'etiquette predite : le cas le plus pauvre pour l'attaquant |
| BPDA, EOT (Athalye 2018) | adaptatif | attaquer une defense non differentiable (feature squeezing) |
| Transfert | black-box | exemple genere sur un modele, teste sur un autre |

Trois familles, du plus fort au plus faible pour l'attaquant : **gradient > sans
gradient > decision**. Un modele se juge sur le PIRE CAS (`eval_suite.py`), pas
sur la seule attaque qui l'arrange. C'est exactement l'erreur qui a fait annoncer
91.0% alors que le pire cas etait 42.0%.

### Une seule variable a la fois : ce qui a change entre nos runs

| Run | Attaque interne | Pas | alpha | Augmentation | Perte | Resultat eps=0.30 |
|---|---|---|---|---|---|---|
| v1 (NumPy) | PGD | 7 | eps/4 | non | pgdat | 67.2% propre / 1.2% PGD |
| A | PGD | 5 | eps/4 | non | pgdat | 98.8% / 65.4% |
| B | PGD | 5 | eps/4 | oui | pgdat | 99.6% / 91.0% (120 epochs) |
| C | PGD | 5 | eps/4 | oui | trades (beta=2) | 96.9% / 2.2% |
| v4 | PGD | 20 | eps/10 | oui | pgdat | 99.4% / **61.8%** (Square 3000) |
| v5 (essai 1) | APGD-DLR | 10 | adaptatif | oui | pgdat | **casse** : val PGD10 46.5% -> 0.6% (attaque interne inoperante) |
| v5 (essai 2) | APGD-CE | 20 | adaptatif | oui | pgdat | val PGD10 **figee a 11.6%** (attaque qui renvoyait le depart aleatoire) |
| v5 (a relancer) | APGD-CE | 20 | adaptatif | oui | pgdat | - |
| abl_a | PGD | 20 | eps/10 (budget 2 eps) | oui | pgdat | **92.0%** PGD-20 / **pire cas 63.2%** |
| abl_b | PGD | 20 | eps/4 (budget 5 eps) | oui | pgdat | 75.6% sous PGD-20 / pire cas **38.8%** |
| abl_c | PGD | 20 | eps (budget 20 eps) | oui | pgdat | 6.8% sous PGD-20 / pire cas **1.6%** (verrou CE=ln10) |

eps = 0.30 partout. Le seul changement de v4 : une attaque interne plus fine
(plus de pas, pas plus petit) - c'est la reponse directe au pire cas de 42.0%,
et elle a paye : **+19.8 points** de pire cas (42.0% -> 61.8%). Detail dans
`memoire.md`.

### L'ablation qui explique tout : le BUDGET DE DEPLACEMENT de l'attaque interne

Meme recette, meme duree, seule la taille du pas de l'attaque interne change
(PGD-20) : budget `pas x 20` de 2 eps -> 92.0%, 5 eps -> 75.6%, 20 eps -> 6.8%
de precision sous PGD-20 eps=0.30. Avec le budget de notre APGD (40 eps, pas de
2 eps) : plateau a ~11%, CE adverse bloquee a ln(10) = 2.30 (le modele repond
uniformement, il abandonne la moitie adverse du batch).

Mesuree sur toute la suite d'attaques (pas seulement PGD-20), la courbe du PIRE
CAS est **en cloche avec un sommet a 2 eps** :

| Budget de l'attaque interne | 1.25 eps (run B) | 2 eps (v4 / abl_a) | 5 eps (abl_b) | 20 eps (abl_c) |
|---|---|---|---|---|
| pire cas eps=0.30 | 42.0% | **63.2%** | 38.8% | 1.6% |

Figure : `results/pire_cas_vs_budget_deplacement.png`.

**La robustesse apprise est donc maximale pour un budget intermediaire (2 eps),
pas pour le maximum. C'est la LISIBILITE de la perturbation qui compte, pas la
force de l'attaque.** Un pas >= eps saute au coin de la boule et y reste : le
motif de signe devient un masque chaotique, inapprenable. Un pas fin construit
la perturbation progressivement : elle reste lisible.

Et c'est pourquoi APGD, excellent pour JUGER (il pousse sa recherche a fond),
detruit l'apprentissage : **on n'entraine pas contre l'attaque qui sert a juger.**
Le sommet de la courbe est a 2 eps = PGD-20 avec pas eps/10 (v4, abl_a).

Les deux extremes valent la peine d'etre montres : a 1.25 eps on obtient une
**robustesse masquee** (91% sous PGD-20 mais 42% de pire cas), a 20 eps un modele
qui a l'air intact (98.4% propre) mais qui est **inutilisable** (1.6% de pire
cas). Et sur abl_c l'ecart gradient <-> Square s'inverse (APGD-DLR 1.6% contre
Square-3000 11.2%) : l'ecart de 22 points est le symptome d'un modele vraiment
robuste, il disparait quand le modele est casse.

### Le faux pas de v5 (essai 1) : une attaque interne peut "reussir" et ne rien apprendre

Le premier essai de v5 (APGD-DLR, 10 pas) s'est **effondre** : la val PGD10 est
montee a 46.5% (epoch 16) puis est tombee a 0.6% (epoch 32) sans jamais remonter,
alors que la precision propre restait a 99.6% et que la perte d'entrainement
continuait de baisser. Ce n'est pas de la malchance : la perte qui baisse sur un
batch a moitie adverse veut dire que **l'attaque interne ne fabriquait plus
d'exemples adverses** (le depart aleatoire de l'APGD etait le meme a chaque batch,
et l'attaque renvoyait ce motif fixe comme "meilleur" exemple). Le modele
apprenait par coeur a le vaincre. Corrige depuis (voir `memoire.md`, 2026-09-12
soir) : graine fraiche a chaque batch, paliers du pas proportionnels au budget,
Nesterov sur les deplacements, et la ligne d'epoch affiche desormais la CE propre,
la CE adverse et le taux de tromperie de l'attaque interne.

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

> **Modèle full** (124 800 images, **92.0% test** — entraîné sur une autre machine
> le 2026-08-25, ~37 min). Attaques lancées le 2026-08-26.

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
> générées sur une autre machine, pas poussées).

---

## La campagne complète : 3 recettes, une seule variable (2026-09-10)

`campagne.sh` répond à une question précise : **est-ce le dataset ou la méthode
qui compte ?** Trois runs identiques, sauf un point chacun.

| | Recette | Propre | FGSM ε=0.30 | PGD ε=0.30 |
|---|---|---|---|---|
| **A** | référence (60000 img, PGD-5, sans augmentation) | **98.8%** | **88.2%** | **65.4%** |
| **B** | A + augmentation de données | 99.5% | 54.8% | 29.8% |
| **C** | B + TRADES (β=2) | 96.9% | 23.4% | 2.2% |
| **B (120 epochs)** | B poussé à 120 epochs | **99.6%** | **96.0%** | **91.0%** |

```bash
./campagne.sh              # NumPy (harden2.py)
./campagne.sh --torch      # PyTorch (harden_torch.py)
./campagne.sh --rapide     # version courte, pour vérifier la chaîne
./campagne.sh --liste      # affiche seulement les runs prévus
```

**Le run A gagne partout, et de loin.** Contre la v1 : **+31.6 pts de précision
propre** et de +64.4 pts (FGSM) à +76.2 pts (PGD) à ε=0.30.

**L'augmentation a NUI à 10 epochs — et GAGNE à 120.** C'est le résultat le plus
instructif du dossier. À budget égal, la loss du run B reste bloquée à **~0.82**
quand celle du run A descend à **0.24** : le modèle augmenté est
**sous-entraîné**, parce que chaque epoch est plus difficile (images déjà
déformées). C'était donc une leçon de **budget**, pas de méthode.

Vérifié en poussant le run B à **120 epochs** : la loss descend à **0.31** et la
robustesse passe à **91.0%** sous PGD ε=0.30, soit **+26 pts devant le run A**.
Autrement dit : l'augmentation paie — mais seulement avec 2-3x plus d'epochs —
et elle ne remplace pas l'adversarial training.

**TRADES reste à reprendre (run C)** : `--lr 0.002` était beaucoup trop bas (la
loss reste figée à ~1.68) et la décroissance du lr aux epochs 5 et 8 achevait de
figer le modèle. À reprendre avec `--lr 0.01` et sans décroissance agressive.

**Un plafond identifié** : la robustesse de validation du run A plafonne à ~70%
dès l'epoch 5 (26.3 -> 36.7 -> 55.3 -> 64.5 -> 68.5 -> 69.2 -> 70.1 -> 70.1 ->
70.7 -> 70.3). Le facteur limitant n'est plus la recette, c'est le **budget
d'épochs**.

---

## Attaques adaptatives : BPDA + EOT (2026-09-11)

Le modèle durci tient 65-67% sous PGD — mais PGD est justement l'attaque contre
laquelle il a été entraîné. Pour savoir si la défense est **réelle**, il faut la
tester avec un attaquant qui la connaît. C'est le test du "gradient masking"
(Athalye, Carlini & Wagner, 2018).

`bpda_eot.py` attaque le modèle **tel qu'il serait déployé** (poids durcis +
feature squeezing à l'inférence) avec quatre attaquants :

- **sans défense** : PGD sur le modèle brut, sans tenir compte du squeezing (le
  squeezing n'est appliqué qu'à l'évaluation) ;
- **naïf** : PGD propage à travers la quantification avec son **vrai** jacobien.
  L'arrondi est une fonction en escalier : sa dérivée est nulle presque partout
  (c'est exactement ce que fait `torch.round`). Le gradient qui remonte à
  l'entrée est donc nul, et l'attaque ne bouge pas d'un pixel ;
- **BPDA** : forward exact, passe arrière approximée par l'identité ;
- **BPDA + EOT** : BPDA avec moyenne du gradient sur des transformations
  aléatoires (profondeur de bits tirée dans {2, 3, 4}, translation +/-2 px).

Résultat (200 images, PGD-20, squeezing 3 bits, accuracy propre 67.0%) :

| eps | sans défense | naïf (vrai jacobien) | BPDA | BPDA + EOT |
|---|---|---|---|---|
| 0.10 | 27.5% | **68.0%** | 27.0% | 41.0% |
| 0.20 | 22.5% | **66.5%** | 21.5% | 37.0% |
| 0.30 | 1.0% | **62.5%** | 1.5% | 10.0% |

```bash
# Modele durci v1 + feature squeezing 3 bits (defaut)
python3 adversarial/scripts/bpda_eot.py

# Plus de tirages EOT, autres eps
python3 adversarial/scripts/bpda_eot.py --bits 3 --eot 8 --eps 0.2 0.3

# Test rapide de la chaine
python3 adversarial/scripts/bpda_eot.py --quick
```

> **Verdict : la défense de la v1 était du gradient masking.** L'attaquant naïf
> laisse 62.5% au modèle à eps=0.30 — elle *paraît* solide. BPDA la fait tomber
> à **1.5%**, soit le niveau d'un modèle non défendu. La couche de feature
> squeezing n'apportait donc **aucune protection réelle** : elle rendait
> simplement le gradient inutilisable pour l'attaquant.
>
> Note honnête : BPDA+EOT est ici *moins* efficace que BPDA seul (10.0% contre
> 1.5% à eps=0.30), parce que la moyenne EOT dilue le signal à nombre de pas
> fixé. La leçon reste la même : une défense stochastique ne protège pas, elle
> rend juste l'attaque plus chère.

**Ce que ça change pour le reste du dossier** : le modèle durci de la piste
torch (98.8% / 65.4%) **n'embarque pas** de feature squeezing et n'est pas
stochastique. Les attaques adaptatives s'y appliquent donc différemment, et le
bon test pour lui est plus loin : CW (bientôt) et AutoAttack.

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
PGD-5, ~1 min par epoch sur une autre machine). C'est le run A du tableau
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

> [info] **Données à télécharger une fois.** `data/kmnist/` (21 Mo) et
> `data/emnist/` (570 Mo) ne sont **pas** dans le dépôt : ce sont des données
> brutes, ignorées par git. Depuis le 2026-09-13, `charger_train` /
> `charger_test` les récupèrent **automatiquement** au premier lancement (via
> `scripts/download_kmnist.py` et `scripts/download_emnist.py`), avec un message
> clair si le réseau manque. Pour les préparer à la main :
>
> ```bash
> python3 scripts/download_kmnist.py     # KMNIST (~21 Mo, CODH Tohoku)
> python3 scripts/download_emnist.py    # EMNIST (~570 Mo)
> ```

Tout le détail (correspondance terme à terme, ce qu'on perd, setup ROCm pour
cartes AMD, réglage du batch) : [`torch/README.md`](torch/README.md).

### Réplication KMNIST (l'ancre Japon) — série lancée le 2026-09-13

Deux corrections de code faites en préparant cette série (13/09) :

1. `data/kmnist/` n'était jamais téléchargé par la piste PyTorch -> voir la note
   « Données à télécharger une fois » ci-dessus.
2. En mode `--sans-attaque` (run propre), le meilleur modèle était sélectionné
   sur la **robustesse de validation** : comme elle vaut 0% partout pour un
   modèle propre, la comparaison `> meilleur` n'était vraie qu'à l'epoch 1 et le
   fichier exporté était le modèle de l'epoch 1 au lieu du modèle fini. La
   sélection se fait maintenant sur la **précision propre** quand
   `--sans-attaque` est actif (et l'alerte « l'attaque interne ne trompe plus »
   est désactivée : sans attaque, elle mesure l'erreur d'entraînement).

Même CNN, mêmes recettes, jeu japonais : la question est de savoir si la **loi
locale** du projet (le budget de déplacement de l'attaque interne, puis l'ordre
des budgets) reproduit à l'identique sur un autre jeu. Deux options ont été
ajoutées pour ça : `--sans-attaque` (entraînement propre = le modèle de
référence) et `--warm-start` sur un `.npz` (les recettes repartent du même
point).

```bash
# 0. Le modèle de référence (entraînement PROPRE, sans attaque)
python3 adversarial/torch/harden_torch.py --dataset kmnist --n-train 60000 \
  --val 1000 --epochs 120 --augment --sans-attaque \
  --out models/kmnist_standard.pt --npz models/kmnist_standard.npz

# 1. Référence adverse EN DUR (budget constant, départ dur : comme abl_a)
python3 adversarial/torch/harden_torch.py --dataset kmnist --n-train 60000 \
  --epochs 120 --augment --pgd-steps 20 --pgd-alpha 0.03 \
  --warm-start models/kmnist_standard.npz --out models/kmnist_fixe_2eps.pt

# 2. Recette douce (plan croissant 0.2 -> 1 eps : comme A8)
python3 adversarial/torch/harden_torch.py --dataset kmnist --n-train 60000 \
  --epochs 120 --augment --pgd-alpha 0.03 --plan-budget "0.2,1" \
  --warm-start models/kmnist_standard.npz --out models/kmnist_plan_doux.pt

# 3. Le test de l'ORDRE : mêmes budgets, ordre inverse (1 -> 0.2 eps)
python3 adversarial/torch/harden_torch.py --dataset kmnist --n-train 60000 \
  --epochs 120 --augment --pgd-alpha 0.03 --plan-budget "1,0.2" \
  --warm-start models/kmnist_standard.npz --out models/kmnist_plan_inverse.pt
```

## Le tableau final : le recap automatique des runs

`eval_suite.py` et `eval_autoattack.py` savent écrire leurs résultats en JSON
(`--json chemin.json`), et `tableau_recap.py` les relit pour imprimer **le**
tableau Markdown, trié par pire cas :

```bash
# Évaluer un modèle (mêmes réglages que les runs MNIST : 500 images,
# Square poussé à 3000, 2 restarts) ET écrire le JSON
python3 adversarial/torch/eval_suite.py --dataset kmnist \
  --weights models/kmnist_plan_doux.pt --n 500 --eps 0.3 \
  --square-steps 3000 --square-restarts 2 --device cuda \
  --label "KMNIST plan doux 0.2->1" \
  --json adversarial/results/logs/kmnist_plan_doux.json

# Le chiffre officiel, sur le gagnant, avec son JSON (même chemin de poids :
# le tableau le rattache tout seul à la bonne ligne)
python3 adversarial/torch/eval_autoattack.py --dataset kmnist \
  --weights models/kmnist_plan_doux.pt --n 10000 --eps 0.3 \
  --device cuda --label "KMNIST plan doux 0.2->1" \
  --json adversarial/results/logs/kmnist_plan_doux_autoattack.json

# Le tableau final (lit tous les JSON de adversarial/results/logs/)
python3 adversarial/torch/tableau_recap.py
python3 adversarial/torch/tableau_recap.py --csv recap.csv   # export tableur
```

Sortie : une ligne par modèle, une colonne par attaque (le nombre de pas est
dans l'en-tête, donc on voit immédiatement si deux lignes sont comparables), le
**pire cas maison** et, quand elle existe, la colonne **AutoAttack (officiel)**
qui est le chiffre d'annonce. Le tableau est du Markdown brut : copier-coller
vers le README ou la model card, rien à reformater.

Prédiction/extrapolation, écrite AVANT de lancer les runs : le départ dur
(série 1) plafonne vers 60-65% de pire cas, la recette douce (série 2) monte
vers ~84%, et l'ordre inverse (série 3) perd ~20 points. Si KMNIST ne se
comporte pas comme MNIST, c'est que la loi n'est pas propre au jeu de données --
et ça, c'est un résultat en soi, à écrire dans le write-up.

---

## La suite d'attaques : juger une défense sur plusieurs familles (2026-09-11)

Un modèle entraîné contre PGD paraît robuste... contre PGD. `eval_suite.py`
passe **trois familles d'attaques** et reporte le **pire cas**.

| Famille | Attaques | Ce que l'attaquant voit |
|---|---|---|
| À gradient (white-box) | FGSM, PGD multi-restarts, **APGD-CE**, **APGD-DLR** | le modèle et son gradient |
| Sans gradient (score-based) | **Square**, **NES** | seulement les scores de sortie |
| Decision-based | **Boundary** | seulement la classe prédite |

- **APGD** (Croce & Hein 2020) : la version moderne de PGD — pas adaptatif,
momentum Nesterov, restarts. Ses deux pertes (CE et DLR) sont le cœur de
l'**AutoAttack**. La DLR reste informative quand la CE sature, ce qui la rend
meilleure sur les modèles vraiment robustes.
- **Square** (Andriushchenko et al. 2020) : modifie de petits carrés à positions
aléatoires, **sans jamais regarder le gradient**. Contrôle indépendant : si une
défense ne tient que face aux attaques à gradient, Square la casse.
- **NES** : estime le gradient par différences finies sur des directions
aléatoires (estimateur antithétique). Coût : 2 appels au modèle par direction.
- **CW-L2** (Carlini & Wagner 2017) : minimise une **distance**, pas une perte.
- **Boundary** (Brendel & Bethge 2018, version simplifiée) : marche aléatoire sur
la frontière de décision, avec pour seule information la classe prédite.

```bash
python3 adversarial/torch/eval_suite.py --weights models/harden2_aug_pgdat_120ep.pt
python3 adversarial/torch/eval_suite.py --weights ... --quick       # verification
python3 adversarial/torch/eval_suite.py --weights ... --famille blackbox
```

Validation de la chaine sur le modele durci v1 (100 images, eps=0.30) :

| Attaque | Accuracy | Famille |
|---|---|---|
| (propre) | 71.0% | - |
| FGSM | 27.0% | gradient |
| PGD-20 (3 restarts) | **1.0%** | gradient |
| APGD-CE (100 pas) | 1.0% | gradient |
| APGD-DLR (100 pas) | 2.0% | gradient |
| Square (1000 pas) | 11.0% | sans gradient |
| NES (40x20) | 28.0% | sans gradient |
| CW-L2 | 51% de succes, distance L2 0.156 | L2 |
| Boundary | 89% de succes, distance L2 4.8 | decision |

Ordre attendu et retrouve : les attaques a gradient detruisent un modele non
robuste ; les attaques sans gradient sont plus faibles a budget de requetes
egal (c'est le prix de la boite noire) ; Boundary "reussit" mais a une grande
distance, ce qui est normal pour une attaque decision-based sur une petite
boule.

### Resultat sur le modele de reference (120 epochs, 500 images, eps=0.30)

| Attaque | Famille | Accuracy |
|---|---|---|
| (propre) | - | **99.8%** |
| FGSM (1 pas) | gradient | 96.0% |
| PGD-20 (3 restarts) | gradient | 91.0% |
| PGD-50 (10 restarts, pas eps/10) | gradient | 92.0% |
| APGD-CE | gradient | 90.0% |
| **APGD-DLR** | gradient | **79.0%** |
| Square (500 pas) | sans gradient | 70.0% |
| **Square (3000 pas, 2 restarts)** | sans gradient | **42.0%** |
| NES | sans gradient | 93.8% |

> **Resultat : le pire cas est 42.0%, pas 91.0%.** Le modele etait annonce a 91%
sous PGD-20 -- c'est-a-dire mesure contre l'attaque de son propre entrainement.
Une attaque **sans gradient**, avec 3000 requetes, le fait tomber a **42.0%**.
L'ecart de 49 points est le resultat le plus important du dossier.
>
> **Pourquoi une attaque sans gradient fait-elle mieux que le gradient ?**
> L'entrainement adversarial a aplati la surface de perte : le gradient renseigne
> mal l'attaquant, tandis qu'une recherche guidee par les seuls scores trouve le
> chemin. C'est la signature d'un modele ou l'attaquant aveugle fait mieux que
> l'attaquant voyant.
>
> **Verification** : la borne de perturbation a ete controlee directement
> (|delta|inf = 0.300000, jamais depassee), le resultat est monotone avec le
> budget (500 pas -> 70.0%, 3000 pas -> 42.0%) et reproduit (39.0% sur 200
> images, 42.0% sur 500).
>
> **Piste principale** : l'attaque d'entrainement etait trop grossiere (PGD-5,
pas de eps/4). Renforcer l'attaque interne (PGD-20, pas de eps/10, option
`--pgd-alpha`) est la prochaine etape.

### Le run v4 a suivi la piste (2026-09-12) : pire cas 42.0% -> 61.8%

Meme architecture, meme budget d'attaque, meme echantillon (500 images). Seule
difference : l'attaque d'ENTRAINEMENT (PGD-20 au pas eps/10, au lieu de PGD-5
au pas eps/4).

| Attaque | 120 epochs (PGD-5) | **v4 (PGD-20, eps/10)** |
|---|---|---|
| (propre) | 99.8% | 99.4% |
| FGSM (1 pas) | 96.0% | 94.2% |
| PGD-20 (3 restarts) | 91.0% | 90.4% |
| PGD-50 (10 restarts) | 92.0%* | 89.6% |
| APGD-CE | 90.0%* | 88.2% |
| **APGD-DLR** | 79.0%* | **81.0%** |
| Square (500 pas) | 70.0% | 83.6% |
| **Square (3000 pas)** | **42.0%** | **61.8%** |
| NES | 93.8% | 93.0% |
| **PIRE CAS** | **42.0%** | **61.8%** |

(*) mesures sur 200 images.

Trois enseignements :

1. **Le diagnostic est valide** : +19.8 points de pire cas, a architecture
   identique. L'attaque d'entrainement trop grossiere etait bien une cause
   majeure, pas une hypothese.
2. **L'ecart PGD <-> Square s'effondre** : 21 points (91.0 contre 70.0) a
   500 pas, contre **6.8 points** (90.4 contre 83.6). La surface de perte est
   moins plate, le gradient renseigne a nouveau l'attaquant.
3. **Il reste un ecart a 3000 pas** : APGD-DLR 81.0% contre Square 61.8%. Une
   recherche guidee par les seuls scores reste plus efficace que le gradient.
   Prochaine etape logique : s'entrainer CONTRE l'attaque la plus forte connu,
   d'ou l'option `--attack apgd-dlr` (ajoutee le 2026-09-12).

Poids versionnes : `models/harden_v4_pgd20.pt` (1,6 Mo).

## Robustesse CERTIFIEE : randomized smoothing

Toutes les defenses precedentes sont **empiriques** : on attaque, on regarde ce
qui reste. Le jour ou une attaque plus forte arrive, le chiffre tombe.
`randomized smoothing` (Cohen, Rosenfeld & Kolter 2019) donne une **garantie**.

Principe : on entraine un classifieur de base sur des images bruitees
(N(0, sigma^2 I)), puis on construit un classifieur **lisse**
`g(x) = argmax_c P(f(x + bruit) = c)`, estime par Monte-Carlo. Un theoreme
donne alors un rayon L2 garanti : pour toute perturbation de norme <= R, la
prediction ne peut pas changer.

```bash
# 1. Entrainer le classifieur de base (sur images bruitees)
python3 adversarial/torch/smoothing.py --entrainer --sigma 0.5 --epochs 90

# 2. Certifier : courbe precision certifiee / rayon L2
python3 adversarial/torch/smoothing.py --certifier --sigma 0.5 --n 1000
```

Les bornes de confiance binomiales sont **exactes** (Clopper-Pearson) et
implementees sans dependance externe : pas besoin de scipy.

[ATTENTION] La borne est en norme **L2**, pas L-infini : elle n'est pas
comparable directement aux 91% sous PGD eps=0.30. C'est une autre forme de
preuve : au lieu de "je n'ai pas trouve d'attaque qui passe", on affirme "aucune
attaque de rayon <= R ne peut passer".

---

## Ce qui reste avant de publier (état au 2026-09-12)

Les étapes listées ici auparavant (BPDA/EOT, Carlini-Wagner, black-box,
randomized smoothing, APGD) sont **faites** : voir "La suite d'attaques" plus
haut. Ce qui reste, par ordre d'importance :

| Priorité | À faire | Pourquoi |
|---|---|---|
| 1 | ~~**Croiser nos chiffres avec `autoattack`**~~ **FAIT le 12/09** pour `bande_cible50` et `abl_a` | notre APGD était une réimplémentation maison : AutoAttack a confirmé le classement et révélé que nos chiffres maison étaient optimistes (abl_a : 63.2% annoncé, **50.71%** officiel) |
| 2 | **Attaquer l'écart gradient <-> Square** : résolu dans son principe le 12/09 (rapport gradient/aléatoire ; marges anisotropes) | reste à le documenter proprement dans le write-up |
| 3 | **Smoothing à relancer** (`--epochs 90`) | le bug de learning rate est corrigé (commit 4f420f8) ; le run précédent s'effondrait |
| 4 | **KMNIST** : entraîner proprement et documenter | l'ancre Japon du repo (aujourd'hui : 70.1% sur 5000 images / 3 epochs, preuve de chaîne seulement) |
| 5 | **Trancher la variante TRADES** | écart avec l'implémentation de référence (voir `defenses.md` 6.5) : aligner le code ou documenter la variante |
| 6 | **Write-up de fond** (FR + EN) | le récit complet "construire -> attaquer -> défendre -> casser sa propre défense", avec la **loi du budget de déplacement de l'attaque interne** comme pièce centrale |
| 7 | **Model card Hugging Face + Space Gradio** | la publication elle-même (poids, recette, robustesse par eps, limites) |

**Fait depuis (2026-09-12)** : run v4 (pire cas 61.8%) puis sa reproduction
`abl_a` (pire cas **63.2%**, nouveau modèle de référence) ; **ablation contrôlée
du pas de l'attaque interne** (budget 2 eps -> 92.0%, 5 eps -> 75.6%, 20 eps ->
6.8%) et sa **courbe du pire cas en cloche** (sommet à 2 eps) ; diagnostic
`diag_attaque_interne.py`.

### Piste ouverte : l'attaque interne à budget ADAPTATIF (idée Maraa, 12/09) - ELLE MARCHE

Au lieu d'un budget fixé une fois pour toutes, on vise par batch une **difficulté
cible** et on arrête l'attaque au pas `k*` qui l'atteint (option `--bande`).
La formalisation, le positionnement par rapport à FAT / IAAT / SAAT et le
protocole d'ablation sont dans **`attaque_adaptative.md`** ; le code est dans
`torch/attaque_adaptative.py` (+ `torch/test_attaque_adaptative.py`).
Point clé : à pas fixe (`eps/10`), faire varier le nombre de pas revient à faire
varier le budget, donc la trajectoire de PGD contient déjà tous les candidats --
le surcoût est **nul** par rapport à PGD-20.

---

## Périmètre de fin de projet et séquence (état au 2026-09-12)

**Jalon d'arrêt = la publication**, pas un chiffre : model card Hugging Face +
Space Gradio de démo + article de fond. Dans la roadmap S1 (sept 2026 → janv
2027), CNN-Handmade doit être terminé en janvier.

### Où on en est

| # | Élément | État |
|---|---|---|
| 1 | Suite d'attaques (BPDA+EOT, CW-L2, black-box Square/NES, Boundary, smoothing) | OK |
| 2 | **Chiffres croisés avec AutoAttack** | **OK depuis le 12/09** (sur `bande_cible50` et `abl_a`) |
| 3 | Ablation du budget de déplacement + courbe en cloche + attaque à budget adaptatif | OK |
| 4 | KMNIST entraîné proprement et documenté (ancre Japon) | à faire |
| 5 | Smoothing relancé (`--epochs 90`) | à faire |
| 6 | Variante TRADES tranchée (aligner ou documenter) | à faire |
| 7 | Write-up de fond FR + EN | à faire |
| 8 | Model card Hugging Face + Space Gradio | à faire |

### Les chiffres à annoncer (AutoAttack `standard`, 10 000 images, eps=0.30)

| Modèle | Jeu | propre | robuste (pire cas officiel) |
|---|---|---|---|
| `bande_cible50` (attaque interne à budget adaptatif) | MNIST | 98.85% | **91.25%** |
| `abl_a` (PGD-20, pas eps/10) | MNIST | 99.53% | 50.71% |
| `kmnist_plan_doux` (plan `0.2 -> 1 eps`) | KMNIST | 95.19% | **51.03%** |

Notes :

- `abl_a` était annoncé à 63.2% par notre suite (500 images). L'écart de
  12 points est le prix de nos attaques maison — c'est exactement pourquoi l'item 2
  existait. Même direction sur KMNIST : 58.4% maison contre **51.03%** officiel
  (**+7.4 points** d'optimisme).
- Sur KMNIST, AutoAttack émet un avertissement ("Square Attack has decreased the
  robust accuracy of 2.24%") : le 51.03% est lui-même légèrement optimiste. À
  écrire dans la model card — une limite annoncée honnêtement vaut mieux qu'un
  chiffre qui se fait casser plus tard.
- **La réplication KMNIST ne se transpose pas en niveau** : la loi de l'ORDRE est
  reproduite et même amplifiée (+41 points entre plan croissant et inverse, contre
  +22 sur MNIST), mais le meilleur modèle KMNIST plafonne à 51% là où MNIST atteint
  82-91%. Et la recette de référence MNIST (`constant 2 eps`) s'effondre à 1.2% sur
  KMNIST — parce qu'à eps=0.30 l'attaque interne est relativement bien plus forte
  sur ce jeu (détail complet : `memoire.md`, entrée du 2026-09-13).

### Les six règles de mesure (leçons du 2026-09-12)

1. **Annoncer sur 10 000 images.** Sur les 500 mêmes images de test, tous les
   modèles étaient tirés vers le haut (APGD-CE d'A1 : 95.6% → 93.8%).
2. **Sélectionner sur la validation, annoncer sur le test.**
3. **Le juge, c'est AutoAttack.** Notre suite sert de diagnostic rapide et
   d'explication, pas de chiffre final.
4. **Jamais de rayon robuste mesuré au PGD seul.** Sur un modèle à surface
   rugueuse il classe à l'envers (`abl_b` : plus grand rayon, 4e pire cas).
   Toujours mesurer le rayon au PGD **et** au Square (`audit_masquage.py`,
   option `--rayon-square`, active par défaut).
5. **La sensibilité (2 s) trie, elle ne prouve pas.** Un modèle qui masque son
   gradient est plat par construction.
6. **Une seule variable à la fois**, et on écrit le critère d'adoption AVANT de
   lancer le run.

### Séquence des runs

- **Phase candidats** (time-boxée : ~6 essais, quelques heures de GPU) : A1 (fait),
  A2 (cible CE-relative), A4/A5 (le 2x2 qui explique le mécanisme), `--large`.
- **Phase gel** : un modèle, un nom définitif, poussé dans le repo ; on ne touche
  plus à la recette.
- **Phase validation** : AutoAttack sur l'artefact gelé (c'est ce qui va dans la
  model card), puis KMNIST, smoothing, TRADES.
- **Phase rédaction** : write-up, model card, Space.

### Le catalogue des runs (arrêté au 2026-09-13, 01h20)

Rappel : notre suite maison (500 images) est optimiste d'environ 3 points ; le
chiffre d'annonce est celui d'AutoAttack sur 10 000 images.

| Run | Recette de l'attaque interne | Coût | propre | pire cas (maison) | pire cas (officiel) | marqueur rayons |
|---|---|---|---|---|---|---|
| v1 (NumPy) | PGD-7, eps/4 | - | 67.2% | 1.2% | - | - |
| run B | PGD-5, eps/4, 120 ep | 20 min | 99.6% | 42.0% | - | - |
| abl_b | PGD-20, eps/4 (5 eps) | 20 min | 99.8% | 38.8% | - | - |
| abl_c | PGD-20, eps (20 eps) | 20 min | 98.4% | 1.6% | - | - |
| v4 | PGD-20, eps/10 (2 eps) | 20 min | 99.4% | 61.8% | - | - |
| abl_a | PGD-20, eps/10 (2 eps) | 20 min | 99.6% | 63.2% | **50.71%** | désaccord |
| A4 | PGD-5, eps/10 (0.5 eps) | 8 min | 99.4% | 49.2% | - | accord |
| A7 | plan `1 -> 2 eps` | 14 min | 99.6% | 62.0% | - | désaccord |
| **A6** | plan `0.2 -> 2 eps` | 11 min | 99.0% | 85.8% | **82.40%** | accord |
| **A8** | plan `0.2 -> 1 eps` | 8 min | 99.0% | **84.4%** | à mesurer | accord |
| **A1** | budget adaptatif (cible 0.5) | 20 min | 98.85% | 93.6% | **91.25%** | accord |
| v5 (essai APGD) | APGD-CE, pas 2 eps | 20 min | 99.2% | 14% | - | - |
| a6_gradient_doux | deux phases `--resume` | - | - | INVALIDE (lr restauré à 0.0005) | - | - |
| **kmnist std** | propre (`--sans-attaque`, KMNIST) | 5 min | 98.6% | 0.0% | - | - |
| **kmnist 1** | constant 2 eps (copie d'`abl_a`) | 20 min | 97.8% | **1.2%** | - | accord |
| **kmnist 2** | plan `0.2 -> 1 eps` (copie d'`A8`) | 8 min | 96.6% | **58.4%** | **51.03%** | accord |
| **kmnist 3** | plan `1 -> 0.2 eps` (copie d'`A9`) | 11 min | 97.2% | **17.0%** | - | accord |
| **kmnist 4** | constant 1 eps (10 pas de eps/10) | 10 min 28 s | 97.8% | **15.2%** | - | - |

Lecture : aucun budget CONSTANT ne dépasse 63% de pire cas, quelles que soient sa
valeur et sa finesse. Les trois recettes qui fonctionnent (84-91%) ont toutes un
départ à bas budget (2 pas) et une croissance.

**Lecture sur la ligne `kmnist 4` (13/09, apres le plan inverse)** : le constant
1 eps sort a **15.2%**, soit quasiment le meme chiffre que le plan INVERSE
(17.0%) et tres loin du plan doux (58.4%). Un budget constant reste mauvais sur
KMNIST quelle que soit sa valeur (2 eps -> 1.2%, 1 eps -> 15.2%), et 1 eps ne se
distingue plus de `1 -> 0.2 eps` : c'est le DEPART haut qui commande, pas le
niveau. Sa diagonale propre (PGD pire cas 62.0% a eps=0.05 -> 58.4% a eps=0.30)
montre qu'il n'est pas mort comme le modele propre (0.0%) : simplement mediocre.

**Lecture sur les 4 lignes KMNIST (13/09)** : la loi de l'ORDRE se reproduit et
s'amplifie (+41 points entre le plan croissant et son inverse, contre +22 sur
MNIST), mais le NIVEAU ne se transpose pas (meilleur KMNIST : 51.03% officiel
contre 82-91% sur MNIST) et la recette de référence `constant 2 eps` s'effondre
(1.2% contre 63.2%). Cause identifiée : à eps=0.30 l'attaque interne est
relativement bien plus forte sur KMNIST, donc le budget 2 eps tombe après la
bascule. Détail : `memoire.md`, entrée du 2026-09-13.

**Suite ouverte (critère écrit avant les runs)** : la marche a suivre pour
localiser le sommet de la cloche sur KMNIST est `plan 0.1 -> 0.5 eps`
(1 -> 5 pas, ~6 min) puis `plan 0.05 -> 0.2 eps` (~5 min). ATTENTION au
plancher de granularite : avec `--pgd-alpha 0.03` (= eps/10), un pas vaut
0.1 eps, donc `0.05` et `0.1` a l'entree donnent tous les deux 1 pas -- les deux
plans ne different que par leur PLAFOND. Pour tester un depart VRAIMENT plus
bas, il faut affiner le pas (`--pgd-alpha 0.015`, un pas = 0.05 eps) ou reduire
l'echelle absolue (`--eps 0.15`, le budget interne etant exprime en multiples de
l'eps d'entrainement).

Hors périmètre (décision explicite, pas en passant) : ensemble de modèles,
entraînement contre Square, au-delà de 1.7M de paramètres, transfert
cross-dataset.

Pistes complémentaires (non bloquantes) : modèle plus gros (`--large`, ~1,7M
paramètres) pour tester l'hypothèse "capacité" ; Free-AT (Wong 2020, bien moins
coûteux) ; ROCm pour passer du CPU au GPU ; transfert cross-dataset MNIST ->
EMNIST ; attaque d'ensemble ; cartographie MITRE ATLAS ; patches physiques.

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
- Zhang et al., *Attacks Which Do Not Kill Training Make Adversarial Learning
  Stronger* (FAT, ICML 2020)
- Cai et al., *Curriculum Adversarial Training* (IJCAI 2018)
- Yu et al., *Strength-Adaptive Adversarial Training* (SAAT, arXiv 2210.01288)
- MITRE ATLAS : atlas.mitre.org (les attaques IA côté défenseur)
