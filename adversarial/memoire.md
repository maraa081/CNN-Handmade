#  Carnet de bord — Adversarial Attacks

> Journal des expériences. **Une entrée par expérience** : date, modèle, paramètres,
> résultat chiffré, observation. C'est ici que se construit la compréhension.
>
> Format d'entrée :
> ```markdown
> ### YYYY-MM-DD — <nom de l'expérience>
> - Modèle / poids : ...
> - Paramètres : ...
> - Résultat : ...
> - Observation / leçon : ...
> ```

---

##  Journal

### 2026-08-24 — Préparation du terrain

- Structure `adversarial/` créée (README, scripts/, results/, ce carnet)
- `fgsm.py` écrit et testé — première attaque opérationnelle

### 2026-08-24 — FGSM non ciblée sur MNIST (modèle full)

- Modèle / poids : `model_weights_full.npz` (98.5% propre sur l'échantillon)
- Paramètres : `--n 1000`, eps ∈ {0.05, 0.1, 0.2, 0.3}, seed 42
- Résultat : l'accuracy s'effondre de 98.5% -> 2.1% à ε=0.30 (flip 96.7%)
- Observation : la chute est **progressive** (94% -> 76% -> 22% -> 2%) : FGSM
  est une attaque « de force », plus l'amplitude autorisée est grande,
  plus le modèle s'effondre. À ε=0.05 (bruit quasi invisible) on perd déjà 4.5 pts.
- Leçon : mon CNN from scratch est **vulnérable** — il suffit d'un bruit de
  ±0.3/255 pour le faire tomber à ~2%. Conforme à la théorie (linearité
grandes dimensions, Goodfellow).

### 2026-08-24 — FGSM non ciblée sur EMNIST letters (modèle rapide)

- Modèle / poids : `emnist_letters_weights.npz` (entraînement rapide 5000 img,
  44.8% propre sur l'échantillon)
- Paramètres : `--n 500`, eps ∈ {0.05, 0.1, 0.2, 0.3}, seed 42
- Résultat : 44.8% -> 2.8% à ε=0.30 (flip 55.8%)
- Observation : l'attaque fonctionne aussi sur 26 classes, mais la chute est
  moins spectaculaire en relatif car le modèle est déjà faible (44.8% propre).
  Le flip (changement de prédiction) reste massif : plus d'une image sur deux
  change de lettre à ε=0.30.
- [warn] Note : l'entraînement full EMNIST (124 800 images, ~2h30) lancé le
  2026-08-24 a échoué silencieusement (log vide, fichier `_full` absent).

### 2026-08-25 — EMNIST full : entraînement complet terminé (machine Maraa)

- Modèle / poids : `models/emnist_letters_weights_full.npz`
- Paramètres : 124 800 images, 10 epochs, batch 128, lr 0.01, ~37 min
  (machine Windows de Maraa, ~5x plus rapide que le WSL)
- Résultat : **92.0% sur 20800 images de test** (hasard = 3.8%)
- Observation : le CNN from scratch reconnaît les 26 lettres avec un niveau
  quasi MNIST (92% vs 98.6% sur 10 classes — la tâche à 26 classes est
  intrinsèquement plus dure). Le modèle full est maintenant disponible pour
  les attaques adversarial sur les lettres (FGSM/PGD EMNIST, transfert
  cross-dataset MNIST <-> EMNIST).

### 2026-08-25 — PGD non ciblée sur MNIST (modèle full) + comparaison FGSM

- Modèle / poids : `models/model_weights_full.npz` (98.6% propre sur l'échantillon)
- Paramètres : `--n 500`, eps ∈ {0.05, 0.1, 0.2, 0.3}, 20 steps, alpha=eps/4,
  démarrage aléatoire, seed 42
- Résultat : l'accuracy s'effondre à **0.0% dès ε=0.20** (flip 99.0%)
- Observation : PGD est **beaucoup plus fort que FGSM** — à ε=0.2, FGSM laisse
  21.6% d'accuracy quand PGD tombe à 0.0%. Les petites étapes avec projection
  convergent vers un vrai maximum local de la loss, là où le pas unique de FGSM
  « dépasse » souvent l'optimum.
- Leçon : FGSM sous-estime la vulnérabilité réelle du modèle. Toute évaluation
  de robustesse sérieuse doit utiliser PGD (ou au moins plusieurs étapes).
  C'est la leçon de Madry et al. 2018 : « les attaques itératives sont la
  vraie mesure de robustesse ».
- [warn] À ε=0.3, PGD avec démarrage aléatoire donne 0.0% exactement — le modèle
  ne reconnaît plus AUCUNE image de l'échantillon.

### 2026-08-25 — Adversarial training : comparaison ÉQUITABLE (mêmes 5000 images)

- Modèles : `defend_mnist_weights.npz` (défendu, 3 epochs, batch 64, eps train 0.15)
  vs `standard_same_data.npz` (baseline standard entraîné sur les MÊMES 5000 images)
- Paramètres : 5000 images train, 500 test, FGSM, eps ∈ {0.05, 0.1, 0.2, 0.3}, seed 42
- Résultat (comparaison équitable, FGSM) :

| eps | standard | défendu | gain |
|---|---|---|---|
| clean | 81.0% | 86.0% | +5.0% |
| 0.05 | 59.6% | 72.8% | +13.2% |
| 0.10 | 43.0% | 59.6% | +16.6% |
| 0.20 | 16.2% | 34.6% | +18.4% |
| 0.30 | 8.4% | 17.4% | +9.0% |

- Observation : à données égales, le défendu gagne PARTOUT — même en accuracy
  propre (+5 pts). Explication : l'adversarial training double la taille effective
  du jeu d'entraînement (chaque batch propre + sa version attaquée), c'est une
  forme d'augmentation de données. Le baseline standard (81% propre) est sous-entraîné
  face à un modèle entraîné sur 60k images (98.6%) : c'est pour ça que la
  comparaison vs `model_weights_full` pénalisait le défendu en clean.
- Leçon : **toute comparaison défense/attaque doit se faire à données égales**,
  sinon on confond l'effet de la défense avec l'effet de la quantité de données.
- [fix] bug `UnboundLocalError: eps_list` dans defend.py (bloc --baseline utilisait
  eps_list avant son assignation) corrigé — le tableau équitable n'avait jamais pu
  s'afficher malgré des entraînements réussis.

### 2026-08-25 — Transfert d'attaque entre modèles MNIST (FGSM)

- Modèles : `full` (SGD, 98.6%), `classic` (SGD classique), `max_config`
  (Adam + Dropout + L2, 99.2%) — mêmes données MNIST, archis différentes
  (max_config a un Dropout)
- Paramètres : `--n 500`, FGSM, eps ∈ {0.1, 0.2, 0.3}, seed 42
- Résultat (taux de transfert = images où la cible change d'avis parmi
  celles que la source a trompées ET que la cible prédisait bien) :

| Source -> Cible | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|
| full -> classic | 25.7% | 50.0% | 72.3% |
| full -> max_config | 8.2% | 15.1% | 50.2% |
| max_config -> full | 8.0% | 31.8% | 60.7% |

- Observation :
  1. **full -> classic transfère fort** (jusqu'à 72%) : deux modèles SGD
     entraînés pareil ont des frontières de décision similaires -> un
     attaquant peut attaquer un modèle public et tromper le modèle cible
     sans y accéder (attaque boîte noire).
  2. **full -> max_config transfère peu** : la régularisation (Dropout + L2)
     lisse la frontière -> les exemples adverses de l'autre modèle y sont
     moins efficaces. La régularisation est une défense partielle.
  3. **max_config -> full** : attaquer le modèle robuste produit des exemples
     moins transférables vers le modèle faible (ils sont « calibrés » sur un
     paysage de loss plus lisse).
- Leçon : la transferabilité dépend de la **similarité des modèles**
  (architecture + entraînement). C'est ce qui rend les attaques boîte
  noire possibles en pratique (Papernot et al., 2016).

### 2026-08-25 — Éval PGD du défendu : les limites du FGSM training

- Modèles : défendu (`defend_mnist_weights.npz`, adversarial training FGSM
  eps 0.15) vs standard complet (`model_weights_full.npz`)
- Paramètres : 500 images, PGD 20 steps random start, eps ∈ {0.05, 0.1, 0.2, 0.3}
- Résultat :

| ε | standard | défendu | gain |
|---|---|---|---|
| 0.05 | 90.0% | 64.8% | -25.2% |
| 0.10 | 44.4% | 36.8% | -7.6% |
| 0.20 | 0.0% | 3.6% | +3.6% |
| 0.30 | 0.0% | 0.0% | +0.0% |

- Observation : **l'adversarial training FGSM ne suffit pas contre PGD.**
  À faible ε (0.05-0.10) le défendu est pire que le standard : il a été
  entraîné avec des exemples FGSM à 1 étape à ε=0.15, donc il n'est ni
  optimisé pour les petits bruits ni robuste aux attaques itératives.
  Seul un léger gain apparaît à ε=0.2 (+3.6 pts) où le standard s'effondre.
- Leçon : FGSM training est une base, pas une fin. Pour résister à PGD
  il faut s'entraîner CONTRE PGD (Madry et al. 2018) — c'est la couche 1
  de la version durcie (`harden.py`).

### 2026-08-25 — Version durcie : adversarial training PGD + feature squeezing

- Objectif : implémenter les sécurités contre TOUTES les attaques du dossier
  (FGSM, PGD, ciblées, transfert) et documenter la démarche.
- Implémentation : `adversarial/scripts/harden.py`, 3 couches :
  1. **Adversarial training PGD** (7 steps, eps 0.3) — entraîner contre
     l'attaque itérative la plus forte (Madry et al. 2018)
  2. **Feature squeezing** (Xu et al. 2018) — quantification 3-4 bits à
     l'inférence pour écraser les perturbations minuscules, sans retrain
  3. **Évaluation multi-attaques** — FGSM, PGD, FGSM ciblée, PGD ciblée,
     transfert depuis le modèle standard (la preuve, pas juste FGSM)
- Paramètres du run complet : 5000 images, 3 epochs, PGD 7 steps, eps 0.3,
  lancé le 2026-08-25 en arrière-plan (log `/tmp/harden_full.log`)
- Résultat (500 images de test) :

| Attaque | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|---|
| FGSM (durci) | 58.8% | 49.2% | 34.8% | 23.8% |
| PGD (durci) | 53.4% | 37.2% | 9.6% | 1.2% |
| FGSM ciblée (durci) | 64.2% | 61.0% | 57.8% | 53.4% |
| PGD ciblée (durci) | 62.4% | 58.0% | 55.2% | 49.4% |
| FGSM (standard full) | 95.4% | 76.2% | 21.6% | 1.8% |
| PGD (standard full) | 90.0% | 44.4% | 0.0% | 0.0% |

  Clean : durci 67.2% vs standard 98.6% (compromis robustesse/accuracy).
  Transfert standard -> durci : 65.8% / 62.6% / 51.6% / 40.4%.

- Observation : le durci domine dès que l'attaque est forte — à ε=0.3 FGSM
  il garde 23.8% vs 1.8% (+22 pts), à ε=0.2 PGD 9.6% vs 0.0% (+9.6 pts).
  Les attaques ciblées échouent largement sur lui (53-64% d'acc même à
  ε=0.3). Le transfert est atténué : 40-66% d'acc contre des attaques
  générées sur un autre modèle.
- Leçon : une défense se prouve contre l'attaquant le plus fort, pas
  contre la version la plus simple de l'attaque. Le prix à payer est
  l'accuracy propre — en pratique on ajuste eps d'entraînement selon
  le niveau de menace (eps 0.15 -> meilleur clean, eps 0.3 -> max robustesse).

---

## 2026-08-26 — PGD vs FGSM sur EMNIST full (26 lettres, modèle 92.0%)

- Commande (machine Maraa) : `pgd.py --dataset emnist --weights models/emnist_letters_weights_full.npz --n 500 --steps 20 --compare`
- Modèle : `emnist_letters_weights_full.npz` (124 800 images, 10 epochs, 92.0% test)
- Échantillon : 500 images de test tirées au hasard (le test set EMNIST est trié par classe)
- Clean sur l'échantillon : **91.0%**
- PGD : 20 itérations, alpha = eps/4, démarrage aléatoire

| Attaque | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|---|
| PGD (acc attaqué) | 61.6% | 10.0% | **0.0%** | **0.0%** |
| PGD (flip) | 29.6% | 81.6% | 93.8% | 95.2% |
| FGSM (acc attaqué) | 74.8% | 48.2% | 10.0% | 4.0% |

- Observation : EMNIST full est **beaucoup plus fragile que MNIST full** —
  à ε=0.10 PGD passe à 10.0% quand MNIST tenait encore à 44.4%.
  Plus de classes (26 vs 10) = frontières de décision plus denses = plus facile à tromper.
- PGD ≫ FGSM confirmé aussi sur EMNIST : à ε=0.20, 0.0% vs 10.0%.
- Même FGSM seul détruit le modèle : 4.0% à ε=0.30 (contre 1.8% sur MNIST).

---

##  Tableau des résultats cumulés

| Date | Attaque | Modèle | ε | Acc propre | Acc attaqué | Flip | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-24 | FGSM untgt | MNIST full | 0.05 | 98.5% | 94.0% | 4.6% | chute déjà visible |
| 2026-08-24 | FGSM untgt | MNIST full | 0.10 | 98.5% | 76.1% | 22.6% | |
| 2026-08-24 | FGSM untgt | MNIST full | 0.20 | 98.5% | 21.9% | 76.9% | |
| 2026-08-24 | FGSM untgt | MNIST full | 0.30 | 98.5% | 2.1% | 96.7% | effondrement |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.05 | 44.8% | 28.6% | 19.2% | |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.10 | 44.8% | 16.2% | 34.8% | |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.20 | 44.8% | 6.2% | 49.8% | |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.30 | 44.8% | 2.8% | 55.8% | |
| 2026-08-26 | PGD untgt (20 st.) | EMNIST full | 0.05 | 91.0% | 61.6% | 29.6% | 26 classes, plus fragile que MNIST |
| 2026-08-26 | PGD untgt (20 st.) | EMNIST full | 0.10 | 91.0% | 10.0% | 81.6% | MNIST tenait à 44.4% |
| 2026-08-26 | PGD untgt (20 st.) | EMNIST full | 0.20 | 91.0% | 0.0% | 93.8% | effondrement total |
| 2026-08-26 | PGD untgt (20 st.) | EMNIST full | 0.30 | 91.0% | 0.0% | 95.2% | 0.0% exact |
| 2026-08-26 | FGSM (rappel) | EMNIST full | 0.05 | 91.0% | 74.8% | — | même échantillon que PGD |
| 2026-08-26 | FGSM (rappel) | EMNIST full | 0.10 | 91.0% | 48.2% | — | |
| 2026-08-26 | FGSM (rappel) | EMNIST full | 0.20 | 91.0% | 10.0% | — | |
| 2026-08-26 | FGSM (rappel) | EMNIST full | 0.30 | 91.0% | 4.0% | — | |
| 2026-08-25 | PGD untgt (20 st.) | MNIST full | 0.05 | 98.6% | 90.0% | 8.8% | |
| 2026-08-25 | PGD untgt (20 st.) | MNIST full | 0.10 | 98.6% | 44.4% | 54.4% | |
| 2026-08-25 | PGD untgt (20 st.) | MNIST full | 0.20 | 98.6% | 0.0% | 99.0% | effondrement total |
| 2026-08-25 | PGD untgt (20 st.) | MNIST full | 0.30 | 98.6% | 0.0% | 99.4% | 0.0% exact |
| 2026-08-25 | FGSM (rappel) | MNIST full | 0.05 | 98.6% | 95.4% | — | même échantillon que PGD |
| 2026-08-25 | FGSM (rappel) | MNIST full | 0.10 | 98.6% | 76.2% | — | |
| 2026-08-25 | FGSM (rappel) | MNIST full | 0.20 | 98.6% | 21.6% | — | |
| 2026-08-25 | FGSM (rappel) | MNIST full | 0.30 | 98.6% | 1.8% | — | |
| 2026-08-25 | Transfert full->classic | MNIST | 0.10 | — | cible 67.6% | 25.7% | boîte noire |
| 2026-08-25 | Transfert full->classic | MNIST | 0.20 | — | cible 40.0% | 50.0% | |
| 2026-08-25 | Transfert full->classic | MNIST | 0.30 | — | cible 22.6% | 72.3% | transfert fort |
| 2026-08-25 | Transfert full->max_config | MNIST | 0.10 | — | cible 96.4% | 8.2% | régularisation protège |
| 2026-08-25 | Transfert full->max_config | MNIST | 0.20 | — | cible 85.4% | 15.1% | |
| 2026-08-25 | Transfert full->max_config | MNIST | 0.30 | — | cible 49.6% | 50.2% | |
| 2026-08-25 | Transfert max_config->full | MNIST | 0.10 | — | cible 95.6% | 8.0% | |
| 2026-08-25 | Transfert max_config->full | MNIST | 0.20 | — | cible 77.6% | 31.8% | |
| 2026-08-25 | Transfert max_config->full | MNIST | 0.30 | — | cible 40.2% | 60.7% |
| 2026-08-25 | Adv. training (équitable) | défendu vs same-data | clean | 81.0% | 86.0% | — | +5.0 pts même en propre |
| 2026-08-25 | Adv. training (équitable) | défendu vs same-data | 0.05 | 59.6% | 72.8% | — | +13.2 pts |
| 2026-08-25 | Adv. training (équitable) | défendu vs same-data | 0.10 | 43.0% | 59.6% | — | +16.6 pts |
| 2026-08-25 | Adv. training (équitable) | défendu vs same-data | 0.20 | 16.2% | 34.6% | — | +18.4 pts |
| 2026-08-25 | Adv. training (équitable) | défendu vs same-data | 0.30 | 8.4% | 17.4% | — | +9.0 pts | |
