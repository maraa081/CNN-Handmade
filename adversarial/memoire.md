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
  Relancé en fond le 2026-08-25 -> les expériences de transfert attendront ce modèle.

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
| 2026-08-25 | Transfert max_config->full | MNIST | 0.30 | — | cible 40.2% | 60.7% | |
