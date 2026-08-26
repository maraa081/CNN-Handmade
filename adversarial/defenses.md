# Défenses en détail — théorie, algorithmes, limites

> Ce document explique **pourquoi** les défenses fonctionnent, **comment**
> elles sont implémentées dans ce repo, et **où sont leurs limites**.
> Complément de `README.md` (vue d'ensemble) et de `memoire.md` (tracé des
> expériences). À lire après `attacks.md` (les défenses se comprennent
> contre les attaques).

---

## 1. Le paysage des défenses

On classe les défenses en 3 familles :

| Famille | Idée | Exemples |
|---|---|---|
| **Entraînement robuste** | Apprendre AU modèle à résister | Adversarial training (FGSM ou PGD) |
| **Transformation d'entrée** | Nettoyer l'entrée à l'inférence | Feature squeezing, débruitage, JPEG |
| **Détection** | Refuser les entrées suspectes | Analyse de la sortie, des gradients |

Ce repo implémente les deux premières familles : **adversarial training**
(`defend.py`, `harden.py`) et **feature squeezing** (`harden.py`). La
détection n'est pas traitée (scope assumé).

---

## 2. Adversarial training — la défense de référence

### 2.1 Le problème formel (Madry et al., 2018)

Un modèle classique minimise la loss sur les données propres :

```
min_θ  E_(x,y) [ L(f_θ(x), y) ]
```

Un modèle robuste doit minimiser la loss **sur le pire cas dans la boule** :

```
min_θ  E_(x,y) [ max_δ, ||δ||_∞<=ε  L(f_θ(x+δ), y) ]
```

- Le **max intérieur** : l'attaquant choisit la pire perturbation δ dans la
  boule (c'est exactement ce que fait PGD — voir `attacks.md`).
- Le **min extérieur** : l'entraînement choisit les poids qui minimisent la
  loss même sur ces pires entrées.

Autrement dit : **on entraîne contre l'attaquant le plus fort possible**.
C'est un jeu min-max, et Madry montre que PGD est le bon solveur du max
intérieur : entraîner contre PGD donne la robustesse la plus élevée possible
pour les attaques de première ordre.

### 2.2 La version simple : FGSM training (`defend.py`)

L'approximation historique (Goodfellow 2014) : remplacer le max intérieur par
**une seule étape FGSM**. À chaque batch :

```
pour chaque batch (x, y) :
    x_adv = FGSM(f_θ, x, y, eps_train)     # 1 pas de gradient signé
    loss = L(f_θ(x), y) + L(f_θ(x_adv), y) # propre + adverses
    mise à jour de θ par descente de gradient
```

Points d'implémentation (defend.py) :

- `eps_train` : l'ε utilisé PENDANT l'entraînement (0.15 dans nos runs).
  C'est le niveau de menace qu'on déclare — l'entraînement ne protège que
  contre les attaques dans cette boule.
- Le modèle voit chaque batch **deux fois** : propre et attaqué. La taille
  effective du jeu double : c'est une forme d'**augmentation de données**
  (observé : le défendu gagne même en accuracy propre quand on compare à
  données égales, +5 pts).
- `eval_defended.py` évalue le modèle défendu (FGSM + PGD) sans ré-entraîner.

### 2.3 Pourquoi FGSM training ne suffit PAS (mesuré)

Résultat de nos runs (PGD sur le modèle défendu FGSM, 500 images) :

| ε | standard (full) | défendu (FGSM train) | gain |
|---|---|---|---|
| 0.05 | 90.0% | 64.8% | -25.2% |
| 0.10 | 44.4% | 36.8% | -7.6% |
| 0.20 | 0.0% | 3.6% | +3.6% |
| 0.30 | 0.0% | 0.0% | +0.0% |

Deux problèmes visibles :

1. **À petit ε, le défendu est PIRE que le standard.** Il a été entraîné avec
   des exemples à ε=0.15 : il est "calibré" pour cette amplitude et se
   comporte mal sur les perturbations plus petites (ses features ont été
   modifiées par l'entraînement adverse).
2. **À grand ε, il s'effondre quand même (0.0% à ε=0.3).** FGSM à 1 étape est
   un adversaire trop faible : le max intérieur n'est pas vraiment maximisé.
   Le modèle n'a appris à se défendre que dans les directions explorées par
   FGSM — PGD trouve les faiblesses résiduelles.

C'est le phénomène de **gradient masking** : la défense ne supprime pas la
vulnérabilité, elle la déplace vers des directions que l'adversaire utilisé
à l'entraînement ne visite pas. Un adversaire plus fort la retrouve.

### 2.4 La version forte : PGD training (`harden.py`, couche 1)

On remplace l'adversaire d'entraînement par **PGD (7 itérations, random
start)** :

```
pour chaque batch (x, y) :
    x_adv = PGD(f_θ, x, y, eps=0.3, steps=7)   # max intérieur fort
    loss = L(f_θ(x), y) + L(f_θ(x_adv), y)
    mise à jour de θ
```

Coût : chaque batch demande ~8 passes forward/backward (1 propre + 7 PGD)
au lieu de 2. C'est le prix computationnel de la robustesse.

---

## 3. Feature squeezing (Xu et al., 2018) — `harden.py`, couche 2

### L'idée

Une perturbation adversarial est un **écart minuscule** sur chaque pixel
(au plus ε). Si on réduit la **profondeur de bits** de l'image avant
l'inférence, ces écarts minuscules sont **écrasés par la quantification**,
tandis que la structure utile de l'image (formes, bords) survit — le réseau
n'a pas besoin de 256 niveaux de gris pour reconnaître un chiffre.

### La formule

```
x_squeezed = round(x * (2^b - 1)) / (2^b - 1)     # b bits (3-4 en pratique)
```

Avec 3 bits : 8 niveaux de gris. Toute perturbation < 1/8 de la plage
disparaît ou est arrondie.

### Propriétés

- **Aucun ré-entraînement** : applicable à un modèle déjà déployé.
- **Complémentaire** de l'adversarial training : l'un agit sur les poids,
  l'autre sur l'entrée. Combinés, ils se couvrent mutuellement (une attaque
  doit survivre à la fois à la robustesse des poids ET au nettoyage d'entrée).
- Limite : si ε est grand (0.3), la perturbation survit partiellement à la
  quantification — le squeezing ne suffit pas seul, il renforce.

---

## 4. La méthodologie d'évaluation (la preuve)

### 4.1 Tester contre l'attaquant le plus fort

Une défense qui ne tient que contre FGSM n'est pas une défense : on évalue
toujours contre **PGD** (l'attaquant le plus fort de première ordre). Nos
runs montrent pourquoi : le défendu FGSM tenait à 22% sous FGSM ε=0.3 mais
tombait à 0% sous PGD.

### 4.2 Tester TOUTES les attaques du dossier

`harden.py` évalue contre : FGSM non ciblée, PGD non ciblée, FGSM ciblée,
PGD ciblée, et **transfert** (attaque générée sur un autre modèle). C'est
la couche 3 : la preuve multi-attaques.

### 4.3 Comparer à données égales

Comparer un modèle défendu (5000 images) au modèle full (60 000 images)
mélange l'effet de la défense et l'effet de la quantité de données. On
entraîne donc un baseline standard sur **exactement les mêmes images**
(`standard_same_data.npz`) :

| ε | standard (mêmes données) | défendu | gain |
|---|---|---|---|
| clean | 81.0% | 86.0% | +5.0% |
| 0.05 | 59.6% | 72.8% | +13.2% |
| 0.10 | 43.0% | 59.6% | +16.6% |
| 0.20 | 16.2% | 34.6% | +18.4% |
| 0.30 | 8.4% | 17.4% | +9.0% |

Leçon méthodologique : **toute comparaison défense/attaque doit se faire à
données égales**, sinon on confond défense et augmentation de données.

---

## 5. La version durcie : résultats et compromis

### 5.1 Le compromis robustesse/accuracy

| ε | standard (full) | durci (PGD train) | gain |
|---|---|---|---|
| clean | 98.6% | 67.2% | -31.4% |
| 0.20 FGSM | 21.6% | 34.8% | +13.2% |
| 0.30 FGSM | 1.8% | 23.8% | +22.0% |
| 0.20 PGD | 0.0% | 9.6% | +9.6% |
| 0.30 PGD | 0.0% | 1.2% | +1.2% |

Le durci **domine dès que l'attaque est forte** (+22 pts à ε=0.3 FGSM,
+9.6 pts à ε=0.2 PGD) mais perd **31 pts en accuracy propre**. Ce n'est pas
un bug : c'est le compromis fondamental. Un modèle entraîné contre des
perturbations à ε=0.3 "consomme" de la capacité à distinguer des images
proches — des paires de lettres quasi identiques deviennent confondues.

En pratique, **on ajuste eps_train au niveau de menace** :
- eps 0.15 → meilleur clean, protection modérée
- eps 0.3 → robustesse max, clean dégradé

### 5.2 Les autres effets mesurés

- **Attaques ciblées échouent** sur le durci (53-64% d'acc même à ε=0.3) :
  la robustesse gomme les "directions sûres" que l'attaquant ciblé exploite.
- **Transfert atténué** (40-66% d'acc contre des attaques générées sur un
  autre modèle) : les frontières du durci sont assez différentes de celles
  du standard pour casser une partie de la transferabilité.

### 5.3 La philosophie : defense in depth

Aucune défense seule n'est parfaite. La démarche du dossier :

1. **Adversarial training PGD** → robustesse des poids (la base, Madry)
2. **Feature squeezing** → nettoyage d'entrée (complément, Xu)
3. **Évaluation multi-attaques** → vérifier, pas croire

La leçon générale : **une défense se prouve contre l'attaquant le plus fort,
pas contre la version la plus simple de l'attaque** — et le prix à payer
(accuracy propre, coût de calcul) doit être documenté, pas caché.

---

## 6. Ce qu'on n'a PAS fait (scope honnête)

- **Robustesse certifiée** (bornes formelles type interval bound propagation)
- **Détection d'attaques** (rejeter les entrées adverses au lieu de prédire)
- **Débruitage appris** (auto-encodeur de nettoyage)
- **Distillation défensive**, **mixup adversarial**, etc.

Ces pistes sont documentées ici pour montrer qu'on connaît les limites de
notre périmètre — un mémoire honnête liste ce qui n'a pas été couvert.

---

## 7. Références

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014)
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018)
- Xu et al., *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks* (2018)
- Papernot et al., *Transferability in Machine Learning* (2016)
- MITRE ATLAS : atlas.mitre.org (taxonomie des attaques/défenses IA côté défenseur)
