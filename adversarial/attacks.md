# Attaques en détail — théorie, algorithmes, implémentation

> Ce document explique **pourquoi** les attaques fonctionnent, **comment** elles
> sont implémentées dans ce repo (CNN from scratch : on contrôle le gradient de
> A à Z), et **comment lire les résultats**. Complément de `README.md`
> (vue d'ensemble) et de `memoire.md` (tracé des expériences).

---

## 1. Modèle de menace (threat model)

Avant d'attaquer, on définit ce que l'attaquant sait et ce qu'il veut.

### Ce que l'attaquant sait

| Scénario | Connaissance | Réaliste ? |
|---|---|---|
| **Boîte blanche (white-box)** | Le modèle complet : poids, architecture, gradients | Rassurant pour apprendre : on a TOUT (notre framework nous donne `dL/dx`) |
| **Boîte noire (black-box)** | Rien que les prédictions (API) | Le cas réel : une API publique ne révèle pas ses poids |

Toutes nos attaques directes (FGSM, PGD, ciblées) sont **boîte blanche**.
Le **transfert** est notre pont vers la boîte noire : on attaque un modèle
qu'on possède, et l'exemple adversarial trompe aussi le modèle distant.

### Ce que l'attaquant veut

| Objectif | Définition | Difficulté |
|---|---|---|
| **Non ciblé (untargeted)** | Faire prédire **n'importe quelle** classe ≠ la bonne | Facile (une seule mauvaise classe suffit) |
| **Ciblé (targeted)** | Faire prédire **une classe précise** (ex. forcer "3") | Plus dur (il faut pousser vers UN point précis) |

C'est pour ça que dans les résultats, les attaques ciblées laissent toujours
plus d'accuracy que les non ciblées.

### La contrainte : imperceptibilité

Une attaque n'a de sens que si l'image modifiée reste **indiscernable** d'une
vraie image. Mathématiquement, on contraint la perturbation dans une **boule L∞** :

```
||x_adv - x||_∞ <= ε        (chaque pixel bouge d'au plus ε)
```

ε est l'**amplitude maximale du bruit par pixel**. Sur une image normalisée
[0, 1], ε = 0.1 signifie que chaque pixel peut bouger de 10% de la plage —
c'est déjà visible ; ε = 0.05 est le seuil où l'œil humain ne voit presque rien.

---

## 2. Notation

```
x          image d'entrée (1, 28, 28) normalisée dans [0, 1]
y          vraie classe (one-hot ou index)
f(x)       sortie softmax du réseau (probabilités)
L(f(x), y) loss (cross-entropy)
∇_x L      gradient de la loss PAR RAPPORT À L'ENTRÉE (le réseau n'est pas modifié)
ε          rayon de la boule L∞ (amplitude max du bruit)
clip(v, a, b)  projection de v dans [a, b] (composante par composante)
```

Point clé : on dérive par rapport à `x`, pas par rapport aux poids. C'est
exactement ce que retourne notre backward si on le branche sur l'entrée.

---

## 3. FGSM — Fast Gradient Sign Method (Goodfellow, 2014)

### L'intuition

Un réseau ReLU est **linéaire par morceaux**. Dans une petite région, il se
comporte comme une fonction linéaire `f(x) ≈ w·x + b`. Pour une fonction
linéaire, la direction qui maximise la loss (pour une norme L∞ bornée) est
simplement `sign(∇L)` : on pousse **chaque pixel** dans la direction où la
loss monte, d'au plus ε. Chaque pixel ne bouge que de ε, mais la somme des
effets sur des centaines de dimensions fait basculer la prédiction.
C'est le "high-dimensional linearity" de Goodfellow : un modèle très
non-linéaire à l'échelle globale reste linéaire à l'échelle d'un pixel.

### La formule

```
x_adv = x + ε · sign(∇_x L(f(x), y))
x_adv = clip(x_adv, 0, 1)
```

### L'algorithme

```
1. forward :  p = f(x)                    # prédiction
2. loss :     L = cross_entropy(p, y)
3. backward : g = dL/dx                   # gradient par rapport à l'entrée
4. sign :     s = sign(g)                 # direction +1/-1 par pixel
5. step :     x_adv = x + ε * s
6. clip :     x_adv = clip(x_adv, 0, 1)   # rester dans l'espace image
```

### Implémentation (fgsm.py)

- Le gradient `dL/dx` vient de notre framework : `model.backward()` calcule
  `dL/dW` pour l'entraînement, mais `dL/dx` s'obtient en propageant le
  gradient depuis la loss jusqu'à l'entrée (mêmes couches, sens inverse).
- `--targeted` inverse le signe : on **minimise** la loss vers la classe
  cible (`x_adv = x - ε·sign(∇L_target)`).
- L'échantillon de test est tiré **aléatoirement** (le test set EMNIST est
  trié par classe : évaluer sur les 500 premières images = évaluer sur 1 classe).

### Forces et limites

| | |
|---|---|
| Force | 1 seule passe de backward : quasi gratuit, très utilisé en pratique |
| Limite | Une seule étape = optimisation grossière du max local. PGD fait mieux |

---

## 4. PGD — Projected Gradient Descent (Madry, 2018)

### L'intuition

FGSM fait **un pas** dans la direction du gradient. PGD fait **plusieurs
petits pas**, en **re-projetant** à chaque fois dans la boule L∞ de rayon ε.
C'est une vraie optimisation de `max_δ L(f(x+δ), y)` contrainte à `||δ||_∞ <= ε`
— l'attaquant "le plus fort possible" dans la boule. Madry et al. ont montré
que PGD est le gold standard : si un modèle résiste à PGD, il résiste à
toutes les attaques de première ordre dans cette boule.

### La formule

```
x_0 = x + U(-ε, ε)                          # démarrage aléatoire dans la boule
x_{t+1} = clip( x_t + α · sign(∇_x L(f(x_t), y)),  x - ε,  x + ε )
```

avec `α` le pas (défaut `ε/4`), et le clip double :
- `[x-ε, x+ε]` : rester dans la boule (projection L∞)
- `[0, 1]` : rester dans l'espace image

### Pourquoi c'est plus fort que FGSM

1. **Plusieurs étapes** : FGSM s'arrête après 1 pas, souvent loin du max.
   PGD continue à grimper la loss tant qu'il reste dans la boule.
2. **Démarrage aléatoire** (`random start`) : FGSM démarre toujours de x,
   PGD explore plusieurs points de départ et converge vers un meilleur max.
   En pratique le random start apporte plusieurs points de robustesse.
3. **Projection** : à chaque pas on force la contrainte — on ne "sort" jamais
   de la boule, donc la perturbation reste imperceptible.

### L'algorithme

```
1. x_t = x + U(-ε, ε)                       # random start
2. pour t = 1..steps :
   a. forward + loss + backward -> g = dL/dx_t
   b. x_t = x_t + α * sign(g)               # pas de gradient
   c. x_t = clip(x_t, x-ε, x+ε)             # projection dans la boule
   d. x_t = clip(x_t, 0, 1)                 # projection image
3. retourner x_t
```

### Implémentation (pgd.py)

- `--steps` : nombre d'itérations (20 par défaut, 40 pour les attaques ciblées).
- `--alpha` : pas (défaut `eps/4`).
- `--compare` : lance FGSM sur le même échantillon pour la comparaison directe.
- `--targeted --target 3` : version ciblée (minimise la loss vers la classe 3).

---

## 5. Attaques ciblées (targeted)

### La différence mathématique

Non ciblée : **maximiser** la loss de la vraie classe.

```
x_adv = x + ε · sign(∇_x L(f(x), y_true))
```

Ciblée : **minimiser** la loss de la classe cible (on veut que le modèle
soit TRÈS confiant sur la classe cible).

```
x_adv = x - ε · sign(∇_x L(f(x), y_target))     # FGSM ciblé
x_{t+1} = clip( x_t - α·sign(∇_x L(f(x_t), y_target)), x-ε, x+ε )   # PGD ciblé
```

### Pourquoi c'est plus dur

Non ciblé : n'importe laquelle des (K-1) autres classes suffit.
Ciblé : il faut pousser la sortie vers UN point précis de la simplex —
la loss cible doit dominer toutes les autres. Résultat mesuré sur nos runs :
les attaques ciblées laissent 53-64% d'accuracy sur le modèle durci quand les
non ciblées tombent à 0-10%.

---

## 6. Transfert d'attaque (black-box)

### Le phénomène

Un exemple adversarial généré contre un modèle A trompe aussi un modèle B,
même si B a été entraîné séparément. C'est la **transferabilité**
(Papernot et al., 2016).

### Pourquoi ça marche

Deux réseaux entraînés sur les mêmes données apprennent des **caractéristiques
similaires** (les mêmes motifs de bord, de forme). Les régions de l'espace
d'entrée où ils se trompent se recouvrent partiellement : la direction
`sign(∇L)` qui trompe A pousse souvent B dans la même zone d'erreur.

### Conséquence sécurité

C'est ce qui rend les attaques **boîte noire** possibles : on n'a pas besoin
du modèle victime, on en entraîne un substitut local, on génère des exemples
adverses dessus, et ils passent chez la victime. Mesuré dans nos runs :
- `full -> classic` (mêmes SGD) : 72% de transfert à ε=0.3 -> très vulnérable
- `full -> max_config` (Dropout + L2) : 50% à ε=0.3 -> la régularisation casse
  une partie du transfert (le modèle a des frontières "plus lisses")

---

## 7. Lire les résultats

Chaque run affiche un tableau :

| Colonne | Signification |
|---|---|
| `acc attaque` | accuracy du modèle sur les images attaquées (le score à battre) |
| `flip` | % d'images dont la prédiction a CHANGÉ (pas forcément fausse : une image mal prédite en propre qui change de classe compte) |
| `cible` | % de réussite de l'objectif (0.0% = jamais poussé vers la classe cible, pour les non ciblées) |
| `perte acc` | acc propre - acc attaqué (la chute due à l'attaque) |

Règle de lecture : **plus `acc attaque` est bas, plus le modèle est
vulnérable**. Un modèle à 0.0% sous PGD ε=0.2 ne reconnaît plus AUCUNE
image de l'échantillon.

---

## 8. Ce qu'on n'a PAS fait (scope honnête)

- Pas d'attaques L2 ou L0 (on s'est limité à L∞, la plus utilisée)
- Pas d'attaques basées sur la confiance seule (boîte noire sans substitut)
- Pas de Carlini-Wagner (optimisation directe, très lente sur un CNN NumPy)
- Le random start PGD n'est testé qu'avec 1 restart (pas d'ensemble complet)

Voir `defenses.md` pour le pendant défensif.
