# L'ordre dans lequel on présente la difficulté construit la robustesse

**Une loi sur le budget de déplacement de l'attaque interne, trouvée, reproduite
sur un second jeu de données, et passée au juge officiel.**

*Brouillon FR du 2026-09-13 (version de travail, chiffres vérifiés, figures à
produire). État et points ouverts : `TRAME-fr.md`. Sources de chaque chiffre :
`adversarial/memoire.md` et les JSON de `adversarial/results/logs/`.*

---

## Résumé

Ce travail part d'un CNN écrit à la main en NumPy, puis le casse, puis le
défend, puis casse sa propre défense. Le résultat central n'est pas un score :
c'est une **loi sur la manière de présenter la difficulté pendant
l'entraînement adversarial**.

Un modèle entraîné contre une attaque interne de budget **décroissant**
(`2 -> 0.2 eps`, 11 minutes de calcul) atteint **63.8%** de pire cas. Le même
modèle, entraîné contre exactement les mêmes budgets dans l'ordre **croissant**
(`0.2 -> 2 eps`, même coût, mêmes 11 minutes), atteint **85.8%**. Vingt-deux
points d'écart, une seule variable changée : l'**ordre**.

Et le juge officiel **amplifie** l'effet au lieu de le réduire. Passés à
AutoAttack sur 10 000 images, les deux membres de la paire donnent **82.40%**
contre **50.35%**, soit **32 points d'écart** — un écart plus grand que celui
mesuré par notre propre suite d'attaques. C'est la première fois dans ce travail
qu'une mesure non biaisée va dans le sens de la mesure biaisée, en plus fort.

Le résultat se reproduit sur un second jeu de données (KMNIST, kana japonais) où
l'écart officiel entre les deux ordres atteint **49.5 points** (51.03% contre
1.49%, mêmes budgets, même coût). En revanche, le *niveau* ne se transpose pas : à
recette identique, les chiffres officiels passent de 82.40% (MNIST) à 58.53%
(KMNIST), soit **-23.9 points**. La formulation retenue est donc : *la recette se
transpose, le niveau non*.

Trois choix méthodologiques structurent le reste : tous les chiffres annoncés
sont ceux d'**AutoAttack** sur 10 000 images ; notre propre suite d'attaques est
présentée comme un outil de diagnostic, avec son biais mesuré (2 à 15.5 points
d'optimisme selon le modèle) ; et le critère d'adoption de chaque recette est
écrit **avant** le run correspondant, y compris quand il se révèle faux.

---

## 1. Introduction

### 1.1 Pourquoi coder un réseau de neurones à la main

Le point de départ est un exercice : écrire un CNN sans framework, en Python et
NumPy seulement. Chaque brique est faite à la main, `im2col` pour la
convolution, la rétropropagation, les optimiseurs (SGD, Momentum, Adam), les
formules d'attaque, et jusqu'aux bornes de confiance du smoothing. Le réseau
compte 421 642 paramètres et atteint 98.6% sur MNIST.

Cet exercice a un intérêt caché pour la suite. Quand on attaque ensuite ce
réseau, on sait exactement ce qu'on attaque : il n'y a pas de couche cachée
d'une bibliothèque, pas de préprocessing non documenté, pas de différence
possible entre ce qu'on croit mesurer et ce qu'on mesure. La modeste taille du
modèle devient un atout : les expériences tiennent en minutes, pas en jours, ce
qui permet de mesurer une courbe complète plutôt qu'un point.

### 1.2 Le fil

Le travail suit un arc simple, et chaque étape est un échec de la précédente.

1. **Construire** : un CNN fait main, 98.6% sur MNIST.
2. **Attaquer** : FGSM le fait tomber à 1.8%, PGD à 0.0%. La précision propre ne
   dit rien de la robustesse.
3. **Défendre** : l'entraînement adversarial remonte la robustesse, mais révèle
   une structure inattendue, une courbe en cloche dont le sommet est à
   l'intérieur de l'intervalle, pas au maximum de force.
4. **Casser sa propre défense** : en cherchant le mécanisme, on découvre que ce
   n'est pas la force de l'attaque d'entraînement qui compte, mais **l'ordre
   dans lequel on présente la difficulté**.

### 1.3 Ce que ce travail apporte

- **Une loi simple, vérifiée et contre-intuitive** : à budgets internes
  identiques et à coût identique, l'ordonnancement du budget de l'attaque
  interne fait 22 points d'écart sur MNIST et 41 sur KMNIST.
- **Sa réplication** sur un second jeu de données, avec ce qui se transpose (la
  loi) et ce qui ne se transpose pas (le niveau, -23.9 points).
- **Une méthode de mesure honnête** : AutoAttack comme juge, notre suite comme
  diagnostic, et notre propre biais chiffré plutôt que caché.
- **Une extension** : rendre le budget adaptatif batch par batch ajoute 7 à 9
  points officiels au prix de quelques minutes.

Ce que ce travail ne prétend pas être : un état de l'art CIFAR-10, une nouvelle
borne théorique, ni un record de robustesse. C'est un travail de **mécanisme**,
sur un petit modèle, avec des chiffres reproductibles et un ancrage explicite
dans la littérature (S.5.1).

---

## 2. Le terrain de jeu et la manière de mesurer

### 2.1 Deux moteurs, les mêmes mathématiques

Le projet contient deux implémentations du même réseau : la version NumPy faite
main, et un portage PyTorch qui sert à entraîner plus vite (environ neuf fois
plus rapide sur CPU, davantage sur GPU). Les paramètres sont interchangeables au
format `.npz`, et l'architecture est strictement identique partout : deux
couches convolutives (32 puis 64 canaux), deux max-poolings, une couche dense,
421 642 paramètres. Aucune comparaison de ce document ne mélange deux
architectures.

Le modèle de menace est le même du début à la fin : perturbation de norme
L-infini au plus **eps = 0.30**, sur des images normalisées dans `[0,1]`. Ce
n'est pas une valeur choisie à l'aveugle : **c'est la valeur du banc d'essai MNIST
de la littérature**, celle utilisée par Madry et al. (2017) et par TRADES (Zhang
et al., 2019) — c'est exactement ce qui rend l'ancrage du S.5.1 légitime. C'est
also le régime où un modèle non défendu s'effondre à **0.0%**, ce qui en fait un
terrain sans ambiguïté pour observer les mécanismes de défense. La dépendance de
la loi à eps (0.1, 0.2) n'a en revanche **pas** été testée : voir les limites.
Toute attaque d'évaluation est rejouée à eps = 0.30, y compris pour les modèles
dont l'attaque *d'entraînement* utilisait un autre budget — c'est ce qui rend les
lignes du tableau final comparables entre elles.

### 2.2 Les six règles de mesure

La moitié du travail a consisté à apprendre à mesurer. Les six règles ci-dessous
sont toutes nées d'une erreur concrète :

1. **Annoncer sur 10 000 images.** Sur les 500 mêmes images de test utilisées
   pendant le développement, tous les modèles étaient tirés vers le haut
   (par exemple 95.6% -> 93.8% pour un des modèles à budget adaptatif).
2. **Sélectionner sur la validation, annoncer sur le test.** Un bug a produit un
   modèle « meilleur » qui était en réalité le modèle de l'epoch 1.
3. **Le juge est AutoAttack.** Notre suite maison est rapide et sert à
   diagnostiquer et à expliquer ; elle ne sert pas à annoncer.
4. **Jamais de rayon robuste mesuré au PGD seul.** Sur un modèle à surface
   rugueuse, le rayon mesuré au PGD classe **à l'envers** : le modèle `abl_b` a
   le plus grand rayon PGD de la série et le quatrième pire cas. Toujours
   mesurer le rayon au PGD *et* au Square.
5. **La sensibilité trie, elle ne prouve pas.** Une mesure de deux secondes
   (sensibilité de la fonction de perte sous perturbation dans la boule) classe
   les modèles dans le même ordre que le pire cas officiel, 5 fois sur 5. Mais un
   modèle qui masque son gradient est plat par construction : cette mesure sert à
   trier et à expliquer, jamais à conclure.
6. **Une seule variable à la fois, et le critère d'adoption écrit avant le run.**
   Les critères écrits avant les runs sont conservés dans ce document, y compris
   ceux qui se sont révélés faux.

### 2.3 Le prix de l'honnêteté, chiffré

Notre suite maison (500 images, Square poussé à 3000 pas) est **optimiste de 3 à
15.5 points** selon le modèle, comparée à AutoAttack sur 10 000 images :

| modèle | pire cas maison | pire cas officiel | écart |
|---|---|---|---|
| `A6` (plan `0.2 -> 2 eps`) | 85.8% | 82.40% | -3.4 |
| KMNIST `0.2 -> 2 eps` | 62.6% | 58.53% | -4.1 |
| `A8` (plan `0.2 -> 1 eps`) | 84.4% | 78.25% | -6.2 |
| KMNIST `0.2 -> 1 eps` | 58.4% | 51.03% | -7.4 |
| KMNIST `1 -> 0.2 eps` (plan inverse) | 17.0% | 1.49% | **-15.5** |
| `A9` (plan `2 -> 0.2 eps`) | 63.8% | 50.35% | **-13.5** |

Et ce biais n'est pas dispersé au hasard : il **sépare les deux familles de
recettes**. Les cinq modèles dont la recette démarre bas sont surestimés de 2.4 à
7.4 points ; les trois recettes à **départ haut** (`abl_a` -12.5, `A9` -13.5, plan
inverse KMNIST -15.5) sont surestimées de 12.5 à 15.5 points. Les deux intervalles
ne se recouvrent pas.

**Ce que ça signifie.** Notre suite d'attaques n'est pas biaisée « en moyenne » :
elle est optimiste **précisément sur la famille que la loi désigne comme
mauvaise**. C'est un argument en faveur de la loi, pas contre elle : les modèles à
départ haut sont exactement ceux dont la robustesse apparente ne survit pas à une
attaque plus forte. C'est aussi la raison technique de la règle de mesure n°3 : le
seul chiffre annonçable est celui du juge.
| `abl_a` (budget constant 2 eps) | 63.2% | 50.71% | -12.5 |

Ce biais n'est pas une anecdote : il est **plus grand sur le modèle le plus
faible**, et le classement qu'il produit est conservé sur les cinq modèles sains
mais se dégrade sur le seul modèle suspect. Le meilleur modèle maison était
annoncé à 63.2% ; le chiffre officiel est **50.71%**. Les deux sont conservés
dans ce document : c'est le prix de la méthode, et le lecteur peut vérifier.

---

## 3. Attaquer : casser le modèle propre

Le modèle fait main atteint 98.6% sur MNIST. Une seule étape de gradient (FGSM,
eps = 0.30) suffit à le faire tomber à **1.8%**. Vingt pas de PGD le font tomber
à **0.0%**.

La suite d'attaques du projet couvre plusieurs familles : à gradient (FGSM, PGD,
APGD-CE, APGD-DLR), à distance minimale (CW-L2), décisionnelle (Boundary), sans
gradient (Square, NES), l'échappement de gradients obfusqués (BPDA + EOT), et
une défense certifiée (randomized smoothing, S.8.1).

**Le premier résultat intéressant n'est pas la chute, c'est l'accord entre les
familles.** Sur un modèle sain, les attaques se rejoignent à quelques points. Le
meilleur modèle sur MNIST (plan `0.2 -> 1 eps`) donne, sur les mêmes 500 images :
FGSM 95.2%, PGD-20 89.2%, APGD-CE 84.4%, APGD-DLR 84.4%, Square 84.6%. Quatre
familles, dont une sans gradient, à moins d'un point les unes des autres.

Sur un modèle atteint de « masquage », elles se contredisent. Le modèle `abl_a`
donne 85.4% sous APGD-DLR et **63.2%** sous Square : 22 points d'écart entre une
attaque à gradient et une attaque sans gradient. Ce désaccord est un **symptôme**,
pas du bruit : c'est lui qui a servi de boussole pendant tout le projet, avant
d'avoir accès au juge officiel.

Enfin, l'entraînement adversarial « naïf » (durcissement contre FGSM seul) donne
l'illusion d'une défense : le modèle résiste à FGSM et reste cassé par PGD. Le
durcissement n'est pas une question de force brute, c'est un problème de recette.
C'est le sujet de la suite.

---

## 4. Défendre : la courbe en cloche

### 4.1 Le premier durcissement, et la leçon des epochs

La première version durcie (NumPy, PGD à 7 pas de pas eps/4, 60 000 images)
monte à 67.2% de précision propre et retombe à **1.2%** de pire cas : à peine
mieux que rien. La campagne suivante (trois recettes, warm start, 60 000 images)
apporte une leçon utile : le run jugé mauvais après 10 epochs atteint **91.0%**
après 120 epochs. Avant de déclarer une recette mauvaise, il faut lui donner son
budget d'epochs. Ce document applique désormais 120 epochs partout.

### 4.2 L'ablation du pas : le sommet est à l'intérieur

On garde tout fixé (120 epochs, PGD-20, eps = 0.30, augmentation, 60 000 images,
graine 42) et on ne change que la **finesse du pas** de l'attaque interne. Le
budget de déplacement, c'est-à-dire la distance maximale que l'attaque peut
parcourir, vaut `pas x nombre de pas`.

| recette | budget interne | pire cas (maison) | précision propre |
|---|---|---|---|
| `run B` | 1.25 eps (pas eps/4) | 42.0% | 99.6% |
| `abl_b` | 5 eps (pas eps/4) | 38.8% | 99.8% |
| `A4` | 0.5 eps (pas eps/10) | 49.2% | 99.4% |
| `v4` / `abl_a` | 2 eps (pas eps/10) | 61.8% / **63.2%** | 99.4% / 99.6% |
| `abl_c` | 20 eps (pas eps) | **1.6%** | 98.4% |

La robustesse apprise est maximale pour un budget **intermédiaire** : ni trop
mou, ni trop dur. Trop mou, l'entraînement apprend une robustesse masquée (le
`run B` affiche 91% sous PGD-20 et 42% de pire cas). Trop dur, l'entraînement se
verrouille.

`abl_c` est le contre-exemple parfait, et le plus instructif : **98.4% de
précision propre**, un modèle qui a l'air intact, pour **1.6%** de pire cas. Le
log d'entraînement explique pourquoi : l'attaque interne ne trompe plus le
modèle, la cross-entropie adverse se bloque à `ln(10) = 2.303` (le niveau du
hasard sur dix classes), et le modèle reste à ce plateau pendant la fin du run.
Il n'apprend plus rien, il répond uniformément.

FIGURE 1 : la courbe en cloche (pire cas en fonction du budget de l'attaque
interne).

### 4.3 Le pas compte autant que le budget

Un détail a failli passer inaperçu : `A4` (pas fin eps/10, 5 pas, budget 0.5 eps)
obtient 49.2% là où `run B` (pas grossier eps/4, budget 1.25 eps) n'obtient que
42.0%. L'ablation précédente confondait deux axes, la **finesse** du pas et le
**nombre** de pas. La formulation retenue est : ce qui compte, c'est la
**lisibilité** de la perturbation, pas seulement sa force. Un pas fin accumule
des directions de gradient cohérentes ; un pas grossier produit une perturbation
que le modèle ne peut pas apprendre à contrer.

### 4.4 Le résultat central : c'est l'ORDRE

**La forme exacte des plans.** Un plan de budget est une rampe **linéaire sur les
epochs** : à l'epoch `t`, le budget vaut `deb + (fin - deb) x (t-1)/(epochs-1)`,
exprimé en multiples de eps. Le pas de l'attaque interne reste **fixe** (eps/10) :
on ne change que le **nombre de pas**, donc `budget = pas x alpha`. Concrètement,
`--plan-budget "0.2,2"` fait passer l'attaque interne de 2 pas (0.2 eps) à l'epoch
1 à 20 pas (2 eps) à l'epoch 120, en une seule rampe monotone. Deux plans ayant
les mêmes bornes mais une autre forme de rampe (logarithmique, par paliers,
loi de puissance) n'ont **pas** été testés : la forme linéaire a été fixée, pas
comparée (voir les limites).

Reste la question la plus intéressante : à budget donné, l'ordre dans lequel on
présente les budgets au modèle compte-t-il ? L'expérience décisive tient en deux
runs, de coût identique, avec exactement le même ensemble de budgets traversés :

| recette | budgets traversés | coût | pire cas (maison) |
|---|---|---|---|
| `A6` : plan `0.2 -> 2 eps` | 2 .. 20 pas | 11 min | **85.8%** |
| `A9` : plan `2 -> 0.2 eps` | 20 .. 2 pas | 11 min | **63.8%** |

**Même somme de calcul, mêmes budgets, ordre inversé : 22 points d'écart.** Le
plan croissant arrive à son budget maximal (le plus dur) *plus tard* ; le plan
décroissant y arrive d'emblée. Et cette observation n'est pas isolée : tous les
mauvais modèles de la série (`abl_a` 63.2%, `A7` 62.0%, `A9` 63.8%) ont un
**départ haut**, tous les bons (`A6` 85.8%, `A8` 84.4%, `A1` 93.6%) commencent à
2 pas (0.2 eps).

La lecture proposée : un départ à budget court joue le rôle d'une régularisation
qui garde l'attaque interne **informative** — elle trompe encore le modèle, donc
elle fournit un signal utile — alors que démarrer à haute force place d'emblée
l'optimisation dans le régime de verrouillage observé sur `abl_c`. Une piste
théorique voisine est l'objectif **min-min** de FAT (Wong et al., ICML 2020) :
l'arrêt anticipé de l'attaque interne change la nature du problème
d'optimisation.

**Deux précautions sur ce résultat, à écrire noir sur blanc.**

1. **Le chiffre-phare, désormais officiel des deux côtés.** `A6` et `A9` sont
tous les deux passés au juge officiel :

| jeu | plan croissant | plan inverse | écart maison | **écart officiel** |
|---|---|---|---|---|
| MNIST | `0.2 -> 2 eps` : **82.40%** | `2 -> 0.2 eps` : **50.35%** | 22.0 | **32.1** |
| KMNIST | `0.2 -> 1 eps` : **51.03%** | `1 -> 0.2 eps` : **1.49%** | 41.0 | **49.5** |

Les deux écarts **grandissent** sous le juge non biaisé. L'objection naturelle
("et si l'écart de 22 points n'était qu'un artefact de votre suite optimiste ?")
se retourne donc : c'est le contraire, et de 10 points sur MNIST comme de 8.5 points
sur KMNIST. Une prédiction avait été écrite avant la mesure d'`A9` (48 à 58%) :
50.35% tombe dedans.
2. **Le mécanisme proposé est une lecture, pas une hypothèse testée.** Il est
cohérent avec la courbe en cloche et avec les logs (`abl_c` verrouille, les
recettes à départ haut n'exploitent pas le signal utile), mais il n'a pas été
isolé de l'alternative évidente : un ordonnancement de budget qui interagit avec
le programme de learning rate, indépendamment du contenu informatif du gradient.
Deux expériences le trancheraient, et elles sont peu coûteuses : (i) un plan
**non monotone** (`0.2 -> 2 -> 0.2 eps` contre `2 -> 0.2 -> 2 eps`), qui distingue
"croissance monotone" de "départ doux" ; (ii) une lecture des trajectoires déjà
enregistrées dans les logs (taux de tromperie et CE adverse epoch par epoch),
qui ne coûte aucun run. Elles sont laissées à un travail ultérieur.

**Précision sur les 22 points : deux mesures indépendantes, pas une.** L'effet a
été reproduit sur un second jeu de données avec des runs distincts (§7.3), où
l'écart passe à 41 points. Ce n'est pas une répétition de graines sur la même
configuration, mais c'est une répétition de l'**effet** sur d'autres données, ce
qui répond à l'objection la plus courante.

La même expérience, avec des plans moins chers, a servi de garde-fou : un plan
`0.2 -> 0.5 eps` (6 min) ne donne que 31.8% et un plan `0.05 -> 0.2 eps` (5 min)
0.0% — baisser le *plafond* détruit la robustesse. C'est le **départ** qui doit
être bas, pas le plafond.

FIGURE 2 : `A6` contre `A9`, mêmes budgets, ordre inversé.

### 4.5 L'extension : un budget adaptatif, batch par batch

Le plan déterministe est une courbe fixée à l'avance. Une variante consiste à
viser, **pour chaque batch**, une difficulté cible et à arrêter l'attaque interne
dès qu'elle est atteinte, au pas `k*`. Le surcoût est nul : à pas fixe, la
trajectoire de PGD contient déjà tous les candidats.

**Paramétrage exact**, pour que le résultat soit reproductible et critiquable :
la cible est un **taux de tromperie** (fraction du batch que l'attaque interne
doit avoir fait basculer), fixée à **0.5** ; l'attaque s'arrête au premier pas `k*`
tel que le taux de tromperie atteint 0.5, et les pas restants sont économisés. La
cible elle-même suit une rampe (0.2 au début, 0.5 à partir de la moitié du run),
parce qu'un modèle encore faible ne peut pas tromper la moitié d'un batch à
l'epoch 1. Un garde-fou alerte si la cible n'est atteinte qu'au plafond de pas sur
une fraction trop grande des batchs (signe qu'on est hors de la fenêtre faisable).

Résultat : **93.6% maison / 91.25% officiel**, contre 82.40% pour le meilleur
plan déterministe. L'asservissement par batch apporte donc +7 à +9 points
officiels pour une dizaine de minutes supplémentaires.

**Ce qui n'a pas été testé sur ce point** : la valeur de la cible. Une variante de
cible plus lisse (cross-entropie relative) et un contrôle à cible très haute (0.9,
qui devrait dégrader) sont décrits dans la documentation de l'attaque mais n'ont
pas été lancés : le résultat annoncé ne vaut donc que pour la cible 0.5, et sa
sensibilité à ce paramètre reste une question ouverte. C'est le meilleur modèle du
projet, et c'est aussi celui dont le paramétrage est le moins ablaté : à corriger
avant toute publication qui le mettrait en avant.

Un modèle plus robuste doit aussi survivre à l'audit de masquage, sans quoi le
gain serait un artefact. Quatre vérifications, toutes passées : les sorties ne
sont pas saturées ; le gradient est informatif (le rapport
gradient/aléatoire est 8.5 fois meilleur que le hasard) ; le transfert depuis un
autre modèle ne bat pas l'attaque en boîte blanche ; et le modèle s'effondre bien
à eps = 0.5 sous un PGD à pas fin (0.6%), ce qui prouve un rayon robuste réel et
non une surface artificiellement plate.

### 4.6 Deux corrections assumées

Deux intuitions ont été écrites, testées, puis abandonnées. Elles font partie du
résultat :

1. « Les rayons robustes sont égaux, donc c'est un effet de fenêtre » : **faux**.
   Le rayon mesuré au PGD seul est invalide quand le gradient ne guide plus
   l'attaque (règle 4).
2. « Le rapport gradient/aléatoire prédit la taille de l'écart entre attaques » :
   **faux** aussi. Le contre-exemple `abl_c` a un rapport de 1.16 et un écart
   *négatif*. Ce rapport prédit seulement si les attaques à gradient sont
   utilisables.

---

## 5. Placer le résultat, et ce qui reste ouvert

### 5.1 Où nous situons par rapport à la littérature

Sur MNIST à eps = 0.30, les implémentations de référence rapportent :

| modèle | propre | robuste |
|---|---|---|
| Madry et al. 2017, PGD-40 | 99.36% | **96.01%** |
| TRADES, Zhang et al. 2019 (`1/lambda = 6`) | 99.48% | **95.60%** |
| **notre** TRADES (variante non conforme, voir S.5.2) | 93.26% | 25.18% |
| notre meilleur (`bande_cible50`), 421k paramètres, 120 epochs | 98.85% | **91.25%** |
| notre `A6` (plan `0.2 -> 2 eps`) | 99.17% | **82.40%** |
| notre `abl_a` (**même** architecture, budget constant 2 eps) | 99.53% | **50.71%** |

Nos lignes TRADES figurent ici par transparence, mais elles ne doivent **pas** être
lues comme un point de comparaison de méthodes : il est établi au S.5.2 que notre
implémentation ne reproduit pas la référence, et l'écart (~70 points) est bien
trop grand pour être un effet de méthode.

Deux phrases suffisent. D'abord, notre meilleur modèle est à environ **cinq
points** de la référence, avec une architecture beaucoup plus petite et 120
epochs : les chiffres de ce document ne sont pas hors-sol. Ensuite, et c'est le
sujet : **dans la même architecture**, le choix de la recette fait 45 points
d'écart (50.71% contre 91.25%). L'article ne revendique pas un record ; il
explique cet écart.

### 5.2 Ce qui est resté ouvert

**Notre implémentation de TRADES ne reproduit pas la référence.** Deux runs,
`beta 2` avec warm start et `beta 6` depuis zéro, donnent **22.01%** et
**25.18%** de robustesse officielle, avec des précisions propres de 96.5% et
93.3% — c'est-à-dire **plus basses** que celles de nos modèles entraînés à la
cross-entropie (99.2% à 99.6%), alors que le terme CE de TRADES est précisément
censé protéger la précision propre. Deux écarts à la référence sont identifiés
dans le code : le **sens de la KL** (nous mesurons `KL(p_adv || p_clean)` là où
la référence minimise `KL(p_clean || p_adv)`) et le fait que **l'attaque interne
maximise la cross-entropie** au lieu du terme KL.

L'écart de ~70 points avec les 95.60% rapportés est trop grand pour être lu comme
une comparaison de méthodes : ces deux runs sont publiés comme une **limite
assumée**, pas comme un résultat sur la perte. Toute phrase du type « l'ordre
compense une perte mal réglée » serait non soutenue par ces mesures.

---

## 6. Le juge officiel

Tous les chiffres annoncés dans ce document viennent d'AutoAttack (version
`standard` : APGD-CE, APGD-T, FAB-T, Square), sur 10 000 images, à eps = 0.30.
Huit modèles ont été croisés (les deux paires à recette appariée y figurent,
c'est ce qui rend le résultat phare vérifiable) :

| modèle | recette | maison | officiel | écart |
|---|---|---|---|---|
| `bande_cible50` | budget adaptatif | 93.6% | **91.25%** | -2.4 |
| `A6` | plan `0.2 -> 2 eps` | 85.8% | **82.40%** | -3.4 |
| `A8` | plan `0.2 -> 1 eps` | 84.4% | **78.25%** | -6.2 |
| KMNIST `0.2 -> 2 eps` | plan `0.2 -> 2 eps` | 62.6% | **58.53%** | -4.1 |
| KMNIST `0.2 -> 1 eps` | plan `0.2 -> 1 eps` | 58.4% | **51.03%** | -7.4 |
| `abl_a` | budget constant 2 eps | 63.2% | **50.71%** | -12.5 |
| `A9` | plan `2 -> 0.2 eps` | 63.8% | **50.35%** | -13.5 |
| KMNIST `1 -> 0.2 eps` | plan inverse | 17.0% | **1.49%** | **-15.5** |

Trois observations.

**Le biais de notre suite est toujours dans le même sens, et il sépare les deux
familles de recettes.** Les cinq modèles à départ bas sont surestimés de 2.4 à 7.4
points ; les trois recettes à **départ haut** (`abl_a` : -12.5, `A9` : -13.5, plan
inverse KMNIST : -15.5) le sont de 12.5 à 15.5 points. Les deux intervalles ne se
recouvrent pas. Notre suite n'est donc pas biaisée « en moyenne » : elle est
optimiste **précisément sur la famille que la loi désigne comme mauvaise**. C'est
un argument en faveur de la loi : les recettes à départ haut sont exactement celles
dont la robustesse apparente ne survit pas à une attaque plus forte.

**Le classement est conservé sur les modèles sains, et il se dégrade exactement
sur la famille que l'article déclare mauvaise.** `abl_a` passe de cinquième en
maison à **sixième** en officiel, `A9` de quatrième à **septième**, et le plan
inverse KMNIST tombe de 17.0% à **1.49%**. À l'inverse, les recettes à départ bas
(`A1`, `A6`, `A8`, et les deux KMNIST) gardent exactement le même ordre.
Formulation à retenir : *la suite maison trie bien les modèles sains et se trompe
en faveur des suspects*.

**Les chiffres officiels eux-mêmes ont des qualités différentes.** AutoAttack a
émis un avertissement sur le modèle KMNIST `0.2 -> 1 eps` (« Square Attack a
réduit la précision robuste de 2.24% »), ce qui signifie que le 51.03% est
lui-même légèrement optimiste. Aucun avertissement sur le 58.53%. Les deux
chiffres KMNIST ne sont donc pas de qualité identique, et la model card le dira.

FIGURE 3 : pire cas maison contre pire cas officiel, avec la diagonale.

---

## 7. Réplication : KMNIST, l'ancre Japon

### 7.1 Pourquoi un second jeu

KMNIST (Kuzushiji-MNIST) contient des caractères japonais cursifs, dans un format
identique à MNIST : images 28x28, dix classes, 60 000 images d'entraînement,
10 000 de test. Mêmes dimensions, même pipeline, même modèle, même eps : c'est un
second jeu **gratuit** pour tester la généralité de la loi. C'est aussi une
accroche cohérente pour un travail réalisé depuis le Japon.

Avant de lancer quoi que ce soit, la prévision a été écrite : « dur ~60-65%,
doux ~84%, ordre inverse -20 points ». Elle s'est révélée à moitié fausse, et
c'est instructif : les prévisions de *niveau* étaient fausses, la prévision
d'*ordre* était juste.

### 7.2 Huit modèles entraînés, neuf points de mesure

| recette | budget interne | pire cas (maison) | officiel |
|---|---|---|---|
| modèle propre (aucune attaque) | - | 0.0% | - |
| constant 2 eps | 20 pas | 1.2% | - |
| plan inverse `1 -> 0.2 eps` | 20 .. 2 pas | 17.0% | **1.49%** |
| constant 1 eps | 10 pas | 15.2% | - |
| plan `0.05 -> 0.2 eps` | 1 .. 2 pas | 0.0% | - |
| plan `0.1 -> 0.5 eps` | 1 .. 5 pas | 25.2% | - |
| plan `0.2 -> 0.5 eps` | 2 .. 5 pas | 31.8% | - |
| plan `0.2 -> 1 eps` | 2 .. 10 pas | 58.4% | **51.03%** |
| plan `0.2 -> 2 eps` | 2 .. 20 pas | **62.6%** | **58.53%** |

### 7.3 Ce qui se transpose

**La loi de l'ordre, et elle s'amplifie.** Le plan croissant `0.2 -> 1 eps` donne
58.4% et son inverse `1 -> 0.2 eps` 17.0% : **41 points d'écart** en maison,
contre 22 sur MNIST. Et surtout, cette paire est la seule du projet dont les deux
côtés sont passés au juge officiel : **51.03% contre 1.49%, soit 49.5 points
d'écart officiel** — un écart plus grand encore que celui mesuré par notre suite.
Mieux : les deux modèles à bas plafond (`0.1 -> 0.5` : 25.2% et `0.05 -> 0.2` :
0.0%) confirment, sur un second jeu, que c'est le *départ* qu'il faut baisser et
non le plafond. Le meilleur plan est aussi le même qu'à MNIST.

### 7.4 Ce qui ne se transpose pas

**Le niveau.** Les deux paires à recette identique, mesurées officiellement des
deux côtés :

| recette | MNIST | KMNIST | écart |
|---|---|---|---|
| plan `0.2 -> 2 eps` | 82.40% | 58.53% | **-23.9 points** |
| plan `0.2 -> 1 eps` | 78.25% | 51.03% | **-27.2 points** |

*La recette se transpose, le niveau non.* C'est la phrase que ce travail cherchait
à pouvoir écrire, et elle est maintenant appuyée par deux paires officielles.

Le mécanisme du décalage est identifiable : à eps = 0.30, l'attaque interne est
relativement beaucoup plus forte sur KMNIST. Sur le modèle propre, un PGD à
0.1 fait tomber MNIST à 44.4% et KMNIST à **9.2%**. Résultat : la recette de
référence à MNIST (budget constant 2 eps, 63.2% de pire cas) **s'effondre** sur
KMNIST à **1.2%** — le budget devient « hors de portée » du jeu, et le modèle
fait le seul arbitrage qui lui reste.

Le log de ce run est éloquent : l'attaque interne trompe 90% du batch du début à
la fin (ce n'est donc *pas* le verrouillage d'`abl_c`), la précision propre de
validation monte à 99.5% — mieux que le modèle de référence propre — pendant que
la robustesse reste bloquée. Un modèle propre *plus* précis et simultanément
inutile : c'est exactement le profil d'un objectif devenu hors d'atteinte.

**La fenêtre de budgets faisables, et sa conséquence sur la portée de la loi.**
L'ensemble des mesures KMNIST délimite empiriquement une fenêtre : un plan dont le
**départ** est bas (2 pas, 0.2 eps) et dont le **plafond** atteint 1 à 2 eps donne
58 à 63% ; un plafond sous 0.5 eps ne produit qu'une robustesse de rayon
minuscule ; un budget **constant** de 2 eps tombe hors de portée et s'effondre à
1.2%. La loi de l'ordre ne s'applique donc pas « dans l'absolu » : **elle
s'applique à l'intérieur d'une fenêtre de budgets faisables, et cette fenêtre
dépend du jeu de données** (elle est plus basse sur MNIST, plus haute sur KMNIST,
où le budget 2 eps constant échoue alors qu'il atteint 63.2% sur MNIST). C'est
une précision importante pour le lecteur : la loi n'est pas un théorème, c'est une
régularité conditionnelle, dont la condition est justement le point à vérifier
avant de transposer une recette.

**Leçon méthodologique** : un sommet de courbe n'est pas un nombre, c'est un
nombre **pour un jeu de données donné**. Toute recette publiée sans son jeu de
validation croisée est une recette qui n'a pas été testée.

FIGURE 4 : les deux courbes, mêmes axes, même forme, hauteurs différentes.

---

## 8. Ce qui tient par garantie, et les limites

### 8.1 La seule garantie du projet

Tous les résultats précédents sont **empiriques** : on attaque, on regarde ce qui
reste. Randomized smoothing (Cohen, Rosenfeld et Kolter, 2019) fournit une
**garantie**. On entraîne un classifieur de base sur des images bruitées
`N(0, sigma^2)`, puis on construit un classifieur lissé
`g(x) = argmax_c P(f(x + bruit) = c)`. Un théorème donne alors un rayon L2 garanti.

Mesuré à `sigma = 0.5`, sur 1 000 images, avec des bornes de confiance exactes
(Clopper-Pearson) et `alpha = 0.001` :

| | valeur |
|---|---|
| précision du classifieur lissé | 99.0% |
| taux d'abstention | 0.5% |
| **rayon L2 certifié médian** | **1.214** |
| précision certifiée à R = 0.30 | 98.5% |
| précision certifiée à R = 1.00 | 83.5% |

C'est le seul chiffre de ce document qui soit une **garantie** et non une
observation : aucune perturbation de norme L2 inférieure ou égale au rayon
certifié ne peut changer la prédiction, par construction. À comparer, en restant
dans la même norme, aux distances L2 que nos attaques CW-L2 doivent atteindre
pour tromper les modèles durcis (de 0.005 à 1.8 selon le modèle) : le modèle
lissé est loin devant.

**Piège de lecture, à ne jamais rater.** La boule L-infini de rayon 0.30 n'est
**pas** incluse dans la boule L2 de rayon 1.214 : une perturbation de norme
L-infini 0.30 peut avoir une norme L2 allant jusqu'à `0.30 x sqrt(784) = 8.4`.
Les résultats L-infini de ce document et cette garantie L2 répondent donc à
**deux questions différentes** : ils ne se comparent pas et ne se remplacent pas.

### 8.2 Limites

- **Le régime testé est étroit** : petites images (28x28), deux jeux seulement,
  un seul eps (0.30), une architecture (421 642 paramètres), un seul moteur
  d'entraînement. La loi est vérifiée sur ce régime, pas démontrée au-delà. Sont
  explicitement hors périmètre : les ensembles de modèles, l'entraînement
  *contre* Square, les modèles au-delà de 1.7M de paramètres, et le transfert
  entre jeux de données.
- **Notre suite maison est optimiste de 2 à 15.5 points** et se trompe en faveur
  des modèles suspects. Elle sert à trier et à expliquer, jamais à annoncer.
- **Le chiffre officiel KMNIST `0.2 -> 1 eps` (51.03%) est lui-même optimiste**,
  AutoAttack signalant une amélioration possible de 2.24% sous Square. Limite
  annoncée, à écrire dans la model card.
- **Robustesse empirique, pas certifiée**, sauf pour le modèle lissé (S.8.1).
- **Notre TRADES ne reproduit pas la référence** (S.5.2) : publié comme limite,
  les deux écarts étant identifiés.
- **Un seul jeu de réplication** et un seul plan gagnant par jeu : la loi de
  l'ordre est reproduite deux fois (MNIST, KMNIST), pas démontrée.
- **Un seul eps** (0.30) : on ne sait pas si la loi tient à eps = 0.1 ou 0.2, où la
  fenêtre de budgets faisables se déplace nécessairement (le verrouillage devient
  plus difficile à provoquer, donc le départ haut pourrait devenir inoffensif).
  Test peu coûteux (un run par eps), non fait.
- **La forme de la rampe n'a pas été comparée** (linéaire contre logarithmique
  contre paliers), pas plus que la valeur de la cible du budget adaptatif
  (S.4.5).
- **Variance d'entraînement non couverte.** Chaque recette est un **run unique**
  (graine 42 pour l'ablation du pas, graine par défaut ailleurs). Les 10 000
  images d'AutoAttack donnent une incertitude d'évaluation négligeable (intervalle
  de Clopper-Pearson d'environ +/- 0.7 point à 82%, le même outil que le S.8.1
  utilise pour le smoothing), mais **la variance de la graine n'est pas mesurée** :
  on ne peut pas exclure qu'une partie de l'écart entre deux recettes vienne de
  l'initialisation ou de l'ordre des mini-batchs. L'argument qui limite le risque :
  l'effet est reproduit sur un second jeu (41 points, S.7.3), et le motif est
  structurel — les cinq modèles sains commencent tous à 2 pas, les quatre modèles
  faibles ont tous un départ haut. Un plan de graines (3 graines x 2 recettes,
  l'ordre croissant contre l'ordre décroissant) reste la confirmation à faire, et
  c'est la plus coûteuse : environ 2 à 3 heures de GPU.

---

## 9. Reproductibilité

Tout ce document est régénérable. Les commandes exactes des runs sont dans
`adversarial/README.md` ; les résultats bruts sont écrits en JSON par
`eval_suite.py` et `eval_autoattack.py` dans `adversarial/results/logs/` ;
`adversarial/torch/tableau_recap.py` reconstruit le tableau final à partir de ces
JSON, ce qui garantit qu'**aucun chiffre de cet article n'a été recopié à la
main**. Le carnet de bord (`adversarial/memoire.md`, plus de 100 ko) conserve,
pour chaque expérience, les paramètres, le résultat et le critère qui avait été
écrit avant de lancer le run — y compris les critères qui se sont révélés faux.

Le projet contient les deux moteurs (S.2.1), la suite d'attaques, les scripts
d'audit (`audit_masquage.py`, `diag_attaque_interne.py`), le plan de budget
(`--plan-budget`), l'attaque à budget adaptatif (`--bande`) et le code de
certification. Aucune étape ne dépend d'un service externe.

---

## 10. Conclusion

Nous avons construit un CNN à la main, l'avons cassé en une seule étape de
gradient, l'avons défendu, puis avons cherché à casser notre propre défense. Ce
chemin a produit un résultat simple : **ce n'est pas la force de l'attaque
d'entraînement qui durcit un réseau, c'est l'ordre dans lequel on lui présente la
difficulté.** Vingt-deux points d'écart sur MNIST, quarante-et-un sur KMNIST,
pour un coût identique.

Le second résultat est une limite, et elle compte autant : **la recette se
transpose, le niveau non** (-23.9 points à recette identique et à juge officiel).
Une recette qui marche quelque part n'est pas une recette qui marche quelque
part.

La suite est éditoriale avant d'être technique : model card sur Hugging Face
(poids, recette, robustesse par eps, limites), démonstration interactive
(« attaquez le CNN en direct »), et version anglaise de ce document.

---

## Annexes

**A. Les corrections assumées.** Rayon PGD invalide en régime rugueux ; rapport
gradient/aléatoire non prédictif ; critère KMNIST partiellement faux ; prévision
de niveau fausse sur KMNIST ; phrase « l'ordre compense une perte mal réglée »
retirée faute de soutien expérimental.

**B. Glossaire.** FGSM, PGD, APGD, CW-L2, Square, NES, Boundary, BPDA+EOT,
AutoAttack, budget de déplacement, curriculum, min-min (FAT), masquage de
gradient, randomized smoothing, rayon certifié.

**C. État du document.** Chiffres vérifiés et datés ; figures 1 à 4 à produire
depuis les JSON ; version anglaise à écrire après validation. Le plan détaillé et
les points ouverts sont dans `TRAME-fr.md`.
