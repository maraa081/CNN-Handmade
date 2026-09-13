# TRAME DE L'ARTICLE (FR) - etat au 2026-09-13

> But de ce fichier : la structure definitive du write-up, avec les chiffres deja
> en place (aucun n'est a recopier a la main : ils viennent de `memoire.md`, du
> catalogue du README et des JSON de `adversarial/results/logs/`). Les cases
> marquees `A MESURER` attendent un run en cours (smoothing, TRADES).
>
> Regle : un chiffre = une source. Les chiffres annonces sans qualificatif sont
> OFFICIELS (AutoAttack `standard`, 10 000 images, eps=0.30). Tout chiffre de
> notre suite maison (500 images) est etiquete "maison".

---

## 0. Le titre et l'accroche

Titres candidats, par ordre de preference :

1. **"L'ordre dans lequel on présente la difficulté construit la robustesse"**
   (sous-titre : *une loi sur le budget de déplacement de l'attaque interne,
   trouvée, reproduite, et passée au juge officiel*)
2. "Construire, attaquer, défendre, casser sa propre défense : un CNN sans
   framework, de bout en bout"
3. "Ce n'est pas la force de l'attaque qui durcit un réseau, c'est son
   ordonnancement"

Accroche (3 phrases, a rediger en dernier) : le contraste qui vend l'article =
un modèle entrainé contre une attaque PLUS FORTE est NETTEMENT MOINS ROBUSTE que
le même modèle entrainé contre la même attaque présentée dans le bon ordre, pour
le même coût en calcul.

**Version courte a se faire relire par quelqu'un qui ne connait pas le domaine** :
"on entraine un réseau de neurones contre un attaquant. Si on presente l'attaque
a pleine force des le premier epoch, le reseau finit fragile. Si on la presente
doucement puis de plus en plus fort, pour exactement le même cout total, le
reseau finit beaucoup plus solide. Et on a la preuve, y compris sur un deuxieme
jeu de donnees (des kana japonais)."

---

## 1. Resume (TL;DR) - a placer en tete, 6 puces chiffrees

- Un CNN ecrit a la main en NumPy (421 642 parametres, `im2col`, backprop et
  optimiseurs faits main) atteint **98.6%** sur MNIST - puis FGSM le fait tomber
  a **1.8%** et PGD a **0.0%** a eps=0.30.
- Un entraînement adversarial agressif (PGD-20, pas eps/10, 120 epochs) monte le
  pire cas a **50.71%** (officiel) mais c'est encore quatre fois moins bien que la
  même recette avec un budget d'attaque interne qui DEMARRE BAS et CROIT :
  plan `0.2 -> 2 eps`, **82.40%** (officiel), pour 11 minutes.
- LE resultat central : a budgets internes identiques et a cout identique, le plan
  croissant `0.2 -> 2 eps` donne **85.8%** et le plan decroissant `2 -> 0.2 eps`
  **63.8%** (maison) - **22 points d'ecart, dus au seul ORDRE**.
- Publier honnetement coute : notre suite maison est optimiste de 3 a 13 points
  selon le modele, et le classement qu'elle donne reste bon sur les modeles sains
  mais se trompe en faveur du seul modele suspect (`abl_a`).
- Replication sur KMNIST (kana japonais, ancre Japon) : la loi tient et
  s'amplifie (**+41 points** contre +22), mais le niveau ne se transpose pas
  (**-23.9 points** a recette identique, officiel des deux cotes).
- Tout est reproductible : commandes exactes, logs JSON, poids, et un
  `tableau_recap.py` qui regenere le tableau final sans recopie manuelle.

---

## 2. Introduction (1,5 page)

- **Pourquoi coder un CNN a la main** : comprendre chaque brique (im2col,
  retropropagation, optimiseurs) au lieu d'appeler une API. Le projet a un
  interet cache : quand on attaque ensuite ce reseau, on sait exactement ce qu'on
  attaque. Argument de credibilite pour le reste de l'article.
- **Le fil rouge** : construire -> attaquer -> defendre -> casser sa propre
  defense. C'est l'arc narratif qui tient les 4 sections techniques.
- **Ce que l'article apporte** (a ecrire en 4 puces) :
  1. une loi simple et verifiee sur l'entrainement adversarial (le budget interne
     doit demarrer BAS et CROITRE ; l'ORDRE suffit a faire 22 points) ;
  2. sa replication sur un deuxieme jeu (KMNIST) avec ce qui se transpose et ce
     qui ne se transpose pas ;
  3. une methode de mesure honnete : le juge est AutoAttack, notre suite maison
     sert de diagnostic rapide, et on chiffre notre propre biais (3 a 13 points) ;
  4. une extension : rendre le budget adaptatif par batch ajoute +7 a +9 points,
     au prix de ~9 minutes de plus.
- **Ce que l'article ne pretend pas** : pas d'etat de l'art CIFAR-10, pas de
  nouvelle borne theorique, pas de benchmark contre les gros modeles. C'est un
  travail de mecanisme, sur petit modele, avec des chiffres verifiables.

---

## 3. Le terrain de jeu et le protocole de mesure (1,5 page)

### 3.1 Les deux moteurs, les memes maths
- CNN fait main en NumPy : MNIST propre **98.6%** ; architecture Conv(32)-Conv(64)-
  Dense, 421 642 parametres, identique partout (aucune comparaison ne melange deux
  architectures).
- Moteur PyTorch jumeau (autograd) : ~9x plus rapide sur CPU, **poids `.npz`
  interchangeables** - c'est ce qui rend la comparaison honnete entre defenseurs
  et attaquants. Aucun entrainement n'est fait sur la machine qui redige.
- Modeles de menace : L-infini, eps=0.30 (le meme partout, MNIST et KMNIST). Toute
  attaque est re-jugee a eps=0.30, meme si l'entrainement a utilise un autre
  budget interne.

### 3.2 Le protocole, et pourquoi il a change en cours de route
- **Les six regles de mesure** (lecons des 12 et 13/09) :
  1. annoncer sur 10 000 images (sur 500, tous les modeles gagnent 1 a 2 points) ;
  2. selectionner sur la validation, annoncer sur le test ;
  3. le juge est AutoAttack, notre suite maison est un diagnostic ;
  4. jamais de rayon robuste mesure au PGD seul (sur un modele rugueux, il classe a
     l'ENVERS : `abl_b` a le plus grand rayon PGD et le 4e pire cas) ;
  5. la sensibilite (2 s de calcul) trie les modeles, elle ne prouve rien ;
  6. une seule variable a la fois, et le critere d'adoption est ecrit AVANT le run.
- **Ce que coute chaque regle** : c'est racontable et c'est ce qui donne du poids
  au reste. Exemple a garder : `abl_a` annonce 63.2% par notre suite, **50.71%**
  officiel. On garde les deux chiffres dans l'article, c'est le prix affiche.

### 3.3 Figures
- FIG 1 : architecture du CNN fait main (deja existante dans le README racine).
- TABLE 1 : les six regles + le cout de l'application (une ligne par regle).

---

## 4. Attaquer : casser le modele propre (2 pages)

- Le modele propre : 98.6%. FGSM (1 pas) : **1.8%**. PGD (20 pas) : **0.0%**.
- La serie d'attaques maison, dans l'ordre de difficulte : FGSM, PGD, APGD-CE,
  APGD-DLR, CW-L2, Boundary, Square/NES (black-box, sans gradient), BPDA+EOT
  (sur un modele non differentiable), randomized smoothing.
- **Point d'article** : l'attaque la plus "forte" n'est pas celle qu'on croit. Sur
  un modele sain, les familles sont d'accord a quelques points (`A8` : 84.4 / 84.4
  / 84.6% en APGD-CE / APGD-DLR / Square maison) ; sur un modele suspect, elles se
  contredisent (`abl_a` : 85.4% en APGD-DLR contre **63.2%** en Square, maison).
  Cette contradiction est le symptome, pas le bruit.
- L'entrainement adversarial "naif" (FGSM seul) : le modele resiste a FGSM mais
  reste casse par PGD -> montre que le durcissement est un probleme de RECETTE.
- TABLE 2 : pire cas du modele propre et du modele durci FGSM.

## 5. Defendre : le durcissement, et la decouverte de la courbe en cloche (2,5 pages)

### 5.1 Les premices (v1 NumPy, campagne A/B/C)
- v1 (NumPy, PGD-7, pas eps/4) : 67.2% propre, pire cas **1.2%**.
- Campagne A/B/C : le facteur limitant etait le **budget d'epochs**, pas la
  recette (le run B, juge mauvais a 10 epochs, atteint **91.0%** a 120). Lecon a
  garder : avant d'annoncer qu'une recette est mauvaise, lui donner ses 120 epochs.

### 5.2 L'ablation du pas : la courbe en cloche (resultat n.1)
- PGD-20 avec des pas differents (budget = pas x nombre de pas) :
  TABLE 3 : `run B` 1.25 eps -> 42.0% ; `abl_b` 5 eps -> 38.8% ; `v4/abl_a`
  2 eps -> 61.8 / **63.2%** ; `A4` 0.5 eps -> 49.2% ; `abl_c` 20 eps -> **1.6%**.
- **Optimum interieur** : la robustesse apprise est maximale pour un budget
  intermediaire (sommet ~2 eps). Ni trop mou, ni trop dur.
- Le contre-exemple parfait `abl_c` : **98.4% propre**, un modele qui a l'air
  intact, et **1.6%** de pire cas. Le modele a fait "le seul arbitrage qui
  restait" : il a arreté d'apprendre a etre robuste.
- FIG 2 : la courbe en cloche (pire cas en fonction du budget de l'attaque
  interne). C'est LA figure de l'article.

### 5.3 Loi v2 : le pas compte autant que le budget
- `A4` (pas fin eps/10, 5 pas, budget 0.5 eps) : 49.2% contre `run B` (pas
  grossier eps/4, budget 1.25 eps) : 42.0%. Les deux axes (finesse du pas, nombre
  de pas) etaient confondus dans l'ablation de 5.2.
- Formulation : **la LISIBILITE de la perturbation compte, pas seulement sa
  force.** A passer dans le resume de section.

### 5.4 Loi v3 (le resultat central) : c'est l'ORDRE
- TABLE 4 (a mettre en avant, c'est le tableau de l'article) :

| plan de budget interne | budgets traverses | cout | pire cas (maison) |
|---|---|---|---|
| `0.2 -> 2 eps` (`A6`) | 2 .. 20 pas | 11 min | **85.8%** |
| `2 -> 0.2 eps` (`A9`) | 20 .. 2 pas | 11 min | **63.8%** |

- **Meme ensemble de budgets, meme cout, ordre inverse : 22 points d'ecart.**
- Contre-intuitif assume : le plan croissant atteint son budget maximal, plus dur,
  plus tard ; le plan decroissant l'atteint d'emblee. Les modeles "mauvais"
  (`abl_a`, `A7`, `A9`) ont TOUS un depart haut ; les bons (`A6`, `A8`, `A1`)
  commencent tous a 2 pas (0.2 eps).
- **Mecanisme propose** : un depart a budget court joue le role d'une
  regularisation qui garde l'attaque interne informative ; demarrer fort place
  l'optimisation dans le regime de verrouillage (`abl_c`). Piste theorique a
  citer : le min-min de FAT (Wong et al., ICML 2020) - l'arret anticipe change
  l'objectif.
- Reproduit deux fois (A6/A9 sur MNIST, plan doux/inverse sur KMNIST).

### 5.5 L'extension : budget adaptatif par batch (idee de Maraa)
- Au lieu d'un plan fixe, viser par batch une difficulte cible et arreter
  l'attaque au pas `k*` qui l'atteint (`--bande --cible 0.5`) : **93.6%** maison /
  **91.25%** officiel, soit **+7 a +9 points** sur le plan deterministe pour ~12
  minutes de plus.
- Surcout nul a pas fixe (la trajectoire de PGD contient deja tous les candidats).
- Formalisation et positionnement (FAT, IAAT, SAAT) : `adversarial/attaque_adaptative.md`.
- **Audit de masquage passe** (5 tests) : sorties non saturees, gradient
  informatif (x8.5 mieux que l'aleatoire), transfert qui ne bat pas le white-box,
  effondrement a eps=0.5 sous PGD a pas fin -> rayon robuste reel, pas de
  masquage. A mettre dans l'article : c'est la section qui coupe les objections.

### 5.6 bis. La loi ne rattrape pas une perte mal reglee (TRADES)

- Meme recette, meme duree, seule la perte change : TRADES (`CE + beta x KL`,
  `beta 2`) au lieu de la cross-entropy sur l'exemple adverse.
- TABLE 4 bis :

| recette (plan `0.2 -> 2 eps`, 120 epochs) | propre | pire cas maison | **officiel** |
|---|---|---|---|
| PGD-AT (`A6`) | 99.2% | 85.8% | **82.40%** |
| TRADES `beta 2` (warm start) | 96.5% | 27.2% | **22.01%** |

- **-60 points a recette identique.** Conclusion : l'ordonnancement du budget est
  un levier PUISSANT mais il ne compense pas un objectif mal reglé. A dire tel
  quel, sans exagerer le propos de l'article.
- Honnetete : c'est une VARIANTE (`beta 2` + warm start, compromis du 10/09 ou
  `beta 6` faisait s'effondrer un modele deja converge : clean 69%). La reference
  (Zhang et al. 2019) s'entraine depuis zero avec `beta 6`. Selon la decision de
  Maraa : soit on documente la variante comme limite (assume), soit on ajoute le
  run de reference (dernier run du projet) et la section gagne un chiffre au lieu
  d'un paragraphe d'excuses. `A MESURER` si la decision est prise.
- Signature a documenter : PGD-20 44.0% contre APGD-DLR 27.2% (**17 points
  d'ecart**, profil rugueux type `abl_a`) et CW-L2 qui trompe 5% des images a une
  distance L2 moyenne de **0.028** : quelques images sont catastrophiquement
  fragiles. Un bon exemple de ce que la suite maison detecte et qu'un chiffre
  unique cache.

### 5.6 ter. La lecture qui explique les modeles (outil, pas preuve)
- Sensibilite de la CE sous perturbation dans la boule (eps=0.3) : classe les 5
  modeles dans le MEME ordre que le pire cas officiel, alors que le rayon PGD
  classe a l'envers. Sert a trier et a expliquer (2 secondes de calcul).
- Marges **anisotropes** chez `abl_a` (~0.40 selon le gradient, <0.30 selon
  certaines directions aleatoires) contre **isotropes** chez `A1`.
- Deux corrections assumees et a raconter (un article honnete dit aussi ce qu'il a
  compris de travers) : (1) "les rayons robustes sont egaux, donc c'est un effet
  de fenetre" etait faux ; (2) "le rapport gradient/aleatoire predit la taille de
  l'ecart" est faux aussi (`abl_c` : rapport 1.16 et ecart negatif).

---

## 6. Le juge : AutoAttack, et ce qu'il nous a coute (2 pages)

- TABLE 5 : les 6 modeles croises (c'est LE tableau de reference) :

| modele | recette | maison | officiel | ecart |
|---|---|---|---|---|
| `bande_cible50` (`A1`) | budget adaptatif | 93.6% | **91.25%** | -2.4 |
| `A6` | plan `0.2 -> 2 eps` | 85.8% | **82.40%** | -3.4 |
| `A8` | plan `0.2 -> 1 eps` | 84.4% | **78.25%** | -6.2 |
| `kmnist_plan_02_2` | KMNIST, plan `0.2 -> 2 eps` | 62.6% | **58.53%** | -4.1 |
| `kmnist_plan_doux` | KMNIST, plan `0.2 -> 1 eps` | 58.4% | **51.03%** | -7.4 |
| `abl_a` | PGD-20 constant, pas eps/10 | 63.2% | **50.71%** | -12.5 |
| `trades_plan_02_2` | plan `0.2 -> 2 eps`, TRADES `beta 2` + warm start | 27.2% | **22.01%** | -5.2 |

- **Le biais de notre suite maison est de 3 a 13 points**, jamais dans l'autre
  sens. On l'annonce dans l'article : c'est ce qui rend les autres chiffres
  credibles.
- **Le classement : les 5 modeles sains gardent EXACTEMENT le meme ordre en maison
  et en officiel ; seul `abl_a` bouge** (4e maison -> dernier officiel). Et c'est
  precisement le modele que nos diagnostics classaient comme rugueux. Phrase a
  ecrire telle quelle : *notre suite maison trie bien les modeles sains et se
  trompe en faveur des suspects*.
- Honnetete sur les chiffres officiels eux-memes : AutoAttack a emis un
  avertissement sur `kmnist_plan_doux` ("Square Attack has decreased the robust
  accuracy of 2.24%") -> 51.03% est lui-meme legerement optimiste, on le dit.
  Aucun avertissement sur `kmnist_plan_02_2` (58.53%) : les deux chiffres KMNIST
  ne sont pas de qualite identique.
- FIG 3 : nuage maison (x) vs officiel (y) avec la diagonale - la figure qui
  montre le biais et le classement.

---

## 7. Replication : KMNIST, l'ancre Japon (2 pages)

- Pourquoi KMNIST : mêmes dimensions (28x28), même pipeline, même eps, mais des
  kana cursifs -> un deuxieme jeu gratuit pour tester la generalite de la loi, et
  une accroche culturelle (auteur base au Japon).
- **Prevision ecrite AVANT les runs** (a citer tel quel, c'est la partie
  scientifique) : "dur ~60-65%, doux ~84%, ordre inverse -20 points".
- TABLE 6 : les 8 points de mesure du pire cas KMNIST :
  0.0% (propre) / 1.2% (constant 2 eps) / 15.2% (constant 1 eps) / 17.0% (plan
  inverse `1 -> 0.2`) / 0.0% (`0.05 -> 0.2`) / 25.2% (`0.1 -> 0.5`) / 31.8%
  (`0.2 -> 0.5`) / **58.4%** (`0.2 -> 1`, 51.03% officiel) / **62.6%**
  (`0.2 -> 2`, **58.53%** officiel).
- **Ce qui se transpose** : la loi de l'ORDRE, et meme AMPLIFIEE (+41 points entre
  plan croissant et inverse, contre +22 sur MNIST). Le plan `0.2 -> 2 eps` est le
  meilleur des deux cotes : la recette se transpose.
- **Ce qui ne se transpose pas** : le NIVEAU. Deux paires appariees, officielles
  des deux cotes :

| recette | MNIST | KMNIST | ecart |
|---|---|---|---|
| plan `0.2 -> 2 eps` | 82.40% | 58.53% | **-23.9 pts** |
| plan `0.2 -> 1 eps` | 78.25% | 51.03% | **-27.2 pts** |

  -> la phrase de l'article : **la recette se transpose, le niveau non** (~-25
  points a recette identique).
- **Mecanisme du decalage** : a eps=0.30 l'attaque interne est relativement bien
  plus forte sur KMNIST (PGD a 0.1 sur le modele propre : 9.2% contre 44.4% sur
  MNIST), donc la recette de reference MNIST (`constant 2 eps`, 63.2% a MNIST)
  s'effondre a **1.2%**. Le budget "hors de portee" du jeu change de valeur.
  Observe dans le log : l'attaque interne trompe 90% du batch jusqu'au bout, val
  clean monte a 99.5% (mieux que le referentiel propre !), val PGD10 bloque a
  9-10% -> ce n'est PAS le verrouillage d'`abl_c`, c'est un arbitrage impose.
- Enseignement methodologique : **un sommet de courbe n'est pas un nombre, c'est un
  nombre pour un jeu de donnees.** A verifier avant de publier une recette.
- FIG 4 : les deux courbes (MNIST et KMNIST) cote a cote, meme axe des budgets ->
  meme forme, hauteur differente.

---

## 8. Limites (0,5 page, a ecrire sans se defendre)

- Petits modeles (421 642 parametres), petites images (28x28), un seul eps
  (0.30) : la loi est etablie sur ce regime, pas au-dela (pas d'ensemble, pas de
  CIFAR, pas au-dela de 1.7M de parametres - decisions explicites, cf. README).
- Notre suite maison est optimiste de 3 a 13 points ; elle est utilisee pour
  TRIER, pas pour annoncer.
- Le chiffre officiel KMNIST du plan doux porte un avertissement AutoAttack
  (2.24%) : il est lui-meme optimiste.
- Robustesse empirique, pas certifiee : le seul resultat avec garantie est le
  randomized smoothing -- **rayon L2 certifie median 1.214, 98.5% des images
  certifiees a R=0.30, 99.0% propre**. A presenter avec le piege de lecture : la
  boule L-infini 0.30 n'est pas incluse dans la boule L2 1.214 (norme L2 jusqu'a
  0.30 x sqrt(784) = 8.4), donc la garantie repond a une autre question que les
  chiffres L-infini de l'article.
- TRADES : voir 5.6 bis. Resultat publie meme s'il est mauvais pour notre
  variante ; le run de reference (`--beta 6`, depuis zero) reste a decider.
- Un seul dataset de replication, et une seule architecture par dataset : la loi
  est verifiee, pas demontree.

## 9. Reproductibilite (1 page)

- Arborescence, commandes exactes (les blocs de `adversarial/README.md`), seeds,
  JSON de resultats dans `adversarial/results/logs/`, poids.
- `python3 adversarial/torch/tableau_recap.py` regenere le tableau final depuis
  les JSON : "aucun chiffre de cet article n'a ete recopie a la main".
- Le carnet de bord (`memoire.md`, 100 ko) et les criteres ecrits AVANT les runs :
  c'est une piece du dispositif, a mentionner.

## 10. Conclusion et suite

- Trois phrases : ce qu'on a construit, ce qu'on a trouve (l'ordre), ce qui reste.
- Suite : model card Hugging Face (poids + recette + robustesse par eps +
  limites), Space Gradio ("attaque mon CNN en direct"), version anglaise du
  write-up, puis terrain CIFAR si le temps le permet.

---

## Annexes

- A. Le catalogue complet des runs (tableau du README, ~25 lignes).
- B. Glossaire : FGSM, PGD, APGD, CW, Square, NES, AutoAttack, budget de
  deplacement, curriculum, min-min, masquage de gradient.
- C. Les deux corrections assumees (fenetre vs rayon ; rapport gradient/aleatoire)
  - elles font partie de l'honnetete du recit.
- D. Code : les 5 fichiers de `adversarial/torch/` (entrainement, harden_torch,
  eval_suite, eval_autoattack, audit_masquage).

---

## Travail restant sur ce document

| # | A faire | Qui |
|---|---|---|
| 1 | ~~Integrer le resultat TRADES~~ fait (5.6 bis) ; le run de reference reste a decider | Maraa |
| 2 | ~~Integrer le smoothing certifie~~ fait (sections 7 et 8) | moi |
| 3 | Ecrire les sections 2 a 8 en prose FR | moi |
| 4 | Produire FIG 2, FIG 3, FIG 4 (matplotlib, depuis les JSON) | moi |
| 5 | Traduire en EN apres validation | moi |
| 6 | Relire et corriger les chiffres | Maraa |

Regle de redaction : pas d'emoji, pas de superlatif ("incroyable", "revolutionnaire"),
les chiffres maison et officiels toujours etiquetes, et chaque affirmation
adverse verifiable par une commande.
