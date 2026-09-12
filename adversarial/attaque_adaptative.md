# Attaque interne a budget adaptatif

> Note de conception (2026-09-12). Idee de Maraa : rendre la difficulte de
> l'attaque d'entrainement adaptative au lieu de la fixer une fois pour toutes.
> Etat : proposition formalisee, module ecrit
> (`torch/attaque_adaptative.py`), testable
> (`torch/test_attaque_adaptative.py`), PAS encore validee par un run.

## 1. L'idee

Pendant l'entrainement adversarial, les parametres de l'attaque interne (pas
`alpha`, nombre de pas `K`, departs multiples...) sont choisis AVANT le run et
ne bougent plus. Maraa propose l'inverse : a chaque batch, chercher la
configuration qui apporte le meilleur signal pour l'etat courant du modele.

    etat du modele -> choix de la configuration d'attaque -> entrainement sur
    le batch -> mise a jour -> nouveau choix au batch suivant

La question laissee ouverte par Maraa est la bonne : quel objectif ? Maximiser
l'accuracy est un mauvais critere (cela pousse a un entrainement facile) ;
maximiser la perte adverse est le critere de Madry, et la loi du budget de
deplacement dit que c'est un piege au-dela de ~2 eps.

## 2. Verdict : pertinente, et pas vierge

L'idee a des cousins directs dans la litterature. Il faut les connaitre, sinon
un relecteur nous renvoie a eux — et surtout ils nous disent ce qui marche deja.

| Travail | Mecanisme | Lien avec l'idee |
|---|---|---|
| FAT, Zhang et al., ICML 2020 ("Attacks Which Do Not Kill Training...") | PGD arrete des la premiere erreur (`friendly` adversarial data) | exactement l'early stopping par batch ; objectif oppose au notre (on vise une difficulte, FAT vise la plus petite qui trompe) |
| IAAT (instance-aware AT), 2021 | meme early stopping, avec seuil par instance selon la marge | version par instance de l'arret anticipe |
| SAAT, arXiv 2210.01288 (2022) | le budget de perturbation est AUGMENTE tant qu'une contrainte de perte adverse n'est pas satisfaite | c'est le plus proche : budget adaptatif pilote par la perte adverse, au niveau du run |
| Curriculum AT, Cai et al., IJCAI 2018 | montee programmee de eps au fil des epochs | le schedule global, sans retour du modele |
| MMA, GAIRAT | eps adaptatif par instance / reponderation par marge | adaptation par exemple, pas par batch |

Donc : **ce n'est pas une idee vierge, c'est une famille**. Ce que notre projet
apporte en plus, c'est la contrainte qui manque a tous ces travaux : notre loi
du budget de deplacement mesuree sur ce modele (voir `memoire.md`, 12/09).

1. **Le parametre a reguler n'est pas la force mais la difficulte lisible.** La
   courbe du pire cas est en cloche avec un sommet a 2 eps (63.2%) et une
   falaise a 5 eps (38.8%) puis 20 eps (1.6%) : viser systematiquement "plus
   fort" detruit l'apprentissage. Un regulateur de difficulte a une CIBLE est
   donc la bonne forme, pas un maximiseur de perte.
2. **Le budget reste borne par la lisibilite (<= 2 eps).** Aucune adaptation
   n'a le droit de sortir de la zone saine : c'est un point d'implementation,
   pas un detail (SAAT, lui, augmente le budget sans cette borne).
3. **Quand le modele depasse la borne, on escalade la DIVERSITE, pas la force**
   (EOT, departs multiples, attaque par scores). C'est une consequence directe
   de notre loi, et a notre connaissance personne ne formule les choses comme
   ca.
4. **Le surcout est nul.** Le point technique qui rend l'idee exploitable :
   couteux au sens naif (essayer 4 configurations = 4 attaques = x4 le temps
   de GPU), c'est gratuit une fois reformule sur la trajectoire de PGD (voir 4).

## 3. Formalisation

Notations : modele `f_theta`, batch `B = {(x_i, y_i)}`, attaque interne de
configuration `c = (alpha, K, ...)`, budget `beta(c) = alpha * K` (deplacement
L-infini maximal de l'attaque).

Indicateurs de difficulte du batch, tous lus sur les logits :

    CE_clean(c=0) = moyenne_i CE(f(x_i), y_i)
    CE_adv(c)     = moyenne_i CE(f(x_i + delta_i(c)), y_i)
    D(c)          = taux de tromperie = moyenne_i [ argmax f(x_i + delta_i(c)) != y_i ]

Objectif de Madry (ce qu'on fait aujourd'hui) :

    min_theta  E_B [ max_{c : beta(c) <= beta_fixe}  CE_adv(c) ]

Objectif propose (asservissement) :

    min_theta  E_B [ CE_adv(c*) ]      avec
    c* = argmin_c | D(c) - rho_t |     sous  beta(c) <= beta_max = 2 eps

ou `rho_t` est la difficulte CIBLE, eventuellement planifiee dans le temps
(warmup : `rho_t` monte de 0.2 a 0.5 sur la premiere moitie du run, voir
`cible_effective`).

Trois variantes d'objectif, a departager par l'experience :

- **(a) Cible de tromperie** : `D(c*) = rho`. Interpretable, mais la tromperie
  est un indicateur non lisse (argmax).
- **(b) Cible de perte adverse** : `CE_adv(c*) = cible` en nats (plafond
  ln(10) = 2.303). C'est la contrainte de SAAT, et c'est l'indicateur lisse
  qu'on suit deja dans les logs. Variante relative : `CE_adv - CE_clean = Delta`.
- **(c) Bande** : n'accepter que `rho_lo <= D <= rho_hi`, et prendre le plus
  petit k qui entre dans la bande. Plus robuste au bruit du batch.

Notre recommandation : **(b) relative**, `Delta = CE_adv - CE_clean`, cible de
l'ordre de 0.8 a 1.0 nat. Raison : `CE_clean` varie d'un batch a l'autre, donc
une cible absolue demande au modele une difficulte differente selon la
composition du batch ; la cible relative garde la difficulte constante.

Pourquoi un asservissement et pas une maximisation : la loi du 12/09 dit que la
robustesse apprise est une fonction EN CLOCHE de `beta`. Maximiser `D` revient a
pousser `beta` vers le haut -> on tombe dans la falaise. Reguler `D` vers une
cible garde `beta` a l'interieur de la zone ou l'apprentissage fonctionne, ET
distribue le budget differemment selon les batchs : les batchs faciles sont
attaques peu (le modele sait deja), les batchs durs sont attaques plus (mais
jamais au-dela de la borne).

Cadre "controle", en deux temps :

    boucle RAPIDE (par batch, gratuite)      : choisir k* (le budget du batch)
    boucle LENTE  (par epoch, quasi gratuite) : corriger la cible ou le modele

Plus formellement, un regulateur proportionnel sur le log-budget :

    log beta_{t+1} = log beta_t + k_p * (rho_t - D_ema_t) + k_i * somme des erreurs

Un seul etat suffit (cible de difficulte), mais on peut aussi reguler le budget
lui-meme. Deux garde-fous non negociables : `beta <= 2 eps` (loi) et l'arret sur
effondrement de la val PGD (deja code : `--collapse-tol`, `--stop-on-collapse`).

## 4. Algorithme : comment ne pas exploser le cout

Le point cle. L'implementation naive de l'idee ("tester plusieurs
configurations a chaque batch") multiplie le cout par le nombre de candidats :
4 candidats x 20 pas = 80 passages avant+arriere au lieu de 20.

On remarque que **la trajectoire de PGD contient deja toutes les configurations
qu'on veut tester**, a condition de faire varier le budget par le NOMBRE de pas
et non par la taille du pas :

    avec alpha FIXE = eps/10, le budget apres k pas vaut k * eps/10
    -> choisir k dans {0..20} = choisir un budget dans {0, 0.1, ..., 2} eps

Donc au lieu de relancer une attaque par candidat, on enregistre la difficulte a
chaque pas de l'attaque en cours, puis on choisit `k*` a posteriori :

    1. depart aleatoire frais (comme pgd : uniforme dans la boule) ;
    2. boucle PGD de K pas ; a chaque pas la difficulte est lue sur les logits
       DEJA calcules pour le gradient -> aucun passage avant supplementaire ;
       on memorise les iterats (K x batch x 784 flottants, ~16 Mo a K=20) ;
    3. k* = plus petit k tel que D(k) >= rho_t (premier passage) ; si la cible
       n'est jamais atteinte, k* = K et on leve le drapeau `plafond` ;
    4. on entraine sur l'iterat k*.

Cout : **exactement le meme que `pgd()`** (K passages avant+arriere) plus la
memoire des iterats. Verifie par le test [3] de
`test_attaque_adaptative.py` (compte des passages avant identique).

Extensions symetriques (meme logique, a tester plus tard) :

- **plusieurs departs** : lancer 2-3 attaques courtes (K/2) au lieu d'une longue
  et garder la meilleure pour la cible -> meme budget de calcul, plus de
  diversite (piste pour l'ecart Square de 22 points) ;
- **selection par batch d'images** : au lieu d'un k* commun, un k* par image
  (generalisation par instance de FAT/IAAT, dont notre regle est la version
  agregee) ;
- **attaque par scores** : garder Square comme juge et, comme signal
  d'entrainement, remplacer l'escalade de force par un melange PGD + scores.

## 5. Branchement dans le pipeline

Rien a recoder : le crochet est le selecteur `attaque()` de la boucle
d'entrainement (`entrainement.py`, etape [3]).

    # a la place de :
    bx_adv = attaque(modele, bx, by, args.eps, args.attack, args.pgd_steps, args.pgd_alpha)

    # avec --bande :
    cible = cible_effective(args.cible, epoch, args.epochs, args.cible_rampe, args.cible_depart)
    bx_adv, info = pgd_bande(modele, bx, by, args.eps, args.pgd_steps, args.pgd_alpha,
                             cible, args.cible_type)
    journal.ajouter(info)

Nouvelles options (defauts prudents) :

| Option | Defaut | Role |
|---|---|---|
| `--bande` | desactive | active l'attaque a budget adaptatif |
| `--cible` | 0.5 | difficulte visee (0-1 si `tromperie`, nats si `ce`) |
| `--cible-type` | tromperie | indicateur regule (`tromperie` ou `ce`) |
| `--cible-depart` / `--cible-rampe` | 0.2 / 0.5 | warmup de la cible |
| `--plafond-tol` | 0.5 | alerte si la cible n'est atteinte qu'au plafond sur N% des batchs |

La ligne d'epoch gagne trois colonnes : `k*` moyen (en pas), budget moyen (en
eps) et part des batchs au plafond. Le module ne touche a rien quand `--bande`
est absent : les runs A0 restent strictement comparables.

## 6. Protocole de validation (cout total ~1 h de calcul)

Meme recette que `abl_a` (PGD-20, 60k images, 120 epochs, augment, lr 0.05,
seed 42), meme suite d'evaluation (Square-3000 = juge). Une seule variable :
la regle d'arret de l'attaque interne.

| Run | Regle | Cout/epoch | Question |
|---|---|---|---|
| A0 = abl_a | fixe, K=20, alpha=eps/10 | 1.0x | reference : pire cas 63.2% |
| A1 | adaptatif, cible tromperie 0.5 | ~0.6x | le budget variable fait-il mieux a cout moindre ? |
| A2 | adaptatif, cible CE-relative | ~0.6x | cible lisse vs cible non lisse |
| A3 | adaptatif, cible tres haute (0.9) | 1.0x | controle : viser trop dur degrade-t-il ? |

Criteres d'adoption : A1/A2 retenus si (i) pire cas >= 63.2% + 2 points, ou (ii)
pire cas equivalent avec un cout <= 70% de A0. A3 doit degrader : sinon la
courbe en cloche ne se reproduit pas au niveau batch et il faut comprendre
pourquoi avant d'ecrire quoi que ce soit.

Avant de lancer : `python -u adversarial/torch/test_attaque_adaptative.py`
(quelques secondes, verifie selection, equivalence avec pgd, cout, plafond).

## 7. Risques et pieges

1. **Statistique de batch bruitee.** L'accuracy d'un batch de 256 fait facilement
   +/- 5 points. Reguler sur la valeur brute d'un batch = regulateur qui
   s'agite. Reponse : cible relative (variante b), moyenne glissante pour la
   boucle lente, et bande plutot que point.
2. **Biais de selection (piege de FAT).** Arreter l'attaque tot et entrainer
   dessus change l'objectif : on n'entraine plus sur le pire cas (min-max) mais
   sur un cas intermediaire (min-min). C'est assume, c'est la these de FAT, mais
   il faut le dire dans l'article : on ne peut plus appeler la recette
   "adversarial training" sans precise que le budget est regule.
3. **Normalisation par lot (BN).** L'attaque adaptative fait varier l'amplitude
   des perturbations d'un batch a l'autre : les statistiques de BN voient une
   distribution plus large. A mesurer sur la val propre ; si degradation, piste
   des BN separees (une branche propre, une branche adverse, cf. AdvProp).
4. **Robust overfitting.** Une cible constante toute la vie du run peut
   sur-specialiser le modele en fin d'entrainement : le warmup de cible et la
   congelation de la cible sur les 20% derniers epochs sont deux reponses.
5. **NE PAS REUTILISER LA REGLE D'ARRET COMME JUGE.** L'attaque adaptative est un
   outil d'ENTRAINEMENT. Le pire cas doit rester mesure par une attaque qui ne
   connait pas notre regle (Square-3000). Sinon on mesure notre propre critere.
6. **Le plafond est un diagnostic, pas un objectif.** S'il tombe souvent, cela
   dit que le modele a depasse le budget lisible : la reponse est la diversite
   (EOT, departs, scores), pas la force.

## 8. Ce qu'on peut esperer

Honnêtement : un gain de pire cas modeste (l'ordre de +2 a +5 points), parce que
`abl_a` est deja au sommet de la courbe globale. Le vrai apport est ailleurs :

- **un budget par batch** : les batchs faciles ne sont plus sur-attaques (cout
  et biais), les batchs durs le sont davantage -> meilleure utilisation du
  meme calcul ;
- **un entrainement moins cher** : l'arret anticipe fait tourner les epochs en
  ~0.6x (a mesurer) ;
- **un recit propre pour l'article** : un seul hyperparametre (la difficulte
  cible) qui interpole entre FAT (cible basse) et Madry (cible maximale), avec
  une courbe en cloche experimentale pour montrer ou se situe l'optimum. C'est
  plus defendable que "on a regle les hyperparametres un par un".
