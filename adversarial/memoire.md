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
  plus le modèle s'effondre. À ε=0.05 (bruit discret) on perd déjà 4.5 pts.
- Leçon : mon CNN from scratch est **vulnérable** — un bruit borné à 0.3 par
  pixel suffit à le faire tomber à ~2%. Conforme à la théorie (linéarité en
  grandes dimensions, Goodfellow).
- [fix] **Échelle de ε** : les images sont normalisées dans `[0, 1]`
  (`images / 255.0` dans `src/data.py`) et FGSM applique
  `clip(x + eps*sign(grad), 0, 1)`. Donc **ε = 0.3 signifie 30% de la plage,
  soit ~76 niveaux de gris sur 255 : c'est une perturbation VISIBLE.** L'erreur
  « ±0.3/255 » qui figurait ici était fausse d'un facteur 255 ; corrigée le
  2026-09-10. Repères : ε=0.05 ~ 13/255 (discret), ε=0.30 ~ 76/255 (très net).

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
  3. **max_config -> full transfère PLUS que l'inverse** : 31.8% contre 15.1%
     à ε=0.20. C'est contre-intuitif : attaquer le modèle *régularisé* produit
     des exemples qui passent *mieux* sur le modèle simple.
     Pourquoi : le taux de transfert est mesuré **parmi les images où la SOURCE
     a été trompée**. max_config étant plus dur à tromper, ce sous-ensemble ne
     contient que des perturbations fortes -> elles traversent aussi le modèle
     simple. **Le taux brut est donc biaisé par la robustesse de la source.**
     Le chiffre qui compte côté défense est l'accuracy ABSOLUE de la cible :
     `full` tombe à 40.2% à ε=0.30 sous une attaque transférée depuis
     max_config (contre 49.6% dans l'autre sens).
- [fix] L'observation n°3 affirmait l'inverse (« attaquer le modèle robuste
  produit des exemples moins transférables »), ce qui **contredisait le tableau
  juste au-dessus**. Corrigé le 2026-09-10.
- [todo] Pistes NON testées au 2026-09-10 :
  - transfert avec **PGD** comme source (plus fort que FGSM) : `--attack pgd`
  - transfert **cross-dataset** MNIST -> EMNIST (le cas réaliste)
  - **attaque d'ensemble** : attaquer plusieurs modèles à la fois pour renforcer
    le transfert
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

### 2026-09-10 — Vérification du transfert (re-run) + corrections de doc

- Re-run des 3 configurations de transfert (500 images, FGSM, seed 42) sur la
  machine OpenClaw : chiffres **identiques** à ceux du 2026-08-25
  (full->classic 25.7 / 50.0 / 72.3 ; full->max_config 8.2 / 15.1 / 50.2 ;
  max_config->full 8.0 / 31.8 / 60.7). Le run est donc reproductible d'une
  machine à l'autre.
- [fix] Échelle de ε corrigée (voir l'entrée du 2026-08-24 : « ±0.3/255 » était
  faux d'un facteur 255).
- [fix] Observation n°3 du transfert corrigée : elle contredisait son propre
  tableau.
- [doc] Section « Sécurité IA » ajoutée au README principal, historique 24-26/08
  ajouté à `docs/memoire-projet.md`, références de courbes corrigées dans
  `adversarial/README.md`.
- 12 courbes de transfert versionnées dans `adversarial/results/`.

### 2026-09-10 - harden2.py : version durcie v2 (code livré, run à venir)

- Diagnostic : la v1 plafonne à 67% propre / 24% sous FGSM ε=0.3 à cause du
  **budget** (5000 images, 3 epochs, départ aléatoire, lr fixe), pas de la
  méthode. Le modèle standard atteint 98.6% avec les 60000 images.
- `harden2.py` écrit : warm start (auto), 60000 images, décroissance du lr,
  TRADES, écrêtage des gradients, augmentation (translations ±2 px),
  sélection du modèle sur la robustesse de validation, éval multi-restarts.
- Gradient TRADES validé par **gradient check numérique** : erreur relative 5e-9.
- [fix] Sans écrêtage des gradients, TRADES **détruit** le modèle : sur un
  modèle déjà convergé le terme KL (beta=6) vaut ~100x le terme CE.
- Coût mesuré : 73 img/s sur le i5-6300U -> PGD-5 = ~90 min par epoch sur 60000
  images. Le run complet doit être lancé sur la machine la plus rapide.
- [todo] Lancer le run complet (PGD-5 puis TRADES), trier les résultats et les
  documenter dans ce carnet. Objectif : battre nettement 67% / 24%.

### 2026-09-10 - augment.py : augmentation de donnees (brusquer le dataset)

- Demande de Maraa : appliquer aleatoirement au dataset des ecarts "pas presents
  de base" (rotation, pixels parasites...). Reflexe important : **aucune
  regeneration du dataset, aucun appel API** — tout se fait en memoire, a la
  volee, pendant l'entrainement.
- Module `augment.py` ecrit : rotation, zoom, translation, bruit impulsionnel
  (sel sur le fond noir + poivre sur le trait), cutout, variation d'epaisseur.
- Cout mesure : 11 ms par batch de 64 -> **10 s par epoch de 60000 images**,
  soit 0.2 % du cout d'un epoch PGD-5 (90 min). Negligeable.
- Effet mesure de chaque transformation (densite d'encre, 500 images) :
  rotation x1.09, zoom x1.09, translation x1.00, bruit x1.09, cutout x0.96,
  epaisseur x1.17.
- [fix] Premiere version : dilatation 3x3 -> densite **x1.80** (le chiffre change
  de forme, un 1 devient un pate). Remplacee par un element en croix avec facteur
  de melange -> x1.17. Lecon : mesurer chaque transformation separement.
- Integre dans `harden2.py` : `--augment`, `--aug-fort`, `--aug-config`.
- Planche de controle : `adversarial/results/augmentation_samples.png`
- Regles : ne JAMAIS augmenter le test/validation ; l'augmentation ne remplace
  pas l'adversarial training (elle aide la precision propre et les perturbations
  naturelles ; une attaque L-inf bornee reste du ressort de PGD-AT).

### 2026-09-10 - Piste PyTorch (`adversarial/torch/`) : le banc d'essai rapide

- Contexte : Maraa veut utiliser son GPU (RX 7800 XT). Or le moteur fait main
  est en NumPy -> CPU uniquement, par construction. Aucun outil ne peut
  accelerer une boucle NumPy : "utiliser le GPU" implique de changer de moteur.
- Decision (choix de Maraa) : **option B** — garder le code fait main intact
  comme reference pedagogique, et ajouter une piste PyTorch avec autograd qui
  rejoue exactement les memes experiences.
- Ecrit : `adversarial/torch/{modele,attaques,entrainement}.py` +
  `harden_torch.py`. Memes options que `harden2.py`, plus `--device`, `--parite`,
  `--npz`.
- **Parite verifiee** (200 images, `model_weights_full.npz`) : accuracy propre
  98.50 % dans les deux implementations, FGSM eps=0.30 -> 1.50 % dans les deux,
  PGD eps=0.20 -> 0.00 % dans les deux. Seul ecart : 0.50 % sur FGSM eps=0.10,
  soit UNE image sur 200 (ordres de sommation differents en float32).
- Debit mesure : 5000 images en PGD-5 = 23 s en PyTorch CPU, contre ~3 min 20
  estime en NumPy -> **environ 9x**.
- Les poids restent interchangeables (.npz dans les deux sens) : c'est ce qui
  garde les deux pistes comparables.
- Doc dediee : `adversarial/torch/README.md` (correspondance terme a terme,
  ce qu'on perd, setup ROCm WSL2 pour le RX 7800 XT / gfx1101, DirectML en
  repli, reglage du batch a 256 sur GPU).
- [warn] PyTorch ne supporte pas Python 3.14 (cette machine) : venv en Python
  3.11 utilise pour la validation.
- [todo] Installer le stack ROCm sur le PC gaming, puis mesurer le gain GPU
  reel et le documenter ici.

### 2026-09-10 (soir) - Premier run chez Maraa : 3 bugs trouves et corriges

Maraa a lance `campagne.sh --torch` sur son Windows. Trois problemes reels
sont sortis du run.

**1. La campagne complete a tout saute (bug de nommage).** La version `--rapide`
et la version complete ecrivaient dans les MEMES fichiers `models/harden2_*.pt` :
la campagne complete croyait donc les runs deja faits. Correction : suffixe
`_rapide` ajoute aux noms de sortie en mode court.

**2. Le learning rate s'effondrait des le premier epoch.** L'ancienne formule
comparaît des fractions a chaque epoch : sur un run de 2 epochs, les DEUX
paliers (0.5 et 0.8) se declenchaient a l'epoch 1, et le lr tombait de 0.05 a
0.0005. Le modele n'apprenait quasiment rien. Correction : paliers calcules une
seule fois en numeros d'epoch entiers, planificateur desactive sous 4 epochs.

**3. TRADES etait FAUX (gradient sur une seule branche).** Le gradient ne
passait que par la branche propre, en traitant la branche adverse comme une
constante. Consequence : la KL poussait la prediction PROPRE vers la prediction
ADVERSE (fausse) - le modele apprenait a se tromper. Mesure : val clean
99.6% -> 8.0% en un seul epoch. Corrections :
- PyTorch : les deux branches restent dans le graphe (forme de l'implementation
  de reference : `F.kl_div(log_softmax(z_clean), softmax(z_adv))`).
- NumPy : deux passes avant/arriere avec accumulation explicite des gradients
  (`d KL / d z_clean = p_clean - p_adv`, `d KL / d z_adv = p_adv*(log(p_adv/p_clean) - KL)`).
Apres correction : val clean 99.5% -> **98.0%** sur le meme test NumPy.

**4. TRADES + warm start = piege.** Sur un modele deja converge, CE ~0.01 et
KL ~1.3 : la KL ecrase tout. Mesures en PyTorch (2 epochs, 10000 images) :
beta=6 -> clean 69%, beta=2 -> 87%, beta=1 -> 93%. Conclusion : avec
`--warm-start auto`, garder **beta <= 2** ; beta=6 suppose un entrainement
depuis zero. Run C de la campagne passe a `--beta 2`.

**Resultats du run de Maraa (utiles malgre les bugs) :**
- Sa machine est ~4x plus rapide que la machine OpenClaw en PyTorch :
  epoch de 10000 images en PGD-3 = ~6-12 s (contre ~24 s ici).
- Le run A (pgdat + warm start) a bien fonctionne : val clean 98.2-98.4%
  conserve, et FGSM eps=0.3 passe de 1.8% (modele non durci) a **25%** en
  2 epochs seulement. La piste est bonne.
- Estimation : la campagne complete (60000 images, 10 epochs, PGD-5) devrait
  prendre **~15 min par run**, soit ~45 min pour les 3 runs.

Corrige dans les commits du 2026-09-10 (campagne.sh, harden2.py,
torch/entrainement.py, defenses.md section 6).

### 2026-09-10 (nuit) - CAMPAGNE COMPLETE : le modele durci est enfin la

Premier vrai run complet, en PyTorch (piste torch), 60000 images, 10 epochs,
PGD-5, warm start depuis `model_weights_full.npz` (98.60% propre).
Machine de Maraa : **~1 min par epoch**, soit 9-10 min par run.

Protocole d'evaluation : 500 images de test, PGD 20 pas, 1 restart.

| | v1 (`harden.py`) | **Run A** | Run B | Run C |
|---|---|---|---|---|
| augmentation | non | non | oui | oui |
| perte | pgdat | pgdat | pgdat | trades beta=2 |
| **propre** | 67.2% | **98.8%** | 99.5% | 96.9% |
| FGSM eps=0.05 | 58.8% | **97.0%** | 97.6% | 91.8% |
| FGSM eps=0.10 | 49.2% | **95.4%** | 87.4% | 80.4% |
| FGSM eps=0.20 | 34.8% | **91.8%** | 60.6% | 48.2% |
| FGSM eps=0.30 | 23.8% | **88.2%** | 54.8% | 23.4% |
| PGD eps=0.05 | 53.4% | **96.6%** | 96.0% | 86.4% |
| PGD eps=0.10 | 37.2% | **94.6%** | 78.0% | 56.8% |
| PGD eps=0.20 | 9.6% | **85.8%** | 44.8% | 9.6% |
| PGD eps=0.30 | 1.2% | **65.4%** | 29.8% | 2.2% |

**Le run A gagne partout, et de loin.** Compare a la v1 : **+31.6 pts de
precision propre**, et de +64.4 pts (FGSM) a +76.2 pts (PGD) a eps=0.30.

Trajectoire de la robustesse de validation du run A (val PGD-10) :
26.3 -> 36.7 -> 55.3 -> 64.5 -> 68.5 -> 69.2 -> 70.1 -> 70.1 -> 70.7 -> 70.3.
Elle plafonne a ~70% des l'epoch 5 : le budget d'epochs est maintenant le
facteur limitant, pas la recette.

**L'augmentation a NUIT (run B).** Contre-intuitif, mais explique : la perte du
run B reste bloquee a ~0.82 alors que celle du run A descend a 0.24. Autrement
dit le modele augmente est **sous-entraine** : l'augmentation rend chaque epoch
plus difficile, donc a budget d'epochs egal il prend du retard. Il gagne en
precision propre (99.5% contre 98.8%) mais perd 35 pts de robustesse a eps=0.3.
Conclusion : l'augmentation paie seulement avec 2-3x plus d'epochs, et elle ne
remplace pas l'adversarial training.

**TRADES reste en echec (run C)** : 2.2% a eps=0.3. Deux causes :
1. `--lr 0.002` est beaucoup trop bas (la perte reste bloquee a ~1.68) ; le lr
   par defaut de TRADES avait ete baisse a cause du probleme d'equilibre
   CE/KL, mais trop loin.
2. La decroissance du lr aux epochs 5 et 8 acheve de figer le modele
   (0.00002 a la fin).
A reprendre avec `--lr 0.01` et sans decroissance agressive.

Fichiers : `models/harden2_ref_pgdat.pt` (le meilleur),
`models/harden2_aug_pgdat.pt`, `models/harden2_aug_trades.pt`.

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

### 2026-09-11 - ATTAQUES ADAPTATIVES : la defense v1 etait du gradient masking

Question posee par Maraa : le modele durci tient 65.4% sous PGD, mais PGD est
justement l'attaque contre laquelle il a ete entraine. Est-ce une vraie defense
ou un gradient obfusque ? (Athalye, Carlini & Wagner, 2018)

Nouveau script : `adversarial/scripts/bpda_eot.py`. Il attaque le modele
DEPLOYE (poids durcis v1 + feature squeezing 3 bits a l'inference) avec quatre
attaquants : PGD sans tenir compte du squeezing, PGD avec le VRAI jacobien de la
quantification (nul presque partout, comme torch.round), BPDA (forward exact,
passe arriere = identite), et BPDA+EOT (moyenne sur bits 2/3/4 et translation
+/-2 px).

Run : 200 images, PGD-20, squeezing 3 bits, accuracy propre 67.0%.

| eps | sans defense | naif (vrai jacobien) | BPDA | BPDA+EOT |
|---|---|---|---|---|
| 0.10 | 27.5% | 68.0% | 27.0% | 41.0% |
| 0.20 | 22.5% | 66.5% | 21.5% | 37.0% |
| 0.30 | 1.0% | 62.5% | 1.5% | 10.0% |

CONSTAT : l'attaquant naif laisse 62.5% a eps=0.30 (la defense parait solide),
BPDA la fait tomber a 1.5% (niveau d'un modele non defendu). La couche de
feature squeezing n'apportait donc AUCUNE protection reelle : elle rendait
seulement le gradient inutilisable. C'est le piege classique du gradient
masking, et c'est la raison pour laquelle une defense doit toujours etre
evaluee sous attaque ADAPTATIVE.

Note : BPDA+EOT est ici moins efficace que BPDA seul (10.0% contre 1.5% a
eps=0.30) car la moyenne EOT dilue le signal a nombre de pas fixe.

Portee : le modele durci de la piste torch (98.8% / 65.4%) n'embarque PAS de
feature squeezing et n'est pas stochastique. Son vrai test est plus loin : CW
puis AutoAttack.

### 2026-09-11 - Regression corrigee : models/defend_pgd_mnist_weights.npz

Le fichier livre dans le depot avait ete ECRASE par erreur le 2026-09-10
(commit 66d1e85, pendant la mise au point de harden2.py). Verifie en direct :
il donnait 28.8% propre au lieu des 67.2% documentes. Restaure depuis b8e573f
(verifie : 67.2% de nouveau). Lecon : apres chaque session, verifier que les
poids versionnes reproduisent bien les chiffres du README.

### 2026-09-11 (soir) - RUN B LONG : 91.0% sous PGD eps=0.30

Suite de la campagne. Le run B (augmentation) avait ete juge mauvais a 10
epochs (29.8% sous PGD). Hypothese : ce n'etait pas l'augmentation, c'etait le
BUDGET. Test : le meme run B, pousse a 120 epochs (60000 images, PGD-5,
augmentation, warm start, lr-drop 0.5/0.8).

Run lance en PyTorch CPU (~0.9 min par epoch), interrompu a l'epoch 60 par une
mise en veille du PC, puis REPRIS a l'epoch 61 via `--resume` (checkpoint
complet + planning du lr conserve).

| Config | epochs | Loss finale | Propre | PGD eps=0.30 |
|---|---|---|---|---|
| v1 (`harden.py`) | 3 | - | 67.2% | 1.2% |
| Run A (sans augment) | 10 | 0.24 | 98.8% | 65.4% |
| Run B (augment) | 10 | 0.82 (bloquee) | 99.5% | 29.8% |
| **Run B (augment)** | **120** | **0.308** | **99.6%** | **91.0%** |

Protocole d'evaluation : 500 images de test, PGD 20 pas, **pire cas sur 3
restarts**.
Detail (pire cas) : FGSM eps=0.30 -> 96.0% ; PGD eps=0.05 -> 94.4%,
0.1 -> 92.4%, 0.2 -> 91.6%, 0.3 -> 91.0%. Entrainement : 57 min 33 s (apres
reprise). Avec 1 seul restart on lisait 94.6 / 92.6 / 91.6 / 91.4 : l'ecart va
de 0.2 a 0.4 pt, ce qui confirme que PGD-20 avait deja converge au premier
essai (chiffre stable, donc fiable).

LECTURE :
1. **L'hypothese est validee.** Le run B passe de 29.8% a 91.0% en ne changeant
   QUE le nombre d'epochs. La cause etait bien le sous-entrainement.
2. **A budget suffisant, l'augmentation GAGNE.** 91.0% contre 65.4% pour le run
   A (sans augmentation). A 10 epochs elle semblait nuire ; a 120 elle apporte
   +26 pts de robustesse. La lecon du run B a 10 epochs etait donc une lecon de
   budget, pas de methode.
3. **La loss ne descend jamais au niveau du run A** (0.308 contre 0.24) : le
   modele augmente est encore en train d'apprendre a la fin. Le budget reste le
   facteur limitant.
4. La robustesse de validation (PGD-10) plafonne autour de 91-93% a partir de
   l'epoch 80 ; le meilleur modele est celui de l'epoch 95 (val PGD 93.2%).
5. Ordre de grandeur : Madry et al. 2018 rapportent environ 93% sous PGD eps=0.3
   sur MNIST avec un CNN plus gros et PGD-40. On est a 91.0% avec 421 642
   parametres et PGD-20.

CHIFFRE OFFICIEL (2026-09-11) : **91.0%** sous PGD-20 eps=0.30, pire cas sur
3 restarts, 500 images de test, modele de l'epoch 95.

### 2026-09-11 (soir) - LA SUITE D'ATTAQUES CASSE LE 91% : le pire cas est 42%

Le modele durci (run B, 120 epochs) etait documente a **91.0% sous PGD-20**
eps=0.30. Cette valeur est mesuree contre l'attaque sur laquelle le modele a ete
ENTRAINE. La suite complete (`eval_suite.py`) montre que c'etait une
surestimation massive.

Run : 500 images de test, eps=0.30, precision propre 99.8%.

| Attaque | Famille | Accuracy | Cout |
|---|---|---|---|
| FGSM (1 pas) | gradient | 96.0% | 1 gradient |
| PGD-20 (3 restarts) | gradient | 91.0% | 60 gradients |
| PGD-50 (10 restarts, pas eps/10) | gradient | 92.0%* | 500 gradients |
| APGD-CE (100 puis 200 pas) | gradient | 90.0% / 89.5%* | adaptatif |
| **APGD-DLR (200 pas)** | gradient | **79.0%*** | adaptatif |
| Square (500 pas) | sans gradient | 70.0% | 500 requetes |
| **Square (3000 pas, 2 restarts)** | sans gradient | **42.0%** | 6000 requetes |
| NES (20x10 puis 50x20) | sans gradient | 93.8% / 94.5%* | 2000 requetes |

(*) mesures sur 200 images (granularite 0.5%), les autres sur 500 images.

LECTURE :
1. **Le 91.0% etait une illusion d'attaque faible.** Donnee a l'attaque l'attaque
   la plus forte trouvee (Square, sans gradient), le modele tombe a **42.0%**,
   soit -49 points. C'est un budget de requetes parfaitement realiste pour un
   attaquant reel.
2. **Le gradient renseigne mal l'attaquant.** Toutes les attaques a gradient
   plafonnent entre 79% et 92%, alors qu'une recherche guidee par les SCORES
   descend a 42%. C'est la signature d'un modele ou la surface de perte est
   devenue tres plate -- caracteristique des modeles entraines adversairement.
   L'attaquant qui ne voit pas le gradient fait donc MIEUX que celui qui le voit.
3. **L'attaque d'entrainement etait trop grossiere.** Le modele a ete entraine
   avec PGD-5 et un pas de eps/4 = 0.075 (5 pas). Il a appris a resister a CETTE
   trajectoire, pas a la boule eps entiere. Piste principale : renforcer
   l'attaque interne (PGD-20, pas eps/10).

Verification : la borne de perturbation a ete controlee directement
(|delta|inf = 0.300000 exactement, jamais depassee). Ce n'est pas un bug de
l'attaque. Tendance monotone avec le budget : 500 pas -> 70.0%, 3000 pas ->
42.0%. Resultat reproduit a 200 images (39.0%) et 500 images (42.0%).

CONSEQUENCE : le chiffre a retenir pour ce modele est **42.0%**, pas 91.0%.
La prochaine etape est un entrainement v4 avec une attaque interne plus forte,
puis une reevaluation avec la meme suite.

---

## 2026-09-12 - Run v4 (PGD-20, pas eps/10) : le pire cas remonte de 42.0% a 61.8%

Le run annonce ci-dessus a ete lance et evalue (torch, 60000 images, PGD-20 avec
`--pgd-alpha 0.03` = eps/10, augmentation, batch 256, GPU AMD RX 7800 XT apres
la mise en place de ROCm sous WSL). Meme suite, meme budget, 500 images :

| Attaque | Modele 120 ep (PGD-5) | **v4 (PGD-20, eps/10)** |
|---|---|---|
| Precision propre | 99.8% | 99.4% |
| FGSM | 96.0% | 94.2% |
| PGD-20 (3 restarts) | 91.0% | 90.4% |
| PGD-50 (10 restarts) | 92.0%* | 89.6% |
| APGD-CE | 90.0%* | 88.2% |
| **APGD-DLR** | 79.0%* | **81.0%** |
| Square (500 pas) | 70.0% | 83.6% |
| **Square (3000 pas, 2 restarts)** | **42.0%** | **61.8%** |
| NES | 93.8% | 93.0% |
| **PIRE CAS** | **42.0%** | **61.8%** |

(*) mesures sur 200 images.

LECTURE :

1. **Le diagnostic est valide.** +19.8 points de pire cas, a architecture
   identique et a budget d'attaque egal. Renforcer l'attaque INTERNE a donc bien
   reduit la specificite du modele a une trajectoire d'attaque donnee.
2. **L'ecart PGD <-> Square s'effondre** : 21 points (91.0 vs 70.0) a 500 pas
   deviennent **6.8 points** (90.4 vs 83.6). La surface de perte est moins
   plate : le gradient renseigne de nouveau l'attaquant.
3. **Mais l'ecart existe encore a 3000 pas** : APGD-DLR 81.0% contre Square
   61.8%, soit 19 points. Une recherche par scores reste plus efficace que le
   gradient. L'attaque d'entrainement n'etait donc pas la seule cause.
4. **APGD-DLR est desormais la meilleure attaque a gradient** (81.0% contre
   90.4% pour PGD-20). C'est logique : c'est l'attaque la plus forte connue, et
   c'est celle a laquelle le modele resiste le moins. Prochaine etape :
   s'entrainer CONTRE elle.

Suite : option `--attack apgd-dlr` (attaques d'entrainement APGD, ajoutee le
2026-09-12) pour un run v5, puis test de capacite (`--large`, 1,7M parametres,
deja implemente). Poids versionnes : `models/harden_v4_pgd20.pt` (1,6 Mo).

Note technique du jour : le bug de peripherique corrige dans
`attaques_avancees.py` (restarts d'APGD et de Square qui creaient un tenseur
GPU avec un generateur CPU) etait invisible sur CPU - voir le piege "tester sur
le peripherique cible" ci-dessus.

---

## 2026-09-12 (soir) - Pourquoi le run v5 s'est effondre : la GRAINE de l'attaque interne

Symptome (run v5 de Maraa, GPU RX 7800 XT, `--attack apgd-dlr --pgd-steps 10
--augment --batch 256 --epochs 120 --seed 42`, tout le reste identique au run B) :
la **val PGD10 monte a 46.5% (epoch 16) puis s'effondre** (20.5% e23, 4.9% e28,
**0.6% e32**) et **ne remonte jamais**. Pendant ce temps la precision propre reste
a 99.5-99.6% et la **perte d'entrainement ne cesse de BAISSER** (0.19 e16 -> 0.08
e46, contre ~0.39 pour le run B au meme stade). Journal :
`results/v5_apgd_dlr_effondrement.log`.

Lecture de la perte qui baisse : sur un batch mixte (50% propre + 50% adverse), la
perte de la moitie adverse devrait rester HAUTE (un exemple adverse est difficile).
Une perte qui tombe a 0.08 signifie que les "exemples adverses" sont devenus
FACILES : l'attaque interne ne fabrique plus rien d'adverse. Le chiffre de
validation (PGD10) ne fait que constater la consequence.

Cause racine : deux defauts cumules de notre port d'APGD, invisibles tant que
l'attaque d'entrainement etait PGD (A/B/v4).

1. **La graine du depart aleatoire etait FIXE (`seed=0` par defaut).** Le depart
   est tire par `torch.empty_like(x).uniform_(-eps, eps, generator=gen)` avec un
   generateur recree a chaque appel avec la meme graine : pour une forme de
   tenseur donnee (batch 256) c'est le MEME motif de bruit a chaque batch, de
   l'epoch 1 a l'epoch 120. Or l'APGD garde le meilleur point de sa trajectoire,
   et a eps=0.3 ce depart (bruit uniforme sur toute l'image) est souvent deja mal
   classe : l'attaque renvoyait donc tres souvent CE MOTIF FIXE comme "meilleur
   exemple adverse". Le modele apprenait par coeur a vaincre un motif constant -
   d'ou la perte qui tend vers zero et la precision propre intacte - et
   n'apprenait plus la robustesse. `pgd()` n'avait pas ce defaut (il tire sur le
   generateur GLOBAL, donc un depart frais a chaque batch) : c'est pour cela que
   les runs A/B/v4 n'ont pas souffert. La SEULE variable qui change en v5 est
   l'attaque interne.
2. **Le pas adaptatif ne s'adaptait jamais a budget court.** Les points de
   controle etaient codoes "tous les 10 pas" en dur (`i % 10`), alors que la
   reference (Croce & Hein 2020) les cale sur ~22% du budget. Avec
   `--pgd-steps 10` il n'y avait donc AUCUN palier utile : le pas restait a
   2*eps (0.6) du debut a la fin et l'attaque ne faisait que rejouer FGSM au coin
   de la boule. En prime le "momentum" melangeait des POINTS puis re-projetait,
   ce qui colle le point aux coins : le momentum ne servait a rien (la reference
   melange des DEPLACEMENTS).

Corrections (meme commit) :

- `attaques_avancees.py::apgd` : `seed=None` par defaut -> graine fraiche a CHAQUE
  appel pour le depart aleatoire (tirage sur le generateur global, donc variable
  d'un batch a l'autre, et reproductible si `torch.manual_seed` est appele au
  demarrage). L'evaluation passe toujours une graine explicite
  (`eval_suite` : 2000 / 3000) : elle reste reproductible au bit pres.
- `attaques_avancees.py::apgd` : paliers du pas proportionnels au budget
  (k = 22% de `steps`, puis k diminue par pas de 3% jusqu'a 6%) et melange
  Nesterov sur les deplacements, avec a = 1 au premier pas (comme la reference).
- `attaques.py` : l'attaque d'entrainement APGD demande explicitement `seed=None`.
- `harden_torch.py` : `torch.manual_seed(args.seed)` au demarrage (le run reste
  reproductible). Et l'etat de l'OPTIMISEUR est desormais restaure par
  `--resume` : il etait ecrit dans le checkpoint mais jamais recharge, donc le
  momentum de SGD repartait de zero en plein milieu d'un run long.
- `entrainement.py` : la ligne d'epoch affiche maintenant **CE propre | CE adv |
  taux de tromperie de l'attaque interne**. C'est le chiffre qui aurait montre le
  probleme des l'epoch 20. Deux garde-fous : [ALERTE] si l'attaque interne ne
  trompe plus 50% du batch adverse, [ALERTE] si la val PGD passe sous le meilleur
  de plus de `--collapse-tol` points (10 par defaut) pendant
  `--collapse-patience` epochs (6), avec `--stop-on-collapse` pour arreter un run qui ne remonte plus
  (le meilleur modele est deja sauvegarde).

Lecon a garder : **le chiffre d'entrainement n'est pas le chiffre de robustesse, et
une attaque interne se juge sur sa PERTE et son taux de tromperie.** Une attaque
interne qui renvoie toujours le meme motif fabrique un modele "robuste" sur le
papier et nu a l'evaluation, exactement comme une attaque trop faible. La
validation robuste seule ne suffit pas : il faut instrumenter l'interieur.

Suite : relancer v5 avec l'attaque corrigee en surveillant la colonne CE adv.
Pour l'ENTRAINEMENT, preferer `--attack apgd-ce` (ou rester sur
`pgd --pgd-steps 20 --pgd-alpha 0.03`, la recette v4) : l'objectif DLR de
Croce & Hein est concu pour EXPOSER une surface de perte deguisee a l'evaluation,
le minimiser en boucle n'est pas le meme probleme. Le meilleur modele du run casse
reste l'epoch 16 (val PGD10 46.5%), deja sauvegarde : rien n'est perdu.

### 2026-09-12 (soir, suite) - Deuxieme defaut : la SEMANTIQUE DE RETOUR de l'attaque

Le correctif de la graine ne suffisait pas. Relance du meme run avec l'attaque
corrigee, et voila les premieres epochs :

```
  Epoch  6/120 | loss 0.8484 | CE propre 0.092 | CE adv 2.361 | attaque 89.0% | val clean 99.30% | val PGD10 11.40%
  Epoch 12/120 | loss 0.8120 | CE propre 0.057 | CE adv 2.322 | attaque 88.9% | val clean 99.40% | val PGD10 11.60%
  Epoch 17/120 | loss 0.8064 | CE propre 0.051 | CE adv 2.318 | attaque 89.0% | val clean 99.40% | val PGD10 11.60%
```

**CE adv = 2.32 = ln(10) = 2.303.** C'est la CE d'un modele qui repond
uniformement (n'importe quoi) : le taux de tromperie de l'attaque est FIGE a 89%
depuis 12 epochs et la val PGD10 est FIGEE a 11.6%. Le modele n'apprend rien de la
moitie adverse : il optimise la moitie propre (CE 0.05) et repond "je ne sais
pas" sur l'autre.

Cause : l'APGD renvoyait **le depart aleatoire**, pas un point de sa marche. La
semantique d'AutoAttack est "garde le PIRE point trouve, depart inclus" - correct
pour EVALUER (on veut le pire cas), faux pour ENTRAINER. A eps=0.3 sur MNIST, le
depart aleatoire (bruit uniforme sur toute l'image) est deja plus destructeur que
tout ce que la marche trouve : l'attaque renvoyait donc du bruit. Un bruit FRAIS
a chaque batch est non apprenable (label noise irresoluble, CE optimale = ln 10) :
le modele apprend seulement a repondre uniformement dessus, n'apprend aucune
robustesse, et le depart reste donc catastrophique - verrou auto-entretenu. Et
c'est le meme defaut qui expliquait le PREMIER effondrement : avec une graine
fixe, le bruit etait constant, donc MEMORISABLE (loss -> 0.08, val PGD10 -> 0.6%).
Meme cause, deux visages.

PGD, lui, renvoie le DERNIER point de sa marche (une perturbation guidee par le
gradient, donc une fonction deterministe de l'image, apprenable) : c'est pourquoi
les runs A/B/v4, tous en PGD, n'ont jamais eu ce probleme.

Correction : nouveau parametre `retour` dans `apgd()`.

- `retour="meilleur"` (defaut) : semantique d'AutoAttack, pour l'EVALUATION
  (`eval_suite` ne change pas, les chiffres publies restent comparables) ;
- `retour="dernier"` : renvoie le dernier point de la marche, comme PGD, et le
  depart aleatoire n'est plus un candidat (premier point de controle apres k pas,
  comme la reference). C'est ce que demande l'attaque d'ENTRAINEMENT
  (`attaques.attaque`).

Nouvel outil `diag_attaque_interne.py` : sur un lot d'images, il compare la CE et
la precision sous (1) le depart aleatoire seul, (2) APGD retour=dernier,
(3) APGD retour=meilleur, (4) PGD pas eps/10, (5) PGD-10 pas eps/4. Si les lignes
1 et 3 sont au niveau de ln(10) alors que la 2 est nettement en dessous, c'est la
preuve directe du mecanisme. A lancer en 30 s :

    python3 adversarial/torch/diag_attaque_interne.py --weights models/harden_v5_apgd_ce.pt

Lecon a ajouter : **une attaque d'evaluation et une attaque d'entrainement n'ont
pas le meme contrat.** L'evaluation cherche le pire point (depart aleatoire inclus) ;
l'entrainement doit fournir une perturbation que le modele PEUT apprendre a
repousser, donc un point visite par la marche. Utiliser une attaque d'evaluation
comme attaque interne, c'est entrainer sur du bruit.

#### Correction (meme soir, 19h30) : l'hypothese "l'attaque renvoyait le bruit" est FAUSSE

Deux mesures l'ont refutee, et il faut les garder :

1. `diag_attaque_interne.py` sur le modele v5 bloque : APGD retour=dernier
   (CE 2.343, acc 14.06%), APGD retour=meilleur (2.343, 14.06%) et PGD-20 eps/10
   (2.457, 14.06%) trouvent les MEMES points. L'attaque renvoyait donc bien de
   vrais exemples adverses, pas le depart aleatoire. (Le correctif `retour`
   reste juste architecturalement, mais il ne change rien ici.)
2. Mesure au moteur NumPy sur le modele STANDARD non robuste
   (`model_weights_full.npz`, 1000 images) : propre 98.50%, **bruit uniforme
   eps=0.3 : 97.00%**, FGSM eps=0.3 : 2.1%. Un bruit uniforme de moyenne nulle
   est donc quasi inoffensif (les convolutions le moyennent) : il ne peut pas etre
   la cause de l'effondrement, et la ligne 1 du diag a 90% est normale.

**Ce qui reste vrai et non explique** : sur le modele entraine avec l'attaque
APGD, la CE de la moitie adverse se bloque a 2.33 = ln(10) et le modele repond
uniformement sur les exemples adverses (il "abandonne" cette moitie) alors qu'il
ajuste parfaitement le propre (CE 0.06) ; la val PGD10 est figee a 11.6%. Avec
PGD-5 (run B, CE adv ~1.14) et PGD-20 eps/10 (v4, 90.4% sous PGD-20 cote test),
la meme recette apprend. La difference ne peut donc etre que dans la
perturbation produite par l'attaque interne.

**Hypothese retenue (a tester)** : ce n'est pas la FORCE de l'attaque interne qui
compte, c'est la DOUCEUR de sa trajectoire. Notre APGD part avec un pas de
2*eps : le premier pas saute au coin de la boule (tous les pixels a la borne), la
perturbation est un masque binaire extreme des le depart, et la moitie adverse du
batch devient impossible a ajuster (verrou : CE = ln(10)). PGD avec un pas fin
(eps/10, recette v4) n'atteint la borne que progressivement : une grande part de
pixels reste a l'interieur, la perturbation est lisible, et le modele apprend.

**Experience controlee (20 min, une seule variable, aucun code a toucher) :**
lancer la MEME commande trois fois en ne faisant varier que le pas de l'attaque
interne (PGD, 20 pas) : `--pgd-alpha 0.03` (eps/10, temoin attendu robuste),
`--pgd-alpha 0.075` (eps/4), `--pgd-alpha 0.3` (eps, saut au coin = ce que fait
notre APGD). Si les trois donnent 90% / intermediaire / ~11%, l'hypothese est
confirmee et l'ablation devient un resultat publiable : "le pas de l'attaque
interne decide si l'entrainement apprend, pas sa force".

Le diag a ete enrichi pour mesurer directement cette douceur : colonnes
`|d|moy`, `borne` (part de pixels a |delta| = eps) et `signes opposes` (50% =
masque aleatoire, moins = champ coherent).

### 2026-09-12 (20h30) - ABLATION : la LOI DU BUDGET DE DEPLACEMENT de l'attaque interne

Experience controlee demandee apres le plateau a 11.6% : meme commande, meme
recette (PGD-20 pas, eps=0.30, augment, batch 256, lr 0.05, 120 epochs, seed 42,
60 000 images), **seul le pas de l'attaque interne change**. Le "budget de
deplacement" est `pas x nombre de pas` : c'est la distance maximale (en L-infini)
que l'attaque peut parcourir dans la boule.

Evaluation finale (500 images, PGD-20, 1 restart) :

| Run | pas alpha | budget | FGSM eps=0.30 | PGD-20 eps=0.30 |
|---|---|---|---|---|
| abL_a | 0.03 (eps/10) | 0.6 = **2 eps** | 94.0% | **92.0%** |
| abl_b | 0.075 (eps/4) | 1.5 = 5 eps | 79.4% | 75.6% |
| abl_c | 0.30 (eps) | 6.0 = 20 eps | 48.6% | **6.8%** |

Et si on ajoute tous les runs precedents (meme recette, meme budget de 120
epochs), la tendance est monotone :

| Run | attaque interne | budget | PGD-20 eps=0.3 | pire cas |
|---|---|---|---|---|
| run B | PGD-5, eps/4 | 0.375 = 1.25 eps | 91.0% | 42.0% (Square 3000) |
| v4 | PGD-20, eps/10 | 0.6 = 2 eps | 90.4% | **61.8%** |
| abl_a | PGD-20, eps/10 | 0.6 = 2 eps | **92.0%** | a mesurer |
| abl_b | PGD-20, eps/4 | 1.5 = 5 eps | 75.6% | a mesurer |
| abl_c | PGD-20, eps | 6.0 = 20 eps | 6.8% | a mesurer |
| v5 | APGD-CE-20, pas 2 eps | 12.0 = 40 eps | ~11% | - |

LECTURE :

1. **La robustesse apprise decroit avec le budget de deplacement de l'attaque
   interne**, et il y a une FALAISE entre 2 eps et 20 eps : a budget 2 eps on
   obtient 92%, a budget 6 eps on tombe a 6.8%. Ce n'est donc pas la "force" de
   l'attaque interne qui compte (au sens du pire cas qu'elle atteint) : c'est la
   LISIBILITE de la perturbation qu'elle fabrique.
2. **Mecanisme.** Avec un pas >= eps, chaque iteration saute au COIN de la boule
   (tous les pixels a la borne) et la suivante re-saute vers un autre coin, en
   suivant le gradient evalue sur une image deja totalement deformer : le motif de
   signe obtenu est chaotique, proche d'un masque de bruit. Le modele ne peut pas
   apprendre une frontiere coherente dessus : il se bloque et repond
   uniformement (CE = ln(10) = 2.303, mesure exacte du plateau a 11.6%). Avec un
   pas fin (eps/10), la perturbation se construit progressivement, en accumulant
   des signes de gradient coherents : elle reste lisible, et le modele apprend.
3. **Pourquoi APGD echoue comme attaque interne alors qu'il est excellent en
   evaluation** : APGD (Croce & Hein) demarre avec un pas de **2 eps** et adapte
   a la hausse la puissance de sa recherche. C'est exactement ce qu'on veut pour
   JUGER un modele (trouver le pire point), et exactement ce qui detruit
   l'apprentissage (budget 40 eps -> plateau ln(10)). Ce n'est pas un bug de
   notre port : c'est un CONTRAT different.
4. **Le bon reglage est un compromis, pas un maximum.** Trop mou, l'entrainement
   apprend une robustesse masquee (run B, budget 1.25 eps : 91% sous PGD-20 mais
   42% seulement de pire cas). Trop dur, il n'apprend plus rien (abl_c, v5).
   Le creux de la courbe est autours de **2 eps** (PGD-20 pas eps/10) : 92% sous
   PGD-20 ET le meilleur pire cas mesure (61.8% pour v4, meme recette).

REGLE A RETENIR : **une attaque interne doit rester DOUCE (budget <= 2 eps, donc
pas <= eps/10 avec 20 pas). On n'entraine pas contre la meme attaque qu'on
utilise pour juger.** Si le budget depasse ~5 eps, l'entrainement se verrouille
(CE adverse = ln(10)) et le tableau de bord le montre des les premieres epochs.

A FAIRE : pire cas de abl_a (eval_suite avec Square 3000) pour verifier qu'il
egale ou depasse v4 (61.8%) ; comparer les colonnes CE adv des logs abl_a et
abl_c (attendu : ~1 pour abl_a, ~2.30 pour abl_c) ; tester un budget plus fin
(eps/20) pour savoir si le creux est plus bas.
