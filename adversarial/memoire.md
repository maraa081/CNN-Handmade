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

### 2026-09-11 (soir) - RUN B LONG : 91.4% sous PGD eps=0.30

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
| **Run B (augment)** | **120** | **0.308** | **99.6%** | **91.4%** |

Protocole d'evaluation : 500 images de test, PGD 20 pas, 1 restart.
Detail : FGSM eps=0.30 -> 96.0% ; PGD eps=0.05 -> 94.6%, 0.1 -> 92.6%,
0.2 -> 91.6%, 0.3 -> 91.4%. Entrainement : 57 min 33 s (apres reprise).

LECTURE :
1. **L'hypothese est validee.** Le run B passe de 29.8% a 91.4% en ne changeant
   QUE le nombre d'epochs. La cause etait bien le sous-entrainement.
2. **A budget suffisant, l'augmentation GAGNE.** 91.4% contre 65.4% pour le run
   A (sans augmentation). A 10 epochs elle semblait nuire ; a 120 elle apporte
   +26 pts de robustesse. La lecon du run B a 10 epochs etait donc une lecon de
   budget, pas de methode.
3. **La loss ne descend jamais au niveau du run A** (0.308 contre 0.24) : le
   modele augmente est encore en train d'apprendre a la fin. Le budget reste le
   facteur limitant.
4. La robustesse de validation (PGD-10) plafonne autour de 91-93% a partir de
   l'epoch 80 ; le meilleur modele est celui de l'epoch 95 (val PGD 93.2%).
5. Ordre de grandeur : Madry et al. 2018 rapportent environ 93% sous PGD eps=0.3
   sur MNIST avec un CNN plus gros et PGD-40. On est a 91.4% avec 421 642
   parametres et PGD-20.

A FAIRE : relancer l'evaluation avec `--restarts 3` (pire cas) pour figer le
chiffre officiel.
