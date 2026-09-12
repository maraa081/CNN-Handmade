# Piste PyTorch — quand le modele grossit

> Cette piste existe pour une raison precise : **le code fait main a une limite
> de debit**. Tant que le modele est petit, ecrire les gradients soi-meme est
> pedagogiquement imbattable et suffisant. Quand le modele grossit, ce n'est
> plus tenable — et c'est la qu'on passe a PyTorch.

Les deux pistes vivent cote a cote. **Aucune ne remplace l'autre.**

| Piste | Dossier | Role |
|---|---|---|
| Faite main (NumPy) | `src/`, `adversarial/scripts/` | la **reference pedagogique** : chaque gradient est ecrit a la main, rien n'est cache |
| PyTorch (autograd) | `adversarial/torch/` | le **banc d'essai rapide** : memes maths, moteur optimise, GPU possible |

---

## 1. Pourquoi il a fallu changer de moteur

Mesures sur la meme machine (i5-6300U, 4 coeurs), apres optimisation :

| | Faite main (NumPy) | PyTorch (CPU) |
|---|---|---|
| Entrainement propre | 172 img/s | ~1 500 img/s |
| 5000 images, PGD-5, 1 epoch | ~3 min 20 | **23 s** |
| 60000 images, 1 epoch propre | 5.8 min | ~40 s |
| GPU | impossible | oui |

Le code NumPy a ete optimise de 2.4x (voir `docs/memoire-projet.md`), mais il
est maintenant **au plafond de son BLAS** : le temps est passe dans les produits
matriciels eux-memes. Doubler la taille du modele double le temps
d'entrainement. Ajouter une couche de convolution, passer a 128 canaux, utiliser
des images en 224x224 : chacune de ces decisions multiplie une duree deja
longue.

PyTorch change trois choses :

1. **L'autograd** : plus besoin d'ecrire `Conv2D.backward()` ni `col2im`. Le
   gradient se deduit du forward.
2. **Des noyaux optimises** : les convolutions sont des primitives natives
   (MIOpen sur AMD, cuDNN sur NVIDIA), pas une boucle im2col maison.
3. **Le GPU** : le meme code tourne sur carte graphique, avec un facteur
   supplementaire de plusieurs dizaines.

---

## 2. Ce qui est identique (volontairement)

L'objectif n'est pas de faire "un autre projet" mais **le meme**, avec un autre
moteur. La correspondance est volontairement terme a terme :

| Fait main | PyTorch | Remarque |
|---|---|---|
| `src/layers.py :: Conv2D` | `nn.Conv2d` | meme im2col a la main / noyau natif |
| `src/layers.py :: MaxPool2D` | `nn.MaxPool2d` | |
| `src/layers.py :: Dense` | `nn.Linear` | |
| `src/model.py :: CNN.backward()` | `loss.backward()` | autograd |
| `src/optimizers.py :: SGD` | `torch.optim.SGD` | |
| `adversarial/scripts/fgsm.py` | `torch/attaques.py :: fgsm` | **meme formule** |
| `adversarial/scripts/pgd.py` | `torch/attaques.py :: pgd` | alpha = eps/4, random start |
| `adversarial/scripts/harden2.py` | `torch/entrainement.py` | memes recettes |
| `adversarial/scripts/augment.py` | `torch/entrainement.py :: augmenter` | memes transformations |

Memes recettes d'entrainement : `pgdat`, `trades`, mix propre/adverse,
decroissance du learning rate, ecrêtage des gradients, selection du modele sur
la robustesse de validation.

Meme architecture, a la virgule pres :

    Conv2d(1 -> 32, k=3, pad=1) -> ReLU -> MaxPool2d(2)
    Conv2d(32 -> 64, k=3, pad=1) -> ReLU -> MaxPool2d(2)
    Flatten (3136) -> Linear(3136 -> 128) -> ReLU -> Linear(128 -> 10)

421 642 parametres, comme la version faite main.

---

## 3. La preuve que c'est bien le meme modele

Deux mecanismes, parce qu'une reecriture non verifiee ne vaut rien.

### 3.1 Les poids sont interchangeables

`modele.py` sait **lire et ecrire le format `.npz` de la version NumPy**
(memes cles `conv_0_kernels`, `dense_7_W`, ... ; biais remis en forme `(C, 1)`).

Consequences :
- on peut charger dans PyTorch un modele entraine a la main et **continuer** dessus ;
- on peut sauvegarder depuis PyTorch et **reutiliser les poids dans les scripts NumPy** ;
- les experiences des deux pistes restent comparables.

### 3.2 Le mode `--parite`

Charge **le meme fichier de poids** dans les deux implementations, calcule
l'accuracy propre et sous attaque sur **le meme echantillon**, et compare :

    python3 adversarial/torch/harden_torch.py --parite

Resultat obtenu (200 images, `model_weights_full.npz`) :

| Mesure | PyTorch | NumPy | Ecart |
|---|---|---|---|
| propre | 98.50% | 98.50% | **0.00%** |
| FGSM eps=0.30 | 1.50% | 1.50% | **0.00%** |
| PGD eps=0.20 | 0.00% | 0.00% | **0.00%** |
| FGSM eps=0.10 | 76.50% | 77.00% | 0.50% |

L'ecart de 0.50% sur FGSM eps=0.10 correspond a **une seule image sur 200** :
les deux implementations somment les gradients dans des ordres differents
(float32), et une image pile sur la frontiere de decision bascule. C'est le
comportement attendu, et c'est la raison d'etre du test.

---

## 4. Ce qu'on perd (et qu'il faut assumer)

- **La pedagogie du backward manuel.** C'est tout l'interet de `src/` : avoir
  ecrit `col2im`, la retropropagation de la convolution, le routage du gradient
  dans le MaxPool. PyTorch le fait pour nous, donc on n'apprend plus rien a ce
  niveau.
- **Le controle fin.** Un `loss.backward()` cache l'ordre des operations. Quand
  quelque chose se comporte bizarrement, on debugue moins facilement.
- **Une dependance lourde.** PyTorch, c'est ~800 Mo installe, et une version
  supportee par ta version de Python.

C'est pour ca que **`src/` reste la reference** : c'est lui qui montre comment
ca marche, `adversarial/torch/` sert a produire des chiffres a une vitesse
utilisable.

---

## 5. Utilisation

> **Pour modifier le code** (et pas seulement le lancer) : lire d'abord
> [`lire-le-code.md`](lire-le-code.md) — visite guidee du trajet d'un batch,
> tableau "ou toucher pour faire X" et trois exercices d'entree.

    # 1. Verifier l'equivalence avec la version faite main (a faire en premier)
    python3 adversarial/torch/harden_torch.py --parite

    # 2. Verifier que tout tourne
    python3 adversarial/torch/harden_torch.py --quick

    # 3. La recette recommandee
    python3 adversarial/torch/harden_torch.py --n-train 60000 --epochs 15 --pgd-steps 5 --augment

    # 4. Variante TRADES
    python3 adversarial/torch/harden_torch.py --n-train 60000 --epochs 15 --loss trades

    # 5. Evaluation honnete d'un modele (accepte aussi un .npz en entree)
    python3 adversarial/torch/harden_torch.py --report models/harden_torch_best.pt --restarts 3

    # 6. Sauvegarder aussi au format NumPy (reutilisable dans les scripts faits main)
    python3 adversarial/torch/harden_torch.py ... --npz models/harden_torch_best.npz

    # 7. La campagne des 3 runs (reference / augmentation / TRADES) avec ce moteur
    ./adversarial/scripts/campagne.sh --torch --liste
    ./adversarial/scripts/campagne.sh --torch

    # 8. Reprendre un run interrompu (veille du PC, arret manuel...)    #    Un checkpoint COMPLET (poids + optimiseur + epoch + lr) est ecrit a
    #    chaque epoch dans <out>_last.pt : rien n'est perdu en cas de coupure.
    python3 adversarial/torch/harden_torch.py ... \
        --resume models/harden_torch_best.pt_last.pt

    #    Ancien format (state_dict seul) : preciser l'epoch et le lr
    python3 adversarial/torch/harden_torch.py ... \
        --resume ancien_modele.pt --start-epoch 61 --lr 0.005

    # 9. JUGER une defense : suite d'attaques multi-familles (le juge)
    python3 adversarial/torch/eval_suite.py --weights models/....pt
    python3 adversarial/torch/eval_suite.py --weights models/....pt --quick
    python3 adversarial/torch/eval_suite.py --weights models/....pt --famille blackbox

    # 9b. ... et ecrire les chiffres en JSON pour le tableau final
    python3 adversarial/torch/eval_suite.py --weights models/....pt --label "mon run" \
        --json adversarial/results/logs/mon_run.json

    # 9c. Le tableau recapitulatif, pret a coller (lit tous les JSON du dossier)
    python3 adversarial/torch/tableau_recap.py
    python3 adversarial/torch/tableau_recap.py --csv adversarial/results/logs/recap.csv

    # 10. Robustesse CERTIFIEE (borne L2 garantie, pas une observation)
    python3 adversarial/torch/smoothing.py --entrainer --sigma 0.5 --epochs 90
    python3 adversarial/torch/smoothing.py --certifier --sigma 0.5 --n 1000

Les options sont **les memes** que `harden2.py` (`--loss`, `--mix`, `--beta`,
`--augment`, `--aug-fort`, `--aug-config`, `--lr`, `--lr-drop`, `--clip`,
`--warm-start`, `--restarts`...), plus :

    --device auto|cpu|cuda|dml      choix du peripherique (auto par defaut)
    --batch 256                     conseille sur GPU
    --resume FICHIER                reprendre un run interrompu (<out>_last.pt)
    --start-epoch N                 epoch de depart (0 = deduit du checkpoint)
    --collapse-tol N                alerte si la val PGD passe sous le meilleur de
                                    plus de N points (10 par defaut, 0 = desactive). La val
                                    PGD10 est bruitee (1 restart, 1000 images) :
                                    augmenter la tolerance plutot que la patience
    --collapse-patience N           epochs sous le meilleur avant l'alerte (6)
    --stop-on-collapse              arreter un run qui s'effondre et ne remonte plus
    --parite                        test d'equivalence avec NumPy
    --npz FICHIER                   exporter les poids au format NumPy

Deux choses a savoir avant de lancer un run long (ajoutees le 2026-09-12) :

- **La ligne d'epoch instrumente l'attaque interne** :
  `CE propre | CE adv | attaque N%`. Le taux de tromperie et la CE adverse sont
  lus sur les logits du batch mixte, donc c'est gratuit. Si la CE adverse rejoint
  la CE propre (ou si le taux de tromperie passe sous 50%), l'attaque interne ne
  fabrique plus d'exemples adverses et le modele n'apprendra plus la robustesse,
  meme si la perte continue de baisser : un [ALERTE] est affiche. C'est le
  symptome exact du run v5 du 12/09 (perte 0.08, val PGD10 0.6%).
- **La graine du depart aleatoire des attaques d'entrainement doit etre FRAICHE a
  chaque batch.** `harden_torch.py` appelle `torch.manual_seed(args.seed)` au
  demarrage (donc le run est reproductible) et les attaques tirent leur depart sur
  le generateur global a chaque appel. Ne jamais figer la graine d'une attaque
  d'entrainement : un depart constant fait apprendre par coeur un motif fixe.
- **Une attaque d'entrainement renvoie un point VISITE, jamais le depart.** `apgd()`
  a un parametre `retour` : `"meilleur"` (defaut, EVALUATION - le pire point de la
  trajectoire, depart inclus, semantique d'AutoAttack) et `"dernier"` (dernier
  point de la marche, comme PGD - ce que demande l'entrainement). A eps eleve le
  depart aleatoire est deja le pire point, donc l'entrainement qui garde le
  "meilleur" s'entraine sur du bruit non apprenable (CE adv = ln(10) = 2.32,
  val figee, constate le 2026-09-12).

Pour verifier tout ca en 30 s sur un modele deja entraine :

    python3 adversarial/torch/diag_attaque_interne.py --weights models/....pt

---

## 6. Preparer le GPU (carte AMD)

Le code detecte tout seul : `cuda` (qui couvre aussi ROCm) > `directml` > `cpu`.

### Voie recommandee : ROCm sous WSL2

Le **RX 7800 XT** (nom de code `gfx1101`) est listé dans la matrice de
compatibilite ROCm en WSL2 (ROCm 6.4.2, Ubuntu 22.04 ou 24.04, avec
AMD Software Adrenalin Edition « for WSL2 »).

    # cote Windows : installer AMD Software Adrenalin Edition pour WSL2
    # puis dans WSL (Ubuntu 22.04 ou 24.04) :
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Sous ROCm, PyTorch expose le GPU via l'API `cuda` (c'est HIP derriere) : le
`True` ci-dessus signifie que la carte est vue.

### Voie de repli : DirectML (Windows natif)

    pip install torch-directml
    python adversarial/torch/harden_torch.py --device dml ...

DirectML fonctionne sur toute carte DirectX 12, mais Microsoft l'a place en
**maintenance mode** : plus de developpement actif, et plus lent que ROCm.

### Deux reglages qui comptent sur GPU

- **Batch** : 256 minimum. A 64 images les noyaux sont trop petits pour occuper
  la carte, et le gain s'effondre.
- **AMP** (precision mixte float16) : a activer si la memoire devient limitante
  — pas encore implemente ici, a ajouter si besoin.

---

## 7. Regles de la piste

1. **`src/` reste la reference.** Aucune modification du moteur fait main au
   nom de la performance PyTorch.
2. **Tout resultat obtenu en PyTorch doit etre signale comme tel** dans
   `adversarial/memoire.md` (piste « torch »), pour ne pas melanger les mesures
   des deux moteurs.
3. **Re-verifier `--parite` a chaque changement d'architecture.** C'est le seul
   garde-fou contre une divergence silencieuse entre les deux implementations.
4. Les poids restent **interchangeables au format `.npz`** : c'est ce qui garde
   les deux pistes comparables.
5. **Un run long doit pouvoir reprendre.** Un checkpoint complet est ecrit a
   chaque epoch, et `--resume` repart de la sans perdre le planning du learning
   rate. Un run de 120 epochs interrompu a l'epoch 60 reprend a l'epoch 61, avec
   le lr de l'epoque et les paliers restants.

---

## 8. Fichiers

    torch/
    |-- README.md          <- ce fichier
    |-- lire-le-code.md    <- visite guidee du code (pour MODIFIER, pas juste lire)
    |-- modele.py          <- meme architecture en nn.Module + conversion .npz
    |-- attaques.py        <- FGSM et PGD (memes formules)
    |-- attaques_avancees.py <- CW, APGD (CE/DLR), Square, NES, Boundary
    |-- entrainement.py    <- pgdat / trades, augmentation, validation robuste
    |-- eval_suite.py      <- suite d'attaques multi-familles (le juge)
    |-- diag_attaque_interne.py <- l'attaque interne d'entrainement est-elle
    |                          informative ? (CE du depart aleatoire vs APGD retour
    |                          dernier/meilleur vs PGD) - 30 s, aucune epoch
    |-- attaque_adaptative.py <- attaque interne a BUDGET ADAPTATIF : on s'arrete
    |                          au pas k* ou la difficulte atteint la cible du batch
    |                          (option --bande ; note : ../attaque_adaptative.md)
    |-- test_attaque_adaptative.py <- verifie selection / equivalence pgd / cout
    |-- audit_masquage.py  <- le modele est-il robuste ou est-ce que nos attaques
    |                          ne le lisent plus ? (saturation, difference finie,
    |                          eps jusqu'a 1.0, pas fin, transfert) - evaluation seule
    |-- smoothing.py       <- robustesse certifiee (randomized smoothing)
    `-- harden_torch.py    <- point d'entree (memes options que harden2.py)

### Utilisation de l'attaque adaptative (`--bande`)

    # controles rapides du module (quelques secondes, CPU)
    python -u adversarial/torch/test_attaque_adaptative.py

    # run A1 : cible de tromperie 50% par batch
    python -u adversarial/torch/harden_torch.py --n-train 60000 --epochs 120 \
      --augment --pgd-steps 20 --pgd-alpha 0.03 --bande --cible 0.5 \
      --device cuda --out models/bande_cible50.pt

Cout identique a `--pgd-steps 20` sans `--bande` (la difficulte est lue sur
les logits deja calcules). Detail et protocole : `../attaque_adaptative.md`.
