# Lire le code PyTorch — visite guidee

> Ce document est fait pour **modifier** le code, pas seulement le lire.
> Il decrit le trajet d'un batch de bout en bout, fichier par fichier, puis
> donne un tableau "ou toucher pour faire X" et trois exercices d'entree.
>
> Point de depart : la piste PyTorch est dans `adversarial/torch/`. Elle rejoue
> **exactement** les memes experiences que la version faite main
> (`adversarial/scripts/`), mais avec l'autograd : le code est plus court et
> beaucoup plus simple a modifier.

---

## 0. Est-ce que `git pull` est risque pendant un entrainement ?

**Non.** Trois raisons :

1. **Python charge les `.py` en memoire au demarrage.** Une fois le script lance,
   modifier les fichiers n'a **aucun** effet sur le process en cours — les
   changements s'appliqueront au **prochain** lancement. C'est justement pour ca
   que `--resume` existe.
2. **Git ne touche pas aux fichiers ignores.** Les poids et checkpoints
   (`models/*.pt`, `models/*_last.pt`) et les logs
   (`adversarial/results/logs/`) sont dans `.gitignore` : un `git pull` ne peut
   pas les ecraser.
3. **L'entrainement ne lit le disque que pour les donnees** (`data/mnist.npz`,
   deja chargees) et ecrit uniquement ses checkpoints.

En pratique : pull, lecture et modification **libres**. La seule chose a eviter
est de **lancer un second entrainement** en meme temps (ils se disputeraient le
CPU).

---

## 1. La carte du code

| Fichier | Role | Taille |
|---|---|---|
| `harden_torch.py` | **point d'entree** : lit les options, cable tout, lance l'entrainement, l'evaluation, et les modes `--parite` / `--report` / `--resume` | ~360 l. |
| `entrainement.py` | **le coeur** : donnees, augmentation, la boucle d'entrainement, la validation, l'evaluation honnete | ~350 l. |
| `attaques.py` | FGSM et PGD (les memes formules qu'a la main, en 60 lignes) | ~60 l. |
| `modele.py` | le CNN en `nn.Module` + la passerelle `.npz` avec la version NumPy | ~120 l. |

Ordre de lecture conseille : `modele.py` -> `attaques.py` -> `entrainement.py`
-> `harden_torch.py`. Du plus simple au plus complet.

---

## 2. Le trajet d'un batch (le coeur de tout)

Tout se passe dans `entrainer()` (fichier `entrainement.py`). Voici ce qui
arrive a un batch d'images, dans l'ordre :

```
  x_tr, y_tr  (60000 images, melangees differemment a chaque epoch)
      |
      |  1. decoupage en lots de `batch` images (256 sur GPU, 64 par defaut)
      v
    bx, by
      |
      |  2. AUGMENTATION (si --augment) : rotation, zoom, bruit, cutout...
      v
    bx (transforme)
      |
      |  3. ATTAQUE : on demande au modele COURANT le pire bruit possible
      |     bx_adv = pgd(modele, bx, by, eps, steps)     <- attaques.py
      v
    bx_adv  (les memes images, deplacees de eps au pire endroit)
      |
      |  4. PERTE : selon la recette
      |     pgdat : on entraine sur [bx ; bx_adv[:mix]]
      |     trades: perte = CE(modele(bx)) + beta * KL(...)
      v
    perte
      |
      |  5. MISE A JOUR : opt.zero_grad() -> perte.backward() ->
      |     clip_grad_norm_ -> opt.step()
      v
    poids mis a jour
      |
      |  6. VALIDATION (une fois par epoch) : accuracy propre + accuracy
      |     SOUS ATTAQUE PGD sur un jeu de validation
      v
    sauvegarde si la robustesse de validation s'ameliore
```

Trois details qui comptent :

- **Le modele passe en `eval()` pendant l'attaque.** Normal : l'attaque doit
  viser le modele tel qu'il se comporte a l'inference, pas avec son dropout
  actif. Il revient en `train()` juste apres.
- **`x_adv` est re-detache a chaque pas de PGD** (`x_adv.detach() + ...`) : on ne
  veut pas deriver par rapport a la perturbation, seulement l'utiliser.
- **La validation decide de la sauvegarde**, pas l'accuracy propre. Un modele
  non robuste a souvent une meilleure accuracy propre : ce serait le mauvais
  critere.

---

## 3. Les quatre fichiers, un par un

### `modele.py` — le reseau et la passerelle avec NumPy

- `class CNN(nn.Module)` : la meme architecture que `src/model.py`, a la
  virgule pres. `forward()` se lit de haut en bas :
  conv -> relu -> pool -> conv -> relu -> pool -> flatten -> fc -> relu -> fc.
- `reset_parameters()` : initialisation **He** (kaiming), identique a
  `randn * sqrt(2/fan_in)` cote NumPy. Les biais a zero.
- `charger_npz()` / `sauver_npz()` : la passerelle. Les cles du `.npz` suivent
  l'**index de la couche dans la liste NumPy** : `conv_0`, `conv_3`, `dense_7`,
  `dense_9`. Ces indices viennent de l'ordre des couches (Conv2D, ReLU,
  MaxPool2D, Conv2D, ReLU, MaxPool2D, Flatten, Dense, ReLU, Dense) : ils sont
  **figes**, ne pas les renommer.
  C'est ce qui permet de charger dans PyTorch un modele entraine a la main, et
  inversement.

### `attaques.py` — FGSM et PGD en 60 lignes

- `_grad_entree()` : **le coeur**. `x.requires_grad_(True)`, on calcule la
  cross-entropy, et `torch.autograd.grad(perte, x)` renvoie le gradient par
  rapport a l'**image** (et non aux poids). Cote NumPy, c'est exactement ce que
  fait `model.backward()` — mais on l'avait ecrit a la main.
- `fgsm()` : une seule etape, `x + eps * sign(grad)`.
- `pgd()` : la boucle. A chaque pas : gradient, petit pas de `alpha = eps/4`,
  puis **projection** dans la boucle L-inf (`clamp(x-eps, x+eps)`) et dans
  l'espace image (`clamp(0, 1)`).
- `attaque()` : le selecteur appele par l'entrainement (`fgsm`, `fgsm-rs`,
  `pgd`). **C'est ici qu'on branche une nouvelle attaque.**

### `entrainement.py` — donnees, augmentation, boucle

- `charger_train()` / `charger_test()` : la selection est **volontairement
  identique** a celle de `harden2.py` (meme `RandomState(0)` pour
  l'entrainement, `RandomState(42)` pour le test), sinon les chiffres des deux
  pistes ne seraient pas comparables.
- `augmenter()` + les helpers `_affine`, `_translation`, `_bruit`, `_cutout`,
  `_epaisseur` : les 6 transformations, version tensorisee (donc sur GPU et sans
  boucle Python).
- `_paliers_lr()` : les epochs ou le learning rate est divise par 10. Calcules
  une seule fois en **numeros d'epochs**, pour que reprendre un run ne
  decale pas le planning.
- `_beta_effectif()` : la rampe de beta pour TRADES (on ne demarre jamais a beta
  plein sur un modele deja converge, sinon la KL ecrase la CE).
- `entrainer()` : **la boucle decrite en section 2**.
- `evaluer()` / `rapport()` : l'evaluation honnete (PGD-20, plusieurs restarts,
  on garde le pire cas).

### `harden_torch.py` — le point d'entree

- `main()` fait tout le cablage dans l'ordre : options -> donnees -> modele ->
  reprise (`--resume`) -> warm start -> optimiseur -> entrainement -> `.npz`
  optionnel -> evaluation finale.
- Les modes speciaux : `--parite` (comparer avec NumPy sur les memes poids),
  `--report` (evaluer un modele deja entraine, sans reentrainer), `--quick`
  (petit run de verification de la chaine).

---

## 4. Ou toucher pour faire X

| Je veux... | Fichier | Ou exactement |
|---|---|---|
| Changer l'architecture (canaux, couches) | `modele.py` | `CNN.__init__` et `forward()`. Attention : `fc1` attend 3136 = 64*7*7, a recalculer si les couches de conv changent |
| Changer la force de l'attaque d'entrainement | `harden_torch.py` | options `--eps`, `--pgd-steps`, `--attack` |
| Adoucir ou durcir l'augmentation | `entrainement.py` | le dictionnaire `CONFIG_AUG` (rotation, zoom, bruit_p, cutout, epaisseur...) ou l'option `--aug-config` |
| **Ajouter** une transformation d'augmentation | `entrainement.py` | ecrire une fonction `_ma_transfo(x)` sur le modele des autres, puis l'appeler dans `augmenter()`. **Mesurer la densite d'encre avant** (regle du repo : `augment.py --n 12`) |
| Changer la recette de perte | `entrainement.py` | dans `entrainer()`, le bloc `if args.loss == "trades": ... else: ...` |
| **Ajouter une attaque** (CW, DeepFool...) | `attaques.py` | ecrire `def cw(modele, x, y, ...)` sur le modele de `pgd()`, puis la brancher dans `attaque()` |
| Changer le critere de selection du modele | `entrainement.py` | `if acc_rob > meilleur:` dans `entrainer()` |
| Changer les paliers du learning rate | `harden_torch.py` | option `--lr-drop` (fractions de la duree totale) |
| Changer l'optimiseur | `harden_torch.py` | option `--optimizer` (sgd, momentum, adam) ; le cablage est dans `main()` |
| Changer les images de validation | `harden_torch.py` | option `--val` (nombre d'images), `--val-steps` (pas de PGD) |

---

## 5. Trois exercices pour commencer

### Exercice 1 — Grossir le modele (30 min)

1. Dans `modele.py`, passer `Conv2d(1, 32, ...)` a `Conv2d(1, 64, ...)` et
   `Conv2d(32, 64, ...)` a `Conv2d(64, 128, ...)`.
2. Recalculer `fc1` : apres deux MaxPool2d(2) sur du 28x28, on obtient du 7x7,
   donc `128 * 7 * 7 = 6272`.
3. Lancer un `--quick` pour verifier que ca tourne.
4. Comparer le nombre de parametres (`[MODEL] ... parametres` au demarrage) et
   le temps par epoch avec l'ancien modele.

> Le test `--parite` ne s'applique plus : l'architecture a change, donc les
> poids NumPy ne correspondent plus. C'est normal.

### Exercice 2 — Ajouter une transformation d'augmentation (30 min)

Idee : un **flou gaussien leger** (noyau 3x3 fixe, applique avec une
probabilite), pour apprendre l'invariance au trait irregulier.

1. Dans `entrainement.py`, ecrire :

       def _flou(x):
           noyau = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]],
                                device=x.device)
           noyau = (noyau / noyau.sum()).view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
           return F.conv2d(x, noyau, padding=1, groups=x.shape[1])

2. L'appeler dans `augmenter()` sous une probabilite (comme les autres).
3. **Avant de l'activer pour de bon** : verifier son effet sur la densite
   d'encre, comme ca a ete fait pour les 5 autres transformations. Un flou a
   tendance a **etouffer** le trait : si la densite chute trop, le chiffre
   change de nature.

### Exercice 3 — Ecrire Carlini-Wagner (1 h)

CW est l'attaque white-box de reference : au lieu de maximiser la perte, elle
**minimise une distance** sous contrainte de mauvaise classification.

1. Dans `attaques.py`, ecrire `cw(modele, x, y, eps, steps=50, lr=0.01)` sur le
   modele de `pgd()` : une variable `w` optimisee par gradient (Adam), projetee
   dans la boucle eps.
2. L'utiliser sur un modele existant via `--report` en comparaison de PGD.
3. Question a laquelle repondre : **CW est-elle vraiment plus forte que PGD-20
   sur nos modeles ?** (sur des reseaux de cette taille, l'ecart est souvent
   faible — c'est une observation interessante a documenter).

---

## 6. Travailler dans VSCode

1. **Ouvrir le dossier** : `File > Open Folder` -> `C:\Users\Maxim\GitHub\CNN`.
2. **Choisir l'interpreteur** : `Ctrl+Shift+P` -> `Python: Select Interpreter`
   -> le chemin en `.venv\Scripts\python.exe`. Sans ca, l'editeur se plaindra de
   ne pas trouver `torch`.
3. **Terminal integre** : `Ctrl+`` puis `source .venv/Scripts/activate`.
   Rappel : taper `python`, pas `python3` (le venv n'expose pas `python3`).
4. **Iterer vite** : lancer en `--quick` pendant les modifications (30 s), et
   seulement en run complet quand la modification est validee.
5. **Ne pas lancer deux entrainements en parallele.**

Le run en cours sur ta machine continue pendant tout ce temps : il a sa propre
copie du code en memoire, tu peux lire et modifier sans le perturber.

---

## 7. Voir un resultat sans attendre des heures

| But | Commande | Duree |
|---|---|---|
| Verifier que la chaine tourne | `--quick` | ~30 s |
| Verifier l'equivalence NumPy | `--parite` | ~30 s |
| Evaluer un modele existant | `--report models/....pt --restarts 3` | 5-10 min |
| Run court realiste | `--n-train 10000 --epochs 5` | ~10 min sur GPU |

C'est ce cycle-la qu'il faut utiliser pour modifier le code : `--quick`, puis
`--n-train 10000`, et seulement ensuite le run complet.
