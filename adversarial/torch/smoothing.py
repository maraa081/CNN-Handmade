"""Randomized smoothing : la seule defense avec une borne GARANTIE.

Toutes les defenses vues jusqu'ici sont EMPIRIQUES : on attaque, et on regarde
ce qui reste. Le jour ou quelqu'un trouve une attaque plus forte, le chiffre
tombe. La verification formelle (Cormack, Bunel, C., Wicker) est une autre
approche : prouver qu'AUCUNE perturbation de norme inferieure a R ne peut
changer la prediction.

`randomized smoothing` (Cohen, Rosenfeld & Kolter 2019) est la version la plus
simple et la plus passe-partout : elle s'applique a N'IMPORTE QUEL modele.

Le principe
-----------
1. On entraine un classifieur de base `f` sur des images BRUITEES :
   a chaque pas, on ajoute un bruit gaussien N(0, sigma^2 I) a l'entree.
2. On construit un classifieur LISSE `g` :

       g(x) = argmax_c  P( f(x + bruit) = c ),   bruit ~ N(0, sigma^2 I)

   En pratique on estime cette probabilite par Monte-Carlo : on tire `n`
   bruits, on compte les votes.
3. THEOREME (Cohen et al. 2019). Si le vote majoritaire depasse la moitie avec
   une marge suffisante, alors pour toute perturbation d de norme L2 <= R,
   g(x + d) = g(x), avec

       R = (sigma / 2) * ( Phi^-1(p_A) - Phi^-1(p_B) )

   ou p_A est une borne INFERIEURE de la probabilite de la classe majoritaire et
   p_B une borne SUPERIEURE de celle du second -- obtenues par un intervalle de
   confiance binomial exact (Clopper-Pearson).

Ce que ca coute
---------------
- La borne est en norme L2, pas L-infini : ce n'est pas directement comparable
  aux 91% sous PGD eps=0.30.
- Le rayon garanti est petit pour les petites valeurs de sigma, et l'accuracy
  propre chute quand sigma grandit. Le compromis est explicite.

Usage
-----
    # 1. Entrainer le classifieur de base (ie. entraine sur images bruitees)
    python3 adversarial/torch/smoothing.py --entrainer --sigma 0.5 --epochs 30

    # 2. Certifier et tracer la courbe precision certifiee / rayon
    python3 adversarial/torch/smoothing.py --certifier --sigma 0.5

    # Verification rapide de la chaine
    python3 adversarial/torch/smoothing.py --entrainer --sigma 0.5 --epochs 2 --quick
"""

import argparse
import math
import os
import sys
import time
from os.path import abspath, dirname, isabs, join
from statistics import NormalDist

import torch
import torch.nn.functional as F

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, join(ROOT_DIR, "adversarial", "torch"))

from adversarial.torch.modele import CNN, charger_npz      # noqa: E402
from adversarial.torch.entrainement import charger_train, charger_test  # noqa: E402

PHI = NormalDist()


# --------------------------------------------------------------------------
#  Intervalle de confiance binomial exact (Clopper-Pearson)
# --------------------------------------------------------------------------
#
# On evite volontairement toute dependance externe (pas de scipy) : la fonction
# beta incomplete est implementee ici par la fraction continue classique
# (Numerical Recipes, chapitre 6.4), puis inversee par dichotomie.

def _betacf(a, b, x):
    """Fraction continue de Lentz pour la fonction beta incomplete."""
    MAXIT, EPS, FPMIN = 200, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    """Fonction beta incomplete regularisee I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p, a, b):
    """Inverse de I_x(a, b) par dichotomie (x croissant en p)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k, n, alpha):
    """Intervalle de confiance exact [bas, haut] pour une proportion k/n."""
    bas = 0.0 if k == 0 else _beta_ppf(alpha / 2.0, k, n - k + 1)
    haut = 1.0 if k == n else _beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return bas, haut


# --------------------------------------------------------------------------
#  Classifieur lisse
# --------------------------------------------------------------------------

def _votes(modele, x, sigma, n, chunk_images=25, chunk_bruit=250, gen=None):
    """Compte les votes de `n` tirages de bruit pour chaque image.

    On travaille par paquets (`chunk_images` x `chunk_bruit`) : tirer
    n=1000 bruits pour 500 images d'un coup representerait 500 000 images en
    memoire d'un seul coup.
    """
    nb_classes = modele.fc2.out_features
    comptes = torch.zeros(x.shape[0], nb_classes, dtype=torch.long, device=x.device)
    with torch.no_grad():
        for deb in range(0, x.shape[0], chunk_images):
            lot = x[deb:deb + chunk_images]
            n_img = lot.shape[0]
            reste = n
            while reste > 0:
                k = min(chunk_bruit, reste)
                reste -= k
                bruit = torch.randn(k, *lot.shape, generator=gen, device=lot.device) * sigma
                # Pas de clamp : meme raison qu'a l'entrainement (voir
                # `entrainer_lisse`). Le modele est entraine sur des entrees
                # bruitees non bornees, la certification doit faire pareil.
                bruite = lot.unsqueeze(0) + bruit
                preds = modele(bruite.reshape(k * n_img, *lot.shape[1:])).argmax(dim=1)
                preds = preds.view(k, n_img)
                for c in range(nb_classes):
                    comptes[deb:deb + n_img, c] += (preds == c).sum(dim=0)
    return comptes


def classifier_lisse(modele, x, sigma, n0=100, n=1000, alpha=0.001,
                     chunk_images=25, chunk_bruit=250, seed=0):
    """Prediction certifiee du classifieur lisse.

    Etape 1 : `n0` tirages pour choisir la classe majoritaire c_A (peu couteux).
    Etape 2 : si la majorite est nette, `n` tirages supplementaires pour des
              bornes de confiance serrees.

    Retourne (prediction, rayon certifie, abstention). Un rayon nul signifie
    "je ne certifie rien" -- c'est le cas quand le vote est trop partage, ou
    quand l'image est mal classee par le classifieur lisse lui-meme.
    """
    modele.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)

    c0 = _votes(modele, x, sigma, n0, chunk_images, chunk_bruit, gen)
    pred = c0.argmax(dim=1)
    p_hat = c0.gather(1, pred.view(-1, 1)).squeeze(1).float() / float(n0)

    # On ne certifie que si la classe majoritaire est nette des le premier tour
    a_second_tour = p_hat > 0.5
    if not bool(a_second_tour.any()):
        return pred, torch.zeros_like(p_hat), torch.ones_like(p_hat, dtype=torch.bool)

    c1 = _votes(modele, x[a_second_tour], sigma, n, chunk_images, chunk_bruit, gen)
    nb = c1.gather(1, pred[a_second_tour].view(-1, 1)).squeeze(1).float()
    c1_sans_a = c1.clone()
    c1_sans_a.scatter_(1, pred[a_second_tour].view(-1, 1), -1)
    nb_second = c1_sans_a.max(dim=1).values.float()

    rayon = torch.zeros_like(nb)
    abstention = torch.ones_like(nb, dtype=torch.bool)
    for i in range(nb.shape[0]):
        k_a, k_b = int(nb[i].item()), int(nb_second[i].item())
        p_a_bas, _ = clopper_pearson(k_a, n, alpha)
        _, p_b_haut = clopper_pearson(k_b, n, alpha)
        if p_a_bas > 0.5 and p_a_bas > p_b_haut:
            rayon[i] = sigma * (PHI.inv_cdf(p_a_bas) - PHI.inv_cdf(p_b_haut)) / 2.0
            abstention[i] = False

    rayon_total = torch.zeros(x.shape[0])
    abstention_totale = torch.ones(x.shape[0], dtype=torch.bool)
    rayon_total[a_second_tour] = rayon
    abstention_totale[a_second_tour] = abstention
    return pred, rayon_total, abstention_totale


# --------------------------------------------------------------------------
#  Entrainement du classifieur de base (images bruitees)
# --------------------------------------------------------------------------

def entrainer_lisse(modele, x_tr, y_tr, sigma, epochs=90, batch=128,
                    lr=None, optimizer="adam", clip=1.0, seed=42, verbose=True):
    """Entraine le classifieur de base SUR DES IMAGES BRUITEES.

    C'est le point cle : sans augmentation par bruit, le classifieur lisse n'a
    aucune raison d'etre robuste (il n'a jamais vu de bruit).

    [PIEGE MESURE LE 2026-09-11] L'entrainement est BEAUCOUP plus sensible au
    learning rate que l'entrainement propre. Constat sur 3000 images / 3 epochs,
    sigma=0.5 (accuracy propre du classifieur de base apres le run) :

        Adam lr=0.010  ->  13.3%   COLLAPSE
        Adam lr=0.001  ->  66.0%
        SGD  lr=0.100  ->   8.7%   COLLAPSE
        SGD  lr=0.050  ->  81.7%

    En cas de collapse, la loss se bloque a ln(10) = 2.3026 des l'epoch 2 : le
    modele predit l'uniforme et n'apprend plus rien. La raison est que l'entree
    bruitee a une magnitude bien plus grande que l'entree propre (ou la plupart
    des pixels valent 0) : le meme learning rate est donc effectivement plus
    grand. On utilise Adam (adaptatif, donc moins sensible a l'echelle) avec un
    lr prudent, plus un ecrêtage des gradients.

    `clip` : ecrêtage de la norme L2 globale des gradients (0 pour desactiver).
    """
    if lr is None:
        lr = 0.001 if optimizer == "adam" else 0.05

    gen = torch.Generator(device="cpu").manual_seed(seed)
    if optimizer == "adam":
        opt = torch.optim.Adam(modele.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(modele.parameters(), lr=lr, momentum=0.9)

    paliers = sorted({max(1, int(round(p * epochs))) for p in (0.5, 0.8)}) if epochs >= 4 else []
    lr_courant = lr
    n = len(x_tr)
    for epoch in range(1, epochs + 1):
        if epoch in paliers:
            lr_courant *= 0.1
            for g in opt.param_groups:
                g["lr"] = lr_courant
            if verbose:
                print(f"  [LR] epoch {epoch} : lr -> {lr_courant:.4f}")
        t0 = time.time()
        ordre = torch.randperm(n, generator=gen)
        modele.train()
        perte_tot = 0.0
        for deb in range(0, n, batch):
            bi = ordre[deb:deb + batch]
            bx, by = x_tr[bi], y_tr[bi]
            # [IMPORTANT] On n'ecrete PAS l'image bruitee : la grande majorite
            # des pixels MNIST valent 0 (le fond), et un clamp(0, 1) rendrait le
            # bruit unilateral au lieu de symetrique. L'implementation de
            # reference n'ecrete pas non plus.
            bruit = torch.randn_like(bx) * sigma
            perte = F.cross_entropy(modele(bx + bruit), by)
            opt.zero_grad(set_to_none=True)
            perte.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(modele.parameters(), clip)
            opt.step()
            perte_tot += perte.item() * len(bx)
        if verbose:
            print(f"  Epoch {epoch:>3}/{epochs} | loss {perte_tot / n:6.4f} | "
                  f"{time.time() - t0:5.1f} s")
    return modele


# --------------------------------------------------------------------------
#  Courbe de precision certifiee
# --------------------------------------------------------------------------

def evaluer_certifie(modele, x, y, sigma, rayon_max=None, pas=0.1,
                     n0=100, n=1000, alpha=0.001, chunk_images=25,
                     chunk_bruit=250, seed=0):
    """Precision certifiee a differents rayons L2.

    Pour un rayon r, la "precision certifiee" est la part des images pour
    lesquelles le modele predit juste ET dont le rayon certifie est >= r.
    Autrement dit : la part des images sur lesquelles on a une GARANTIE.
    """
    pred, rayon, abstention = classifier_lisse(
        modele, x, sigma, n0, n, alpha, chunk_images, chunk_bruit, seed)
    juste = (pred == y) & ~abstention
    if rayon_max is None:
        rayon_max = max(0.5, float(rayon.max().item()))
    rayons = [i * pas for i in range(int(rayon_max / pas) + 1)]
    res = {}
    for r in rayons:
        res[r] = float((juste & (rayon >= r)).float().mean().item())
    return res, pred, rayon, abstention


# --------------------------------------------------------------------------
#  Ligne de commande
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Randomized smoothing (Cohen et al. 2019)")
    p.add_argument("--entrainer", action="store_true")
    p.add_argument("--certifier", action="store_true")
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--clip", type=float, default=1.0,
                   help="ecrêtage de la norme des gradients (0 = desactive)")
    p.add_argument("--n-test", type=int, default=200, help="images certifiees")
    p.add_argument("--n0", type=int, default=100)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--out", default=None, help="modele de base (defaut selon sigma)")
    p.add_argument("--weights", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs, args.n_train, args.n_test = 2, 5000, 40
        args.n0, args.n = 50, 300

    defaut = f"models/smooth_mnist_s{int(round(args.sigma * 100)):03d}.pt"
    chemin = args.out or args.weights or defaut
    chemin = chemin if isabs(chemin) else join(ROOT_DIR, chemin)

    (x_tr, y_tr), _ = charger_train(args.n_train, 1000)
    x_te, y_te = charger_test("mnist", args.n_test)

    if args.entrainer:
        print("=" * 70)
        print(f"  RANDOMIZED SMOOTHING - entrainement du classifieur de base")
        print(f"  sigma = {args.sigma} | {args.n_train} images | {args.epochs} epochs "
              f"| {args.optimizer} lr={args.lr or 'defaut'} clip={args.clip}")
        print("=" * 70)
        modele = CNN()
        modele = entrainer_lisse(modele, x_tr, y_tr, args.sigma, args.epochs,
                                 args.batch, args.lr, args.optimizer, args.clip,
                                 seed=args.seed)
        torch.save(modele.state_dict(), chemin)
        print(f"[SAVE] {chemin}")
    elif not os.path.exists(chemin):
        print(f"[ERREUR] modele introuvable : {chemin}")
        return 1

    if args.certifier or args.entrainer:
        modele = CNN()
        ck = torch.load(chemin, map_location="cpu", weights_only=False)
        modele.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        modele.eval()

        print("\n" + "=" * 70)
        print(f"  CERTIFICATION - sigma={args.sigma}, n0={args.n0}, n={args.n}, "
              f"alpha={args.alpha}")
        print("=" * 70)
        t0 = time.time()
        res, pred, rayon, abstention = evaluer_certifie(
            modele, x_te, y_te, args.sigma, n0=args.n0, n=args.n,
            alpha=args.alpha, seed=args.seed)
        acc_lisse = float((pred == y_te).float().mean().item())
        print(f"[EVAL] precision du classifieur lisse : {acc_lisse:.1%}")
        print(f"[EVAL] taux d'abstention               : {float(abstention.float().mean()):.1%}")
        print(f"[EVAL] rayon certifie median           : {float(rayon.median()):.3f}")
        print(f"\n  {'rayon L2':>9} | {'precision certifiee':>20}")
        print("  " + "-" * 34)
        for r, v in res.items():
            print(f"  {r:>9.2f} | {v:>20.1%}")
        print(f"\n[TEMPS] {time.time() - t0:.1f} s")
        print("\n  Rappel : c'est une GARANTIE, pas une observation. Toute")
        print("  perturbation de norme L2 <= rayon certifie laisse la prediction")
        print("  inchangee -- par construction, pas par experience.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
