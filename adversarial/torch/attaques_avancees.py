"""Attaques avancees : de quoi juger une defense serieusement.

`attaques.py` contient FGSM et PGD, les attaques de base. Ici on ajoute ce
qu'il faut pour ne pas se raconter d'histoires :

  - APGD   : la version moderne de PGD (pas adaptatif, momentum, restarts),
             en deux variantes de perte (CE et DLR). C'est le coeur de
             l'AutoAttack de Croce & Hein (2020).
  - Square : attaque boite noire SANS GRADIENT (score-based), qui sert de
             controle independant : si une defense ne tient que contre les
             attaques a gradient, Square la casse.
  - NES    : boite noire score-based, estime le gradient par differences
             finies sur des directions aleatoires.
  - CW     : Carlini & Wagner (2017), l'attaque white-box de reference,
             en version L2 (metrique complementaire du L-infini).
  - Boundary : boite noire DECISION-based (Brendel & Bethge 2018, version
             simplifiee) : n'utilise QUE le label predit.

Piege classique, et c'est tout l'objet de ce fichier : une defense evaluee
uniquement contre l'attaque sur laquelle elle a ete entrainee (PGD) para^it
solide pour de mauvaises raisons. La bonne pratique est d'utiliser plusieurs
familles d'attaques et de garder le PIRE cas.

Le detail theorique est dans `adversarial/attacks.md`.
"""

import sys
from os.path import abspath, dirname, join

import torch
import torch.nn.functional as F

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))

from adversarial.torch.attaques import accuracy  # noqa: E402


# --------------------------------------------------------------------------
#  Objectifs de l'attaquant (ce qu'on cherche a MAXIMISER)
# --------------------------------------------------------------------------

def _marges(logits, y):
    """Renvoie (Z_y, max des logits des AUTRES classes)."""
    zy = logits.gather(1, y.view(-1, 1)).squeeze(1)
    autres = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), float("-inf"))
    return zy, autres.max(dim=1).values


def objectif(logits, y, loss="ce"):
    """Objectif de l'attaquant, a MAXIMISER.

    - "ce"  : la cross-entropy (le plus classique)
    - "dlr" : Difference of Logits Ratio (Croce & Hein 2020). Normalisee par
              l'ecart entre le 1er et le 3e logit : elle reste informative meme
              quand le modele est deja confiant, la ou la CE sature. C'est ce
              qui la rend plus forte sur les modeles robustes.
    """
    if loss == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    zy, zmax_autre = _marges(logits, y)
    tri = logits.sort(dim=1, descending=True).values
    denom = (tri[:, 0] - tri[:, 2]).clamp(min=1e-12)
    return -((zy - zmax_autre) / denom)


def _grad_obj(modele, x, y, loss):
    """Gradient de l'objectif par rapport a l'IMAGE."""
    x = x.detach().requires_grad_(True)
    obj = objectif(modele(x), y, loss)
    (g,) = torch.autograd.grad(obj.sum(), x)
    return g.detach()


def _projeter(z, x, eps):
    """Projection dans la boule L-infini autour de x, puis dans [0, 1]."""
    return z.clamp(x - eps, x + eps).clamp(0.0, 1.0)


def _obj(modele, z, y, loss="ce"):
    """Objectif sans gradient (pour les comparaisons internes des attaques)."""
    with torch.no_grad():
        return objectif(modele(z), y, loss)


# --------------------------------------------------------------------------
#  APGD : PGD avec pas adaptatif, momentum et restarts (Croce & Hein 2020)
# --------------------------------------------------------------------------

def apgd(modele, x, y, eps, loss="ce", steps=100, restarts=1, rho=0.75, seed=0,
         random_start=False):
    """APGD (Auto-PGD) : la version moderne de PGD.

    Trois ameliorations par rapport a PGD classique :

    1. PAS ADAPTATIF. Tous les 10 pas, on regarde si l'objectif a progresse.
       Si oui, on garde le point et on continue. Si non, on REVIENT au
       meilleur point connu et on DIVISE LE PAS par 2. PGD, lui, garde un pas
       fixe (eps/4) : il peut osciller sans converger.
    2. MOMENTUM (Nesterov). On ajoute une fraction du deplacement precedent,
       ce qui accelere la progression.
    3. RESTARTS. On relance l'attaque depuis plusieurs points de depart
       aleatoires, et on garde le meilleur resultat.

    `loss` : "ce" (APGD-CE) ou "dlr" (APGD-DLR). Ce sont les deux attaques
    white-box de l'AutoAttack.
    """
    n = x.shape[0]
    dev = x.device
    gen = torch.Generator(device="cpu").manual_seed(seed)

    meilleur_global = x.clone()
    obj_global = torch.full((n,), float("-inf"), device=dev)

    for r in range(restarts):
        if r == 0 and not random_start:
            x_r = x.clone()
        else:
            # generateur CPU + tenseur cree sur CPU : la sequence aleatoire est
            # identique sur CPU et sur GPU (reproductible), et surtout le
            # generateur ne peut pas etre CPU quand le tenseur est CUDA.
            b = torch.empty_like(x, device="cpu").uniform_(
                -eps, eps, generator=gen).to(dev)
            x_r = _projeter(x + b.to(dev), x, eps)

        x_prev = x_r.clone()
        x_best = x_r.clone()
        f_best = _obj(modele, x_r, y, loss)
        eta = torch.full((n,), 2.0 * eps, device=dev)

        for i in range(steps):
            # -- Point de controle : on evalue, et on divise le pas si besoin --
            if (i % 10 == 0) or (i == steps - 1):
                f_cur = _obj(modele, x_r, y, loss)
                ameliore = f_cur >= f_best
                m = ameliore.view(-1, 1, 1, 1)
                x_best = torch.where(m, x_r, x_best)
                f_best = torch.where(ameliore, f_cur, f_best)
                retour = (~ameliore).view(-1, 1, 1, 1)
                x_r = torch.where(retour, x_best, x_r)
                x_prev = torch.where(retour, x_best, x_prev)
                eta = torch.where(~ameliore, (eta / 2).clamp(min=1e-8), eta)
                if bool((eta < 1e-8).all()):
                    break

            # -- Pas de gradient signe + momentum Nesterov --
            g = _grad_obj(modele, x_r, y, loss)
            z = _projeter(x_r + eta.view(-1, 1, 1, 1) * g.sign(), x, eps)

            ameliore = _obj(modele, z, y, loss) > _obj(modele, x_r, y, loss)
            m = ameliore.view(-1, 1, 1, 1)
            x_prev_old = x_prev
            x_prev = x_r
            z_nesterov = _projeter(z + rho * (z - x_prev_old), x, eps)
            x_r = torch.where(m, z_nesterov, z)

        # -- Fin du restart : on compare au meilleur global --
        f_fin = _obj(modele, x_best, y, loss)
        mieux = f_fin > obj_global
        meilleur_global = torch.where(mieux.view(-1, 1, 1, 1), x_best, meilleur_global)
        obj_global = torch.where(mieux, f_fin, obj_global)

    return meilleur_global.detach()


# --------------------------------------------------------------------------
#  Square Attack : boite noire, SANS gradient (Andriushchenko et al. 2020)
# --------------------------------------------------------------------------

def square(modele, x, y, eps, steps=500, restarts=1, p_init=0.8, seed=0):
    """Square Attack : une attaque boite noire qui ne regarde jamais le gradient.

    Principe : on modifie un PETIT CARRE de l'image, aux positions aleatoires,
    en poussant les pixels a +-eps. On garde la modification seulement si
    l'objectif progresse. La taille du carre decroit au fil des iterations :
    gros carres au debut (exploration), petits a la fin (finition).

    Pourquoi c'est important en plus des attaques a gradient : si une defense
    ne resiste qu'aux attaques a gradient, Square la casse -- c'est un
    controleur independant. C'est le composant boite noire de l'AutoAttack.
    """
    n, c, h, w = x.shape
    dev = x.device
    gen = torch.Generator(device="cpu").manual_seed(seed)

    meilleur = x.clone()
    obj_global = torch.full((n,), float("-inf"), device=dev)

    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    x_clip = x.clamp(0.0, 1.0)

    for r in range(restarts):
        if r == 0:
            delta = torch.zeros_like(x)
        else:
            b = torch.empty_like(x, device="cpu").uniform_(
                -eps, eps, generator=gen).to(dev)
            delta = (x_clip + b).clamp(0.0, 1.0) - x_clip
        obj = _obj(modele, x_clip + delta, y, "ce")

        for i in range(steps):
            p = p_init * (1.0 - i / steps)
            taille = max(1, int((p * h * w) ** 0.5))
            dy = torch.randint(0, h - taille + 1, (n,), generator=gen).to(dev)
            dx = torch.randint(0, w - taille + 1, (n,), generator=gen).to(dev)
            signe = torch.where(torch.rand(n, generator=gen).to(dev) < 0.5, -1.0, 1.0)

            masque = ((yy >= dy.view(-1, 1, 1)) & (yy < (dy + taille).view(-1, 1, 1))
                      & (xx >= dx.view(-1, 1, 1)) & (xx < (dx + taille).view(-1, 1, 1)))
            masque = masque.unsqueeze(1)                       # (N, 1, H, W)

            extremite = (x_clip + 2.0 * eps * signe.view(-1, 1, 1, 1)).clamp(0.0, 1.0)
            cand = torch.where(masque, extremite, x_clip + delta)
            cand = _projeter(cand, x_clip, eps)
            obj_c = _obj(modele, cand, y, "ce")

            ameliore = obj_c >= obj
            delta = torch.where(ameliore.view(-1, 1, 1, 1), cand - x_clip, delta)
            obj = torch.where(ameliore, obj_c, obj)

        mieux = obj > obj_global
        meilleur = torch.where(mieux.view(-1, 1, 1, 1), x_clip + delta, meilleur)
        obj_global = torch.where(mieux, obj, obj_global)

    return meilleur.detach().clamp(0.0, 1.0)


# --------------------------------------------------------------------------
#  NES : boite noire score-based, gradient estime par differences finies
# --------------------------------------------------------------------------

def nes(modele, x, y, eps, steps=40, samples=20, sigma=0.005,
        lr=None, seed=0):
    """NES (Natural Evolution Strategies) : boite noire, score-based.

    L'attaquant ne voit ni les gradients ni les poids : uniquement la SORTIE
    du modele (les scores). Il estime le gradient par differences finies sur
    des directions aleatoires (estimateur antithetique) :

        grad ~= E_u [ (F(x + sigma*u) - F(x - sigma*u)) / (2*sigma) * u ]

    puis fait des pas de type PGD avec ce gradient estime.

    Cout : `2 * samples` appels au modele par pas, contre 1 pour une attaque
    a gradient. C'est le prix de la boite noire.
    """
    n = x.shape[0]
    dev = x.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    alpha = lr if lr is not None else eps / 10.0

    delta = torch.zeros_like(x)
    obj = _obj(modele, x + delta, y, "ce")
    forme = x.shape[1:]

    for _ in range(steps):
        u = torch.randn(samples, n, *forme, generator=gen).to(dev)
        base = (x + delta).unsqueeze(0)
        with torch.no_grad():
            op = objectif(modele((base + sigma * u).reshape(samples * n, *forme)),
                          y.repeat(samples), "ce").view(samples, n)
            om = objectif(modele((base - sigma * u).reshape(samples * n, *forme)),
                          y.repeat(samples), "ce").view(samples, n)
        g = (((op - om) / (2.0 * sigma)).view(samples, n, 1, 1, 1) * u).mean(dim=0)

        cand = _projeter(x + delta + alpha * g.sign(), x, eps)
        obj_c = _obj(modele, cand, y, "ce")
        ameliore = obj_c >= obj
        delta = torch.where(ameliore.view(-1, 1, 1, 1), cand - x, delta)
        obj = torch.where(ameliore, obj_c, obj)

    return (x + delta).detach().clamp(0.0, 1.0)


# --------------------------------------------------------------------------
#  CW : Carlini & Wagner (2017), version L2
# --------------------------------------------------------------------------

def cw_l2(modele, x, y, steps=100, c=1.0, kappa=0.0, lr=0.01):
    """Attaque Carlini & Wagner L2 : la reference white-box.

    A la difference de PGD (qui maximise la perte dans une boule de rayon fixe),
    CW cherche la perturbation la PLUS PETITE possible qui fait changer la
    prediction :

        minimise  ||delta||_2^2  +  c * f(x + delta)
        avec      f = max(Z_y - max_{i!=y} Z_i + kappa, 0)   (marge)

    Elle utilise un changement de variable x_adv = 0.5*(tanh(w) + 1) : ainsi
    les pixels restent TOUJOURS dans [0, 1], sans projection -- on peut donc
    utiliser un optimiseur adaptatif (Adam), ce qui converge mieux.

    Retourne (x_adv, distance L2 par image). Aucune garantie que l'image soit
    effectivement mal classee : c'est au lecteur de compter (voir `eval_suite`).
    """
    w0 = torch.atanh((2.0 * x - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    w = w0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)

    meilleur = x.clone()
    meilleure_dist = torch.full((x.shape[0],), float("inf"), device=x.device)

    for _ in range(steps):
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = modele(x_adv)
        zy, zmax_autre = _marges(logits, y)
        f = torch.clamp(zy - zmax_autre + kappa, min=0.0)
        dist = ((x_adv - x) ** 2).flatten(1).sum(1)

        opt.zero_grad(set_to_none=True)
        (dist + c * f).sum().backward()
        opt.step()

        with torch.no_grad():
            trompe = modele(x_adv).argmax(dim=1) != y
            garder = trompe & (dist < meilleure_dist)
            meilleur = torch.where(garder.view(-1, 1, 1, 1), x_adv, meilleur)
            meilleure_dist = torch.where(garder, dist, meilleure_dist)

    return meilleur.detach(), meilleure_dist.detach()


# --------------------------------------------------------------------------
#  Boundary Attack : boite noire DECISION-based (Brendel & Bethge 2018)
# --------------------------------------------------------------------------

def boundary(modele, x, y, steps=200, theta=0.01, init_bruite=True, seed=0):
    """Boundary Attack (version simplifiee) : seule la CLASSE PREDITE est vue.

    L'attaquant n'a ni les gradients, ni les scores : uniquement la decision du
    modele ("chat" ou "chien"). Il part d'un point mal classe, puis fait une
    marche aleatoire SUR LA FRONTIERE de decision en essayant de se rapprocher
    de l'image d'origine.

    Version simplifiee : la recherche du point de depart et la dichotomie sont
    reduites a un pas par iteration (l'original en fait une vraie recherche).
    Sur des petites boules (eps=0.30), les attaques decision-based sont
    mecaniquement plus faibles que les attaques a gradient -- c'est normal, et
    c'est justement l'interet du controle.

    Retourne (x_adv, distance L2 par image). Les images ou le point de depart
    n'etait pas adversarial sont laissees intactes (distance 0).
    """
    dev = x.device
    gen = torch.Generator(device="cpu").manual_seed(seed)

    if init_bruite:
        depart = torch.rand(x.shape, generator=gen).to(dev).clamp(0.0, 1.0)
    else:
        depart = torch.full_like(x, 0.5)
    with torch.no_grad():
        ok = modele(depart).argmax(dim=1) != y
    x_adv = torch.where(ok.view(-1, 1, 1, 1), depart, x)

    def dist_l2(a):
        return ((a - x) ** 2).flatten(1).sum(1).sqrt()

    for _ in range(steps):
        d = (x - x_adv).flatten(1)
        norme_d = d.norm(dim=1, keepdim=True)

        # 1) deplacement aleatoire, orthogonal a la direction vers x
        u = torch.randn(x.shape, generator=gen).to(dev).flatten(1)
        u = u - (u * d).sum(1, keepdim=True) / (norme_d ** 2 + 1e-12) * d
        u = u / (u.norm(dim=1, keepdim=True) + 1e-12)
        pas = (norme_d * theta).view(-1, 1, 1, 1)
        cand = (x_adv + pas * u.view_as(x)).clamp(0.0, 1.0)
        with torch.no_grad():
            reste = modele(cand).argmax(dim=1) != y
        x_adv = torch.where(reste.view(-1, 1, 1, 1), cand, x_adv)

        # 2) un pas de rapprochement vers x (la frontiere est re-testee)
        cand2 = (x_adv + (x - x_adv) * theta).clamp(0.0, 1.0)
        with torch.no_grad():
            reste2 = modele(cand2).argmax(dim=1) != y
        x_adv = torch.where(reste2.view(-1, 1, 1, 1), cand2, x_adv)

    return x_adv.detach(), dist_l2(x_adv).detach()


# --------------------------------------------------------------------------
#  Selecteur et utilitaires
# --------------------------------------------------------------------------

def attaque_avancee(modele, x, y, eps, type_attaque, **kw):
    """Selecteur, pour rester compatible avec l'interface de `attaques.attaque`."""
    if type_attaque == "apgd-ce":
        return apgd(modele, x, y, eps, loss="ce", **kw)
    if type_attaque == "apgd-dlr":
        return apgd(modele, x, y, eps, loss="dlr", **kw)
    if type_attaque == "square":
        return square(modele, x, y, eps, **kw)
    if type_attaque == "nes":
        return nes(modele, x, y, eps, **kw)
    raise ValueError(f"attaque inconnue : {type_attaque}")
