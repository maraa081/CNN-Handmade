"""Les memes attaques que la version NumPy, en PyTorch.

Formules strictement identiques a `adversarial/scripts/fgsm.py` et `pgd.py` :

    FGSM :  x_adv = clip(x + eps * sign(grad_x L), 0, 1)
    PGD  :  x_0   = clip(x + U(-eps, eps), 0, 1)
            x_t+1 = clip(x_t + alpha * sign(grad_x L), x - eps, x + eps)
            x_t+1 = clip(x_t+1, 0, 1)          avec alpha = eps / 4

Difference : ici le gradient vient de l'autograd de PyTorch, alors que la
version NumPy le calcule a la main dans `model.backward()`. Le resultat doit
etre le meme (c'est verifie par le mode `--parite`).
"""

import torch
import torch.nn.functional as F


def _grad_entree(modele, x, y):
    """Gradient de la cross-entropy par rapport a l'IMAGE (pas aux poids)."""
    x = x.detach().requires_grad_(True)
    perte = F.cross_entropy(modele(x), y, reduction="sum")
    (g,) = torch.autograd.grad(perte, x)
    return g


def fgsm(modele, x, y, eps, random_start=False):
    """FGSM non ciblee."""
    if random_start:
        x0 = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x0 = x
    g = _grad_entree(modele, x0, y)
    return (x0.detach() + eps * g.sign()).clamp(0.0, 1.0)


def pgd(modele, x, y, eps, steps=20, alpha=None, random_start=True):
    """PGD L-inf non ciblee (Madry et al. 2018)."""
    alpha = alpha if alpha is not None else eps / 4.0
    if random_start:
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x.clone()
    for _ in range(steps):
        g = _grad_entree(modele, x_adv, y)
        x_adv = (x_adv.detach() + alpha * g.sign())
        x_adv = x_adv.clamp(x - eps, x + eps).clamp(0.0, 1.0)
    return x_adv.detach()


def attaque(modele, x, y, eps, type_attaque="pgd", steps=20):
    if type_attaque == "fgsm":
        return fgsm(modele, x, y, eps)
    if type_attaque == "fgsm-rs":
        return fgsm(modele, x, y, eps, random_start=True)
    return pgd(modele, x, y, eps, steps=steps)


@torch.no_grad()
def accuracy(modele, x, y):
    return (modele(x).argmax(dim=1) == y).float().mean().item()
