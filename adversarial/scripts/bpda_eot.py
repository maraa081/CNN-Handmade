#!/usr/bin/env python3
"""
bpda_eot.py - Attaques ADAPTATIVES : BPDA et EOT (Athalye et al. 2018)

Le probleme
-----------
Un modele peut sembler robuste pour la MAUVAISE raison : parce que le gradient
ne dit plus rien a l'attaquant. C'est le "gradient masking" / "obfuscated
gradients" (Athalye, Carlini & Wagner, 2018). Trois causes classiques :

  1. la defense n'est pas derivable (ex : quantification, arrondi, seuillage) ;
  2. la defense est stochastique (ex : transformation aleatoire de l'entree) ;
  3. le gradient s'effondre (ex : disparition ou explosion numerique).

Concretement : sous attaque a un pas (FGSM), le modele parait solide. Mais ce
n'est pas de la robustesse, c'est un angle mort du gradient. Un attaquant
informe le contourne. Cette distinction est CENTRALE : une defense qui n'est
pas cassee par les attaques automatiques n'est pas une defense prouvee.

Les deux contournements
-----------------------
BPDA - Backward Pass Differentiable Approximation
  Quand une transformation g (ici le feature squeezing, un arrondi) n'est pas
  derivable, on l'approxime par l'identite DANS LA PASSE ARRIERE :

      avant  : u = g(x)  puis  grad_x = J_g(x)^T . grad_u   (J_g = 0 presque partout)
      BPDA   : u = g(x)  puis  grad_x ~= grad_u             (on pose J_g = I)

  Le forward reste EXACT (on evalue bien le vrai modele defendu), seule la
  retropropagation est approximee. C'est le contournement de reference.

EOT - Expectation Over Transformation
  Quand la defense est stochastique, on moyenne le gradient sur plusieurs
  tirages de la transformation :

      grad ~= 1/N * somme_i grad_x f( T_i(x) )

  Une defense aleatoire ne protege pas : elle rend juste l'attaque plus chere.

Ce que fait ce script
---------------------
Il attaque le mode defense du depot : poids durcis (adversarial training PGD)
+ feature squeezing a l'inference. Quatre attaquants sont compares :

  A. pgd_sans_defense : PGD sur le modele brut, sans tenir compte du squeezing.
     Il evalue ensuite AVEC squeezing. C'est l'attaquant "de bonne foi" naive.
  B. pgd_naif         : PGD avec le vrai jacobien de la quantification (nul
     presque partout, comme torch.round sous PyTorch). Le gradient s'annule :
     c'est le piege du gradient masking, demontre.
  C. pgd_bpda         : forward exact, passe arriere approximee par l'identite.
  D. bpda_eot         : C, avec en plus une moyenne sur des transformations
     aleatoires (profondeur de bits et translation tirees au hasard).

Lecture du resultat : si B tient mais que C ou D s'effondrent, la defense
n'etait que du gradient masking.

Usage
-----
    python3 adversarial/scripts/bpda_eot.py                          # modele durci v1
    python3 adversarial/scripts/bpda_eot.py --bits 3 --eot 8
    python3 adversarial/scripts/bpda_eot.py --eps 0.3 --steps 20
    python3 adversarial/scripts/bpda_eot.py --quick                  # test rapide
    python3 adversarial/scripts/bpda_eot.py --weights models/model_weights_full.npz

Sorties :
    - tableau chiffre dans le terminal
    - courbes dans adversarial/results/bpda_eot.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath, exists

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))  # -> CNN-Handmade/
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))

from adversarial.scripts.fgsm import build_model, load_data, accuracy


EPS_LIST = [0.05, 0.1, 0.2, 0.3]


# --------------------------------------------------------------------------
#  La defense d'entree (celle qu'on cherche a casser)
# --------------------------------------------------------------------------

def feature_squeeze(x, bits=3):
    """Quantification de l'image sur 2^bits niveaux (Xu et al. 2018).

    Non derivable : la derivee de l'arrondi est nulle presque partout.
    """
    levels = 2 ** bits
    return np.round(x * (levels - 1)) / (levels - 1)


def decaler(x, dy, dx):
    """Translation entiere avec remplissage par des zeros.

    y[i, j] = x[i - dy, j - dx]  (hors bornes -> 0)

    La translation est ici notre transformation stochastique d'inference
    (l'equivalent d'un recadrage aleatoire). Son adjointe exacte est
    `decaler(g, -dy, -dx)` : elle sert a ramener le gradient dans le repere
    de l'image, ce qui est indispensable pour que EOT reste correct.
    """
    H, W = x.shape[-2], x.shape[-1]
    out = np.zeros_like(x)
    yd = slice(max(0, dy), H + min(0, dy))
    ys = slice(max(0, -dy), H + min(0, -dy))
    xd = slice(max(0, dx), W + min(0, dx))
    xs = slice(max(0, -dx), W + min(0, -dx))
    out[..., yd, xd] = x[..., ys, xs]
    return out


# --------------------------------------------------------------------------
#  Gradient de la loss par rapport a l'ENTREE (une passe avant + une arriere)
# --------------------------------------------------------------------------

def grad_entree(model, u, y):
    """d(Loss) / d(entree), calcule par le backward fait main du CNN."""
    logits = model.forward(u)
    C = logits.shape[1]
    model.loss_fn.forward(logits, np.eye(C)[y])
    grad = model.loss_fn.backward()
    return model.backward(grad)


def _pas_pgd(x_adv, grad, x, eps, alpha):
    """Un pas de PGD non cible, projete dans la boule L_inf et dans [0, 1]."""
    x_adv = x_adv + alpha * np.sign(grad)
    return np.clip(x_adv, np.clip(x - eps, 0.0, 1.0), np.clip(x + eps, 0.0, 1.0))


# --------------------------------------------------------------------------
#  A. PGD naif "sans defense" : on ignore le squeezing dans la boucle
# --------------------------------------------------------------------------

def pgd_sans_defense(model, x, y, eps, steps=20, rng=None):
    """PGD classique sur le modele brut (le squeezing n'est applique qu'a l'eval)."""
    if rng is None:
        rng = np.random.RandomState()
    alpha = eps / 4.0
    x_adv = x + rng.uniform(-eps, eps, size=x.shape)
    for _ in range(steps):
        grad = grad_entree(model, x_adv, y)
        x_adv = _pas_pgd(x_adv, grad, x, eps, alpha)
    return x_adv


# --------------------------------------------------------------------------
#  B. PGD "naif" avec le VRAI jacobien de la defense : le gradient s'annule
# --------------------------------------------------------------------------

def pgd_naif(model, x, y, eps, steps=20, rng=None, bits=3):
    """PGD propage a travers la quantification avec son vrai jacobien.

    L'arrondi est une fonction en escalier : sa derivee est nulle presque
    partout (c'est exactement ce que fait `torch.round` en PyTorch, dont le
    backward renvoie 0). Le gradient qui remonte a l'entree est donc NUL et
    l'attaque ne bouge pas d'un pixel : c'est le gradient masking.

    Ce n'est pas un bug de l'attaquant, c'est le piege que la defense tend.
    """
    if rng is None:
        rng = np.random.RandomState()
    alpha = eps / 4.0
    x_adv = x + rng.uniform(-eps, eps, size=x.shape)
    for _ in range(steps):
        u = feature_squeeze(x_adv, bits)          # avant : exact
        grad_u = grad_entree(model, u, y)
        jacobien = np.zeros_like(x_adv)           # d(arrondi)/dx = 0 presque partout
        grad = grad_u * jacobien
        x_adv = _pas_pgd(x_adv, grad, x, eps, alpha)
    return x_adv


# --------------------------------------------------------------------------
#  C. BPDA : forward exact, passe arriere approximee (J_g ~= I)
# --------------------------------------------------------------------------

def pgd_bpda(model, x, y, eps, steps=20, rng=None, bits=3):
    """BPDA : on evalue le vrai modele defendu, on retropropage l'identite.

    Difference avec B : on remplace J_g par l'identite au lieu de sa vraie
    valeur. Le gradient redevient utilisable, l'attaque progresse.
    """
    if rng is None:
        rng = np.random.RandomState()
    alpha = eps / 4.0
    x_adv = x + rng.uniform(-eps, eps, size=x.shape)
    for _ in range(steps):
        u = feature_squeeze(x_adv, bits)          # avant : exact
        grad = grad_entree(model, u, y)           # arriere : J_g ~= I
        x_adv = _pas_pgd(x_adv, grad, x, eps, alpha)
    return x_adv


# --------------------------------------------------------------------------
#  D. BPDA + EOT : moyenne du gradient sur la defense stochastique
# --------------------------------------------------------------------------

def pgd_bpda_eot(model, x, y, eps, steps=20, rng=None, bits=3,
                 eot=8, bits_var=(2, 3, 4), shift=2):
    """BPDA + EOT sur un modele dont l'inference est aleatoire.

    A chaque pas, on tire `eot` transformations T_i (profondeur de bits
    aleatoire + translation aleatoire), on calcule le gradient pour chacune,
    on ramene le gradient dans le repere de l'image, puis on moyenne.

    Cela revient a minimiser E_T[Loss(f(T(x)))] au lieu de Loss(f(T_0(x))).
    """
    if rng is None:
        rng = np.random.RandomState()
    alpha = eps / 4.0
    x_adv = x + rng.uniform(-eps, eps, size=x.shape)

    for _ in range(steps):
        grad = np.zeros_like(x_adv)
        for _ in range(eot):
            b = int(rng.choice(bits_var)) if bits_var else bits
            dy, dx = int(rng.randint(-shift, shift + 1)), int(rng.randint(-shift, shift + 1))
            u = feature_squeeze(decaler(x_adv, dy, dx), b)   # forward exact
            g = grad_entree(model, u, y)                     # BPDA
            grad += decaler(g, -dy, -dx)                     # adjointe de la translation
        grad /= float(eot)
        x_adv = _pas_pgd(x_adv, grad, x, eps, alpha)
    return x_adv


# --------------------------------------------------------------------------
#  Ecriture du modele : le squeezing est bien applique a l'inference
# --------------------------------------------------------------------------

def acc_sous_defense(model, x, y, bits=3):
    """Accuracy du modele tel qu'il serait deploye (avec le squeezing)."""
    return accuracy(model, feature_squeeze(x, bits), y)


def evaluer(model, x, y, eps_list, args, rng):
    """Accuracy du modele deploye, sous chacun des 4 attaquants."""
    res = {"sans_defense": [], "naif": [], "bpda": [], "bpda_eot": []}
    for eps in eps_list:
        xa = pgd_sans_defense(model, x, y, eps, args.steps, np.random.RandomState(rng.randint(1 << 30)))
        res["sans_defense"].append(acc_sous_defense(model, xa, y, args.bits))

        xa = pgd_naif(model, x, y, eps, args.steps, np.random.RandomState(rng.randint(1 << 30)), args.bits)
        res["naif"].append(acc_sous_defense(model, xa, y, args.bits))

        xa = pgd_bpda(model, x, y, eps, args.steps, np.random.RandomState(rng.randint(1 << 30)), args.bits)
        res["bpda"].append(acc_sous_defense(model, xa, y, args.bits))

        xa = pgd_bpda_eot(model, x, y, eps, args.steps, np.random.RandomState(rng.randint(1 << 30)),
                          args.bits, args.eot, tuple(args.bits_var), args.shift)
        res["bpda_eot"].append(acc_sous_defense(model, xa, y, args.bits))
    return res


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Attaques adaptatives BPDA + EOT (Athalye et al. 2018)")
    p.add_argument("--weights", default="models/defend_pgd_mnist_weights.npz",
                   help="poids du modele durci (.npz)")
    p.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    p.add_argument("--n", type=int, default=200, help="images de test")
    p.add_argument("--eps", nargs="+", type=float, default=EPS_LIST)
    p.add_argument("--steps", type=int, default=20, help="pas de PGD (tous les attaquants)")
    p.add_argument("--bits", type=int, default=3, help="bits du feature squeezing")
    p.add_argument("--eot", type=int, default=8, help="tirages par pas pour EOT")
    p.add_argument("--bits-var", nargs="+", type=int, default=[2, 3, 4],
                   help="profondeurs de bits tirees au hasard (EOT)")
    p.add_argument("--shift", type=int, default=2, help="translation aleatoire max (px, EOT)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="version courte (test de la chaine)")
    args = p.parse_args()

    if args.quick:
        args.n = 50
        args.steps = 5
        args.eot = 2
        args.eps = [0.3]

    chemin = args.weights if os.path.isabs(args.weights) else join(ROOT_DIR, args.weights)
    if not exists(chemin):
        print(f"[ERREUR] poids introuvables : {chemin}")
        print("         (le modele durci v1 est dans models/defend_pgd_mnist_weights.npz)")
        return 1

    print("=" * 70)
    print("  ATTAQUES ADAPTATIVES - BPDA + EOT (Athalye et al. 2018)")
    print("=" * 70)

    x, y = load_data(args.dataset, args.n)
    model = build_model(args.dataset)
    model.load_weights(chemin)
    print(f"[LOAD] {chemin}")
    print(f"[DATA] {len(x)} images, {args.dataset}")

    clean_sans = accuracy(model, x, y)
    clean_avec = acc_sous_defense(model, x, y, args.bits)
    print(f"[EVAL] propre, sans squeezing : {clean_sans:.1%}")
    print(f"[EVAL] propre, avec squeezing : {clean_avec:.1%}")
    print(f"[CONF] PGD-{args.steps} | squeezing {args.bits} bits | EOT {args.eot} "
          f"(bits {args.bits_var}, translation +/-{args.shift} px)")

    rng = np.random.RandomState(args.seed)
    res = evaluer(model, x, y, args.eps, args, rng)

    # -- Tableau : le modele DEPLOYE (avec squeezing) sous chaque attaquant --
    print(f"\n[RESULTAT] accuracy du modele deploye (avec squeezing {args.bits} bits)")
    print(f"{'eps':>6} | {'sans def.':>10} | {'naif':>8} | {'BPDA':>8} | {'BPDA+EOT':>9}")
    print("-" * 55)
    for i, eps in enumerate(args.eps):
        print(f"{eps:>6} | {res['sans_defense'][i]:>10.1%} | {res['naif'][i]:>8.1%} | "
              f"{res['bpda'][i]:>8.1%} | {res['bpda_eot'][i]:>9.1%}")

    i_max = len(args.eps) - 1
    eps_max = args.eps[i_max]
    print(f"\n[VERDICT] a eps={eps_max} :")
    print(f"  attaquant naif (vrai jacobien)      : {res['naif'][i_max]:.1%}  <- le piege")
    print(f"  BPDA                                : {res['bpda'][i_max]:.1%}")
    print(f"  BPDA + EOT                          : {res['bpda_eot'][i_max]:.1%}")
    if res["naif"][i_max] - res["bpda_eot"][i_max] > 0.10:
        print("  -> La defense s'effondre sous attaque adaptative : son apparente")
        print("     robustesse etait du GRADIENT MASKING, pas de la robustesse.")
    elif res["bpda_eot"][i_max] > 0.5:
        print("  -> La defense resiste a BPDA+EOT : le resultat est credible.")
    else:
        print("  -> Resultat intermediaire : a confirmer avec plus d'images et de pas.")

    # -- Courbes --
    out_dir = join(ROOT_DIR, "adversarial", "results")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    style = {"sans_defense": ("o", "tab:gray", "PGD sans defense (ignore le squeezing)"),
             "naif": ("^", "tab:red", "PGD naif (vrai jacobien -> gradient nul)"),
             "bpda": ("s", "tab:orange", "BPDA (jacobien ~ identite)"),
             "bpda_eot": ("D", "tab:green", "BPDA + EOT (defense stochastique)")}
    for cle, (marqueur, couleur, label) in style.items():
        ax.plot(args.eps, res[cle], marker=marqueur, linewidth=2, color=couleur, label=label)
    ax.axhline(clean_avec, linestyle=":", color="black", alpha=0.5,
               label=f"propre avec squeezing ({clean_avec:.1%})")
    ax.set_xlabel("eps (amplitude L_inf)")
    ax.set_ylabel("Accuracy du modele deploye")
    ax.set_title("BPDA + EOT : casser une defense a gradient obfusque")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = join(out_dir, "bpda_eot.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] {out}")
    print("[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
