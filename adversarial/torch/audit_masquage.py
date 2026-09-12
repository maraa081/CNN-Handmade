#!/usr/bin/env python3
"""audit_masquage.py - le modele est-il robuste, ou est-ce que nos attaques
ne savent plus le lire ?

Contexte (2026-09-12) : le run `bande_cible50` (attaque interne a budget
adaptatif, cible de tromperie 0.5) sort des chiffres suspects :

    propre 98.6% | FGSM 0.3 : 98.2% (-0.4 pt seulement !) | PGD-20 : 96.2%
    APGD-DLR : 95.4% | Square-3000 : 93.6% | NES : 97.4% | CW-L2 2.6%
    Boundary : 90.4%

Or FGSM a eps=0.30 ne coute que 0.4 point a ce modele, alors qu'il en coutait
5.6 a `abl_a` (94.0% contre 99.6%). Une perturbation pleine grandeur qui ne
change presque rien, c'est le symptome classique d'une sortie SATUREE : le
gradient et les scores ne renseignent plus l'attaquant, et TOUTES les attaques
de notre suite (qui lisent le gradient ou les scores) perdent leur signal.
C'est la famille "obfuscated gradients" d'Athalye et al. 2018 -- et une
robustesse qui n'existe que dans les yeux de nos attaques.

Cet audit passe cinq tests, tous en EVALUATION seule (aucun entrainement) :

  [1] Saturation des sorties : max-probabilite, CE propre, ecart de logits.
      Une softmax saturee (max-prob ~ 1.0, ecart de logits enorme) empeche
      l'attaquant de lire quoi que ce soit.
  [2] Difference finie contre gradient analytique : la direction du gradient
      fait-elle mieux qu'une direction ALEATOIRE de meme norme pour faire
      monter la perte ? Si non, le gradient ne pointe plus la vulnerabilite.
  [3] Balayage de eps jusqu'a 1.0 (borne non contrainte) + controle par bruit
      aleatoire de moyenne nulle : la forme de la chute est-elle credible ?
  [4] Attaque a PAS FIN : si un PGD a alpha=eps/100 casse le modele alors que
      le PGD a alpha=eps/4 ne le casse pas, le probleme est le pas grossier,
      pas la robustesse (notre loi du budget de deplacement dit exactement que
      le pas grossier saute au coin de la boule et n'explore rien).
  [5] TRANSFERT : des exemples fabriques sur un AUTRE modele (reference) sont-ils
      plus efficaces que l'attaque white-box ? Si oui, c'est le test decisif du
      masquage : une defense qui masque son gradient se fait battre par un
      transfert, parce que le transfert n'utilise pas son gradient.

Usage
-----
    python -u adversarial/torch/audit_masquage.py \
        --weights models/bande_cible50.pt --device cuda

    # avec comparaison a un modele de reference (transfert + saturation)
    python -u adversarial/torch/audit_masquage.py \
        --weights models/bande_cible50.pt --reference models/abl_a_eps10.pt --device cuda

Sortie : un verdict par test, puis une synthese. Aucune conclusion n'est
"le modele est robuste" sans que les cinq tests soient propres.
"""

import argparse
import sys
from os.path import abspath, dirname, join

import torch
import torch.nn.functional as F

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))
sys.path.insert(0, join(ROOT_DIR, "adversarial", "torch"))

from adversarial.torch.attaques import fgsm, pgd, accuracy          # noqa: E402
from adversarial.torch.attaques_avancees import square              # noqa: E402
from adversarial.torch.entrainement import charger_test             # noqa: E402
from adversarial.torch.eval_suite import charger_modele             # noqa: E402

SUSPECTS = []
DRAPEAUX = []        # profil a surveiller (n'est PAS une preuve de masquage)
PREUVES = []         # element qui, lui, conclut au masquage


def drapeau(test, message, detail):
    """Signe de profil : ne conclut a RIEN tout seul (voir la synthese).

    Lecon du 12/09 : un modele MNIST durci est naturellement tres confiant et sa
    surface peut etre rugueuse (le gradient renseigne mal) SANS masquer quoi que
    ce d'autre. Ce qui fait un masquage, c'est le profil + le fait que TOUTES les
    attaques echouent, y compris a eps=1.0. On separe donc les deux.
    """
    DRAPEAUX.append(f"{test} : {detail}")
    print(f"    [NOTE] {message}")


def alerte(test, message):
    SUSPECTS.append(f"{test} : {message}")
    print(f"    [SUSPECT] {message}")


def ok(test, message):
    print(f"    [OK] {message}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--reference", default="models/abl_a_eps10.pt",
                   help="modele de comparaison (saturation + transfert)")
    p.add_argument("--dataset", choices=["mnist", "kmnist", "emnist"], default="mnist")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "dml"])
    p.add_argument("--rayon-square", type=int, default=500,
                   help="pas de Square pour le rayon robuste de [3b] (0 = ignorer). "
                        "Indispensable sur les modeles a surface rugueuse : le rayon"
                        " mesure au PGD y est faux (voir 2026-09-12 23h00).")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    modele = charger_modele(args.weights, device, args.dataset)
    x, y = charger_test(args.dataset, args.n)
    x, y = x.to(device), y.to(device)
    print("=" * 72)
    print("  AUDIT : robustesse reelle ou masquage de gradient ?")
    print("=" * 72)
    print(f"  modele     : {args.weights}")
    print(f"  reference  : {args.reference}")
    print(f"  {len(x)} images de test | peripherique : {device}")

    # ---------------------------------------------------------------- [1]
    print("\n[1] Saturation des sorties (les attaques lisent-elles quelque chose ?)")
    with torch.no_grad():
        logits = modele(x)
        proba = F.softmax(logits, dim=1)
        credibilite, _ = proba.max(dim=1)
        ce_propre = F.cross_entropy(logits, y).item()
        tri, _ = logits.sort(dim=1, descending=True)
        ecart = (tri[:, 0] - tri[:, 1]).mean().item()
        surconf = (credibilite > 0.999).float().mean().item()
    print(f"    precision propre   : {accuracy(modele, x, y):.2%}")
    print(f"    CE propre          : {ce_propre:.4f}  (nos modeles sont TRES surs :"
          " 0.012 pour abl_a)")
    print(f"    max-prob moyenne   : {credibilite.mean().item():.4f}")
    print(f"    ecart logits 1-2   : {ecart:.3f}")
    print(f"    part max-prob>0.999: {surconf:.1%}")

    ref = None
    if args.reference:
        try:
            ref = charger_modele(args.reference, device, args.dataset)
            with torch.no_grad():
                lg = ref(x)
                pr = F.softmax(lg, dim=1).max(dim=1)[0]
                tri2, _ = lg.sort(dim=1, descending=True)
                ce_ref = F.cross_entropy(lg, y).item()
            print(f"    [reference] CE {ce_ref:.4f} | max-prob {pr.mean().item():.4f} "
                  f"| ecart logits {(tri2[:, 0] - tri2[:, 1]).mean().item():.3f}")
        except Exception as e:                                     # noqa: BLE001
            print(f"    [reference] non chargee ({type(e).__name__} : {e})")
            ref = None

    if surconf > 0.8 or ecart > 20.0:
        drapeau(1, "sorties tres confiantes (courant sur MNIST durci : un modele "
                   "a 99.6% en propre est presque toujours > 0.999) -- a confronter "
                   "au rayon robuste du test [3b]",
                f"sorties tres confiantes (max-prob>0.999 sur {surconf:.0%}, "
                f"ecart de logits {ecart:.1f})")
    else:
        ok(1, "sorties non saturees : les attaques ont de quoi lire le modele")
    # Remarque (2026-09-12) : la CE propre seule ne dit RIEN de la saturation. Nos
    # modeles durcis sont tous tres surs (CE 0.012 pour abl_a, 0.045 ici) : un
    # seuil sur la CE produirait un faux positif sur la reference elle-meme.

    # ---------------------------------------------------------------- [2]
    print("\n[2] Difference finie vs gradient analytique (le gradient pointe-t-il ?)")
    xg = x.detach().requires_grad_(True)
    perte = F.cross_entropy(modele(xg), y)
    (g,) = torch.autograd.grad(perte, xg)
    direction = g.sign()
    torch.manual_seed(0)
    aleatoire = torch.sign(torch.randn_like(x))
    with torch.no_grad():
        ce_depart = F.cross_entropy(modele(x), y).item()
        ce_grad = F.cross_entropy(modele((x + args.eps * direction).clamp(0, 1)), y).item()
        ce_alea = F.cross_entropy(modele((x + args.eps * aleatoire).clamp(0, 1)), y).item()
    d_grad, d_alea = ce_grad - ce_depart, ce_alea - ce_depart
    rapport = d_grad / d_alea if abs(d_alea) > 1e-9 else float("inf")
    print(f"    CE depart                       : {ce_depart:.4f}")
    print(f"    CE apres eps*sign(gradient)     : {ce_grad:.4f}  (delta {d_grad:+.4f})")
    print(f"    CE apres eps*sign(aleatoire)    : {ce_alea:.4f}  (delta {d_alea:+.4f})")
    print(f"    gradient / aleatoire            : x{rapport:.2f}")
    print(f"    |gradient| moyen (par pixel)    : {g.abs().mean().item():.3e}")
    if d_grad <= 0:
        drapeau(2, "la perturbation dans le sens du gradient ne fait pas monter la "
                   "perte : le gradient ne pointe plus la vulnerabilite",
                "gradient non informatif (delta CE <= 0)")
    elif rapport < 2.0:
        drapeau(2, f"la direction du gradient fait a peine mieux qu'une direction "
                   f"aleatoire (x{rapport:.2f}) : surface rugueuse, l'attaquant a "
                   "gradient va patiner. ATTENTION : cela PREDIT un ecart entre "
                   "attaques a gradient et attaques par scores (verifier la ligne "
                   "APGD vs Square de la suite), sans que le modele soit masque",
                f"gradient peu informatif (x{rapport:.2f} contre une direction aleatoire)")
    else:
        ok(2, f"le gradient reste informatif (x{rapport:.2f} mieux que l'aleatoire)")
    print(f"    [INDICE] rapport gradient/aleatoire : abl_a = 1.28 ; bande_cible50 = 8.52 ;"
          f" A4 = 6.25. Ce rapport dit si les attaques a gradient sont UTILISABLES ;"
          f" il ne predit PAS le pire cas (A4 : x6.25 et 49.2% seulement).")

    # --------------------------------------------------------------- [2b]
    # La CE est la MAUVAISE echelle sur ces modeles. Lecon du 12/09 (23h50) :
    # A4 a un gradient "informatif" (x6.25 sur la CE) et pourtant APGD-DLR
    # (57.2%) et Square (49.2%) le cassent, alors que le PGD sur la CE le rate
    # (80.2%). Autrement dit : la surface peut etre PLATE EN CE et fragile en
    # MARGE. On mesure donc aussi la marge (logit de la vraie classe moins le
    # meilleur autre logit), le long du gradient et en direction aleatoire.
    print("\n[2b] Difference finie sur la MARGE (l'echelle qui compte vraiment)")

    def marge(logits, y):
        vrai = logits.gather(1, y.view(-1, 1)).squeeze(1)
        autres = logits.clone()
        autres.scatter_(1, y.view(-1, 1), float("-inf"))
        return (vrai - autres.max(dim=1)[0]).mean().item()

    with torch.no_grad():
        m0 = marge(modele(x), y)
        m_grad = marge(modele((x + args.eps * direction).clamp(0, 1)), y)
        m_alea = marge(modele((x + args.eps * aleatoire).clamp(0, 1)), y)
    d_m_grad, d_m_alea = m_grad - m0, m_alea - m0
    rapport_m = d_m_grad / d_m_alea if abs(d_m_alea) > 1e-9 else float("inf")
    print(f"    marge propre                    : {m0:+.3f}")
    print(f"    marge apres eps*sign(gradient)  : {m_grad:+.3f}  (delta {d_m_grad:+.3f})")
    print(f"    marge apres eps*sign(aleatoire) : {m_alea:+.3f}  (delta {d_m_alea:+.3f})")
    print(f"    marge detruite par l'aleatoire   : {100 * d_m_alea / abs(m0) if m0 else 0:+.1f}%")
    print(f"    gradient / aleatoire (marge)     : x{rapport_m:.2f}")
    if m0 and d_m_alea / abs(m0) < -0.5 and d_m_grad / abs(m0) < -0.5:
        print("    [ALERTE] UNE DIRECTION ALEATOIRE detruit la marge : le modele n'est"
              " pas robuste en marge, meme si la CE bouge peu. Les attaques sur la"
              " marge (APGD-DLR) et par scores (Square) devraient le casser ->"
              " c'est exactement le profil de A4 (49.2% de pire cas).")
    elif m0 and d_m_alea / abs(m0) > -0.2:
        print("    [OK] la marge resiste meme a une direction aleatoire : profil de"
              " A1 (les attaques sur la CE, sur la marge et par scores echouent"
              " toutes).")

    # ---------------------------------------------------------------- [3]
    print("\n[3] Balayage de eps (jusqu'a la borne non contrainte) + bruit aleatoire")
    print(f"    {'eps':>6} | {'propre':>8} | {'FGSM':>8} | {'PGD-20':>8} | "
          f"{'PGD fin':>8} | {'bruit':>8}")
    lignes = []
    for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        torch.manual_seed(1)
        a_fgsm = accuracy(modele, fgsm(modele, x, y, eps), y)
        a_pgd = accuracy(modele, pgd(modele, x, y, eps, 20), y)
        a_fin = accuracy(modele, pgd(modele, x, y, eps, 200, alpha=eps / 100.0), y)
        bruit = (x + eps * torch.sign(torch.randn_like(x))).clamp(0, 1)
        a_bruit = accuracy(modele, bruit, y)
        print(f"    {eps:>6.2f} | {accuracy(modele, x, y):>8.1%} | {a_fgsm:>8.1%} | "
              f"{a_pgd:>8.1%} | {a_fin:>8.1%} | {a_bruit:>8.1%}")
        lignes.append((eps, a_fgsm, a_pgd, a_fin, a_bruit))

    # Le verdict d'insensibilite se lit sur TOUTE la courbe, pas sur une seule
    # ligne. Correction du 12/09 : exiger "insensible a eps=0.3" suffisait a
    # declarer un faux positif sur un modele qui s'effondre a eps=0.5 (ce qui est
    # la signature d'un RAYON ROBUSTE reel et non d'un masquage).
    tres_faibles = [l for l in lignes if l[0] >= 0.3 and l[1] > 0.95 and l[2] > 0.95]
    resiste_partout = all(l[1] > 0.90 for l in lignes if l[0] >= 0.5)
    if tres_faibles and resiste_partout:
        PREUVES.append("le modele reste insensible a eps=1.0 : la il n'y a plus de "
                       "robustesse a trouver, la decision ne depend plus de l'entree")
        print("    [SUSPECT] insensible jusqu'a eps=1.0")
    elif tres_faibles:
        print("    [OK] insensible jusqu'a eps=0.3 mais s'effondre au-dela : ce "
              "n'est PAS un masquage, c'est un RAYON ROBUSTE reel (voir [3b])")
    for eps, a_fgsm, _a_pgd, _a_fin, a_bruit in lignes:
        if eps == args.eps and a_bruit < a_fgsm - 0.05:
            PREUVES.append(f"a eps={eps}, le bruit aleatoire fait plus de degats "
                           f"({a_bruit:.1%}) que l'attaque dirigee ({a_fgsm:.1%})")
            print(f"    [SUSPECT] bruit aleatoire plus efficace que l'attaque dirigee")

    # ---------------------------------------------------------------- [3b]
    # Le vrai chiffre a comparer entre modeles n'est pas le pire cas a un eps
    # arbitraire, c'est le RAYON ROBUSTE : le eps ou le modele passe sous 50%.
    #
    # CORRECTIF du 2026-09-12 (23h00) : le rayon mesure au SEUL PGD est faux sur
    # les modeles a surface rugueuse. Mesure : abl_a et bande_cible50 avaient le
    # meme rayon PGD (~0.40) alors qu'AutoAttack leur donne 50.7% et 91.25% au
    # meme eps. Quand le gradient ne guide plus (rapport du test [2] proche de 1),
    # PGD sous-estime la vulnerabilite et le rayon annonce est trop grand. On
    # mesure donc AUSSI au Square (recherche par scores, insensible au gradient).
    print("\n[3b] Rayon robuste approche (eps ou la precision passe sous 50%)")
    with_square = args.rayon_square > 0
    entete = f"    {'eps':>6} | {'PGD fin':>8}"
    if with_square:
        entete += f" | {'Square':>8} | {'PIRE':>8}"
    print(entete)
    rayon = rayon_sq = None
    for eps_i in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        torch.manual_seed(4)
        a_fin = accuracy(modele, pgd(modele, x, y, eps_i, 100, alpha=eps_i / 50.0), y)
        if with_square:
            a_sq = accuracy(modele, square(modele, x, y, eps_i,
                                           steps=args.rayon_square,
                                           restarts=1, seed=6000), y)
            ligne_eps = (f"    {eps_i:>6.2f} | {a_fin:>8.1%} | {a_sq:>8.1%} | "
                         f"{min(a_fin, a_sq):>8.1%}")
            if rayon_sq is None and a_sq < 0.5:
                rayon_sq = eps_i
        else:
            ligne_eps = f"    {eps_i:>6.2f} | {a_fin:>8.1%}"
        print(ligne_eps)
        if rayon is None and a_fin < 0.5:
            rayon = eps_i
    if rayon is None:
        print("    rayon robuste (PGD) > 0.60")
    else:
        print(f"    rayon robuste (PGD)    ~ {rayon:.2f} (borne sup. ; pas de 0.05)")
    if with_square:
        if rayon_sq is None:
            print("    rayon robuste (Square) > 0.60")
        else:
            print(f"    rayon robuste (Square) ~ {rayon_sq:.2f}")
        if rayon_sq is not None and rayon is not None and rayon - rayon_sq >= 0.10:
            drapeau("3b", "le rayon mesure au PGD surestime la robustesse : la marche de "
                           f"{rayon - rayon_sq:.2f} entre PGD ({rayon:.2f}) et Square "
                           f"({rayon_sq:.2f}) dit que le gradient ne guide plus. "
                           "Le rayon PGD n'est PAS publiable sur ce modele ; c'est le "
                           "pire cas sous l'attaque la plus forte qui compte",
                    f"rayon PGD {rayon:.2f} contre rayon Square {rayon_sq:.2f}")
        elif rayon is not None and rayon_sq is not None:
            ok("3b", f"les deux attaques donnent le meme rayon (~{rayon:.2f}) : la "
                      "mesure est fiable dans les deux familles")

    # ---------------------------------------------------------------- [4]
    print("\n[4] Pas fin contre pas grossier (notre loi du budget de deplacement)")
    torch.manual_seed(2)
    par_pas = {}
    for nom, alpha, steps in [("eps/4 (defaut)", args.eps / 4, 20),
                              ("eps/10", args.eps / 10, 20),
                              ("eps/50", args.eps / 50, 100),
                              ("eps/100", args.eps / 100, 200),
                              ("eps/400", args.eps / 400, 400)]:
        par_pas[nom] = accuracy(modele, pgd(modele, x, y, args.eps, steps, alpha=alpha), y)
        print(f"    PGD {nom:<16} : {par_pas[nom]:>6.1%}")
    meilleur = max(par_pas.values())
    grossier = par_pas["eps/4 (defaut)"]
    if grossier - meilleur > 0.05:
        alerte(4, f"un pas plus FIN gagne {grossier - meilleur:.1%} sur le pas "
                  "grossier : c'est le PAS qui etouffe l'attaque, pas la robustesse")
    else:
        ok(4, f"aucun pas ne gagne plus de 5 pts sur le pas grossier "
              f"(meilleur {meilleur:.1%})")

    # ---------------------------------------------------------------- [5]
    print("\n[5] Transfert depuis un autre modele (le test decisif du masquage)")
    if ref is None:
        print("    ignore (pas de modele de reference)")
    else:
        torch.manual_seed(3)
        adv_fgsm = fgsm(ref, x, y, args.eps)
        adv_pgd = pgd(ref, x, y, args.eps, 20)
        a_t_fgsm = accuracy(modele, adv_fgsm, y)
        a_t_pgd = accuracy(modele, adv_pgd, y)
        a_wb_fgsm = accuracy(modele, fgsm(modele, x, y, args.eps), y)
        a_wb_pgd = accuracy(modele, pgd(modele, x, y, args.eps, 20), y)
        print(f"    transfert FGSM (source -> cible) : {a_t_fgsm:>6.1%} "
              f"(white-box : {a_wb_fgsm:.1%})")
        print(f"    transfert PGD  (source -> cible) : {a_t_pgd:>6.1%} "
              f"(white-box : {a_wb_pgd:.1%})")
        # Comparaison a armes egales : transfert PGD contre white-box PGD.
        if a_t_pgd < a_wb_pgd - 0.05:
            PREUVES.append(f"un transfert bat l'attaque white-box de "
                           f"{a_wb_pgd - a_t_pgd:.1%} : la robustesse n'est pas dans "
                           "les poids")
            print("    [SUSPECT] le transfert bat le white-box")
        else:
            ok(5, "le transfert ne bat pas l'attaque white-box : pas de masquage")

    # ---------------------------------------------------------------- synthese
    print("\n" + "=" * 72)
    if PREUVES:
        print(f"  SYNTHESE : MASQUAGE PROBABLE ({len(PREUVES)} preuve(s))")
        for s in PREUVES:
            print(f"    - {s}")
        print("\n  -> NE PAS publier ni entrainer davantage sur ce modele avant")
        print("     d'avoir tranche : un modele qui masque son gradient peut")
        print("     tomber a 30% sous une attaque adaptative (AutoAttack).")
    elif DRAPEAUX or SUSPECTS:
        print("  SYNTHESE : aucun masquage (le modele s'effondre des que eps augmente,"
              " donc il y a un rayon robuste reel a trouver), MAIS a surveiller :")
        for s in DRAPEAUX:
            print(f"    - [profil] {s}")
        for s in SUSPECTS:
            print(f"    - [mesure] {s}")
        print("\n  -> ces signes decrivent une surface RUGUEUSE et tres confiante, pas une")
        print("     defense opaque. Consequence attendue : les attaques par SCORES doivent")
        print("     battre les attaques a gradient sur ce modele. A confirmer par la suite")
        print("     d'attaques (ligne APGD vs Square) et par AutoAttack.")
    else:
        print("  SYNTHESE : aucun signe de masquage sur ces cinq tests.")
        print("  -> la robustesse est plausible ; la croiser avec AutoAttack")
        print("     reste l'etape 1 du perimetre avant toute publication.")
    print("=" * 72)
    return 1 if PREUVES else 0


if __name__ == "__main__":
    sys.exit(main())
