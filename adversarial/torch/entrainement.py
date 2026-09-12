"""Entrainement durci en PyTorch : memes recettes que `harden2.py`.

Recettes disponibles (identiques a la version NumPy) :
  - pgdat  : entrainement sur exemples propres + adverses (Madry)
  - trades : CE(propre) + beta * KL(propre || adverse) (Zhang et al. 2019)

Plus : warm start, decroissance du learning rate, ecrasement des gradients,
augmentation de donnees, selection du modele sur la robustesse de validation.

L'augmentation reproduit les memes transformations que `augment.py` (rotation,
zoom, translation, bruit impulsionnel, cutout, epaisseur) mais avec des
operations PyTorch, donc sur GPU et sans boucle Python sur les images.
"""

import sys
import time
from os.path import abspath, dirname, join

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, join(ROOT_DIR, "src"))

from data import MNISTLoader, normalize, add_channel_dim  # noqa: E402
from adversarial.torch.attaques import attaque, accuracy  # noqa: E402
from adversarial.torch.attaque_adaptative import (pgd_bande, cible_effective,  # noqa: E402
                                                 JournalBande)


CONFIG_AUG = {
    "rotation": 12.0, "translation": 2, "zoom": 0.10,
    "bruit_p": 0.02, "bruit_intensite": (0.5, 1.0),
    "cutout": 6, "epaisseur": 0.3, "epaisseur_melange": 0.6, "prob": 0.5,
}


# --------------------------------------------------------------------------
#  Donnees (meme selection que harden2.py, pour une comparaison equitable)
# --------------------------------------------------------------------------

def _assurer_donnees(dataset):
    """Telecharge le jeu de donnees s'il manque, sinon ne fait rien.

    `data/kmnist/` et `data/emnist/` sont volontairement ignores par git (ce
    sont des donnees brutes, 21 Mo et 570 Mo) : un clone frais ne les contient
    pas. Sans cet appel, `harden_torch.py` mourrait sur un FileNotFoundError de
    `src/data.py`, tres loin de la vraie cause. On reutilise les
    telechargeurs deja ecrits pour la piste NumPy (`scripts/download_kmnist.py`,
    `adversarial/scripts/download_emnist.py`) plutot que d'en ecrire un autre.

    MNIST n'est pas concerne (deja present dans le depot).
    """
    if dataset == "kmnist":
        from scripts.download_kmnist import ensure_data
    elif dataset == "emnist":
        from scripts.download_emnist import ensure_data
    else:
        return
    if not ensure_data():
        raise SystemExit(
            f"Donnees {dataset.upper()} absentes et recuperation impossible "
            "(pas de reseau ?).\n"
            "  Telecharge-les a la main puis relance :\n"
            f"    python3 scripts/download_{dataset}.py"
        )


def charger_train(n_train, n_val, dataset="mnist"):
    """Reproduit exactement la selection de donnees de harden2.py.

    `dataset` : "mnist" (defaut), "kmnist" (kana japonais, meme format 28x28 et
    10 classes) ou "emnist". Branche KMNIST ajoutee le 2026-09-13 pour la
    replication de l'etude sur le jeu japonais (ancre Japon du repo).
    """
    _assurer_donnees(dataset)
    if dataset == "kmnist":
        from data import KMNISTLoader
        loader, dossier = KMNISTLoader(), join(ROOT_DIR, "data", "kmnist")
    elif dataset == "emnist":
        from data import EMNISTLoader
        loader, dossier = EMNISTLoader("letters"), join(ROOT_DIR, "data", "emnist")
    else:
        loader, dossier = MNISTLoader(), join(ROOT_DIR, "data")
    (x_all, y_all), _ = loader.load(dossier)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(x_all), size=min(n_train + n_val, len(x_all)), replace=False)
    x = np.ascontiguousarray(normalize(add_channel_dim(x_all[idx])).transpose(0, 3, 1, 2))
    y = y_all[idx]
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    return (x[n_val:], y[n_val:]), (x[:n_val], y[:n_val])


def charger_test(dataset="mnist", n=500):
    """Meme echantillon de test que les scripts NumPy (load_data de fgsm.py).

    Reimplemente ici plutot qu'importe, pour que la piste PyTorch n'ait besoin
    que de torch et numpy (les scripts NumPy importent matplotlib au chargement).
    La selection est identique : RandomState(42) sur l'ordre du test set.
    """
    _assurer_donnees(dataset)
    if dataset == "kmnist":
        from data import KMNISTLoader
        (_, _), (x_test, y_test) = KMNISTLoader().load(join(ROOT_DIR, "data", "kmnist"))
    elif dataset == "mnist":
        loader = MNISTLoader()
        (_, _), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))
    else:
        from data import EMNISTLoader
        loader = EMNISTLoader("letters")
        (_, _), (x_test, y_test) = loader.load(join(ROOT_DIR, "data", "emnist"))

    rng = np.random.RandomState(42)
    idx = rng.choice(len(x_test), size=min(n, len(x_test)), replace=False)
    x = np.ascontiguousarray(
        normalize(add_channel_dim(x_test[idx])).transpose(0, 3, 1, 2))
    return torch.from_numpy(x).float(), torch.from_numpy(y_test[idx]).long()


# --------------------------------------------------------------------------
#  Augmentation (memes transformations que augment.py)
# --------------------------------------------------------------------------

def _affine(x, degre_max, zoom_max):
    """Rotation + echelle via grid_sample (le fond reste noir)."""
    n, _, h, w = x.shape
    dev = x.device
    a = torch.deg2rad(torch.empty(n, device=dev).uniform_(-degre_max, degre_max))
    s = 1.0 + torch.empty(n, device=dev).uniform_(-zoom_max, zoom_max)
    cos, sin = torch.cos(a) * s, torch.sin(a) * s
    zero = torch.zeros_like(cos)
    theta = torch.stack([
        torch.stack([cos, sin, zero], dim=1),
        torch.stack([-sin, cos, zero], dim=1),
    ], dim=1)                                   # (N, 2, 3)
    grille = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grille, align_corners=False, padding_mode="zeros")


def _translation(x, decalage_max):
    n, _, h, w = x.shape
    dev = x.device
    dx = torch.randint(-decalage_max, decalage_max + 1, (n,), device=dev).float()
    dy = torch.randint(-decalage_max, decalage_max + 1, (n,), device=dev).float()
    cos = torch.ones_like(dx)
    zero = torch.zeros_like(dx)
    theta = torch.stack([
        torch.stack([cos, zero, -2.0 * dx / w], dim=1),
        torch.stack([zero, cos, -2.0 * dy / h], dim=1),
    ], dim=1)
    grille = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grille, align_corners=False, padding_mode="zeros")


def _bruit(x, p, intensite):
    """Sel et poivre : pixels allumes sur le fond, quelques pixels du trait eteints."""
    fond = x < 0.2
    sel = (torch.rand_like(x) < p) & fond
    bas, haut = intensite
    val = torch.rand_like(x) * (haut - bas) + bas
    x = torch.where(sel, val, x)
    poi = (torch.rand_like(x) < p * 0.5) & (x > 0.2)
    return torch.where(poi, torch.zeros_like(x), x)


def _cutout(x, taille_max):
    n, _, h, w = x.shape
    dev = x.device
    t = torch.randint(taille_max // 2, taille_max + 1, (n,), device=dev)
    y0 = (torch.rand(n, device=dev) * (h - t).float()).long()
    x0 = (torch.rand(n, device=dev) * (w - t).float()).long()
    yy = torch.arange(h, device=dev).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, 1, w)
    masque = ((yy >= y0.view(-1, 1, 1, 1)) & (yy < (y0 + t).view(-1, 1, 1, 1))
              & (xx >= x0.view(-1, 1, 1, 1)) & (xx < (x0 + t).view(-1, 1, 1, 1)))
    return torch.where(masque, torch.zeros_like(x), x)


def _epaisseur(x, melange):
    """Epaissit ou amincit le trait (element en croix = max_pool 3x3, padding 1)."""
    n = x.shape[0]
    dil = F.max_pool2d(x, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-x, 3, stride=1, padding=1)
    choix = (torch.rand(n, 1, 1, 1, device=x.device) < 0.5).float()
    epais = x + melange * (dil - x)
    fin = x + melange * (ero - x)
    return choix * epais + (1 - choix) * fin


def augmenter(x, cfg=None):
    """Pipeline aleatoire, memes transformations que augment.py."""
    c = dict(CONFIG_AUG)
    if cfg:
        c.update(cfg)
    if c["rotation"] or c["zoom"]:
        x = _affine(x, c["rotation"], c["zoom"])
    if c["translation"] and torch.rand(()) < c["prob"]:
        x = _translation(x, c["translation"])
    if c["epaisseur"] and torch.rand(()) < c["epaisseur"]:
        x = _epaisseur(x, c["epaisseur_melange"])
    if c["bruit_p"] and torch.rand(()) < c["prob"]:
        x = _bruit(x, c["bruit_p"], c["bruit_intensite"])
    if c["cutout"] and torch.rand(()) < c["prob"] * 0.5:
        x = _cutout(x, c["cutout"])
    return x.clamp(0.0, 1.0)


# --------------------------------------------------------------------------
#  Entrainement
# --------------------------------------------------------------------------

def _paliers_lr(spec, epochs):
    """Epochs ou le learning rate est divise par 10.

    Calcule une seule fois, en numeros d'epoch entiers (pas en fractions) :
    avec l'ancienne formule, un run de 2 epochs declenchait les DEUX paliers
    (0.5 et 0.8) des le premier epoch, et le lr s'effondrait a 0.0005 avant
    d'avoir servi. Le modele n'apprenait alors quasiment rien.
    Sous 4 epochs, le planificateur est desactive (trop court pour decroitre).
    """
    if not spec or epochs < 4:
        return []
    return sorted({max(1, int(round(float(p) * epochs)))
                   for p in spec.split(",") if p.strip()})


def pas_planifies(args, epoch):
    """Nombre de pas de l'attaque interne pour cette epoch, si un plan de
    budget est demande (`--plan-budget "0.2,2"`, bornes en multiples de eps).

    CURRICULUM DETERMINISTE : le budget de l'attaque interne monte au fil du
    run, lineairement. Le pas reste FIXE (eps/10), on ne change QUE le nombre
    de pas -- donc le budget vaut `nombre de pas x pas`.

    Pourquoi une option, alors qu'on peut relancer en deux phases avec
    `--resume` : la reprise RESTAURE le learning rate du checkpoint, et les
    paliers du lr sont calcules sur la duree de la phase. Une phase courte
    (30 epochs) fait donc tomber le lr a 0.0005 (paliers aux epochs 15 et 24),
    et la phase suivante repart a ce lr minuscule : le modele n'apprend plus.
    Mesure du 2026-09-13 : A6 en deux phases = 71.2% au lieu des ~91% vises.
    Un seul run supprime toute cette classe d'erreur.
    """
    if not getattr(args, "plan_budget", ""):
        return args.pgd_steps
    try:
        deb, fin = (float(v) for v in args.plan_budget.split(","))
    except ValueError:
        print(f"  [PLAN] [warn] --plan-budget mal forme ({args.plan_budget}) : ignore")
        return args.pgd_steps
    alpha = args.pgd_alpha if args.pgd_alpha else args.eps / 10.0
    t = (epoch - 1) / max(1, args.epochs - 1)
    budget = deb + (fin - deb) * t          # en multiples de eps
    return max(1, int(round(budget * args.eps / alpha)))


def _beta_effectif(beta, epoch, epochs, warmup):
    """Rampe de beta pour TRADES.

    Sur un modele deja entraine, la cross-entropy vaut ~0 alors que la KL vaut
    ~1 : si on demarre a beta plein, le terme KL ecrase tout et le modele
    s'effondre (val clean 98% -> 50% observe le 2026-09-10). On demarre donc a
    10% de beta et on monte progressivement.
    """
    if warmup <= 0:
        return beta
    # Rampe etalee sur AU MOINS 3 epochs, et partant de ZERO :
    # le premier epoch est du CE pur (aucune KL), puis beta monte.
    # Mesure : sans ce depart a zero, un modele deja converge s'effondre
    # (val clean 98% -> 29% en 2 epochs, constate le 2026-09-10). La raison est
    # que sur un modele converge la CE vaut ~0.01 quand la KL vaut ~1.3 : le
    # terme KL ecrase tout et le modele minimise la KL en devenant constant.
    frac = (epoch - 1) / max(3.0, warmup * epochs)
    return beta * min(1.0, frac)


def entrainer(modele, opt, train, val, args, device):
    x_tr, y_tr = train
    x_val, y_val = val
    n = len(x_tr)
    meilleur = float(getattr(args, "meilleur_init", -1.0) or -1.0)
    lr = args.lr
    # Reprise eventuelle : on redemarre la boucle a l'epoch demandee.
    # Les paliers de lr sont calcules sur la duree TOTALE : reprendre a
    # l'epoch 61 d'un run de 120 garde donc le palier de l'epoch 96.
    start = max(1, int(getattr(args, "start_epoch", 1) or 1))
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    paliers = _paliers_lr(args.lr_drop, args.epochs)
    if args.lr_drop and not paliers:
        print(f"  [LR] planificateur desactive ({args.epochs} epochs : trop court)")
    elif paliers:
        print(f"  [LR] paliers aux epochs {paliers}")

    if start > 1:
        hist = f"{meilleur:.2%}" if meilleur >= 0 else "aucun (reprise d'un ancien checkpoint)"
        print(f"  [REPRISE] debut a l'epoch {start} (lr {lr:.5f}, meilleur val PGD {hist})")
    if start > args.epochs:
        print(f"  [REPRISE] rien a faire : {start} > {args.epochs} epochs")
        return meilleur

    epochs_sous = 0        # epochs consecutifs sous le meilleur (garde-fou)
    pas_ep_prec = None     # derniere valeur annoncee du plan de budget

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()

        if paliers and epoch in paliers:
            lr *= 0.1
            for g in opt.param_groups:
                g["lr"] = lr
            print(f"  [LR] epoch {epoch} : lr -> {lr:.5f}")

        beta = _beta_effectif(args.beta, epoch, args.epochs, args.beta_warmup)
        if args.loss == "trades" and epoch == 1:
            print(f"  [TRADES] beta effectif : {beta:.2f} -> {args.beta:.2f} "
                  f"(rampe sur {args.beta_warmup:.0%} du run)")

        ordre = torch.randperm(n, generator=gen)
        # [plan de budget] curriculum deterministe : le nombre de pas de
        # l'attaque interne monte au fil du run (le PAS reste fixe, donc le
        # budget `pas x alpha` monte avec lui).
        pas_ep = args.pgd_steps
        if getattr(args, "plan_budget", ""):
            pas_ep = pas_planifies(args, epoch)
            if pas_ep != pas_ep_prec:
                alpha_eff = args.pgd_alpha if args.pgd_alpha else args.eps / 10.0
                print(f"  [PLAN] budget attaque interne -> {pas_ep} pas "
                      f"({pas_ep * alpha_eff / args.eps:.2f} eps)")
                pas_ep_prec = pas_ep
        perte_tot = 0.0
        # Diagnostic de l'attaque interne (affiche sur la ligne d'epoch) :
        # perte sur la moitie PROPRE, perte et taux de tromperie sur la moitie
        # ADVERSE. Les deux pertes sont gratuites : elles sont lues sur les
        # logits du batch mixte, qui servent deja au calcul de la perte.
        ce_propre_som = 0.0
        ce_adv_som = 0.0
        n_propre_tot = 0
        n_adv_tot = 0
        n_adv_succ = 0.0
        journal = JournalBande()
        modele.train()

        # ---- Un batch = les etapes [1] a [5] de la visite guidee ----
        for debut in range(0, n, args.batch):
            # [1] un lot d'images (l'ordre a ete melange une fois par epoch)
            bi = ordre[debut:debut + args.batch]
            bx = x_tr[bi].to(device)
            by = y_tr[bi].to(device)

            # [2] augmentation : les memes images, deformees a la volee
            if args.augment:
                bx = augmenter(bx, args.aug_cfg)

            # [3] exemple adverse (genere avec le modele courant)
            #     eval() pendant l'attaque : on vise le modele tel qu'il se
            #     comporte a l'inference (dropout desactive), puis on revient
            #     en train().
            modele.eval()
            if getattr(args, "sans_attaque", False):
                # Entrainement PROPRE, sans aucune attaque : sert a produire le
                # modele de reference et le point de depart (warm start) des
                # runs robustes. Sinon tout run passe par une attaque, meme
                # avec --eps 0.
                bx_adv = bx
            elif getattr(args, "bande", False):
                # Attaque a budget ADAPTATIF : on s'arrete au pas k* ou la
                # difficulte (tromperie ou CE) atteint la cible de l'epoch.
                # Cout identique a pgd() : voir attaque_adaptative.py.
                cible = cible_effective(args.cible, epoch, args.epochs,
                                        getattr(args, "cible_rampe", 0.5),
                                        getattr(args, "cible_depart", 0.2))
                bx_adv, info = pgd_bande(modele, bx, by, args.eps, args.pgd_steps,
                                         getattr(args, "pgd_alpha", None), cible,
                                         getattr(args, "cible_type", "tromperie"))
                journal.ajouter(info)
            else:
                bx_adv = attaque(modele, bx, by, args.eps, args.attack, pas_ep,
                                 getattr(args, "pgd_alpha", None))
            modele.train()

            # [4] perte : pgdat (propre + adverse) ou trades (CE + beta*KL)
            if args.loss == "trades":
                # [fix CRITIQUE] Le gradient doit passer par LES DEUX branches.
                # x_adv est detache (on ne derive pas par rapport a la
                # perturbation), mais logits_adv = modele(x_adv) reste dans le
                # graphe : les parametres sont partages.
                # En detachant logits_adv, on obtient une catastrophe : la KL
                # pousse alors la prediction PROPRE vers la prediction ADVERSE
                # (qui est fausse), donc le modele apprend a se tromper.
                # Mesure : val clean 99.6% -> 8.0% en un seul epoch.
                # Variante de TRADES (Zhang 2019) : KL(p_adverse || p_propre),
                # les deux branches restant dans le graphe (voir le [fix] plus
                # haut). [a trancher] l'implementation de reference
                # (yaodongyu/TRADES) utilise le sens INVERSE,
                # KL(p_propre || p_adverse), et genere la perturbation en
                # maximisant cette KL ; ici la perturbation vient d'un PGD sur
                # la CE (selecteur `attaque`). Detail : defenses.md, 6.5.
                logits = modele(bx)
                logits_adv = modele(bx_adv)
                ce = F.cross_entropy(logits, by)
                kl = F.kl_div(F.log_softmax(logits, dim=1),
                              F.softmax(logits_adv, dim=1),
                              reduction="batchmean")
                perte = ce + beta * kl
                # diagnostic : les deux moities sont deja calculees ici
                with torch.no_grad():
                    ce_propre_som += ce.item() * len(bx)
                    n_propre_tot += len(bx)
                    ce_adv_som += F.cross_entropy(logits_adv, by).item() * len(bx)
                    n_adv_succ += (logits_adv.argmax(1) != by).float().sum().item()
                    n_adv_tot += len(bx)
            else:
                n_propre = 0
                if args.mix < 1.0:
                    k = int(round(len(bx) * args.mix))
                    cx = torch.cat([bx, bx_adv[:k]], dim=0)
                    cy = torch.cat([by, by[:k]], dim=0)
                    n_propre = len(bx)
                else:
                    cx, cy = bx_adv, by
                logits_batch = modele(cx)
                perte = F.cross_entropy(logits_batch, cy)
                # diagnostic : on relit les logits du batch mixte, deja calcules
                with torch.no_grad():
                    if n_propre:
                        ce_propre_som += F.cross_entropy(
                            logits_batch[:n_propre], cy[:n_propre]).item() * n_propre
                        n_propre_tot += n_propre
                    n_adv = len(cx) - n_propre
                    if n_adv:
                        ce_adv_som += F.cross_entropy(
                            logits_batch[n_propre:], cy[n_propre:]).item() * n_adv
                        n_adv_succ += (logits_batch[n_propre:].argmax(1)
                                       != cy[n_propre:]).float().sum().item()
                        n_adv_tot += n_adv

            # [5] mise a jour des poids (zero_grad -> backward -> clip -> step)
            opt.zero_grad(set_to_none=True)
            perte.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(modele.parameters(), args.clip)
            opt.step()
            perte_tot += perte.item() * len(bx)

        # [6] validation : propre + sous attaque. C'est ce chiffre (et non
        #     l'accuracy propre) qui decide quel modele est sauvegarde -- sauf en
        #     mode --sans-attaque, voir le critere juste apres.
        modele.eval()
        with torch.no_grad():
            acc_clean = accuracy(modele, x_val.to(device), y_val.to(device))
        xa = attaque(modele, x_val.to(device), y_val.to(device),
                     args.eps, "pgd", args.val_steps)
        acc_rob = accuracy(modele, xa, y_val.to(device))

        # [6bis] Critere de selection du meilleur modele.
        #   - run robuste : la robustesse de validation (regle historique) ;
        #   - --sans-attaque : la precision PROPRE.
        # Pourquoi : en mode propre, la val PGD vaut 0% partout (c'est normal,
        # un modele propre n'est pas robuste). Avec `acc_rob > meilleur`, la
        # comparaison n'est vraie qu'une seule fois, a l'epoch 1 -- le modele
        # exporte etait donc celui de l'epoch 1 (val clean 94.6%), pas le modele
        # fini (99%+), et les runs robustes demarraient d'un warm start
        # volontairement affaibli. Bug trouve le 2026-09-13 en lançant la serie
        # KMNIST, avant que le run n'ait produit quoi que ce soit.
        sans_attaque = getattr(args, "sans_attaque", False)
        acc_critere = acc_clean if sans_attaque else acc_rob
        nom_critere = "val clean" if sans_attaque else f"val PGD{args.val_steps}"

        dt = time.time() - t0
        reste = (args.epochs - epoch) * dt / 60
        detail = ""
        if n_adv_tot and sans_attaque:
            # La "moitie adverse" est ici le batch propre lui-meme : ces colonnes
            # mesurent l'ERREUR D'ENTRAINEMENT, pas la reussite d'une attaque.
            # Les nommer "CE adv / attaque" induisait en erreur (13/09).
            ce_p = (ce_propre_som / n_propre_tot) if n_propre_tot else float("nan")
            detail = (f" | CE {ce_p:5.3f} "
                      f"| erreur train {n_adv_succ / n_adv_tot:6.1%}")
        elif n_adv_tot:
            ce_p = (ce_propre_som / n_propre_tot) if n_propre_tot else float("nan")
            detail = (f" | CE propre {ce_p:5.3f} | CE adv {ce_adv_som / n_adv_tot:5.3f} "
                      f"| attaque {n_adv_succ / n_adv_tot:6.1%}")
        if getattr(args, "bande", False):
            resume = journal.resume()
            if resume:
                detail += f" | {resume}"
        print(f"  Epoch {epoch:>2}/{args.epochs} | loss {perte_tot / n:6.4f}{detail} | "
              f"val clean {acc_clean:6.2%} | val PGD{args.val_steps} {acc_rob:6.2%} | "
              f"{dt / 60:5.1f} min | reste ~{reste:4.0f} min")

        # [ALERTE 1] L'attaque interne fabrique-t-elle encore des exemples
        # adverses ? Si elle ne trompe plus la moitie du batch adverse, le
        # modele n'apprend plus que du propre : c'est le signal AVANT-COUREUR de
        # l'effondrement de la robustesse (constate le 2026-09-12 : la CE adv
        # rejoint la CE propre, puis la val PGD10 tombe a 0.6%). On previent,
        # on ne corrige pas tout seul : c'est un probleme de recette d'attaque,
        # pas de patience.
        # En mode --sans-attaque il n'y a AUCUNE attaque interne : le signal
        # mesure juste l'erreur d'entrainement en baisse, donc aucune alerte
        # (sinon tous les runs de reference crient au loup pendant 120 epochs).
        if n_adv_tot and not sans_attaque and n_adv_succ / n_adv_tot < 0.5:
            print(f"           [ALERTE] l'attaque interne ne trompe plus que "
                  f"{n_adv_succ / n_adv_tot:.0%} du batch adverse : la robustesse "
                  "apprise est en train de disparaitre (voir memoire.md, 2026-09-12).")

        # [ALERTE 2] Effondrement de la robustesse de validation : "s'effondre et
        # ne remonte plus" -> inutile de bruler le GPU, le meilleur modele est
        # deja sauvegarde.
        # [ALERTE 3] Attaque adaptative au plafond : le modele a depasse le
        # budget disponible. Ne PAS augmenter la force (la loi du budget de
        # deplacement dit que c'est contre-productif au-dela de ~2 eps) :
        # augmenter la diversite (EOT, departs multiples, attaque par scores).
        if getattr(args, "bande", False):
            msg = journal.alerte(getattr(args, "plafond_tol", 0.5))
            if msg:
                print(f"           {msg}")

        if args.collapse_tol and meilleur >= 0 and acc_critere < meilleur - args.collapse_tol:
            epochs_sous += 1
            print(f"           [ALERTE] {nom_critere} {acc_critere:.1%} : "
                  f"{meilleur - acc_critere:.1%} sous le meilleur ({meilleur:.1%}), "
                  f"{epochs_sous} epoch(s) de suite.")
            if args.stop_on_collapse and epochs_sous >= args.collapse_patience:
                print(f"           [ARRET] --stop-on-collapse : plus rien ne progresse depuis "
                      f"{epochs_sous} epochs. Meilleur modele conserve : {args.out} "
                      f"({nom_critere} {meilleur:.1%}).")
                return meilleur
        else:
            epochs_sous = 0

        # Checkpoint complet : permet de REPRENDRE apres une coupure (veille du
        # PC, arret manuel...) sans repartir de zero. Le fichier "best" reste au
        # format state_dict simple (compatible --report et --npz).
        torch.save({"format": 2, "model": modele.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "lr": lr, "meilleur": meilleur}, args.out + "_last.pt")
        if acc_critere > meilleur:
            meilleur = acc_critere
            torch.save(modele.state_dict(), args.out)
            print(f"           -> meilleur modele sauvegarde ({nom_critere} {acc_critere:.2%})")

    return meilleur


# --------------------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------------------

def evaluer(modele, x_te, y_te, eps_list, steps, restarts, device):
    res = {"fgsm": {}, "pgd": {}}
    for eps in eps_list:
        res["fgsm"][eps] = accuracy(modele, attaque(modele, x_te, y_te, eps, "fgsm"), y_te)
        pires = []
        for r in range(restarts):
            torch.manual_seed(1000 + r)
            pires.append(accuracy(modele, attaque(modele, x_te, y_te, eps, "pgd", steps), y_te))
        res["pgd"][eps] = min(pires)
    return res


def rapport(modele, x_te, y_te, eps_list, steps, restarts, label, device):
    print(f"\n[EVAL] {label} ({len(x_te)} images, PGD {steps} pas, {restarts} restart(s))")
    print(f"{'eps':>6} | {'FGSM':>8} | {'PGD (pire cas)':>15}")
    print("-" * 36)
    res = evaluer(modele, x_te, y_te, eps_list, steps, restarts, device)
    for eps in eps_list:
        print(f"{eps:>6} | {res['fgsm'][eps]:>8.1%} | {res['pgd'][eps]:>15.1%}")
    return res
