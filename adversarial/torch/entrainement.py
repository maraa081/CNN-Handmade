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


CONFIG_AUG = {
    "rotation": 12.0, "translation": 2, "zoom": 0.10,
    "bruit_p": 0.02, "bruit_intensite": (0.5, 1.0),
    "cutout": 6, "epaisseur": 0.3, "epaisseur_melange": 0.6, "prob": 0.5,
}


# --------------------------------------------------------------------------
#  Donnees (meme selection que harden2.py, pour une comparaison equitable)
# --------------------------------------------------------------------------

def charger_train(n_train, n_val, dataset="mnist"):
    """Reproduit exactement la selection de donnees de harden2.py."""
    loader = MNISTLoader()
    (x_all, y_all), _ = loader.load(join(ROOT_DIR, "data"))
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
    if dataset == "mnist":
        loader = MNISTLoader()
        (_, _), (x_test, y_test) = loader.load(join(ROOT_DIR, "data"))
    else:
        from data import EMNISTLoader
        from adversarial.scripts.fgsm import ensure_data
        ensure_data()
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
            bx_adv = attaque(modele, bx, by, args.eps, args.attack, args.pgd_steps,
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
        #     l'accuracy propre) qui decide quel modele est sauvegarde.
        modele.eval()
        with torch.no_grad():
            acc_clean = accuracy(modele, x_val.to(device), y_val.to(device))
        xa = attaque(modele, x_val.to(device), y_val.to(device),
                     args.eps, "pgd", args.val_steps)
        acc_rob = accuracy(modele, xa, y_val.to(device))

        dt = time.time() - t0
        reste = (args.epochs - epoch) * dt / 60
        detail = ""
        if n_adv_tot:
            ce_p = (ce_propre_som / n_propre_tot) if n_propre_tot else float("nan")
            detail = (f" | CE propre {ce_p:5.3f} | CE adv {ce_adv_som / n_adv_tot:5.3f} "
                      f"| attaque {n_adv_succ / n_adv_tot:6.1%}")
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
        if n_adv_tot and n_adv_succ / n_adv_tot < 0.5:
            print(f"           [ALERTE] l'attaque interne ne trompe plus que "
                  f"{n_adv_succ / n_adv_tot:.0%} du batch adverse : la robustesse "
                  "apprise est en train de disparaitre (voir memoire.md, 2026-09-12).")

        # [ALERTE 2] Effondrement de la robustesse de validation : "s'effondre et
        # ne remonte plus" -> inutile de bruler le GPU, le meilleur modele est
        # deja sauvegarde.
        if args.collapse_tol and meilleur >= 0 and acc_rob < meilleur - args.collapse_tol:
            epochs_sous += 1
            print(f"           [ALERTE] val PGD{args.val_steps} {acc_rob:.1%} : "
                  f"{meilleur - acc_rob:.1%} sous le meilleur ({meilleur:.1%}), "
                  f"{epochs_sous} epoch(s) de suite.")
            if args.stop_on_collapse and epochs_sous >= args.collapse_patience:
                print(f"           [ARRET] --stop-on-collapse : plus rien ne progresse depuis "
                      f"{epochs_sous} epochs. Meilleur modele conserve : {args.out} "
                      f"(val PGD {meilleur:.1%}).")
                return meilleur
        else:
            epochs_sous = 0

        # Checkpoint complet : permet de REPRENDRE apres une coupure (veille du
        # PC, arret manuel...) sans repartir de zero. Le fichier "best" reste au
        # format state_dict simple (compatible --report et --npz).
        torch.save({"format": 2, "model": modele.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "lr": lr, "meilleur": meilleur}, args.out + "_last.pt")
        if acc_rob > meilleur:
            meilleur = acc_rob
            torch.save(modele.state_dict(), args.out)
            print(f"           -> meilleur modele sauvegarde (val PGD {acc_rob:.2%})")

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
