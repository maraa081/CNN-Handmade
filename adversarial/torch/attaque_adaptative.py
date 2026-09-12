"""Attaque interne a BUDGET ADAPTATIF (-bande).

Idee (Maraa, 2026-09-12) : au lieu de fixer une fois pour toutes les parametres
de l'attaque d'entrainement (pas alpha et nombre de pas K), on fabrique pour
CHAQUE BATCH l'attaque la plus utile a l'etat courant du modele.

Ce que la loi du budget de deplacement (memoire.md, 2026-09-12) impose :

    ce n'est pas la FORCE de l'attaque interne qui fait apprendre, c'est la
    LISIBILITE de la perturbation qu'elle fabrique. Un budget <= 2 eps apprend
    (92% sous PGD-20), un budget >= 5 eps verrouille l'entrainement (CE adverse
    = ln(10), le modele repond uniformement).

Donc l'objectif n'est PAS "l'attaque la plus forte possible" (max de la perte
adverse) mais "une difficulte CIBLE" : on cherche l'attaque la plus courte qui
met le modele en difficulte d'une quantite voulue. C'est un ASSERVISSEMENT, pas
une maximisation.

Implementation (et c'est la partie interessante pour le cout) :

    la trajectoire de PGD contient deja toutes les attaques qu'on veut tester.
    En gardant un pas FIXE (alpha = eps/10) et en n'arretenant l'attaque qu'a
    un pas k*, le budget vaut k* x alpha : choisir k* dans {0..K} revient donc
    a choisir un budget dans {0, eps/10, 2 eps/10, ..., 2 eps}. Tester toutes
    ces configurations ne coute RIEN de plus : la difficulte du point courant
    est lue sur les logits deja calcules pour le gradient.

    Cout total = exactement celui de pgd() aujourd'hui (K pas avant+arriere),
    plus la memoire des K iterats (K x batch x 784 flottants, ~16 Mo a K=20
    et batch=256). On ne paie pas un facteur K, on paie le meme prix et on
    choisit a posteriori.

La difficulte est mesuree par deux indicateurs, lus sur le batch :
  - "tromperie" : part des images dont la prediction change (argmax != y),
    objectif le plus proche de ce qu'on appelle robustesse ;
  - "ce" : la cross-entropy adverse moyenne, plus lisse (proche de ln(10)=2.30
    quand le modele a abandonne).

Regle de selection : le plus PETIT k tel que difficulte(k) >= cible (premier
passage au-dessus de la cible, donc la perturbation la plus courte qui suffit).
Si la cible n'est jamais atteinte en K pas, on renvoie le dernier point et on
le signale (`plafond`) : c'est le signal que le modele a depasse le budget
disponible, et la reponse a ce signal n'est PAS d'augmenter la force (interdit
par la loi) mais d'augmenter la DIVERSITE (EOT, plusieurs departs, attaque par
scores) -- voir attaque_adaptative.md.
"""

import torch
import torch.nn.functional as F


def cible_effective(cible, epoch, epochs, rampe=0.5, depart=0.2):
    """Cible de difficulte avec montee en puissance (warmup).

    Les premieres epochs, le modele est mauvais : lui demander d'atteindre 50%
    de tromperie des le batch 1 donnerait des perturbations enormes sur un
    modele qui se trompe deja tout seul. On part donc d'une cible basse
    (`depart`) et on rejoint `cible` sur `rampe` de la duree du run.

    rampe = 0 desactive le warmup (cible constante).
    """
    if not rampe:
        return cible
    progres = min(1.0, max(0.0, epoch / max(1.0, epochs * rampe)))
    return depart + (cible - depart) * progres


def pgd_bande(modele, x, y, eps, steps=20, alpha=None, cible=0.5,
              cible_type="tromperie"):
    """PGD L-inf non ciblee, arretee au pas k* ou la difficulte atteint `cible`.

    Renvoie `(x_adv, info)` ou `info` contient le pas choisi, le budget
    correspondant, la trajectoire de difficulte et le drapeau `plafond`.

    - `cible` : difficulte visee (si None ou <= 0 : comportement de pgd(),
      on renvoie simplement le dernier point -> sert de reference A0) ;
    - `cible_type` : "tromperie" (0-1) ou "ce" (en nats, ln(10) = 2.303) ;
    - `alpha` : par defaut eps/10 (pas fin = le sweet spot mesure) ;
    - le depart aleatoire est TIRE FRAIS a chaque appel (correctif du
      2026-09-12 : un depart fixe faisait de l'entrainement sur du bruit).

    Attention : contrairement a pgd(), le point renvoye n'est PAS le dernier
    point de la trajectoire mais l'iterat k*, memorise. Les K iterats sont
    gardes en memoire (detaches), d'ou le surcout memoire indique en tete.
    """
    if alpha is None:
        alpha = eps / 10.0
    # Garde-fou : sous no_grad() les logits n'ont pas de grad_fn et autograd
    # echoue avec "element 0 of tensors does not require grad". Le message
    # brut de PyTorch ne dit pas d'ou vient l'erreur, celui-ci si.
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "pgd_bande a ete appele sous torch.no_grad() : l'attaque a besoin du "
            "graphe pour deriver par rapport a l'image. Sortir du bloc no_grad() "
            "(les mesures de difficulte sont deja protegees en interne).")
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)

    iterats = [x_adv]          # iterats[k] = le point apres k pas de PGD
    ce_traj = []
    tromperie_traj = []

    for _ in range(steps):
        # Un seul passage avant+arriere par pas, comme pgd(). On garde les
        # logits pour lire la difficulte gratuitement (ils servent deja au
        # gradient), ce qui evite une passe avant supplementaire.
        x_req = x_adv.detach().requires_grad_(True)
        logits = modele(x_req)
        perte = F.cross_entropy(logits, y, reduction="sum")
        (g,) = torch.autograd.grad(perte, x_req)

        with torch.no_grad():
            ce_traj.append(F.cross_entropy(logits, y).item())
            tromperie_traj.append((logits.argmax(1) != y).float().mean().item())

        x_adv = (x_req.detach() + alpha * g.sign())
        x_adv = x_adv.clamp(x - eps, x + eps).clamp(0.0, 1.0)
        iterats.append(x_adv)

    # La trajectoire mesuree porte sur les points 0..steps-1 (l'iterat `steps`
    # n'a pas ete evalue : on ne fait pas de passe avant en plus pour lui).
    trajectoire = ce_traj if cible_type == "ce" else tromperie_traj

    k_choisi = steps
    plafond = False
    if cible is not None and cible > 0:
        atteint = [t for t, d in enumerate(trajectoire) if d >= cible]
        if atteint:
            k_choisi = atteint[0]
        else:
            plafond = True      # la cible n'est pas atteinte au budget maximal

    info = {
        "k": k_choisi,
        "steps": steps,
        "alpha": alpha,
        "budget": k_choisi * alpha,
        "budget_eps": (k_choisi * alpha) / eps if eps else 0.0,
        "cible": cible,
        "cible_type": cible_type,
        "plafond": plafond,
        "ce_trajectoire": ce_traj,
        "tromperie_trajectoire": tromperie_traj,
        "difficulte_choisie": trajectoire[k_choisi] if k_choisi < steps else None,
    }
    return iterats[k_choisi], info


class JournalBande:
    """Cumule les statistiques de l'attaque adaptative sur une epoch.

    Sert a la ligne d'epoch (pas de suivie, part du budget au plafond) et a
    l'alerte "le modele a depasse le budget" : si une grande part des batchs
    finit au plafond, la reponse est d'escalader la diversite de l'attaque, pas
    sa force (loi du budget de deplacement).
    """

    def __init__(self):
        self.n_batch = 0
        self.k_som = 0.0
        self.budget_som = 0.0
        self.n_plafond = 0
        self.difficulte_som = 0.0
        self.n_difficulte = 0

    def ajouter(self, info):
        self.n_batch += 1
        self.k_som += info["k"]
        self.budget_som += info["budget_eps"]
        if info["plafond"]:
            self.n_plafond += 1
        if info["difficulte_choisie"] is not None:
            self.difficulte_som += info["difficulte_choisie"]
            self.n_difficulte += 1

    def resume(self):
        if not self.n_batch:
            return ""
        k_moy = self.k_som / self.n_batch
        budget_moy = self.budget_som / self.n_batch
        part_plafond = self.n_plafond / self.n_batch
        txt = (f"k* {k_moy:4.1f} pas (budget {budget_moy:4.2f} eps) "
               f"| plafond {part_plafond:4.0%}")
        if self.n_difficulte:
            txt += f" | difficulte {self.difficulte_som / self.n_difficulte:.2f}"
        return txt

    def alerte(self, seuil=0.5):
        if self.n_batch and self.n_plafond / self.n_batch >= seuil:
            return ("[ALERTE] la cible n'est atteinte qu'au plafond sur "
                    f"{self.n_plafond / self.n_batch:.0%} des batchs : le modele a "
                    "depasse le budget. Escalader la DIVERSITE de l'attaque "
                    "(EOT, departs multiples, attaque par scores), PAS sa force.")
        return None
