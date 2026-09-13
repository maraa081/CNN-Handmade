#!/usr/bin/env python3
"""Genere les figures de l'article (adversarial/article/figures/).

Source unique des chiffres : `adversarial/memoire.md` et les JSON de
`adversarial/results/logs/`. Aucun chiffre n'est recopie a la main ailleurs
que dans le dictionnaire DONNEES ci-dessous, ou chaque entree porte son nom de
modele et la source (maison = notre suite 500 images, officiel = AutoAttack
10 000 images, eps=0.30).

Usage :
    python3 adversarial/article/figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLEU = "#2b6cb0"
ROUGE = "#c53030"
VERT = "#2f855a"
GRIS = "#718096"
ORANGE = "#c05621"

def _enregistrer(fig, nom):
    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, nom)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"[OK] {chemin}")


# ---------------------------------------------------------------------------
# Donnees (pire cas maison, 500 images, eps=0.30)
# ---------------------------------------------------------------------------

# Ablation du budget : budget de l'attaque interne en multiples de eps
CLOCHE = [
    ("run B\n(pas eps/4)", 1.25, 42.0),
    ("abl_b\n(pas eps/4)", 5.0, 38.8),
    ("A4\n(pas eps/10)", 0.5, 49.2),
    ("v4 / abl_a\n(pas eps/10)", 2.0, 63.2),
    ("abl_c\n(pas eps)", 20.0, 1.6),
]

# Plans : plan de budget, budget maximal, pire cas, budget de depart
PLANS_MNIST = [
    ("A8 : 0.2 -> 1 eps\n(2 -> 10 pas)", 1.0, 84.4, 0.2),
    ("A6 : 0.2 -> 2 eps\n(2 -> 20 pas)", 2.0, 85.8, 0.2),
    ("A7 : 1 -> 2 eps\n(10 -> 20 pas)", 2.0, 62.0, 1.0),
    ("A9 : 2 -> 0.2 eps\n(20 -> 2 pas)", 2.0, 63.8, 2.0),
]

# L'ordre : (croissant, decroissant) x (maison, officiel), par jeu
ORDRE = {
    "MNIST": {"croissant": (85.8, 82.40), "decroissant": (63.8, 50.35),
              "legende": ("A6 : plan 0.2 -> 2 eps", "A9 : plan 2 -> 0.2 eps")},
    "KMNIST": {"croissant": (58.4, 51.03), "decroissant": (17.0, 1.49),
               "legende": ("plan 0.2 -> 1 eps", "plan 1 -> 0.2 eps")},
}

# Nuage maison vs officiel : (nom, maison, officiel, famille)
NUAGE = [
    ("A1 (adaptatif)", 93.6, 91.25, "bas"),
    ("A6 (0.2->2)", 85.8, 82.40, "bas"),
    ("A8 (0.2->1)", 84.4, 78.25, "bas"),
    ("KMNIST (0.2->2)", 62.6, 58.53, "bas"),
    ("KMNIST (0.2->1)", 58.4, 51.03, "bas"),
    ("abl_a (constant 2 eps)", 63.2, 50.71, "haut"),
    ("A9 (2->0.2)", 63.8, 50.35, "haut"),
    ("KMNIST (1->0.2)", 17.0, 1.49, "haut"),
    ("TRADES beta 2", 27.2, 22.01, "autre"),
    ("TRADES beta 6", 30.0, 25.18, "autre"),
]

# Recette identique sur les deux jeux (officiel des deux cotes)
PAIRES = [
    ("plan 0.2 -> 2 eps", 82.40, 58.53),
    ("plan 0.2 -> 1 eps", 78.25, 51.03),
]


# ---------------------------------------------------------------------------
# FIGURE 1 : la courbe en cloche (le sommet est a l'interieur)
# ---------------------------------------------------------------------------

def figure_cloche():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    xs = [c[1] for c in CLOCHE]
    ys = [c[2] for c in CLOCHE]
    ax.plot(xs, ys, "o-", color=BLEU, linewidth=2, markersize=8,
            label="budget constant (ablation du pas)")
    for nom, x, y in CLOCHE:
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=BLEU)
        ax.annotate(nom, (x, y), textcoords="offset points",
                    xytext=(0, -26), ha="center", fontsize=7.5, color=GRIS)

    for nom, x, y, depart in PLANS_MNIST:
        couleur = VERT if depart <= 0.2 else ROUGE
        ax.plot([x], [y], "D", color=couleur, markersize=9, zorder=5)
        ax.annotate(f"{nom}\n{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(14, -4), ha="left", fontsize=7.5, color=couleur)

    ax.axvline(2.0, color=GRIS, linestyle=":", linewidth=1)
    ax.annotate("sommet : budget ~2 eps", (2.0, 12), rotation=90, fontsize=8,
                color=GRIS, ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "20"])
    ax.set_xlabel("budget de l'attaque interne (en multiples de eps)\n"
                  "vert : plan a depart bas -- rouge : plan a depart haut")
    ax.set_ylabel("pire cas (suite maison, 500 images, eps=0.30)")
    ax.set_ylim(0, 100)
    ax.set_title("FIGURE 1 -- La robustesse apprise est maximale pour un budget intermediaire\n"
                 "(MNIST, 421 642 parametres, 120 epochs)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    _enregistrer(fig, "fig1_cloche.png")


# ---------------------------------------------------------------------------
# FIGURE 2 : l'ordre (le test decisif)
# ---------------------------------------------------------------------------

def figure_ordre():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, jeu in zip(axes, ["MNIST", "KMNIST"]):
        d = ORDRE[jeu]
        groupes = ["suite maison\n(500 images)", "AutoAttack\n(10 000 images)"]
        x = [0, 1]
        larg = 0.36
        vals_c = [d["croissant"][0], d["croissant"][1]]
        vals_d = [d["decroissant"][0], d["decroissant"][1]]
        b1 = ax.bar([i - larg / 2 for i in x], vals_c, larg, color=BLEU,
                    label=d["legende"][0])
        b2 = ax.bar([i + larg / 2 for i in x], vals_d, larg, color=ROUGE,
                    label=d["legende"][1])
        for barres in (b1, b2):
            for b in barres:
                ax.annotate(f"{b.get_height():.1f}%",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=9)
        ecart_m = vals_c[0] - vals_d[0]
        ecart_o = vals_c[1] - vals_d[1]
        ax.annotate(f"ecart maison : {ecart_m:+.1f} pts",
                    (0, min(vals_d[0], 0) + 4), ha="center", fontsize=8.5, color=GRIS)
        ax.annotate(f"ecart OFFICIEL : {ecart_o:+.1f} pts",
                    (1, min(vals_d[1], 0) + 4), ha="center", fontsize=9,
                    color="black", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groupes, fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("pire cas a eps=0.30")
        ax.set_title(f"{jeu}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("FIGURE 2 -- Memes budgets, meme cout de calcul, ordre inverse : "
                 "le juge officiel AMPLIFIE l'ecart", y=1.02)
    _enregistrer(fig, "fig2_ordre.png")


# ---------------------------------------------------------------------------
# FIGURE 3 : maison contre officiel (le biais separe les familles)
# ---------------------------------------------------------------------------

def figure_nuage():
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 100], [0, 100], linestyle="--", color=GRIS, linewidth=1,
            label="diagonale (aucun biais)")
    styles = {"bas": (VERT, "depart bas (plan croissant)"),
              "haut": (ROUGE, "depart haut (constant ou plan inverse)"),
              "autre": (GRIS, "TRADES (variante non conforme)")}
    vus = set()
    for nom, maison, officiel, famille in NUAGE:
        couleur, etiquette = styles[famille]
        ax.plot([maison], [officiel], "o", color=couleur, markersize=8,
                label=etiquette if etiquette not in vus else None)
        vus.add(etiquette)
        ax.annotate(f"{nom}\n({maison:.1f} -> {officiel:.2f}, "
                    f"{officiel - maison:+.1f})",
                    (maison, officiel), textcoords="offset points",
                    xytext=(-8, 8), ha="right" if maison > 40 else "left",
                    fontsize=7.5, color=couleur)
    ax.set_xlabel("pire cas maison (500 images)")
    ax.set_ylabel("pire cas officiel (AutoAttack, 10 000 images)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("FIGURE 3 -- Le biais de la suite maison separe les deux familles\n"
                 "depart bas : -2.4 a -7.4 pts -- depart haut : -12.5 a -15.5 pts")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    _enregistrer(fig, "fig3_maison_vs_officiel.png")


# ---------------------------------------------------------------------------
# FIGURE 4 : la recette se transpose, le niveau non
# ---------------------------------------------------------------------------

def figure_transposition():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = range(len(PAIRES))
    larg = 0.36
    m = [p[1] for p in PAIRES]
    k = [p[2] for p in PAIRES]
    b1 = ax.bar([i - larg / 2 for i in x], m, larg, color=BLEU, label="MNIST")
    b2 = ax.bar([i + larg / 2 for i in x], k, larg, color=ORANGE, label="KMNIST")
    for barres in (b1, b2):
        for b in barres:
            ax.annotate(f"{b.get_height():.2f}%",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10)
    for i, (nom, mn, km) in enumerate(PAIRES):
        ax.annotate(f"{km - mn:+.1f} pts", (i, max(mn, km) + 9), ha="center",
                    fontsize=11, fontweight="bold", color=ROUGE)
    ax.set_xticks(list(x))
    ax.set_xticklabels([p[0] for p in PAIRES])
    ax.set_ylim(0, 100)
    ax.set_ylabel("pire cas OFFICIEL (eps=0.30, 10 000 images)")
    ax.set_title("FIGURE 4 -- A recette identique, l'ordre des recettes est conserve\n"
                 "et le niveau chute d'environ 25 points : la recette se transpose, le niveau non")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    _enregistrer(fig, "fig4_transposition.png")


if __name__ == "__main__":
    figure_cloche()
    figure_ordre()
    figure_nuage()
    figure_transposition()
    print("[FIN] figures dans adversarial/article/figures/")
