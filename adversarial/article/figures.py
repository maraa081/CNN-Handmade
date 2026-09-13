#!/usr/bin/env python3
"""Figures for the write-up (adversarial/article/figures/).

Design rules (demande Maraa, 13/09) :
  - English labels (the talk and the English version use the same figures),
  - as simple as possible: few words on the canvas, numbers live in the legend
    or in axis labels, never in overlapping text blocks,
  - every number comes from THIS file, with its model name and its source
    (in-house suite = 500 images ; official = AutoAttack, 10 000 images, eps=0.30).

Usage:
    python3 adversarial/article/figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLEU = "#2b6cb0"
VERT = "#2f855a"
ROUGE = "#c53030"
ORANGE = "#c05621"
GRIS = "#4a5568"


def _enregistrer(fig, nom):
    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, nom)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160)
    plt.close(fig)
    print(f"[OK] {chemin}")


# ---------------------------------------------------------------------------
# Data. worst case = accuracy under the strongest attack found (in-house, 500
# images, eps=0.30) ; official = AutoAttack, 10 000 images, eps=0.30.
# ---------------------------------------------------------------------------

# Constant-budget ablation: (budget in multiples of eps, worst case)
CONSTANTS = [(0.5, 49.2), (1.25, 42.0), (2.0, 63.2), (5.0, 38.8), (20.0, 1.6)]

# Budget plans: (label, start budget, end budget, worst case)
# A7 (1 -> 2 eps, 62.0%) is left out of the figure to keep it readable: it tells
# the same story as A9 (starting high is what hurts) and stays in the tables.
PLANS = [
    ("A6: 0.2 -> 2", 0.2, 2.0, 85.8),
    ("A9: 2 -> 0.2", 2.0, 0.2, 63.8),
]

# Order test: increasing vs decreasing, per dataset: (in-house, official)
ORDRE = {
    "MNIST": ("A6: 0.2 -> 2 eps", "A9: 2 -> 0.2 eps", (85.8, 82.40), (63.8, 50.35)),
    "KMNIST": ("0.2 -> 1 eps", "1 -> 0.2 eps", (58.4, 51.03), (17.0, 1.49)),
}

# In-house vs official, with family: (short label, in-house, official, family)
NUAGE = [
    ("A1 (adaptive)", 93.6, 91.25, "low"),
    ("A6 (0.2 -> 2)", 85.8, 82.40, "low"),
    ("A8 (0.2 -> 1)", 84.4, 78.25, "low"),
    ("KMNIST (0.2 -> 2)", 62.6, 58.53, "low"),
    ("KMNIST (0.2 -> 1)", 58.4, 51.03, "low"),
    ("abl_a (constant 2 eps)", 63.2, 50.71, "high"),
    ("A9 (2 -> 0.2)", 63.8, 50.35, "high"),
    ("KMNIST (1 -> 0.2)", 17.0, 1.49, "high"),
    ("TRADES beta 2", 27.2, 22.01, "other"),
    ("TRADES beta 6", 30.0, 25.18, "other"),
]

# Same recipe, two datasets (official both sides)
PAIRES = [("0.2 -> 2 eps", 82.40, 58.53), ("0.2 -> 1 eps", 78.25, 51.03)]


# ---------------------------------------------------------------------------
# FIGURE 1 -- the bell: worst case vs inner-attack budget
# ---------------------------------------------------------------------------

def figure_cloche():
    fig, ax = plt.subplots(figsize=(9, 6))

    xs = [c[0] for c in CONSTANTS]
    ys = [c[1] for c in CONSTANTS]
    ax.plot(xs, ys, "o-", color=BLEU, linewidth=2.5, markersize=10,
            label="constant budget (PGD-20)")

    for libelle, deb, fin, val in PLANS:
        couleur = VERT if deb < 0.5 else ROUGE
        ax.annotate("", xy=(fin, val), xytext=(deb, val),
                    arrowprops=dict(arrowstyle="-|>", color=couleur, linewidth=3))
        ax.plot([deb], [val], "o", color=couleur, markersize=7)
        ax.annotate(f"{libelle}   {val:.1f}%", xy=(max(deb, fin), val),
                    xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=11, color=couleur)

    ax.axvline(2.0, color=GRIS, linestyle=":", linewidth=1.2)
    ax.annotate("sweet spot", xy=(2.0, 95), color=GRIS, fontsize=11, ha="center")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "20"])
    ax.set_xlim(0.15, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("inner-attack budget (multiples of eps)")
    ax.set_ylabel("worst-case accuracy (%)")
    ax.set_title("MNIST: best robustness at an intermediate budget\n"
                 "arrows = budget plans (green: starting low)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=11)
    _enregistrer(fig, "fig1_budget_bell.png")


# ---------------------------------------------------------------------------
# FIGURE 2 -- order matters (increasing vs decreasing, same budgets, same cost)
# ---------------------------------------------------------------------------

def figure_ordre():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), sharey=True)
    for ax, jeu in zip(axes, ["MNIST", "KMNIST"]):
        lab_c, lab_d, vals_c, vals_d = ORDRE[jeu]
        x = [0, 1]
        larg = 0.34
        b1 = ax.bar([i - larg / 2 for i in x], list(vals_c), larg,
                    color=BLEU, label=f"increasing ({lab_c})")
        b2 = ax.bar([i + larg / 2 for i in x], list(vals_d), larg,
                    color=ROUGE, label=f"decreasing ({lab_d})")
        for barres in (b1, b2):
            for b in barres:
                ax.annotate(f"{b.get_height():.1f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["in-house\n(500 im.)", "AutoAttack\n(10 000 im.)"])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_title(f"{jeu}   (gap: {vals_c[0] - vals_d[0]:+.1f} in-house, "
                     f"{vals_c[1] - vals_d[1]:+.1f} official)")
    axes[0].set_ylabel("worst-case accuracy (%)")
    fig.suptitle("Same budgets, same training cost, reversed order", y=1.02, fontsize=13)
    _enregistrer(fig, "fig2_order.png")


# ---------------------------------------------------------------------------
# FIGURE 3 -- in-house vs official: the bias separates the two families
# ---------------------------------------------------------------------------

def figure_nuage():
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot([0, 100], [0, 100], "--", color=GRIS, linewidth=1.2, label="no bias")
    couleurs = {"low": VERT, "high": ROUGE, "other": GRIS}
    entrees = []
    for i, (nom, maison, officiel, famille) in enumerate(NUAGE, start=1):
        couleur = couleurs[famille]
        ax.plot([maison], [officiel], "o", color=couleur, markersize=10)
        ax.annotate(str(i), xy=(maison, officiel), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
        entrees.append((f"{i}. {nom}   {maison:.1f}  ->  {officiel:.2f}", couleur))
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("in-house worst case (%), 500 images")
    ax.set_ylabel("official worst case (%), AutoAttack, 10 000 images")
    ax.set_title("green: starts low (bias -2.4 to -7.4 pts)\n"
                 "red: starts high (bias -12.5 to -15.5 pts)", fontsize=12)
    ax.grid(alpha=0.3)
    lignes = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=8,
                         label=t) for t, c in entrees]
    ax.legend(handles=lignes, loc="upper left", bbox_to_anchor=(0.0, -0.16),
              ncol=2, fontsize=10, frameon=False)
    _enregistrer(fig, "fig3_inhouse_vs_official.png")


# ---------------------------------------------------------------------------
# FIGURE 4 -- recipe transfers, level does not
# ---------------------------------------------------------------------------

def figure_transposition():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = list(range(len(PAIRES)))
    larg = 0.34
    m = [p[1] for p in PAIRES]
    k = [p[2] for p in PAIRES]
    b1 = ax.bar([i - larg / 2 for i in x], m, larg, color=BLEU, label="MNIST")
    b2 = ax.bar([i + larg / 2 for i in x], k, larg, color=ORANGE, label="KMNIST")
    for barres in (b1, b2):
        for b in barres:
            ax.annotate(f"{b.get_height():.1f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=11)
    for i, (_, mn, km) in enumerate(PAIRES):
        ax.annotate(f"{km - mn:+.1f} pts", (i, max(mn, km) + 8), ha="center",
                    fontsize=12, fontweight="bold", color=ROUGE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"same recipe:\n{p[0]}" for p in PAIRES])
    ax.set_ylim(0, 100)
    ax.set_ylabel("official worst-case accuracy (%)")
    ax.set_title("Recipe transfers, level does not")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)
    _enregistrer(fig, "fig4_recipe_transfers.png")


if __name__ == "__main__":
    figure_cloche()
    figure_ordre()
    figure_nuage()
    figure_transposition()
    print("[FIN] figures dans adversarial/article/figures/")
