# 🥷 Adversarial Attacks — Attaquer (et défendre) mon CNN from scratch

> **Objectif :** apprendre la sécurité des modèles en attaquant mon propre CNN.
> Je contrôle le gradient de A à Z (aucun framework) → je peux implémenter les attaques moi-même.
>
> **Documentation soignée** — chaque script est commenté, chaque résultat est expliqué,
> les expériences sont tracées dans `memoire.md`. Pas de brouillon. 🧹

---

## 📁 Organisation

```
adversarial/
├── README.md          ← ce fichier : vue d'ensemble, concepts, résultats clés
├── memoire.md         ← carnet de bord : chaque expérience tracée (date, paramètres, résultat)
├── scripts/
│   ├── fgsm.py        ← attaque FGSM (1 étape de gradient)        ✅ opérationnel
│   ├── pgd.py         ← attaque PGD (itérative, plus forte)       ✅ opérationnel
│   ├── transfer.py    ← transfert d'attaque entre modèles     [à faire]
│   └── defend.py      ← adversarial training (défense)        [à faire]
└── results/           ← images + chiffres générés par les scripts (gitignoré)
```

## 🧠 Les concepts (à maîtriser)

### Évasion adversarial (adversarial examples)

Un **exemple adversarial** est une entrée modifiée de façon **imperceptible**
(bruit de quelques millièmes) qui fait se tromper le modèle avec haute confiance.

```
image originale          bruit (×10 grossi)          image attaquée
    ┌─────┐                   ┌─────┐                   ┌─────┐
    │ 'a' │       +    ε·sign(∇)  │     │       =        │ 'h' │  (identique à l'œil)
    └─────┘                   └─────┘                   └─────┘
```

### FGSM — Fast Gradient Sign Method (Goodfellow, 2014)

L'attaque fondatrice. Une seule étape :

```
x_adv = x + ε · sign(∇_x L(f(x), y))
```

- `∇_x L` : gradient de la loss **par rapport à l'entrée** (ce que mon backward retourne)
- `sign()` : on ne garde que la direction (+1/-1 par pixel)
- `ε` (epsilon) : l'amplitude du bruit — plus c'est grand, plus l'attaque est forte (et visible)
- `x_adv = clip(x_adv, 0, 1)` : on reste dans l'espace image valide

**Pourquoi ça marche :** le modèle est linéaire par morceaux ; une toute petite poussée
dans la direction du gradient cumule des effets sur toutes les dimensions et fait
basculer la sortie. C'est le "high-dimensional linearity" de Goodfellow.

### PGD — Projected Gradient Descent (Madry, 2018) ✅ implémenté

Version itérative de FGSM : plusieurs petites étapes avec projection dans la boule
L∞ de rayon ε. Attaque plus forte (le "gold standard" des attaques).

```
x_0 = x + U(-ε, ε)                       # démarrage aléatoire
x_{t+1} = clip(x_t + α·sign(∇L), x-ε, x+ε)  # pas de gradient projeté
```

### Transfert d'attaque *[à faire]*

Un exemple adversarial généré contre MON modèle trompe aussi d'autres modèles.
C'est ce qui rend les attaques dangereuses en pratique (attaques boîte noire).

### La défense : adversarial training *[à faire]*

Réentraîner le modèle **avec** des exemples adverses → il devient robuste.
C'est le pendant défensif — indispensable pour raconter les deux côtés.

---

## 🚀 Lancer une attaque

```bash
# Attaque FGSM sur le modèle MNIST complet
python3 adversarial/scripts/fgsm.py --dataset mnist --weights models/model_weights_full.npz

# Attaque PGD sur MNIST (20 itérations, démarrage aléatoire)
python3 adversarial/scripts/pgd.py --dataset mnist --weights models/model_weights_full.npz

# Comparaison directe FGSM vs PGD sur les mêmes images
python3 adversarial/scripts/pgd.py --compare

# Sur le modèle EMNIST letters
python3 adversarial/scripts/fgsm.py --dataset emnist --weights models/emnist_letters_weights.npz
python3 adversarial/scripts/pgd.py --dataset emnist --weights models/emnist_letters_weights.npz

# Attaque ciblée (forcer la prédiction vers une classe précise)
python3 adversarial/scripts/fgsm.py --targeted --target 3
python3 adversarial/scripts/pgd.py --targeted --target 3 --steps 40
```

Résultats dans `adversarial/results/` : images comparatives + résumé chiffré.

---

## 📊 Résultats clés (mis à jour à chaque expérience)

### FGSM — MNIST (models/model_weights_full.npz, 1000 images)

| ε | Acc attaqué | Flip | Observation |
|---|---|---|---|
| 0.05 | 94.0% | 4.6% | bruit invisible, déjà -4.5 pts |
| 0.10 | 76.1% | 22.6% | 1 image sur 4 change de prédiction |
| 0.20 | 21.9% | 76.9% | effondrement |
| 0.30 | 2.1% | 96.7% | le modèle ne reconnaît presque plus rien |

> **Propre : 98.5%** — une seule étape de gradient suffit à détruire le modèle.

### FGSM — EMNIST letters (models/emnist_letters_weights.npz, 500 images)

| ε | Acc attaqué | Flip | Observation |
|---|---|---|---|
| 0.05 | 28.6% | 19.2% | |
| 0.10 | 16.2% | 34.8% | |
| 0.20 | 6.2% | 49.8% | |
| 0.30 | 2.8% | 55.8% | |

> **Propre : 44.8%** (modèle rapide 5000 images — le full n'a pas abouti, relancé).

### PGD vs FGSM — MNIST (à compléter après le run)

| Attaque | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|---|
| FGSM | 94.0% | 76.1% | 21.9% | 2.1% |
| PGD (20 steps) | — | — | — | — |

---

## 🔗 Références

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014)
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018)
- MITRE ATLAS : atlas.mitre.org (les attaques IA côté défenseur)
