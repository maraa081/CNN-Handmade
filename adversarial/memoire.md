# 📓 Carnet de bord — Adversarial Attacks

> Journal des expériences. **Une entrée par expérience** : date, modèle, paramètres,
> résultat chiffré, observation. C'est ici que se construit la compréhension.
>
> Format d'entrée :
> ```markdown
> ### YYYY-MM-DD — <nom de l'expérience>
> - Modèle / poids : ...
> - Paramètres : ...
> - Résultat : ...
> - Observation / leçon : ...
> ```

---

## 🗓️ Journal

### 2026-08-24 — Préparation du terrain

- Structure `adversarial/` créée (README, scripts/, results/, ce carnet)
- `fgsm.py` écrit et testé — première attaque opérationnelle

### 2026-08-24 — FGSM non ciblée sur MNIST (modèle full)

- Modèle / poids : `model_weights_full.npz` (98.5% propre sur l'échantillon)
- Paramètres : `--n 1000`, eps ∈ {0.05, 0.1, 0.2, 0.3}, seed 42
- Résultat : l'accuracy s'effondre de 98.5% → 2.1% à ε=0.30 (flip 96.7%)
- Observation : la chute est **progressive** (94% → 76% → 22% → 2%) : FGSM
  est une attaque « de force », plus l'amplitude autorisée est grande,
  plus le modèle s'effondre. À ε=0.05 (bruit quasi invisible) on perd déjà 4.5 pts.
- Leçon : mon CNN from scratch est **vulnérable** — il suffit d'un bruit de
  ±0.3/255 pour le faire tomber à ~2%. Conforme à la théorie (linearité
grandes dimensions, Goodfellow).

### 2026-08-24 — FGSM non ciblée sur EMNIST letters (modèle rapide)

- Modèle / poids : `emnist_letters_weights.npz` (entraînement rapide 5000 img,
  44.8% propre sur l'échantillon)
- Paramètres : `--n 500`, eps ∈ {0.05, 0.1, 0.2, 0.3}, seed 42
- Résultat : 44.8% → 2.8% à ε=0.30 (flip 55.8%)
- Observation : l'attaque fonctionne aussi sur 26 classes, mais la chute est
  moins spectaculaire en relatif car le modèle est déjà faible (44.8% propre).
  Le flip (changement de prédiction) reste massif : plus d'une image sur deux
  change de lettre à ε=0.30.
- ⚠️ Note : l'entraînement full EMNIST (124 800 images, ~2h30) lancé le
  2026-08-24 a échoué silencieusement (log vide, fichier `_full` absent).
  Relancé en fond le 2026-08-25 → les expériences de transfert attendront ce modèle.

---

## 📊 Tableau des résultats cumulés

| Date | Attaque | Modèle | ε | Acc propre | Acc attaqué | Flip | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-24 | FGSM untgt | MNIST full | 0.05 | 98.5% | 94.0% | 4.6% | chute déjà visible |
| 2026-08-24 | FGSM untgt | MNIST full | 0.10 | 98.5% | 76.1% | 22.6% | |
| 2026-08-24 | FGSM untgt | MNIST full | 0.20 | 98.5% | 21.9% | 76.9% | |
| 2026-08-24 | FGSM untgt | MNIST full | 0.30 | 98.5% | 2.1% | 96.7% | effondrement |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.05 | 44.8% | 28.6% | 19.2% | |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.10 | 44.8% | 16.2% | 34.8% | |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.20 | 44.8% | 6.2% | 49.8% | |
| 2026-08-24 | FGSM untgt | EMNIST rapide | 0.30 | 44.8% | 2.8% | 55.8% | |
