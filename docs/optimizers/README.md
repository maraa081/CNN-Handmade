# 📚 Fiches Optimiseurs — CNN-Handmade

> Fiches de référence pour chaque optimiseur implémenté.
> Quand tu ajoutes un nouvel optimiseur, copie `TEMPLATE.md` et remplis-le.

## Index

| Optimiseur | Fiche | Statut |
|---|---|---|
| **SGD** — Stochastic Gradient Descent | [`sgd.md`](sgd.md) | ✅ |
| **Momentum** — SGD avec élan | [`momentum.md`](momentum.md) | ✅ |
| **Adam** — Adaptive Moment Estimation | [`adam.md`](adam.md) | ✅ |

---

## Comparaison rapide

| Critère | SGD | Momentum | Adam |
|---|---|---|---|
| Learning rate typique | 0.01 | 0.01 | 0.001 |
| Hyperparams supplémentaires | — | `momentum=0.9` | `β₁=0.9, β₂=0.999, ε=1e-8` |
| Adaptatif (lr par paramètre) | ❌ | ❌ | ✅ |
| Lisse les oscillations | ❌ | ✅ (élan) | ✅ (mom + RMS) |
| Convergence | Lente | Plus rapide | Rapide |
| Robuste au tuning | ❌ (sensible) | Moyen | ✅ |
| Coût mémoire | 1× (les poids) | 2× (poids + vitesse) | 3× (poids + 2 moments) |

---

## Template pour les nouveaux optimiseurs

Quand tu implémentes un nouvel optimiseur (ex: RMSprop, Nadam, Lookahead…) :

1. Copie [`TEMPLATE.md`](TEMPLATE.md)
2. Renomme-le (`rmsprop.md`, `nadam.md`, …)
3. Remplis chaque section
4. Ajoute-le dans la table du `README.md` ci-dessus
5. Encode-le dans `src/optimizers.py`
