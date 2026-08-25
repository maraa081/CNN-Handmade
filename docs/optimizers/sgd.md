#  SGD — Stochastic Gradient Descent (vanilla)

> L'optimiseur de base. Rien de plus que « suivre la pente locale ».
> Si t'as zéro idée de quel optimiseur choisir, tu commences par SGD.

##  Formule de mise à jour

```
θ <- θ - lr · ∇θ

θ  = paramètre (kernels, W, bias…)
lr = learning rate
∇θ = gradient du paramètre
```

### Version avec weight_decay (L2 régularisation)

```
θ <- θ - lr · (∇θ + weight_decay · θ)
```

Le terme `weight_decay · θ` pénalise les poids trop grands en les attirant vers zéro à chaque étape. C'est une régularisation qui lutte contre l'overfitting.

##  Intuition

Imagine une bille qui descend une vallée (la loss). SGD la fait rouler en suivant la pente locale, d'un pas de taille fixe `lr`. Pas de mémoire, pas d'accélération — juste la pente du point où elle se trouve. Si la pente change brusquement, la bille change de direction aussi sec.

Ça marche, mais c'est lent et ça peut osciller autour du minimum (surtout si la loss a une forme allongée — ravin).

##  Hyperparamètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `lr` | 0.01 | Taille du pas de descente. Trop petit -> lent, trop grand -> diverge. |
| `weight_decay` | 0.0 | Force L2 qui ramène les poids vers zéro (régularisation). 0.001 est un bon début. |

##  Code flow

```python
for layer in layers:
    if layer a des kernels (Conv2D):
        # kernels : (C_out, C_in, k, k)
        layer.kernels -= lr * (layer.d_kernels + wd * layer.kernels)
        # bias    : (C_out,)
        layer.bias    -= lr * (layer.d_bias    + wd * layer.bias)

    if layer a des W (Dense):
        # W : (out_features, in_features)
        layer.W -= lr * (layer.dW + wd * layer.W)
        # b : (out_features,)
        layer.b -= lr * (layer.db + wd * layer.b)
```

### Points clés

- **Aucun état interne** — pas de buffers, pas de mémoire. C'est ce qui le rend si simple et léger en RAM.
- **Le `weight_decay` est appliqué ici**, pas dans la loss. C'est équivalent à ajouter `0.5 * wd * ||θ||²` à la loss et à dériver.
- Seules les couches **paramétrées** (Conv2D, Dense) sont mises à jour — ReLU, MaxPool, Flatten, Dropout n'ont pas de paramètres.

##  Quand l'utiliser

- Baseline / debug — tu veux vérifier que ta rétropropagation marche
- Problème simple avec peu de données
- Tu veux tuner finement le lr et tu as le temps

## OK Avantages

- **Le plus simple** à comprendre et implémenter
- **Zéro mémoire supplémentaire** — ne stocke que les poids
- **Un seul hyperparamètre** (lr) à tuner
- L'ajout de features (momentum, lr adaptatif) se fait en extension

## FAIL Inconvénients

- **Lent à converger** — pas d'accélération
- **Sensible au learning rate** — un mauvais lr et ça diverge ou ça stagne
- **Peut osciller** dans les ravins (directions de courbure différentes)
- **Pas de lr adaptatif** — même pas pour tous les paramètres
- **Peut rester coincé** dans des minima locaux ou des plateaux

##  Résultats CNN-Handmade

(À remplir après les expériences — lien vers le graphique, accuracy atteinte, etc.)

##  Fichier source

- `src/optimizers.py` — classe `SGD`
