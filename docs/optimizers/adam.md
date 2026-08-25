#  Adam — Adaptive Moment Estimation

> Le couteau suisse des optimiseurs. Combine Momentum (vitesse) +
> RMSprop (lr adaptatif par paramètre). En pratique : ça marche
> sans rien tuner, dans 90% des cas.

##  Formule de mise à jour

```
m <- β₁ · m + (1 - β₁) · g           <- 1er moment (moyenne des gradients)
v <- β₂ · v + (1 - β₂) · g²          <- 2e moment (variance des gradients)

m <- m / (1 - β₁ᵗ)                    <- correction de biais
v <- v / (1 - β₂ᵗ)

θ <- θ - lr · m / (√v + ε)
```

Où :
- `g` = gradient courant (`∇θ + weight_decay·θ` si L2)
- `t` = pas de temps (numéro de l'update)
- `β₁` = decay rate du 1er moment
- `β₂` = decay rate du 2e moment
- `ε` = petit epsilon de stabilité (évite division par zéro)

### Version avec weight_decay

```
g = ∇θ + weight_decay · θ
m <- β₁ · m + (1 - β₁) · g
v <- β₂ · v + (1 - β₂) · g²
… (suite inchangée)
```

##  Intuition

Adam fait deux choses en même temps :

1. **Comme Momentum** : il accumule la direction moyenne des gradients (`m`). Si on va toujours dans la même direction, on accélère. Si le gradient change de signe à chaque pas, `m` reste petit -> pas de sur-réaction.

2. **Comme RMSprop** : il suit la variance des gradients (`v`). Si un paramètre a des gradients très variables (grande variance), `v` est grand -> `lr / √v` réduit le pas. Si un paramètre a des gradients stables, `v` est petit -> le pas est plus grand.

Résultat : **chaque paramètre a son propre learning rate adaptatif**. Les paramètres qui oscillent apprennent doucement, ceux qui sont stables apprennent vite.

**La correction de biais** (`m`, `v`) corrige le fait qu'au début, `m` et `v` sont initialisés à zéro et mettent du temps à « chauffer ». Sans correction, les premières mises à jour seraient trop petites.

##  Hyperparamètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `lr` | **0.001** | Learning rate. **Plus bas que SGD/Momentum** (0.01 -> divergence fréquente). |
| `beta1 (β₁)` | 0.9 | Décroissance de la moyenne des gradients. 0.9 ≈ regarde ~10 pas en arrière. |
| `beta2 (β₂)` | 0.999 | Décroissance de la variance. 0.999 ≈ regarde ~1000 pas en arrière. |
| `eps (ε)` | 1e-8 | Stabilité numérique. Empêche la division par zéro quand `v` ≈ 0. |
| `weight_decay` | 0.0 | L2 régularisation identique aux autres optimiseurs. |

[warn] **Si Adam diverge avec lr=0.001**, essaye 0.0001 plutôt que de toucher aux betas.

##  Code flow

```python
# Initialisation (1ère update) : buffers pour chaque paramètre
# m : {"conv_0_k": np.zeros_like(k), "conv_0_b": np.zeros_like(b), …}
# v : {"conv_0_k": np.zeros_like(k), …}
# t : 0

# À chaque update :
t += 1

for i, layer in enumerate(layers):
    if Conv2D:
        g_k = layer.d_kernels + wd * layer.kernels
        g_b = layer.d_bias    + wd * layer.bias

        # Mise à jour des moments
        m[k] = β₁·m[k] + (1-β₁)·g_k
        v[k] = β₂·v[k] + (1-β₂)·g_k²      # g_k² = element-wise square

        # Correction de biais
        m_hat = m[k] / (1 - β₁ᵗ)
        v_hat = v[k] / (1 - β₂ᵗ)

        # Mise à jour
        layer.kernels -= lr · m_hat / (√v_hat + ε)
        # (pareil pour bias)

    if Dense:  # même logique pour W et b
        ...
```

### Points clés

- **3× la mémoire** des poids : chaque paramètre a `m` + `v` en plus (vs 2× pour Momentum, 1× pour SGD)
- **La puissance `t`** : `β₁ᵗ` et `β₂ᵗ` − avec `t` qui grandit, `1 - βᵗ` -> 1, la correction de biais devient négligeable
- **Même système de clés uniques** que Momentum : `f'conv_{i}_k'`, `f'dense_{i}_b'`, etc.
- **La division par `√v + ε`** est élément-wise : chaque coefficient du paramètre a son lr effectif
- **reset()** réinitialise `m`, `v`, **et `t`**

##  Quand l'utiliser

- **Par défaut** — si tu ne sais pas quel optimiseur choisir, prends Adam
- Tu n'as pas le temps de tuner les hyperparams
- Tes gradients sont de magnitudes très variables selon les couches
- Problème complexe avec beaucoup de paramètres

## OK Avantages

- **Learning rate adaptatif par paramètre** — chaque poids a son propre pas
- **Robuste** — fonctionne bien avec les valeurs par défaut dans la plupart des cas
- **Momentum + RMSprop combinés** pour le meilleur des deux mondes
- **Bon compromis** entre vitesse de convergence et stabilité
- **Référence** dans la littérature (Kingma & Ba, 2015) — l'optimiseur le plus utilisé

## FAIL Inconvénients

- **Mémoire : 3× les poids** (contre 1× pour SGD)
- **Ne généralise pas toujours aussi bien** que SGD avec un bon scheduler
- **Hyperparamètres en plus** : betas et eps ; même si les défauts marchent, les comprendre aide au debug
- **Peut trop réduire le lr** pour certains paramètres si `v` devient très grand — problème résolu par AdamW
- **Correction de biais nécessaire** rend l'implémentation légèrement moins triviale

##  Détail : pourquoi `β₂=0.999` ?

`β₂` contrôle la fenêtre de la variance. Avec 0.999, il faut `t ≈ 1000` pas pour que `v` oublie complètement un gradient passé.

Ça veut dire que si la variance d'un paramètre était grande il y a 500 pas mais devient petite, Adam mettra du temps à « lâcher » le fait que ce paramètre oscillait — donc à augmenter son lr.

##  Résultats CNN-Handmade

(À remplir après les expériences — comparaison SGD/Momentum/Adam)

##  Fichier source

- `src/optimizers.py` — classe `Adam`
