# 🔧 Momentum — SGD avec élan

> SGD avec de la mémoire : au lieu de suivre chaque gradient local, on accumule
> une « vitesse » dans la direction générale. Plus rapide, moins d'oscillations.

## 📐 Formule de mise à jour

```
v ← α · v - lr · ∇θ
θ ← θ + v
```

Où :
- `v` = vitesse (accumulation des gradients passés)
- `α` = coefficient d'élan (momentum)
- `lr` = learning rate
- `∇θ` = gradient courant

### Version avec weight_decay (L2 régularisation)

```
∇θ_effectif = ∇θ + weight_decay · θ
v ← α · v - lr · ∇θ_effectif
θ ← θ + v
```

## 🧠 Intuition

Reprends l'image de la bille qui descend la vallée.

SGD, c'est la bille qui s'arrête à chaque instant pour regarder la pente à ses pieds et faire un pas — aucun souvenir de la direction qu'elle prenait avant.

**Momentum**, c'est la bille qui a une **inertie**. Elle accumule la direction des pas précédents dans une « vitesse ». Si elle va vers la droite depuis 3 pas, elle continue à droite même si localement la pente dit « un peu à gauche ». Ça :

1. **Lisse les oscillations** (utile dans les ravins)
2. **Accélère dans les directions stables** (gain de vitesse)
3. **Traverse les plateaux** (l'élan la porte au-delà des zones plates)

Avec `α=0.9`, la vitesse se souvient d'environ 10 pas en arrière (décroissance exponentielle).

## ⚙️ Hyperparamètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `lr` | 0.01 | Learning rate de base (même ordre de grandeur que SGD) |
| `momentum (α)` | 0.9 | Coefficient d'élan. 0 = SGD vanilla, 0.9 = standard, 0.99 = très élan |
| `weight_decay` | 0.0 | L2 régularisation (inchangée par rapport à SGD) |

## 🔄 Code flow

```python
# Initialisation (1ère update) : création des buffers de vitesse
# v est un dict : {"conv_0_k": np.zeros_like(kernels), "conv_0_b": np.zeros_like(bias), …}

# À chaque update :
for i, layer in enumerate(layers):
    if Conv2D:
        # ---- kernels ----
        grad = layer.d_kernels + wd * layer.kernels
        v[conv_i_k] = α * v[conv_i_k] - lr * grad
        layer.kernels += v[conv_i_k]          # ← on ADDITIONNE v, on ne soustrait pas lr·grad

        # ---- bias ----
        grad = layer.d_bias + wd * layer.bias
        v[conv_i_b] = α * v[conv_i_b] - lr * grad
        layer.bias += v[conv_i_b]

    if Dense:  # même logique pour W et b
        ...
```

### Points clés

- **Mémoire supplémentaire :** on stocke `v` pour chaque paramètre (≈ 2× les poids en RAM)
- **Clés uniques :** `f'conv_{i}_kernels'`, `f'conv_{i}_bias'`, `f'dense_{i}_W'`, `f'dense_{i}_b'` — basées sur l'indice `i` de la couche dans la liste
- **Pas de mise à jour pour ReLU, MaxPool, Flatten, Dropout :** pas de paramètres
- **Attention aux shapes Collision :** un Dense(3136→128) a W.shape=(128,3136), un Conv2D(32→64,3×3) a kernels.shape=(64,32,3,3). Les clés uniques par `{type}_{i}_{param}` évitent les collisions
- **reset() :** `self.v = {}` à appeler entre deux entraînements

## 💡 Quand l'utiliser

- Tu veux converger plus vite que SGD
- Ta loss oscille beaucoup (courbes en dents de scie)
- Tu as un ravin (une direction de courbure faible vs une forte)
- Compromis simple entre SGD et Adam

## ✅ Avantages

- **Converge plus vite** que SGD
- **Réduit les oscillations** dans les ravins
- **1 seul hyperparamètre supplémentaire** (momentum)
- Toujours aussi simple à implémenter

## ❌ Inconvénients

- **Toujours pas de lr adaptatif** — même lr pour tous les paramètres
- **Pas de correction pour les plateaux** — si le momentum n'est pas suffisant, ça stagne
- **Peut overshoot** — si l'élan est trop fort, la bille dépasse le minimum
- **Mémoire doublée** (poids + vitesses)

## 📊 Résultats CNN-Handmade

(À remplir après les expériences)

## 📁 Fichier source

- `src/optimizers.py` — classe `Momentum`
