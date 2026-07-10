# 📋 Template — Nouvel optimiseur

> Copie ce fichier, renomme-le, et remplis chaque section.
> Conserve les emojis et la structure pour avoir des fiches homogènes.

```markdown
# 🔧 NOM_OPTIMISEUR — Description courte

> Petite phrase d'accroche — pourquoi cet optimiseur existe, quel problème il résout.

## 📐 Formule de mise à jour

```
θ ← …   (formule complète)
```

### Version avec weight_decay (L2)

```
θ ← …   (idem avec le terme de régularisation)
```

## 🧠 Intuition

> 2-3 phrases qui expliquent COMMENT ça marche, sans maths.
> L'image mentale que tu veux garder en tête.

## ⚙️ Hyperparamètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `lr` | — | … |
| … | … | … |

## 🔄 Code flow

```
1. Pour chaque couche paramétrée (Conv2D, Dense) :
   a. Récupérer le gradient stocké (d_kernels, dW, etc.)
   b. Ajouter le weight_decay si présent
   c. Calculer la mise à jour avec la formule de l'optimiseur
   d. Appliquer la mise à jour au paramètre
```

### Points clés dans le code

- **Buffers internes :** où sont stockés les états (vitesse, moments…) ?
- **Initialisation :** quand et comment sont créés les buffers ?
- **Reset :** que faut-il réinitialiser entre deux entraînements ?

## 💡 Quand l'utiliser

- …
- …

## ✅ Avantages

- …

## ❌ Inconvénients

- …

## 📊 Résultats CNN-Handmade

(À remplir après expérience — lien vers le graphique, accuracy, etc.)

## 📁 Fichier source

- `src/optimizers.py` — classe `NomOptimiseur`
```
