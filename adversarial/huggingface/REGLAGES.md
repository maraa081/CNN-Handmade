# Publication Hugging Face : ce qu'il me faut de Maraa

Tout est ecrit et pret a partir (`pousser_hf.py`, carte du modele, espace de
demo). Ce qui manque n'est pas du code, ce sont quatre decisions.

## 1. Le compte et les noms de depots

Dans `pousser_hf.py`, bloc REGLAGES :

- `DEPOT_MODELE` (defaut propose : `maraa081/mnist-cnn-robuste`)
- `DEPOT_ESPACE` (defaut propose : `maraa081/mnist-robuste-demo`)

Si le pseudo Hugging Face n'est pas `maraa081`, il faut le corriger aux DEUX
endroits (le script et la ligne `MODELE_HF` dans `espace/app.py`).

## 2. Quels poids publier

`models/*.pt` est ignore par git : les poids ne peuvent pas vivre sur GitHub.
Le bloc `POIDS` de `pousser_hf.py` suppose ces cinq fichiers dans `models/` :

| nom dans le depot | fichier local attendu | quoi |
|---|---|---|
| `bande_cible50.pt` | `models/bande_cible50.pt` | champion (91.25% officiel) |
| `a6_plan_budget.pt` | `models/a6_plan_budget.pt` | plan `0.2 -> 2` (82.40%) |
| `a8_plan_0p2_1.pt` | `models/a8_plan_0p2_1.pt` | plan `0.2 -> 1` (78.25%) |
| `abl_a_eps10.pt` | `models/abl_a_eps10.pt` | constante 2 eps (50.71%) |
| `standard.pt` | `models/standard.pt` | temoin non defendu |

**A confirmer** : les noms exacts des deux premiers (le champion vient de
`--bande --cible 0.5`, le fichier a peut-etre un autre nom) et l'existence d'un
`standard.pt` (a defaut, `models/standard_same_data.npz` existe deja dans le
depot et peut servir de temoin, mais l'espace charge des `.pt`).

Trop de poids alourdit le depot sans rien prouver : si on veut se limiter a
trois fichiers, garder `bande_cible50.pt`, `a6_plan_budget.pt` et `standard.pt`
(retirer les autres du bloc `POIDS` ET de `FICHIERS` dans `espace/app.py`).

## 3. Le jeton

Jamais sur la ligne de commande (historique du shell) :

```bash
HF_TOKEN=hf_xxx python3 adversarial/huggingface/pousser_hf.py --envoyer
```

Jeton avec droit d'ecriture : https://huggingface.co/settings/tokens

## 4. La licence et la langue

- Le depot GitHub n'a **aucun fichier LICENSE**. La carte annonce `license: mit`
  par defaut : a confirmer ou a changer (Apache-2.0, CC-BY-4.0...).
- La carte du modele est ecrite en **anglais** (convention Hugging Face, public
  international) avec le resultat principal en clair. Version francaise possible
  en cinq minutes si prefere.
- Le champ `emoji` de la carte d'espace a ete **volontairement omis** (regle du
  depot : aucun emoji dans le code ni les docs). Hugging Face en met un par
  defaut ; ajoutable depuis l'interface si souhaite.

## Ordre propose

1. `git pull` ;
2. `python3 adversarial/huggingface/pousser_hf.py` -> simulation : dit ce qui
   manque (aucun envoi) ;
3. confirmer les noms de poids, corriger `POIDS` si besoin ;
4. `HF_TOKEN=... python3 adversarial/huggingface/pousser_hf.py --envoyer` ;
5. verifier les deux pages a l'ecran, puis annoncer.
