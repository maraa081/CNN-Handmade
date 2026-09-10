#!/usr/bin/env bash
# campagne.sh — lance la campagne d'entrainement durci en serie, en arriere-plan.
#
#   ./adversarial/scripts/campagne.sh              lance tout (saute ce qui est deja fait)
#   ./adversarial/scripts/campagne.sh --rapide     version courte (pour tester la chaine)
#   ./adversarial/scripts/campagne.sh --forcer     refait tout, meme si les poids existent
#   ./adversarial/scripts/campagne.sh --liste      affiche seulement la liste des runs
#
# Chaque run ecrit son log dans adversarial/results/logs/. Si on relance le
# script, les runs deja termines sont sautes : on peut donc l'interrompre.
#
# Pour suivre l'avancement depuis un autre terminal :
#   tail -f adversarial/results/logs/*.log

set -u
RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RACINE" || exit 1

LOGS="adversarial/results/logs"
mkdir -p "$LOGS"

MODE="${1:-normal}"
FORCER=0
[ "$MODE" = "--forcer" ] && FORCER=1

# Version courte : 10x moins de calcul, pour verifier que la chaine tourne.
EPOCHS=10
IMAGES=60000
ETAPES=5
MOTEUR="adversarial/scripts/harden2.py"
EXT="npz"
SUFIXE=""
# Permet de pointer un autre interpreteur (venv) : PYTHON=python ./campagne.sh
if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then PYTHON=python3; else PYTHON=python; fi
fi
LISTE=0
for a in "$@"; do
    case "$a" in
        --rapide) EPOCHS=2; IMAGES=10000; ETAPES=3; SUFIXE="_rapide" ;;
        --forcer) FORCER=1 ;;
        --torch)  MOTEUR="adversarial/torch/harden_torch.py"; EXT="pt" ;;
        --liste)  LISTE=1 ;;
    esac
done

# Le suffixe evite que la version courte et la version complete ecrivent dans le
# meme fichier : sinon la campagne complete saute tous les runs en croyant
# qu'ils sont deja faits (bug constate le 2026-09-10 chez Maraa).

# nom|options
RUNS=(
"a_ref_pgdat|--n-train $IMAGES --epochs $EPOCHS --pgd-steps $ETAPES --out models/harden2_ref_pgdat$SUFIXE.$EXT"
"b_aug_pgdat|--n-train $IMAGES --epochs $EPOCHS --pgd-steps $ETAPES --augment --out models/harden2_aug_pgdat$SUFIXE.$EXT"
"c_aug_trades|--n-train $IMAGES --epochs $EPOCHS --pgd-steps $ETAPES --augment --loss trades --beta 2 --out models/harden2_aug_trades$SUFIXE.$EXT"
)

if [ "$LISTE" -eq 1 ]; then
    echo "Moteur : $MOTEUR"
    echo "Runs prevus (ressources : $IMAGES images, $EPOCHS epochs, PGD-$ETAPES) :"
    for r in "${RUNS[@]}"; do echo "  - ${r%%|*}  ->  ${r#*|}"; done
    exit 0
fi

echo "================================================================"
echo "  CAMPAGNE DURCIE   ($IMAGES images, $EPOCHS epochs, PGD-$ETAPES)"
echo "  moteur    : $MOTEUR"
echo "  python    : $PYTHON"
echo "  suffixe   : '${SUFIXE:-aucun}'"
echo "  demarrage : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  logs      : $LOGS/"
echo "================================================================"
echo

for entree in "${RUNS[@]}"; do
    nom="${entree%%|*}"
    opts="${entree#*|}"
    poids=$(echo "$opts" | grep -oE '\-\-out [^ ]+' | cut -d' ' -f2)

    echo "------------------------------------------------------------"
    echo ">>> $nom"
    echo "    options : $opts"
    if [ -f "$poids" ] && [ "$FORCER" -eq 0 ]; then
        echo "    [SKIP] $poids existe deja (utiliser --forcer pour refaire)"
        continue
    fi

    t0=$(date +%s)
    # shellcheck disable=SC2086
    "$PYTHON" "$MOTEUR" $opts 2>&1 | tee "$LOGS/$nom.log"
    rc=${PIPESTATUS[0]}
    t1=$(date +%s)
    m=$(( (t1 - t0) / 60 ))

    if [ "$rc" -eq 0 ]; then
        echo "    [OK] $nom termine en ${m} min"
    else
        echo "    [ECHEC] $nom (code $rc) apres ${m} min - voir $LOGS/$nom.log"
    fi
    echo
done

echo "================================================================"
echo "  CAMPAGNE TERMINEE - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo
echo "Resultats a comparer :"
echo "  $PYTHON $MOTEUR --report models/harden2_ref_pgdat$SUFIXE.$EXT --restarts 3"
echo "  $PYTHON $MOTEUR --report models/harden2_aug_pgdat$SUFIXE.$EXT --restarts 3"
echo "  $PYTHON $MOTEUR --report models/harden2_aug_trades$SUFIXE.$EXT --restarts 3"
