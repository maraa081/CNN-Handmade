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
if [ "$MODE" = "--rapide" ]; then
    EPOCHS=2
    IMAGES=10000
    ETAPES=3
fi

# nom|options
RUNS=(
"a_ref_pgdat|--n-train $IMAGES --epochs $EPOCHS --pgd-steps $ETAPES --out models/harden2_ref_pgdat.npz"
"b_aug_pgdat|--n-train $IMAGES --epochs $EPOCHS --pgd-steps $ETAPES --augment --out models/harden2_aug_pgdat.npz"
"c_aug_trades|--n-train $IMAGES --epochs $EPOCHS --pgd-steps $ETAPES --augment --loss trades --out models/harden2_aug_trades.npz"
)

if [ "$MODE" = "--liste" ]; then
    echo "Runs prevus (ressources : $IMAGES images, $EPOCHS epochs, PGD-$ETAPES) :"
    for r in "${RUNS[@]}"; do echo "  - ${r%%|*}  ->  ${r#*|}"; done
    exit 0
fi

echo "================================================================"
echo "  CAMPAGNE DURCIE   ($IMAGES images, $EPOCHS epochs, PGD-$ETAPES)"
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
    python3 adversarial/scripts/harden2.py $opts 2>&1 | tee "$LOGS/$nom.log"
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
echo "  python3 adversarial/scripts/harden2.py --report models/harden2_ref_pgdat.npz  --restarts 3"
echo "  python3 adversarial/scripts/harden2.py --report models/harden2_aug_pgdat.npz  --restarts 3"
echo "  python3 adversarial/scripts/harden2.py --report models/harden2_aug_trades.npz --restarts 3"
