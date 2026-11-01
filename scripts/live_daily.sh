#!/usr/bin/env bash
# Simulation en direct, un jour : nouvelles cotations, décision signée et ancrée, puis
# sauvegarde dans le dépôt. Lancé chaque nuit par le Planificateur de tâches Windows
# (voir README, « Automatiser chaque nuit ») ; à la main : bash scripts/live_daily.sh
#
# Un jour manqué (PC éteint) est rattrapé au passage suivant, marqué « en retard ».
# Sans nouveau jour (week-end, jour férié), rien n'est ajouté et rien n'est poussé.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
log="outputs/live_daily.log"
{
    echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC ==="
    git pull --no-edit origin main
    uv run gold-agents live
    git add outputs/ledger/live ledger
    if git diff --cached --quiet; then
        echo "Rien de nouveau à sauvegarder."
    else
        git commit -m "Simulation en direct : $(date -u '+%Y-%m-%d')"
        git push origin main
    fi
} >> "$log" 2>&1
