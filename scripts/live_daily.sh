#!/usr/bin/env bash
# Simulation en direct, un jour : nouvelles cotations, décision signée et ancrée, puis
# sauvegarde dans le dépôt, depuis votre PC : bash scripts/live_daily.sh
# Pour que ça tourne PC éteint : .github/workflows/live.yml (README, « Automatiser »).
# N'utiliser qu'un des deux à la fois, sinon les deux poussent le même jour.
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
    git add outputs/ledger/live ledger data/raw/XAU_1d_update.csv
    if git diff --cached --quiet; then
        echo "Rien de nouveau à enregistrer."
    else
        git commit -m "Simulation en direct : $(date -u '+%Y-%m-%d')"
    fi
    # Toujours : envoie aussi un commit local resté en attente (sans effet si à jour).
    git push origin main
} >> "$log" 2>&1
