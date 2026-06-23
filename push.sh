#!/bin/bash
# Push CNN-Handmade changes to GitHub
# Usage: bash push.sh "message de commit"

cd "$(dirname "$0")"

MSG="${1:-Mise à jour CNN-Handmade}"

git add -A
git commit -m "$MSG" && git push origin main
