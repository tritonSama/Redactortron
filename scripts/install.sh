#!/usr/bin/env bash
# Unix installer for Redactortron
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXTRAS=()
WITH_POPPLER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --api) EXTRAS+=(--api); shift ;;
    --dev) EXTRAS+=(--dev); shift ;;
    --with-poppler) WITH_POPPLER=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

python3 scripts/install_deps.py "${EXTRAS[@]}" ${WITH_POPPLER:+--with-poppler}

echo
echo "If Poppler is missing:"
echo "  macOS:  brew install poppler"
echo "  Debian: sudo apt-get install -y poppler-utils"
echo
echo "Start GUI:  python3 -m redactortron ui"
