#!/bin/bash
# Copy the figures made by plots.py (figures/*.pdf) into the article's figures folder.
#
# Usage: ./copy_figures.sh [--dry-run] [NAME ...]
#   NAME     copy only figures/NAME.pdf (default: every PDF in figures/)
#   --dry-run  list what would be copied, copy nothing
# Article folder: $ARTICLE_FIGURES_DIR, default ~/Documents/Artigos/Decrypto/figures.
# Only copies; never deletes. A file the article already has is replaced only if ours is newer.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)/figures"
DEST_DIR="${ARTICLE_FIGURES_DIR:-$HOME/Documents/Artigos/Decrypto/figures}"

dry_run=0
names=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) names+=("${arg%.pdf}") ;;
  esac
done

[ -d "$SRC_DIR" ] || { echo "no $SRC_DIR: run plots.py first" >&2; exit 1; }
[ -d "$DEST_DIR" ] || { echo "article figures folder not found: $DEST_DIR" >&2; exit 1; }

if [ ${#names[@]} -eq 0 ]; then
  shopt -s nullglob
  files=("$SRC_DIR"/*.pdf)
else
  files=()
  for n in "${names[@]}"; do
    [ -f "$SRC_DIR/$n.pdf" ] || { echo "not found: $SRC_DIR/$n.pdf" >&2; exit 1; }
    files+=("$SRC_DIR/$n.pdf")
  done
fi
[ ${#files[@]} -gt 0 ] || { echo "no PDFs in $SRC_DIR" >&2; exit 1; }

for src in "${files[@]}"; do
  dest="$DEST_DIR/$(basename "$src")"
  if [ -e "$dest" ] && [ ! "$src" -nt "$dest" ]; then
    echo "up to date: $(basename "$src")"
    continue
  fi
  action=$([ -e "$dest" ] && echo "replace" || echo "add")
  if [ $dry_run -eq 1 ]; then
    echo "would $action: $(basename "$src") -> $DEST_DIR/"
  else
    cp -p "$src" "$dest"
    echo "$action: $(basename "$src") -> $DEST_DIR/"
  fi
done
