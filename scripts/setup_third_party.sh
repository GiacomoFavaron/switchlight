#!/usr/bin/env bash
# Clone third-party model repos into third_party/.
# Run from project root: bash scripts/setup_third_party.sh
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

echo "Cloning third-party repos into $THIRD_PARTY"

# DSINE — surface normal estimation
if [ ! -d "$THIRD_PARTY/DSINE" ]; then
    echo "  Cloning DSINE..."
    git clone --depth 1 https://github.com/baegwangbin/DSINE.git "$THIRD_PARTY/DSINE"
else
    echo "  DSINE already present, skipping."
fi

# Intrinsic / Ordinal Shading — albedo estimation
if [ ! -d "$THIRD_PARTY/Intrinsic" ]; then
    echo "  Cloning Intrinsic..."
    git clone --depth 1 https://github.com/compphoto/Intrinsic.git "$THIRD_PARTY/Intrinsic"
else
    echo "  Intrinsic already present, skipping."
fi

echo ""
echo "Done. third_party/ contents:"
ls -la "$THIRD_PARTY"
echo ""
echo "Note: pretrained weights are downloaded automatically by each model on first use."
