#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/third_party/mujoco_menagerie"

if [ -d "$TARGET_DIR" ]; then
    echo "mujoco_menagerie already exists at $TARGET_DIR"
    echo "To update, delete the directory and re-run this script."
    exit 0
fi

echo "Cloning mujoco_menagerie into $TARGET_DIR ..."
mkdir -p "$SCRIPT_DIR/third_party"
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "$TARGET_DIR"
echo "Done. Panda model available at: $TARGET_DIR/franka_emika_panda/"
