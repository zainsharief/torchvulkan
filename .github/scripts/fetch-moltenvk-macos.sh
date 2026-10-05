#!/usr/bin/env bash
# The output path MUST match TORCHVULKAN_MOLTENVK in [tool.cibuildwheel.macos]
# environment in pyproject.toml.
set -euo pipefail

MOLTENVK_VERSION="1.4.2"
OUT_DIR="/tmp/moltenvk"

mkdir -p "$OUT_DIR"
curl -sSL -o "$OUT_DIR/MoltenVK-macos.tar" \
  "https://github.com/KhronosGroup/MoltenVK/releases/download/v${MOLTENVK_VERSION}/MoltenVK-macos.tar"
tar -xf "$OUT_DIR/MoltenVK-macos.tar" -C "$OUT_DIR"
lipo -thin "$(uname -m)" "$OUT_DIR/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib" -output "$OUT_DIR/libMoltenVK.dylib"

test -f "$OUT_DIR/libMoltenVK.dylib"
