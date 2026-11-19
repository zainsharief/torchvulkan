#!/usr/bin/env bash
# The output path MUST match TORCHVULKAN_MOLTENVK in [tool.cibuildwheel.macos]
# environment in pyproject.toml. The dylib ships inside the macOS wheel, so the
# download is pinned by SHA-256; update the hash with the version.
set -euo pipefail

# zainsharief/MoltenVK-torchvulkan: MoltenVK + VK_KHR_cooperative_matrix (arm64).
MOLTENVK_VERSION="1.4.3-torchvulkan.1"
MOLTENVK_SHA256="1ea0c46f202f838b6f271a09776cd45613c65d28493aedcb95a3bea5dd67449b"
OUT_DIR="${MOLTENVK_OUT_DIR:-/tmp/moltenvk}"
ARCHIVE="$OUT_DIR/MoltenVK-macos-arm64.tar.gz"

mkdir -p "$OUT_DIR"
# an archive already here with the pinned hash is reused, so local reconfigures don't download again
if ! echo "${MOLTENVK_SHA256}  $ARCHIVE" | shasum -a 256 -c - > /dev/null 2>&1; then
  curl -sSLf -o "$ARCHIVE" \
    "https://github.com/zainsharief/MoltenVK-torchvulkan/releases/download/v${MOLTENVK_VERSION}/MoltenVK-macos-arm64.tar.gz"
  echo "${MOLTENVK_SHA256}  $ARCHIVE" | shasum -a 256 -c -
fi
tar -xzf "$ARCHIVE" -C "$OUT_DIR"
cp "$OUT_DIR/MoltenVK/libMoltenVK.dylib" "$OUT_DIR/libMoltenVK.dylib"

test -f "$OUT_DIR/libMoltenVK.dylib"
