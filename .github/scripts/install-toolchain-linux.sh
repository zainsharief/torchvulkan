#!/usr/bin/env bash
# Installs the torchvulkan build toolchain inside the cibuildwheel manylinux
# container (AlmaLinux 8 / manylinux_2_28): the Vulkan SDK (headers + loader +
# the spirv-cross executable) and the Slang compiler (slangc).
#
# The install paths here MUST match the VULKAN_SDK / SLANG_DIR / PATH entries in
# [tool.cibuildwheel.linux].environment in pyproject.toml.
set -euo pipefail

SLANG_VERSION="2026.8.1"
VULKAN_SDK_VERSION="1.3.296.0"

dnf install -y wget tar xz unzip

wget -q -O /tmp/vulkansdk.tar.xz \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
mkdir -p /opt/vulkan
tar -xf /tmp/vulkansdk.tar.xz -C /opt/vulkan --strip-components=1

wget -q -O /tmp/slang.zip \
  "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip"
mkdir -p /opt/slang
unzip -q /tmp/slang.zip -d /opt/slang

test -f /opt/vulkan/x86_64/include/vulkan/vulkan.h
test -x /opt/vulkan/x86_64/bin/spirv-cross
test -x /opt/slang/bin/slangc
