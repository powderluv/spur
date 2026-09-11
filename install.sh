#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install Spur — AI-native job scheduler
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash
#
#   # Install nightly:
#   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- nightly
#
#   # Install a specific version:
#   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- v0.1.0
#
#   # Install to a custom directory:
#   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | INSTALL_DIR=/opt/spur/bin bash
#
#   # Uninstall:
#   curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- uninstall

set -euo pipefail

REPO="ROCm/spur"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"

BINARIES="spur spurctld spurd spurstepd"
SYMLINKS="sbatch srun squeue scancel sinfo sacct scontrol"

log()  { echo "==> $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Spur installer — AI-native job scheduler

USAGE:
    install.sh [OPTIONS] [VERSION]

VERSION:
    latest          Install the latest stable release (default)
    nightly         Install the latest nightly build
    v0.1.0          Install a specific version
    uninstall       Remove Spur binaries

OPTIONS:
    -h, --help      Show this help message
    -d, --dir DIR   Install directory (default: ~/.local/bin)

ENVIRONMENT:
    INSTALL_DIR     Override install directory (same as --dir)

EXAMPLES:
    # Install latest stable
    curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash

    # Install nightly
    curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- nightly

    # Install to /opt/spur/bin
    curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- -d /opt/spur/bin

    # Uninstall
    curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh | bash -s -- uninstall
EOF
    exit 0
}

do_uninstall() {
    log "Uninstalling Spur from ${INSTALL_DIR}/"
    local removed=0
    for f in ${BINARIES} ${SYMLINKS}; do
        if [ -e "${INSTALL_DIR}/${f}" ]; then
            rm -f "${INSTALL_DIR}/${f}"
            echo "  removed ${f}"
            removed=$((removed + 1))
        fi
    done
    if [ "${removed}" -eq 0 ]; then
        log "No Spur files found in ${INSTALL_DIR}/"
    else
        log "Removed ${removed} file(s). Spur has been uninstalled."
    fi
    exit 0
}

# --- Parse arguments ---
VERSION="latest"
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)     usage ;;
        -d|--dir)      INSTALL_DIR="$2"; shift 2 ;;
        uninstall)     do_uninstall ;;
        *)             VERSION="$1"; shift ;;
    esac
done

# --- Platform check ---
OS=$(uname -s)
ARCH=$(uname -m)
[ "$OS" = "Linux" ] || err "Spur currently supports Linux only (got ${OS})"
[ "$ARCH" = "x86_64" ] || err "Spur currently supports x86_64 only (got ${ARCH})"

# --- glibc check (binaries built on AlmaLinux 8, require glibc >= 2.28) ---
# Two-step extraction avoids pipefail + SIGPIPE interaction (head -1 closes
# early, ldd gets SIGPIPE, pipeline exit != 0, fallback echo "0" appends).
_ldd_line=$(ldd --version 2>&1 | head -1 || true)
GLIBC_VER=$(echo "$_ldd_line" | grep -oE '[0-9]+\.[0-9]+$' || echo "0")
if [ "$(printf '%s\n' "2.28" "${GLIBC_VER}" | sort -V | head -1)" != "2.28" ]; then
    err "Spur requires glibc >= 2.28 (found ${GLIBC_VER}). Supported: Ubuntu 20.04+, Debian 10+, RHEL 8+, Fedora 28+"
fi

# --- Resolve version ---
if [ "$VERSION" = "latest" ]; then
    log "Fetching latest release..."
    # Use the Atom feed: /releases/latest ignores pre-releases (all our
    # releases are pre-releases) and the unauthenticated API is rate-limited
    # per IP. Feed is newest-first; take the first non-nightly tag.
    VERSION=$(curl -fsSL "https://github.com/${REPO}/releases.atom" \
        | grep -oE 'releases/tag/[^"]+' | sed 's#releases/tag/##' \
        | grep -v '^nightly$' | head -1) \
        || err "Could not resolve latest release from https://github.com/${REPO}/releases"
    [ -n "$VERSION" ] || err "No releases found at https://github.com/${REPO}/releases"
fi
log "Installing Spur ${VERSION}"

# --- Download ---
TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

if [ "$VERSION" = "nightly" ]; then
    # Nightly tarball name carries date+sha; read it from expanded_assets
    # rather than the rate-limited API.
    TARBALL=$(curl -fsSL "https://github.com/${REPO}/releases/expanded_assets/nightly" \
        | grep -oE 'spur-nightly-[^"]+-linux-amd64\.tar\.gz' | head -1) \
        || err "Could not find nightly release assets"
    [ -n "$TARBALL" ] || err "Could not find nightly release assets"
else
    TARBALL="spur-${VERSION}-linux-amd64.tar.gz"
fi

URL="https://github.com/${REPO}/releases/download/${VERSION}/${TARBALL}"
log "Downloading ${TARBALL}..."
curl -fSL -o "${TMPDIR}/${TARBALL}" "${URL}" \
    || err "Download failed. Check that release ${VERSION} exists at https://github.com/${REPO}/releases"

# --- Verify checksum if available ---
SHA_URL="${URL}.sha256"
if curl -fsSL -o "${TMPDIR}/${TARBALL}.sha256" "${SHA_URL}" 2>/dev/null; then
    log "Verifying checksum..."
    (cd "${TMPDIR}" && sha256sum -c "${TARBALL}.sha256") || err "Checksum mismatch"
fi

# --- Extract ---
log "Extracting..."
tar xzf "${TMPDIR}/${TARBALL}" -C "${TMPDIR}"

# --- Install ---
INSTALL_DIR="${INSTALL_DIR%/}"
mkdir -p "${INSTALL_DIR}"
# Find the extracted directory (name varies for nightly)
EXTRACTED=$(find "${TMPDIR}" -maxdepth 1 -type d -name 'spur-*' | head -1)
[ -n "${EXTRACTED}" ] || err "Could not find extracted directory"
cp -f "${EXTRACTED}"/bin/* "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/spur" "${INSTALL_DIR}/spurctld" "${INSTALL_DIR}/spurd"

PLUGIN_DIR="$(dirname "${INSTALL_DIR}")/lib/spur"
if [ -f "${EXTRACTED}/lib/spur/spur_mpi_pmix.so" ]; then
    mkdir -p "${PLUGIN_DIR}"
    cp -f "${EXTRACTED}/lib/spur/spur_mpi_pmix.so" "${PLUGIN_DIR}/"
    log "Installed MPI plugin to ${PLUGIN_DIR}/spur_mpi_pmix.so"
fi

# --- Verify ---
if ! "${INSTALL_DIR}/spur" --version >/dev/null 2>&1; then
    # Binary exists but --version may not be implemented yet
    if [ -x "${INSTALL_DIR}/spur" ]; then
        log "Binaries installed (version flag not yet supported)"
    else
        err "Installation verification failed"
    fi
fi

# --- PATH hint ---
log "Installed to ${INSTALL_DIR}/"
log "Binaries: ${BINARIES}"
log "Symlinks: ${SYMLINKS}"
if [ -f "${EXTRACTED}/etc/spur.conf.example" ]; then
    log "Example config in tarball: etc/spur.conf.example (copy to /etc/spur/spur.conf)"
fi

if ! echo "$PATH" | tr ':' '\n' | grep -qx "${INSTALL_DIR}"; then
    echo ""
    echo "Add to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    echo ""
    echo "Or add to ~/.bashrc:"
    echo "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
fi

log "Done."
