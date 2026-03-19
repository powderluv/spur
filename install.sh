#!/bin/bash
# Install Spur — AI-native job scheduler
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/powderluv/spur/main/install.sh | bash
#
#   # Install nightly:
#   curl -fsSL https://raw.githubusercontent.com/powderluv/spur/main/install.sh | bash -s -- nightly
#
#   # Install a specific version:
#   curl -fsSL https://raw.githubusercontent.com/powderluv/spur/main/install.sh | bash -s -- v0.1.0
#
#   # Install to a custom directory:
#   curl -fsSL https://raw.githubusercontent.com/powderluv/spur/main/install.sh | INSTALL_DIR=/opt/spur/bin bash

set -euo pipefail

REPO="powderluv/spur"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"
VERSION="${1:-latest}"

log()  { echo "==> $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

# --- Platform check ---
OS=$(uname -s)
ARCH=$(uname -m)
[ "$OS" = "Linux" ] || err "Spur currently supports Linux only (got ${OS})"
[ "$ARCH" = "x86_64" ] || err "Spur currently supports x86_64 only (got ${ARCH})"

# --- Resolve version ---
if [ "$VERSION" = "latest" ]; then
    log "Fetching latest release..."
    VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4) \
        || err "No releases found. Create one with: gh release create v0.1.0"
fi
log "Installing Spur ${VERSION}"

# --- Download ---
TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

if [ "$VERSION" = "nightly" ]; then
    # Nightly tarballs include date+sha in the name — find the .tar.gz asset
    TARBALL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/nightly" \
        | grep '"name"' | grep '\.tar\.gz"' | grep -v sha256 | head -1 | cut -d'"' -f4) \
        || err "Could not find nightly release assets"
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
mkdir -p "${INSTALL_DIR}"
# Find the extracted directory (name varies for nightly)
EXTRACTED=$(find "${TMPDIR}" -maxdepth 1 -type d -name 'spur-*' | head -1)
[ -n "${EXTRACTED}" ] || err "Could not find extracted directory"
cp -f "${EXTRACTED}"/bin/* "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/spur" "${INSTALL_DIR}/spurctld" "${INSTALL_DIR}/spurd" \
         "${INSTALL_DIR}/spurdbd" "${INSTALL_DIR}/spurrestd"

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
log "Binaries: spur, spurctld, spurd, spurdbd, spurrestd"
log "Symlinks: sbatch, srun, squeue, scancel, sinfo, sacct, scontrol"

if ! echo "$PATH" | tr ':' '\n' | grep -qx "${INSTALL_DIR}"; then
    echo ""
    echo "Add to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    echo ""
    echo "Or add to ~/.bashrc:"
    echo "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
fi

log "Done."
