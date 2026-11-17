#!/bin/sh
# bench/setup-swift.sh: check for Swift, and install it if it is missing.
#
# Only the benchmarks in bench/ need Swift. Building llama.cpp does not.
#
# Usage:  bench/setup-swift.sh [--check]
#   --check   only report whether a usable Swift is installed (exit 1 if not)
#
# Installs as you, never as root: swiftly, swift.org's toolchain installer for
# Linux, into ~/.local/share/swiftly, and through it the latest Swift release.
# swiftly's download is checked against swift.org's signing keys before it runs.
# swiftly lists any system packages Swift still needs; installing those is yours.

set -eu

SWIFT_MIN_MAJOR=6

if [ "$(id -u)" -eq 0 ]; then
    echo "error: run as your normal user, not root: Swift installs into your home directory." >&2
    exit 1
fi

# a swiftly install from an earlier run, in a shell whose PATH does not have it yet
swiftly_bin="${SWIFTLY_BIN_DIR:-${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/bin}"
if ! command -v swift >/dev/null 2>&1 && [ -x "$swiftly_bin/swift" ]; then
    PATH="$swiftly_bin:$PATH"
    export PATH
fi

swift_ok() {
    command -v swift >/dev/null 2>&1 || return 1
    v=$(swift --version 2>/dev/null | sed -n 's/.*Swift version \([0-9][0-9]*\)\..*/\1/p' | head -n 1)
    [ -n "$v" ] && [ "$v" -ge "$SWIFT_MIN_MAJOR" ]
}

if swift_ok; then
    echo "Swift: $(swift --version 2>/dev/null | head -n 1)"
    exit 0
fi

if [ "${1:-}" = "--check" ]; then
    echo "Swift: not installed, or older than $SWIFT_MIN_MAJOR (run bench/setup-swift.sh to install it)." >&2
    exit 1
fi

if [ "$(uname -s)" != "Linux" ]; then
    echo "error: this installer is for Linux; elsewhere, install Swift from https://www.swift.org/install/" >&2
    exit 1
fi

arch=$(uname -m)
case "$arch" in
    x86_64|aarch64) ;;
    *) echo "error: no swiftly build for $arch" >&2; exit 1 ;;
esac

for tool in curl gpg tar; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool is needed to install Swift" >&2; exit 1; }
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT INT TERM

echo "Downloading swiftly ($arch) and its signature from swift.org..."
curl -fsSL -o "$work/swiftly.tar.gz"     "https://download.swift.org/swiftly/linux/swiftly-$arch.tar.gz"
curl -fsSL -o "$work/swiftly.tar.gz.sig" "https://download.swift.org/swiftly/linux/swiftly-$arch.tar.gz.sig"
# swift.org serves its keys gzip-encoded; --compressed decodes them.
curl -fsSL --compressed -o "$work/all-keys.asc" "https://www.swift.org/keys/all-keys.asc"

# a throwaway keyring holding only swift.org's published keys
export GNUPGHOME="$work/gnupg"
mkdir -m 700 "$GNUPGHOME"
gpg --batch --quiet --import "$work/all-keys.asc"
if ! gpg --batch --verify "$work/swiftly.tar.gz.sig" "$work/swiftly.tar.gz" 2>"$work/verify.log"; then
    cat "$work/verify.log" >&2
    echo "error: swiftly's signature does not verify against swift.org's keys" >&2
    exit 1
fi
echo "Signature verified against swift.org's keys."
unset GNUPGHOME

tar -xzf "$work/swiftly.tar.gz" -C "$work"
"$work/swiftly" init --assume-yes --quiet-shell-followup

# swiftly's env.sh is bash; put its bin directory on PATH directly instead
swiftly_bin="${SWIFTLY_BIN_DIR:-${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/bin}"
PATH="$swiftly_bin:$PATH"
export PATH
# init does not always create the swift, swiftc, ... links in its bin directory; link does
[ -e "$swiftly_bin/swift" ] || "$swiftly_bin/swiftly" link
hash -r 2>/dev/null || true

if swift_ok; then
    echo "Swift: $(swift --version 2>/dev/null | head -n 1)"
    echo "New shells pick it up from your profile; in this one: export PATH=\"$swiftly_bin:\$PATH\""
else
    swift --version 2>&1 | head -n 3 >&2 || true
    echo "error: Swift is installed but does not run. Install the system packages swiftly" >&2
    echo "       listed above (that needs root), then run bench/setup-swift.sh --check." >&2
    exit 1
fi
