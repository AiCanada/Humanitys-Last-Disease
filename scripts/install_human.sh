#!/usr/bin/env bash
# Install the `human` program inside a Linux (or WSL Ubuntu) system.
#
# This is the thin part. Everything that builds the environment already lives in
# setup.sh and the scripts it calls, and this does not repeat any of it: it adds
# only the two things setup.sh assumes are already there -- the handful of system
# packages it shells out to, and a checkout to run in -- and then hands over.
#
# Run it from a checkout to install that checkout, or from anywhere with
# --repository to clone one first.
set -euo pipefail

#: System commands setup.sh and the scripts it calls shell out to, and the apt
#: package that provides each. A minimal Ubuntu image has tar but often not curl
#: or git, and `make` only arrives with build-essential. Checked by command
#: rather than by package so a machine that already has them installs nothing.
declare -A NEEDED=(
    [git]=git
    [curl]=curl
    [tar]=tar
    [make]=build-essential
)

#: Packages that provide no command of their own, so `command -v` cannot see
#: them and they are checked with dpkg instead. This is the set the container
#: recipes install for the same reason -- see Dockerfile and
#: apptainer/rfd4-proteina.def, which explain each one:
#:
#:   ca-certificates  curl against https fails without it, with an error that
#:                    does not mention certificates
#:   build-essential  torch.compile builds C++ and CUDA glue at *run* time, not
#:                    only at install time, so a missing compiler surfaces as a
#:                    failed generation long after setup looked successful
#:   libxrender1 libx11-6 libxext6
#:                    OpenBabel's runtime libraries. Nothing imports them until
#:                    a small-molecule path runs, which is the worst time to
#:                    discover they are absent.
#:
#: Not installing these is what makes a from-scratch Ubuntu differ from the
#: containers, and none of the setup scripts installs a single apt package.
LIBRARY_PACKAGES=(ca-certificates build-essential libxrender1 libx11-6 libxext6)

repository="https://github.com/AiCanada/RFD4-Proteina.git"
target=""
clone=false
setup_args=()

usage() {
    cat <<'TEXT'
Usage: bash scripts/install_human.sh [options] [-- setup.sh options]

  --repository URL   Clone from URL instead of installing the checkout this
                     script sits in. Default: this project's origin.
  --target DIR       Where to clone. Default: ~/human/RFD4-Proteina
  --skip-packages    Do not touch apt. Fails later if a command is missing.
  -h, --help         This text.

Anything after -- is passed to setup.sh unchanged, so:

  bash scripts/install_human.sh -- --skip-weights --skip-search-envs

installs the environment without the model weights or the optional
AlphaFold search environments. See bash setup.sh --help for all of them.
TEXT
}

skip_packages=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repository) repository="${2:?--repository needs a URL}"; clone=true; shift 2 ;;
        --target) target="${2:?--target needs a directory}"; clone=true; shift 2 ;;
        --skip-packages) skip_packages=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; setup_args=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- packages
installed_package() {
    # dpkg's own answer, so a package that is merely unpacked or removed-but-
    # configured does not read as present.
    [[ "$(dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null)" == "installed" ]]
}

missing=()
for command_name in "${!NEEDED[@]}"; do
    command -v "$command_name" >/dev/null 2>&1 || missing+=("${NEEDED[$command_name]}")
done
if command -v dpkg-query >/dev/null 2>&1; then
    for package in "${LIBRARY_PACKAGES[@]}"; do
        installed_package "$package" || missing+=("$package")
    done
else
    # Not a dpkg system. The command checks above still apply; the libraries
    # cannot be checked and are left to the person, named rather than assumed.
    say "Not a dpkg system, so these were not checked: ${LIBRARY_PACKAGES[*]}"
fi
# One apt call for the set, de-duplicated: build-essential appears both as the
# provider of `make` and in its own right.
if [[ ${#missing[@]} -gt 0 ]]; then
    mapfile -t missing < <(printf '%s\n' "${missing[@]}" | sort -u)
fi

if [[ ${#missing[@]} -eq 0 ]]; then
    say "System packages are already present."
elif [[ "$skip_packages" == true ]]; then
    say "Missing system packages, not installing them (--skip-packages): ${missing[*]}"
else
    say "Installing system packages: ${missing[*]}"
    sudo_command=()
    if [[ "$(id -u)" != 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_command=(sudo)
            echo "sudo will ask for your Ubuntu password."
        else
            echo "These packages are missing and this is not root and there is no sudo:" >&2
            printf '  %s\n' "${missing[@]}" >&2
            echo "Install them yourself, or re-run with --skip-packages." >&2
            exit 1
        fi
    fi
    "${sudo_command[@]}" apt-get update
    DEBIAN_FRONTEND=noninteractive "${sudo_command[@]}" apt-get install -y --no-install-recommends "${missing[@]}"
fi

# ---------------------------------------------------------------- checkout
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$here")"

if [[ "$clone" == true ]]; then
    target="${target:-$HOME/human/RFD4-Proteina}"
    if [[ -d "$target/.git" ]]; then
        say "A checkout is already at $target. Leaving it as it is and installing it."
    else
        say "Cloning $repository into $target"
        mkdir -p "$(dirname "$target")"
        if ! git clone --depth 1 "$repository" "$target"; then
            cat >&2 <<TEXT

The clone failed. If that repository is private, git needs credentials this
script deliberately does not handle: sign in with the GitHub CLI (gh auth login)
or set up an SSH key, then run this again. Nothing was installed.
TEXT
            exit 1
        fi
    fi
    repo_root="$target"
elif [[ ! -f "$repo_root/setup.sh" ]]; then
    echo "No setup.sh beside this script, so this is not a checkout." >&2
    echo "Pass --repository to clone one, or run this from inside a checkout." >&2
    exit 1
fi

# ---------------------------------------------------------------- hand over
say "Building the environment with setup.sh"
echo "This installs uv, Python 3.12 and the CUDA build of torch, so it downloads"
echo "several GB and takes a while. setup.sh skips anything already in place, so"
echo "it is safe to run again if it is interrupted."
cd "$repo_root"
bash setup.sh --native "${setup_args[@]}"

say "Done."
printf 'The program starts with:\n  cd %q && .venv/bin/human --ui\n' "$repo_root"
echo "On Windows, scripts/Install-HumanShortcut.ps1 puts that on your desktop."
