#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_PYTHON="3.10"
DEFAULT_VENV_DIR="${ROOT_DIR}/.venv-herobm"
DEFAULT_DEPS_DIR="${ROOT_DIR}/deps"
DEFAULT_GEQTRAIN_REF="main"
DEFAULT_CGMAP_REF="main"

PYTHON_VERSION="${DEFAULT_PYTHON}"
VENV_DIR="${DEFAULT_VENV_DIR}"
DEPS_DIR="${DEFAULT_DEPS_DIR}"
GEQTRAIN_REF="${DEFAULT_GEQTRAIN_REF}"
CGMAP_REF="${DEFAULT_CGMAP_REF}"
TORCH_BACKEND="auto"
TORCH_VERSION=""
INSTALL_TORCH=1
RECREATE_VENV=0
UPDATE_REPOS=0

usage() {
  cat <<USAGE
Usage: ./venv_setup.sh [options]

Create a uv-based virtual environment for HEroBM and install:
- GEqTrain (https://github.com/limresgrp/GEqTrain.git)
- CGmap    (https://github.com/limresgrp/CGmap.git)
- HEroBM   (this repository)

Options:
  --python VERSION         Python version to use (default: ${DEFAULT_PYTHON})
  --venv-dir PATH          Virtual environment directory (default: ${DEFAULT_VENV_DIR})
  --deps-dir PATH          Directory where GEqTrain/CGmap are cloned (default: ${DEFAULT_DEPS_DIR})
  --geqtrain-ref REF       GEqTrain git ref/branch/tag (default: ${DEFAULT_GEQTRAIN_REF})
  --cgmap-ref REF          CGmap git ref/branch/tag (default: ${DEFAULT_CGMAP_REF})
  --torch-backend BACKEND  uv torch backend: auto|cpu|cu118|cu121|cu124|cu126|cu128|rocm (default: auto)
  --torch-version VERSION  Torch version to install (default: latest)
  --no-torch               Skip torch installation
  --recreate               Remove existing venv before creating a new one
  --update-repos           Fetch and update GEqTrain/CGmap if already cloned
  -h, --help               Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --deps-dir)
      DEPS_DIR="$2"
      shift 2
      ;;
    --geqtrain-ref)
      GEQTRAIN_REF="$2"
      shift 2
      ;;
    --cgmap-ref)
      CGMAP_REF="$2"
      shift 2
      ;;
    --torch-backend)
      TORCH_BACKEND="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --no-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --recreate)
      RECREATE_VENV=1
      shift
      ;;
    --update-repos)
      UPDATE_REPOS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to find uv after installation. Add ~/.local/bin or ~/.cargo/bin to PATH and retry." >&2
    exit 1
  fi
}

sync_repo() {
  local url="$1"
  local dir="$2"
  local ref="$3"

  mkdir -p "$(dirname "${dir}")"

  if [[ ! -d "${dir}/.git" ]]; then
    echo "Cloning ${url} -> ${dir}"
    git clone "${url}" "${dir}"
  else
    echo "Repository already exists: ${dir}"
    if [[ "${UPDATE_REPOS}" -eq 1 ]]; then
      echo "Fetching updates for ${dir}"
      git -C "${dir}" fetch --all --tags
    fi
  fi

  if [[ -z "${ref}" ]]; then
    return
  fi

  if git -C "${dir}" show-ref --verify --quiet "refs/heads/${ref}"; then
    git -C "${dir}" checkout "${ref}"
    if [[ "${UPDATE_REPOS}" -eq 1 ]]; then
      git -C "${dir}" pull --ff-only || true
    fi
    return
  fi

  if git -C "${dir}" show-ref --verify --quiet "refs/tags/${ref}"; then
    git -C "${dir}" checkout "tags/${ref}"
    return
  fi

  if git -C "${dir}" ls-remote --exit-code --heads origin "${ref}" >/dev/null 2>&1; then
    git -C "${dir}" fetch origin "${ref}"
    git -C "${dir}" checkout -B "${ref}" "origin/${ref}"
    return
  fi

  git -C "${dir}" checkout "${ref}"
}

install_pdbfixer() {
  local python_bin="$1"
  if uv pip install --python "${python_bin}" pdbfixer; then
    return
  fi

  echo "PyPI install for pdbfixer failed; trying GitHub source..."
  uv pip install --python "${python_bin}" "git+https://github.com/openmm/pdbfixer.git"
}

ensure_uv

if [[ "${RECREATE_VENV}" -eq 1 && -d "${VENV_DIR}" ]]; then
  echo "Removing existing venv: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR} (Python ${PYTHON_VERSION})"
  if ! uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed; then
    echo "Python ${PYTHON_VERSION} not available locally. Installing via uv..."
    uv python install "${PYTHON_VERSION}"
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed
  fi
else
  echo "Using existing venv: ${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found in venv: ${PYTHON_BIN}" >&2
  exit 1
fi

GEQTRAIN_DIR="${DEPS_DIR}/GEqTrain"
CGMAP_DIR="${DEPS_DIR}/CGmap"

sync_repo "https://github.com/limresgrp/GEqTrain.git" "${GEQTRAIN_DIR}" "${GEQTRAIN_REF}"
sync_repo "https://github.com/limresgrp/CGmap.git" "${CGMAP_DIR}" "${CGMAP_REF}"

echo "Installing base packaging tools..."
uv pip install --python "${PYTHON_BIN}" -U pip setuptools wheel

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  TORCH_SPEC="torch"
  if [[ -n "${TORCH_VERSION}" ]]; then
    TORCH_SPEC="torch==${TORCH_VERSION}"
  fi

  echo "Installing ${TORCH_SPEC} (backend: ${TORCH_BACKEND})"
  if [[ "${TORCH_BACKEND}" == "auto" ]]; then
    uv pip install --python "${PYTHON_BIN}" "${TORCH_SPEC}"
  else
    uv pip install --python "${PYTHON_BIN}" "${TORCH_SPEC}" --torch-backend "${TORCH_BACKEND}"
  fi
fi

echo "Installing GEqTrain and CGmap (editable)..."
uv pip install --python "${PYTHON_BIN}" -e "${GEQTRAIN_DIR}"
uv pip install --python "${PYTHON_BIN}" -e "${CGMAP_DIR}"

echo "Installing HEroBM runtime dependencies..."
uv pip install --python "${PYTHON_BIN}" openmm matplotlib seaborn plotly
install_pdbfixer "${PYTHON_BIN}"

echo "Installing HEroBM (editable)..."
uv pip install --python "${PYTHON_BIN}" -e "${ROOT_DIR}"

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Installed repositories:"
echo "  GEqTrain: ${GEQTRAIN_DIR}"
echo "  CGmap:    ${CGMAP_DIR}"
