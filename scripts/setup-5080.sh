#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_model="$root/models/Inkling-2-Mini"
output="${TWINKLING_MODEL_DIR:-$root/models/Inkling-2-Mini-nvfp4}"

if [[ "${1:-}" == --help ]]; then
  echo 'Build the RTX 5080 engine, download the pinned Inkling 2 Mini pack and convert the host-expert layers.'
  echo 'Set TWINKLING_MODEL_DIR to choose where to save the converted pack.'
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo 'Use --help for setup options.' >&2
  exit 2
fi

INGINE_MIN_GPU_MIB=15000 INGINE_INSTALL_PRESET=install-5080 INGINE_GPU_PROFILE=skip \
  INGINE_CONFIG="$root/.tools/twinkling.env" INGINE_MODEL_DIR="$source_model" \
  bash "$root/scripts/install.sh"

if [[ -s "$output/requant-report.json" && -s "$output/experts-nvfp4.safetensors" && -s "$output/model.safetensors.index.json" ]]; then
  printf 'Converted model is ready at %s\n' "$output"
  exit 0
fi
if [[ -e "$output" || -L "$output" ]]; then
  printf 'Choose a new TWINKLING_MODEL_DIR. The existing path is incomplete: %s\n' "$output" >&2
  exit 1
fi

if [[ -x "$root/.tools/bin/uv" ]]; then
  uv="$root/.tools/bin/uv"
else
  uv="$(command -v uv)"
fi
venv="$root/.tools/convert-5080"
[[ -x "$venv/bin/python" ]] || "$uv" venv --python 3.12 "$venv"
"$uv" pip install --python "$venv/bin/python" 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"$uv" pip install --python "$venv/bin/python" 'safetensors==0.8.0' 'numpy==2.5.5'

mkdir -p "$(dirname -- "$output")"
stage="$(mktemp -d "$(dirname -- "$output")/.twinkling-model.XXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
"$venv/bin/python" -I "$root/tools/requant_nvfp4.py" --model-dir "$source_model" --out "$stage" --layers 20-23
mv -T -- "$stage" "$output"
trap - EXIT
printf 'Converted model is ready at %s\n' "$output"
