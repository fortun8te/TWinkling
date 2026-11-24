#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
model="${TWINKLING_MODEL_DIR:-$root/models/Inkling-2-Mini-nvfp4}"
server="${TWINKLING_SERVER:-$root/native/build-5080/apps/serve/ingine-serve}"
plan="${TWINKLING_EXPERT_PLAN:-$root/config/rtx5080-experts.json}"
port="${TWINKLING_PORT:-18298}"

if [[ "${1:-}" == --help ]]; then
  echo 'Start TWinkling on localhost with the 64K RTX 5080 text profile.'
  echo 'Optional environment settings: TWINKLING_MODEL_DIR, TWINKLING_SERVER, TWINKLING_EXPERT_PLAN, TWINKLING_PORT.'
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo 'Use --help for the launcher options.' >&2
  exit 2
fi

[[ -x "$server" && -r "$plan" && -r "$model/model.safetensors.index.json" ]] || {
  echo 'The server, converted model or expert plan is missing. Run scripts/setup-5080.sh first.' >&2
  exit 1
}
args=(--engine --model-dir "$model" --host 127.0.0.1 --port "$port" --max-ctx 65536
      --dense-fp8 --kv-fp8 --spec --spec-threshold 0.8 --no-vision --park-mb 0 --embed-host
      --prefill-chunk 2048 --prefill-stage-mib 256 --expert-plan "$plan" --max-queue 8)
if [[ -r "$model/inkling-2-mini-abliterated.safetensors" ]]; then
  args+=(--ablation "$model/inkling-2-mini-abliterated.safetensors" --model-id inkling-2-mini-abliterated)
fi
exec "$server" "${args[@]}"
