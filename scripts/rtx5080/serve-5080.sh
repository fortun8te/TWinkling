#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
model="${TWINKLING_MODEL_DIR:-$root/models/Inkling-2-Mini-nvfp4}"
server="${TWINKLING_SERVER:-$root/native/build-5080/apps/serve/ingine-serve}"
plan="${TWINKLING_EXPERT_PLAN:-$root/config/rtx5080-experts.json}"
port="${TWINKLING_PORT:-18298}"
ctx="${TWINKLING_CONTEXT:-65536}"
chunk="${TWINKLING_PREFILL_CHUNK:-2048}"
cache="${TWINKLING_KV:-fp8}"

if [[ "${1:-}" == --help ]]; then
  echo 'Start TWinkling on localhost with the 64K RTX 5080 text profile.'
  echo 'Optional environment settings: TWINKLING_MODEL_DIR, TWINKLING_SERVER, TWINKLING_EXPERT_PLAN, TWINKLING_PORT.'
  echo 'Experimental: TWINKLING_CONTEXT, TWINKLING_PREFILL_CHUNK, TWINKLING_KV=fp8|v4, TWINKLING_PREFILL_OVERLAP=1, TWINKLING_PARK_HOST_MIB.'
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
args=(--engine --model-dir "$model" --host 127.0.0.1 --port "$port" --max-ctx "$ctx"
      --dense-fp8 --spec --spec-threshold 0.8 --no-vision --park-mb 0 --embed-host
      --prefill-chunk "$chunk" --prefill-stage-mib 256 --expert-plan "$plan" --max-queue 8)
case "$cache" in
  fp8) args+=(--kv-fp8) ;;
  v4) args+=(--kv-v4) ;;
  *) echo 'TWINKLING_KV must be fp8 or v4.' >&2; exit 2 ;;
esac
if [[ "${TWINKLING_PREFILL_OVERLAP:-0}" == 1 ]]; then args+=(--prefill-overlap); fi
if [[ -n "${TWINKLING_PARK_HOST_MIB:-}" ]]; then args+=(--park-host-mb "$TWINKLING_PARK_HOST_MIB"); fi
if [[ -r "$model/inkling-2-mini-abliterated.safetensors" ]]; then
  args+=(--ablation "$model/inkling-2-mini-abliterated.safetensors" --model-id inkling-2-mini-abliterated)
fi
exec "$server" "${args[@]}"
