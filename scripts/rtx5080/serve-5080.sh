#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
model="${TWINKLING_MODEL_DIR:-$root/models/Inkling-2-Mini-nvfp4}"
server="${TWINKLING_SERVER:-$root/native/build-5080/apps/serve/ingine-serve}"
port="${TWINKLING_PORT:-18298}"
profile="${TWINKLING_PROFILE:-64k}"

if [[ "${1:-}" == --help ]]; then
  echo 'Start TWinkling on localhost. TWINKLING_PROFILE=64k (default), 32k (fastest), 128k, or 256k.'
  echo 'Optional environment settings: TWINKLING_MODEL_DIR, TWINKLING_SERVER, TWINKLING_EXPERT_PLAN, TWINKLING_PORT.'
  echo 'Experimental: TWINKLING_CONTEXT, TWINKLING_PREFILL_CHUNK, TWINKLING_KV=fp8|v4|k4, TWINKLING_PREFILL_OVERLAP=1, TWINKLING_PARK_HOST_MIB.'
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo 'Use --help for the launcher options.' >&2
  exit 2
fi

case "$profile" in
  # 32k/128k/256k are the measured RTX 5080 recipes in benchmarks/rtx5080/long-context.
  32k) capacity=32768; width=2048; precision=v4; overlap=0; stage=0; elastic=1; experts="" ;;
  64k) capacity=65536; width=2048; precision=fp8; overlap=0; stage=256; elastic=0; experts=rtx5080-experts.json ;;
  128k) capacity=131072; width=2048; precision=k4; overlap=1; stage=128; elastic=0; experts=rtx5080-experts-128k-p3.json ;;
  256k) capacity=262144; width=2048; precision=k4; overlap=1; stage=128; elastic=0; experts=rtx5080-experts-256k-p3.json ;;
  *) echo 'TWINKLING_PROFILE must be 32k, 64k, 128k, or 256k.' >&2; exit 2 ;;
esac
plan="${TWINKLING_EXPERT_PLAN:-${experts:+$root/config/$experts}}"
ctx="${TWINKLING_CONTEXT:-$capacity}"
chunk="${TWINKLING_PREFILL_CHUNK:-$width}"
cache="${TWINKLING_KV:-$precision}"

[[ -x "$server" && ( -z "$plan" || -r "$plan" ) && -r "$model/model.safetensors.index.json" ]] || {
  echo 'The server, converted model or expert plan is missing. Run scripts/setup-5080.sh first.' >&2
  exit 1
}
args=(--engine --model-dir "$model" --host 127.0.0.1 --port "$port" --max-ctx "$ctx"
      --dense-fp8 --spec --spec-threshold 0.8 --no-vision --park-mb 0 --embed-host
      --prefill-chunk "$chunk" --prefill-stage-mib "$stage" --max-queue 8)
if [[ -n "$plan" ]]; then args+=(--expert-plan "$plan"); fi
case "$cache" in
  fp8) args+=(--kv-fp8) ;;
  v4) args+=(--kv-v4) ;;
  k4) args+=(--kv-k4) ;;
  *) echo 'TWINKLING_KV must be fp8, v4 or k4.' >&2; exit 2 ;;
esac
if [[ "$elastic" == 1 ]]; then args+=(--elastic-kv); fi
if [[ "${TWINKLING_PREFILL_OVERLAP:-$overlap}" == 1 ]]; then args+=(--prefill-overlap); fi
if [[ -n "${TWINKLING_PARK_HOST_MIB:-}" ]]; then args+=(--park-host-mb "$TWINKLING_PARK_HOST_MIB"); fi
if [[ "$profile" == 64k && -r "$model/inkling-2-mini-abliterated.safetensors" ]]; then
  args+=(--ablation "$model/inkling-2-mini-abliterated.safetensors" --model-id inkling-2-mini-abliterated)
fi
exec "$server" "${args[@]}"
