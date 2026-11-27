#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PC_BIN="${TWINKLING_PC:-$HOME/bin/pc}"
REMOTE_HOST="${TWINKLING_REMOTE_HOST:-degitaar}"
REMOTE_REPO="${TWINKLING_REMOTE_REPO:-/root/twinkling-20261127/p3}"
REMOTE_INSTALLED="${TWINKLING_REMOTE_INSTALLED:-/root/ingine}"
WSL_DISTRO="${TWINKLING_WSL_DISTRO:-Ubuntu-26.04}"
STAGING_WIN="${TWINKLING_STAGING_WIN:-C:/Users/pc-agent/AgentWorkspace/twinkling-20261127}"
STAGING_WSL="${TWINKLING_STAGING_WSL:-/mnt/c/Users/pc-agent/AgentWorkspace/twinkling-20261127}"
RESERVE="${TWINKLING_RESERVE:-/Users/michael/bin/pure-reserve}"
RESERVE_MINUTES="${TWINKLING_RESERVE_MINUTES:-45}"
RESERVE_QUIET="${TWINKLING_RESERVE_QUIET:-300}"
TRANSPORT="${TWINKLING_TRANSPORT:-pc}"
SERVE_PORT="${TWINKLING_VALIDATE_PORT:-18368}"
RESULTS_DIR="${TWINKLING_RESULTS_DIR:-/tmp/twinkling-phase3}"

usage() {
  cat <<'EOF'
Usage: validate_phase3.sh [--reserved] [sync|build|tests|ab|sweep|qualify|all]

Subcommands (default: all):
  sync     Sync TWinkling-long to DeGitaar (incremental copy, excludes .git/native/build*/models/.tools/__pycache__).
  build    Configure (if needed) and build native/build-5080 (ingine-serve, ingine-bench, all 7 GPU oracle tests).
  tests    Under pure-reserve (250 W), run all 7 kernel/model tests (Phase 3 ON) + toggle-off sanity checks.
  ab       Under pure-reserve (250 W), run Wave A/B/C isolation A/B at 128K (baseline vs +pack6 vs +ring8 vs +all).
  sweep    Under pure-reserve (250 W), sweep 32k ALL-128, 128k-{11.2,p3,p3-hi}, 256k-{10.0,p3,p3-hi}; enforce >=450 MiB free.
  qualify  Under pure-reserve (250 W), run qualify_long.py (--output-tokens 1536 --headroom 2048) on 32k/128k/256k winners.
  all      Run sync -> build -> tests -> ab -> sweep -> qualify.
EOF
}

guard_pc_hold() {
  local hold="$HOME/.config/pc-hold"
  if [[ -f "$hold" ]]; then
    local val
    val="$(tr '[:upper:]' '[:lower:]' < "$hold" | tr -d '[:space:]')"
    if [[ "$val" == "off" || "$val" == *off* ]]; then
      echo "ERROR: $hold is set ($val). Refusing to contact DeGitaar or call pc-wake/pure-reserve." >&2
      exit 1
    fi
  fi
}

wsl_exec() {
  local script="$1"
  if [[ "$TRANSPORT" == "ssh" ]]; then
    ssh -o BatchMode=yes "$REMOTE_HOST" bash -s -- < <(printf 'set -euo pipefail\n%s\n' "$script")
    return $?
  fi
  local tmp
  tmp="$(mktemp /tmp/tw-p3-cmd.XXXXXX)"
  printf 'set -euo pipefail\n%s\n' "$script" > "$tmp"
  "$PC_BIN" put "$tmp" "$STAGING_WIN/validate-p3-cmd.sh" >/dev/null
  rm -f "$tmp"
  "$PC_BIN" ps "wsl.exe -d $WSL_DISTRO -u root -- bash $STAGING_WSL/validate-p3-cmd.sh; exit \$LASTEXITCODE"
}

run_reserved() {
  local cmd="$1"
  shift || true
  guard_pc_hold
  if [[ -n "${PURE_RESERVATION:-}" ]]; then
    "cmd_${cmd}" "$@"
    return $?
  fi
  [[ -x "$RESERVE" ]] || { echo "ERROR: pure-reserve not found at $RESERVE" >&2; exit 1; }
  "$RESERVE" --owner "twinkling-phase3" --minutes "$RESERVE_MINUTES" --quiet "$RESERVE_QUIET" -- \
    "$0" --reserved "$cmd" "$@"
}

setup_gpu_session() {
  guard_pc_hold
  if [[ "$TRANSPORT" == "pc" ]]; then
    ORIG_PL="$(python3 - <<'PY'
import pathlib, re, runpy, subprocess
home = pathlib.Path.home()
pc = str(home / "bin/pc")
res = subprocess.run([pc, "ps", "nvidia-smi.exe -i 0 --query-gpu=power.limit --format=csv,noheader,nounits; exit $LASTEXITCODE"],
                     capture_output=True, text=True, check=True, timeout=20)
m = re.search(r"(?m)^\s*(\d+(?:\.\d+)?)\s*$", res.stdout)
orig = float(m.group(1)) if m else 250.0
runpy.run_path(str(home / "bin/pure-engine-update"))["switch"]("none")
if abs(orig - 250.0) > 0.05:
    subprocess.run([pc, "ps", "nvidia-smi.exe -i 0 -pl 250; exit $LASTEXITCODE"], check=True, timeout=20)
print(f"{orig:g}")
PY
)"
    trap 'cleanup_gpu_session' EXIT
  fi
}

cleanup_gpu_session() {
  set +e
  wsl_exec "pkill -TERM -f 'ingine-serve.*--port ${SERVE_PORT}' 2>/dev/null || true" >/dev/null 2>&1 || true
  if [[ "$TRANSPORT" == "pc" && -n "${ORIG_PL:-}" && "${ORIG_PL}" != "250" ]]; then
    "$PC_BIN" ps "nvidia-smi.exe -i 0 -pl ${ORIG_PL}; exit \$LASTEXITCODE" >/dev/null 2>&1 || true
  fi
}

cmd_sync() {
  guard_pc_hold
  echo "=== [sync] $ROOT -> $REMOTE_REPO ($TRANSPORT) ==="
  if [[ "$TRANSPORT" == "ssh" ]]; then
    rsync -az --delete \
      --exclude='.git' --exclude='native/build*' --exclude='models/' --exclude='.tools/' --exclude='__pycache__/' \
      "$ROOT/" "$REMOTE_HOST:$REMOTE_REPO/"
    return 0
  fi
  mkdir -p "$RESULTS_DIR"
  local tarfile="$RESULTS_DIR/tw-phase3-sync.tar.gz"
  COPYFILE_DISABLE=1 tar \
    --exclude='.git' --exclude='native/build*' --exclude='models' --exclude='.tools' --exclude='__pycache__' \
    -czf "$tarfile" -C "$ROOT" native benchmarks config scripts
  "$PC_BIN" put "$tarfile" "$STAGING_WIN/tw-phase3-sync.tar.gz" >/dev/null
  wsl_exec "
    mkdir -p '$REMOTE_REPO'
    unpack=\$(mktemp -d /tmp/tw-sync.XXXXXX)
    tar --warning=no-unknown-keyword -xzf '$STAGING_WSL/tw-phase3-sync.tar.gz' -C \"\$unpack\"
    python3 - \"\$unpack\" '$REMOTE_REPO' <<'PY'
import filecmp, shutil, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
changed = 0
for p in src.rglob('*'):
    if p.is_file():
        rel = p.relative_to(src)
        o = dst / rel
        if not o.is_file() or not filecmp.cmp(p, o, shallow=False):
            o.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, o)
            changed += 1
shutil.rmtree(src)
print(f'synced incremental files updated: {changed}')
PY
    chmod +x '$REMOTE_REPO/scripts/rtx5080/serve-5080.sh' '$REMOTE_REPO/scripts/rtx5080/validate_phase3.sh' 2>/dev/null || true
  "
}

cmd_build() {
  guard_pc_hold
  echo "=== [build] preset install-5080 + tests in $REMOTE_REPO/native/build-5080 ==="
  wsl_exec "
    export PATH=\"$REMOTE_INSTALLED/.tools/venv/bin:$REMOTE_INSTALLED/.tools/bin:\$PATH\"
    cmake_bin=\$(command -v cmake || echo '$REMOTE_INSTALLED/.tools/venv/bin/cmake')
    cuda_root='$REMOTE_INSTALLED/.cuda-toolkit/nvidia/cu13'
    nvcc_bin=\$(find '$REMOTE_INSTALLED/.cuda-toolkit' -name nvcc -type f -print -quit)
    [[ -n \"\$nvcc_bin\" ]] || nvcc_bin=\"\$cuda_root/bin/nvcc\"
    host_cxx=\$(sed -n 's/^CMAKE_CUDA_HOST_COMPILER:[^=]*=//p' '$REMOTE_INSTALLED/native/build-5080/CMakeCache.txt' 2>/dev/null | head -1 || true)
    extra=()
    [[ -n \"\$host_cxx\" && -x \"\$host_cxx\" ]] && extra+=(\"-DCMAKE_CUDA_HOST_COMPILER=\$host_cxx\")
    bdir='$REMOTE_REPO/native/build-5080'
    if [[ ! -r \"\$bdir/build.ninja\" ]]; then
      \"\$cmake_bin\" -S '$REMOTE_REPO/native' -B \"\$bdir\" --preset install-5080 \
        -DINGINE_BUILD_TESTS=ON \
        \"-DCMAKE_CUDA_COMPILER=\$nvcc_bin\" \
        \"-DCUDAToolkit_ROOT=\$cuda_root\" \
        \"-DINGINE_TEST_MODEL_DIR=$REMOTE_INSTALLED/models/Inkling-2-Mini-nvfp4\" \
        \"\${extra[@]}\"
    fi
    \"\$cmake_bin\" --build \"\$bdir\" -j 12 --target \
      ingine-serve ingine-bench \
      kernels_moe kernels_attn model_layer model_runtime \
      prefill_moe prefill_rows prefill_attention
  "
}

REMOTE_PREAMBLE='
set -euo pipefail
installed="'"$REMOTE_INSTALLED"'"
model="$installed/models/Inkling-2-Mini-nvfp4"
repo="'"$REMOTE_REPO"'"
bdir="$repo/native/build-5080"
smi=/usr/lib/wsl/lib/nvidia-smi
cuda="$installed/.cuda-toolkit/nvidia/cu13"
export PATH="$installed/.tools/venv/bin:$installed/.tools/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
port="'"$SERVE_PORT"'"

check_idle_gpu() {
  local sample free watts
  sample=$("$smi" --query-gpu=memory.free,power.limit --format=csv,noheader,nounits)
  read -r free watts <<< "${sample/,/}"
  echo "GPU precheck: ${free} MiB free, ${watts} W limit"
  [[ "$free" =~ ^[0-9]+$ && "$free" -ge "${TWINKLING_MIN_INIT_FREE_MIB:-15000}" ]] || {
    echo "ERROR: GPU free VRAM ${free} MiB < 15000 MiB" >&2
    exit 75
  }
}

run_serve_case() {
  local tag="$1" env_str="$2" ctx="$3" kv_flag="$4" plan="$5" chunk="$6" stage="$7" depth="$8" qcases="$9" qmode="${10}" extra_flags="${11:-}"
  pkill -TERM -f "ingine-serve.*--port $port" 2>/dev/null || true
  sleep 0.5
  local args=(--model-dir "$model" --max-ctx "$ctx" --dense-fp8 --embed-host "$kv_flag"
              --prefill-chunk "$chunk" --prefill-stage-mib "$stage" --prefill-overlap
              --spec --spec-threshold 0.8 --engine --host 127.0.0.1 --port "$port"
              --model-id twinkling-phase3 --no-vision --park-mb 0 --max-queue 8)
  [[ "$plan" != "NONE" ]] && args+=(--expert-plan "$repo/config/$plan")
  [[ -n "$extra_flags" ]] && args+=($extra_flags)
  local log="/tmp/tw-${tag}.log" out="/tmp/tw-${tag}.json"
  local t0 t1
  t0=$(date +%s%N)
  local env_arr=()
  if [[ -n "$env_str" ]]; then read -r -a env_arr <<< "$env_str"; fi
  env ${env_arr[@]+"${env_arr[@]}"} taskset -c 0-7 "$bdir/apps/serve/ingine-serve" "${args[@]}" >"$log" 2>&1 &
  local pid=$!
  trap "kill -TERM $pid 2>/dev/null || true; wait $pid 2>/dev/null || true; trap - RETURN" RETURN
  local ready=0
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { tail -40 "$log" >&2; return 1; }
    if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q twinkling-phase3; then
      ready=1; break
    fi
    sleep 0.25
  done
  [[ "$ready" == 1 ]] || { echo "ERROR: $tag serve startup timeout" >&2; tail -40 "$log" >&2; return 1; }
  t1=$(date +%s%N)
  local load_ms=$(( (t1 - t0) / 1000000 ))
  local vram used free
  vram=$("$smi" --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)
  read -r used free <<< "${vram/,/}"
  echo "[$tag] load_ms=${load_ms} vram_used=${used}MiB vram_free=${free}MiB"
  if [[ "$free" -lt 450 ]]; then
    echo "FAIL [$tag]: memory.free=${free} MiB < 450 MiB WDDM safety floor" >&2
    return 2
  fi
  local cflags=()
  for c in $qcases; do cflags+=(--case "$c"); done
  python3 "$repo/benchmarks/rtx5080/qualify_long.py" \
    --url "http://127.0.0.1:$port/v1/chat/completions" \
    --model twinkling-phase3 --depth "$depth" "${cflags[@]}" \
    --mode "$qmode" --timeout 420 --output-tokens 1536 --headroom 2048 --out "$out"
}
'

cmd_tests() {
  setup_gpu_session
  echo "=== [tests] Unit + FP64 oracle tests (default Phase 3 ON + toggle-off checks) ==="
  wsl_exec "
    $REMOTE_PREAMBLE
    check_idle_gpu
    tests=(
      tests/kernels/kernels_moe
      tests/kernels/kernels_attn
      tests/model/model_layer
      tests/model/model_runtime
      tests/prefill/prefill_moe
      tests/prefill/prefill_rows
      tests/prefill/prefill_attention
    )
    for t in \"\${tests[@]}\"; do
      echo \"--- running \$t (default Phase 3 ON) ---\"
      \"\$bdir/\$t\"
    done
    echo '--- toggle-off regression checks ---'
    INGINE_PACK_SCALES=0 \"\$bdir/tests/model/model_layer\"
    INGINE_RING_FP8=1 \"\$bdir/tests/model/model_runtime\"
    INGINE_PREFILL_W4A4=1 INGINE_PREFILL_FUSED_ACTQ=0 \"\$bdir/tests/prefill/prefill_moe\"
    INGINE_PREFILL_FUSED_GATE=0 \"\$bdir/tests/prefill/prefill_rows\"
  "
}

cmd_ab() {
  setup_gpu_session
  echo "=== [ab] Phase 3 Wave isolation A/B (128K, rtx5080-experts-11.2.json) ==="
  wsl_exec "
    $REMOTE_PREAMBLE
    check_idle_gpu
    run_serve_case ab-0-base  'INGINE_PACK_SCALES=0 INGINE_RING_FP8=0 INGINE_PREFILL_FUSED_ACTQ=0 INGINE_PREFILL_FUSED_GATE=0' \
      131072 --kv-k4 rtx5080-experts-11.2.json 2048 128 131072 'code' cold
    run_serve_case ab-1-pack6 'INGINE_PACK_SCALES=1 INGINE_RING_FP8=0 INGINE_PREFILL_FUSED_ACTQ=0 INGINE_PREFILL_FUSED_GATE=0' \
      131072 --kv-k4 rtx5080-experts-11.2.json 2048 128 131072 'code' cold
    run_serve_case ab-2-ring8 'INGINE_PACK_SCALES=1 INGINE_RING_FP8=1 INGINE_PREFILL_FUSED_ACTQ=0 INGINE_PREFILL_FUSED_GATE=0' \
      131072 --kv-k4 rtx5080-experts-11.2.json 2048 128 131072 'code' cold
    run_serve_case ab-3-all   'INGINE_PACK_SCALES=1 INGINE_RING_FP8=0 INGINE_PREFILL_FUSED_ACTQ=1 INGINE_PREFILL_FUSED_GATE=1' \
      131072 --kv-k4 rtx5080-experts-11.2.json 2048 128 131072 'code' cold
  "
}

cmd_sweep() {
  setup_gpu_session
  echo "=== [sweep] Residency & VRAM headroom sweep (ALL-128 @ 32k, p3 plans @ 128k/256k) ==="
  wsl_exec "
    $REMOTE_PREAMBLE
    check_idle_gpu
    fail=0
    run_serve_case sw-32k-all128 '' 32768 --kv-v4 NONE 2048 0 16384 'code recall' cold '--elastic-kv' || fail=1
    run_serve_case sw-128k-112   '' 131072 --kv-k4 rtx5080-experts-11.2.json 2048 128 131072 'code recall' cold || fail=1
    run_serve_case sw-128k-p3    '' 131072 --kv-k4 rtx5080-experts-128k-p3.json 2048 128 131072 'code recall' cold || fail=1
    run_serve_case sw-128k-p3-hi '' 131072 --kv-k4 rtx5080-experts-128k-p3-hi.json 2048 128 131072 'code recall' cold || fail=1
    run_serve_case sw-256k-100   '' 262144 --kv-k4 rtx5080-experts-10.0.json 2048 128 262144 'code recall' cold || fail=1
    run_serve_case sw-256k-p3    '' 262144 --kv-k4 rtx5080-experts-256k-p3.json 2048 128 262144 'code recall' cold || fail=1
    run_serve_case sw-256k-p3-hi '' 262144 --kv-k4 rtx5080-experts-256k-p3-hi.json 2048 128 262144 'code recall' cold || fail=1
    exit \$fail
  "
}

cmd_qualify() {
  setup_gpu_session
  local plan128="${TWINKLING_QUALIFY_128K_PLAN:-rtx5080-experts-128k-p3.json}"
  local plan256="${TWINKLING_QUALIFY_256K_PLAN:-rtx5080-experts-256k-p3.json}"
  echo "=== [qualify] Full qualify_long.py (32k ALL-128, 128k=$plan128, 256k=$plan256) ==="
  wsl_exec "
    $REMOTE_PREAMBLE
    check_idle_gpu
    run_serve_case q-32k-all128 '' 32768 --kv-v4 NONE 2048 0 16384 'recall code decode tool' both '--elastic-kv'
    run_serve_case q-128k '' 131072 --kv-k4 '$plan128' 2048 128 131072 'recall code decode tool' both
    run_serve_case q-256k '' 262144 --kv-k4 '$plan256' 2048 128 262144 'recall code decode tool' both
  "
}

RESERVED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --reserved) RESERVED=1; shift ;;
    *) break ;;
  esac
done

SUB="${1:-all}"
shift || true

case "$SUB" in
  sync) cmd_sync "$@" ;;
  build) cmd_build "$@" ;;
  tests|ab|sweep|qualify)
    if [[ "$RESERVED" == 1 ]]; then
      "cmd_${SUB}" "$@"
    else
      run_reserved "$SUB" "$@"
    fi
    ;;
  all)
    cmd_sync
    cmd_build
    if [[ "$RESERVED" == 1 ]]; then
      cmd_tests
      cmd_ab
      cmd_sweep
      cmd_qualify
    else
      guard_pc_hold
      [[ -x "$RESERVE" ]] || { echo "ERROR: pure-reserve not found at $RESERVE" >&2; exit 1; }
      "$RESERVE" --owner "twinkling-phase3" --minutes "$RESERVE_MINUTES" --quiet "$RESERVE_QUIET" -- bash -c '
        set -euo pipefail
        "$0" --reserved tests
        "$0" --reserved ab
        "$0" --reserved sweep
        "$0" --reserved qualify
      ' "$0"
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
