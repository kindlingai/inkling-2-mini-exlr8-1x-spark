#!/usr/bin/env bash
set -euo pipefail

# Single-Spark (TP=1) launcher for Inkling-2-Mini exlr8 + Melded DFlash2.
# Works inside ghcr.io/kindlingai/gr8:main (/opt/gr8) or a bare-metal gr8 checkout.

GR8_ROOT="${GR8_ROOT:-/opt/gr8}"
G6M4="${G6M4:-$GR8_ROOT/bin/g6m4_ink2}"
FRONT="${FRONT:-$GR8_ROOT/bin/g6front-rs}"
SHIM="${SHIM:-$GR8_ROOT/frontend/g6s2shim.py}"
PY="${PY:-python3}"

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your local kindlingai/inkling-2-mini-exlr8-k2-k3-k4 directory}"
VARIANT="${VARIANT:-k4}"
MTP="${MTP:-7}"
CAP="${CAP:-16384}"
SESSIONS="${SESSIONS:-8}"
CHUNK="${CHUNK:-8}"
PREFILL="${PREFILL:-4096}"
PORT="${PORT:-8100}"
WORK_DIR="${WORK_DIR:-/tmp/ink2-gr8}"

export GR8_INK2_DF2_Q8="${GR8_INK2_DF2_Q8:-0}"
export GR8_INK2_PF_MX8_MOE="${GR8_INK2_PF_MX8_MOE:-2}"
export GR8_INK2_MX8_DENSE="${GR8_INK2_MX8_DENSE:-2}"
export GR8_INK2_MX8_LM_HEAD="${GR8_INK2_MX8_LM_HEAD:-0}"

mkdir -p "$WORK_DIR"
SOCK="$WORK_DIR/s2.sock"
rm -f "$SOCK"

cleanup() {
  trap - EXIT INT TERM
  kill "${FRONT_PID:-}" "${SHIM_PID:-}" "${ENG_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Inkling-2-Mini (variant=$VARIANT, mtp=$MTP, cap=$CAP, prefill=$PREFILL, port=$PORT)..."

# Zero GR8_INK2_DFLASH2 flag needed: g6m4_ink2 auto-discovers dflash2-melded.safetensors in MODEL_DIR.
"$G6M4" \
  --rank 0 --ips 127.0.0.1 \
  --v2-dense "$MODEL_DIR" \
  --variant "$VARIANT" \
  --mtp "$MTP" \
  --cap "$CAP" \
  --sessions "$SESSIONS" \
  --chunk "$CHUNK" \
  --prefill "$PREFILL" \
  --s2-sock "$SOCK" &
ENG_PID=$!

for _ in $(seq 1 120); do
  [ -S "$SOCK" ] && break
  kill -0 "$ENG_PID" 2>/dev/null || { echo "Engine exited early" >&2; exit 1; }
  sleep 0.5
done
[ -S "$SOCK" ] || { echo "Timed out waiting for $SOCK" >&2; exit 1; }

"$PY" "$SHIM" --sock "$SOCK" --work "$WORK_DIR" &
SHIM_PID=$!

"$FRONT" \
  --listen "0.0.0.0:$PORT" \
  --dialect inkling \
  --model-dir "$MODEL_DIR" \
  --served-model "inkling-2-mini" \
  --work "$WORK_DIR" &
FRONT_PID=$!

wait -n "$ENG_PID" "$SHIM_PID" "$FRONT_PID"
