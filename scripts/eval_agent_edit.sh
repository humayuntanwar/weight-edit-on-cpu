#!/usr/bin/env bash
# Build the GGUF for one agent attempt's submitted edit and evaluate it with condition A (paper's sampling, 4,096 cap,
# 60 problems x 2 samples), the same protocol as the Part 1 base and paper-edit runs.
# usage: eval_agent_edit.sh <attempt-name> [cpus] [threads]
set -euo pipefail
cd "$(dirname "$0")/.."
N=$1; CPUS=${2:-0-7}; TH=${3:-8}
W=${WSP_HOME:-/home/box/wsp_agent}; PY=${PY:-../venv/bin/python}; GG=../gguf/agent_$N-q8.gguf
mkdir -p runs/agent/$N runs/agent_eval
$PY scripts/agent_edit_summary.py $W/base/model.safetensors-00001-of-00001.safetensors $W/attempts/$N/work/edited runs/agent/$N
[ -f $GG ] || WSP_THREADS=$TH taskset -c $CPUS $PY harness/build_model.py $W/base $W/attempts/$N/work/edited $GG
taskset -c $CPUS $PY scripts/eval_math.py --model $GG --tag agent_$N --cond A --cap 4096 --samples 2 --n 60 \
  --parallel 32 --threads $TH --port 8093 --out runs/agent_eval/A_cap4096_agent_$N.jsonl
