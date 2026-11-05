#!/usr/bin/env bash
# Build the GGUF for one agent attempt's submitted edit and evaluate it with condition A (paper's sampling, 4,096 cap)
# on the same MATH-500 subset as Part 1. Default: 60 problems x 1 sample (sample 0; seeds match the Part 1 sample 0).
# usage: eval_agent_edit.sh <attempt-name> [cpus] [threads]
set -euo pipefail
cd "$(dirname "$0")/.."
N=$1; CPUS=${2:-0-7}; TH=${3:-8}; SAMPLES=${SAMPLES:-1}; PAR=${PARALLEL:-24}
W=${WSP_HOME:-/home/box/wsp_agent}; PY=${PY:-../venv/bin/python}; GG=../gguf/agent_$N-q8.gguf
if [ -z "$(ls $W/attempts/$N/work/edited/*.npy 2>/dev/null)" ]; then echo "$N submitted no edit: its model is the base model"; exit 0; fi
mkdir -p runs/agent/$N runs/agent_eval
[ -f runs/agent/$N/edit_manifest.json ] || OMP_NUM_THREADS=$TH taskset -c $CPUS $PY scripts/agent_edit_summary.py \
  $W/base/model.safetensors-00001-of-00001.safetensors $W/attempts/$N/work/edited runs/agent/$N
[ -f $GG ] || WSP_THREADS=$TH taskset -c $CPUS $PY harness/build_model.py $W/base $W/attempts/$N/work/edited $GG
taskset -c $CPUS $PY scripts/eval_math.py --model $GG --tag agent_$N --cond A --cap 4096 --samples $SAMPLES --n 60 \
  --parallel $PAR --threads $TH --port 8093 --out runs/agent_eval/A_cap4096_agent_$N.jsonl
