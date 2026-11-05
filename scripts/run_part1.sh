#!/usr/bin/env bash
# Part 1: base vs paper-edited (both Q8_0 GGUF) on the MATH-500 subset (resumable: finished rows are skipped).
# Pinned to cores 0-4 (5 threads) so the agent harness can use cores 5-7 at the same time.
# Sample counts were cut after measuring ~200 tok/s aggregate on this box (B, C: 1 sample; A at 8k: 20 problems).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../venv/bin/python}; G=${GGUF_DIR:-../gguf}; O=runs/part1; mkdir -p $O
run() { taskset -c ${CPUS:-0-4} $PY scripts/eval_math.py --model $G/$1-q8.gguf --tag $1 --cond $2 --cap $3 --samples $4 --n $5 --parallel $6 --threads ${THREADS:-5} --out $O/$2_cap$3_$1.jsonl; }
for m in base edited; do run $m A 4096 2 60 32; done
for m in base edited; do run $m C 4096 1 60 30; done
for m in base edited; do run $m B 4096 1 60 30; done
for m in base edited; do run $m A 8192 1 20 20; done
echo PART1_DONE
