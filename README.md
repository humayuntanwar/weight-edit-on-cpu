# weight-edit-on-cpu

Re-running the "Sparks of Weight-Space Programming" math claim (Qwen3.5-0.8B, MIT CSAIL, Nov 2 2026) on a CPU-only
box (8 cores, 15 GB RAM, no GPU), and letting a separate LLM agent try its own blind weight edit with a small
forward-pass budget.

The post reports MATH-500 7.9% -> 48.1% after subtracting a "looping" direction from 64 output columns of
`mlp.down_proj` in layers 15 and 19, measured with thinking on, a 32,768-token cap and 4 samples per problem.

{{HEADLINE}}

## What was run

* **Checkpoints.** `Qwen/Qwen3.5-0.8B` (rev `2fc0636`) and `RikiyaT/Qwen3.5-0.8B-weight-space-programming`. All 488
  tensors compared by sha256 (`scripts/diff_checkpoints.py`, `runs/checkpoint_diff.json`): exactly two differ,
  `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj`, 64 columns each, 65,260 + 64,928 = **130,188 changed
  bfloat16 entries** (the post's 131,072 counts the touched entries; this matches the authors' own manifest).
* **Engine.** llama.cpp b11870 (`llama-server`, raw-prompt completions), both models converted with
  `convert_hf_to_gguf.py` and quantized to **Q8_0** (2x the BF16 speed here). The edit survives conversion: rebuilding
  the edited GGUF from base + the two tensors gives bit-identical tensor data.
* **Problems.** 60 MATH-500 problems, 12 per level, `random.Random(0)` (`runs/math500_subset60.jsonl`). Prompt and
  chat template as in the authors' code: problem + "Please reason step by step, and put your final answer within
  \boxed{}.". **Grader:** last `\boxed{}` after the last `</think>` (whole text if none), checked with `math-verify`
  0.9.0 like the authors (`scripts/mathgrade.py`). On the authors' own 200 published first samples it agrees with
  their grades 200/200.
* **Conditions** (all with a 4,096-token cap, 1 sample per problem with the same seed for every model):

| | thinking | temperature | top_p | top_k | presence penalty | source |
|---|---|---|---|---|---|---|
| A | on | 0.6 | 0.95 | 20 | 1.5 | the post's evaluation |
| B | on | 1.0 | 0.95 | 20 | 1.5 | model card, thinking mode, text tasks |
| C | off | 1.0 | 1.0 | 20 | 2.0 | model card, non-thinking mode, text tasks |

min_p 0 and repetition penalty 1.0 everywhere; the presence penalty covers the whole generated text, as in vLLM.

## Results (Part 1: base vs the paper's edit)

{{TABLE1}}

{{PAIRED}}

**The cap is the whole story at CPU budgets.** The authors published the full text of the first sample for 200
MATH-500 problems (vLLM, 32,768 cap). Generation is causal, so cutting those texts at T tokens gives real T-cap
samples. Same grader, n = 200 per cell (`scripts/authors_truncated.py`, `runs/authors_first_samples_truncated.json`):

{{TABLE_TRUNC}}

## Part 2: a blind agent edits the weights

{{PART2}}

## Is this fair to the authors?

{{FAIR}}

## Caveats

{{CAVEATS}}

## How to run

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt   # torch from the CPU index
hf download Qwen/Qwen3.5-0.8B --local-dir ../models/base
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --local-dir ../models/edited
python scripts/diff_checkpoints.py ../models/{base,edited}/model.safetensors-00001-of-00001.safetensors runs/checkpoint_diff.json
# convert both with llama.cpp's convert_hf_to_gguf.py (--outtype bf16) and llama-quantize ... Q8_0 into ../gguf/
python scripts/make_subset.py runs/math500_subset60.jsonl
scripts/run_part1.sh                                   # Part 1 (resumable)
python harness/prepare_practice.py ../practice         # GSM8K + MATH train only
harness/run_attempt.sh a1 10 60                        # one blind agent attempt: budget 10 passes, 60 minutes
scripts/eval_agent_edit.sh a1 0-7 8                    # evaluate its edit with condition A
python scripts/score.py && python scripts/make_charts.py
```

## Time on this box

{{TIME}}
