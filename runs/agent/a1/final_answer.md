# Final edit: ablate the "hesitation / Wait" direction from layers 15-23 output writes

## What changed
- Tensors (18): for each layer l = 15..23: `mlp.down_proj.weight`, plus `linear_attn.out_proj.weight` (l = 16,17,18,20,21,22) or `self_attn.o_proj.weight` (l = 15,19,23). 51.9M weights modified (rank-1 change per tensor).
- Edit: W' = (I - d d^T) W, where d is a unit vector in the 1024-d residual stream. This removes everything these layers write along d (alpha = 1.0). Embeddings, the final norm and lm_head are untouched.
- d = normalized sum of unit diff-of-means directions at the residual after layer 20 and after layer 22: (positions where the next token is " Wait"/"Wait") minus (control positions with the same preceding tokens, e.g. indentation, "$.", "\n", that are *not* followed by Wait). Computed from teacher-forced reads of 32 of the model's own MATH generations (pass 2).

## Evidence
- Diagnosis (pass 1, 32 MATH problems, original): 32/32 answers were cut at 1,024 tokens. The model writes very long structured "thinking processes" with repeated self-doubt loops ("Wait, let's re-verify", "No, wait", "Actually"), often after it already has the right value. At the eval's 4,096-token limit, running out of tokens is the main way it fails.
- Localization (pass 2): a linear direction predicts the next token being "Wait" against matched controls, with held-out AUC 0.84 (L14) → 0.89 (L20) → 0.91 (L22). It builds up gradually through layers 14-22 (|d| grows 0.6 → 2.9; cosine between d20 and d22 is 0.89). MLP neuron attributions are spread out (the top 10 neurons explain only a small part of each layer's contribution), so I edited at the level of the direction, not single neurons.
- Causal effect, dose-response (paired, same seeds):
  - alpha 0.7 (pass 3, 16 MATH level 2-3): Wait 36→28, Actually 8→2; correct answer appearing in text 7→9.
  - alpha 1.0 (pass 4, 16 GSM8K): Wait 41→20, Actually 16→0; correct answer appearing in text 8→11; reaching "final answer" wording 3→8; mean characters within 1,024 tokens 3,681→3,397. The text stays coherent and still checks itself, just with fewer re-verification loops.
- Limitation: every generation on both models hit the 1,024-token cap (even GSM8K), so boxed-answer accuracy could not be measured here (0 vs 0). The expected gain is that more solutions finish inside 4,096 tokens. The evidence is that the targeted hesitation tokens fall and the model gets further per token. A drop in accuracy from losing useful self-corrections cannot be ruled out with this sample size.
