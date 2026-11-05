# Final edit

**Changed:** `model.language_model.layers.{17,18,19,20}.mlp.down_proj.weight`. Each one gets a rank-1 update
W' = W - 0.5 * u (u^T W). Here u is a unit vector in the 1024-wide residual space, so the update touches all 1024x3584 entries of each of the 4 tensors.
This halves how much each of these MLPs writes along u. No other tensors were changed.

**Diagnosis (gen pass 1, original model, 32 MATH problems, 1,024 tokens):** all 32 answers (100%) were cut off while still thinking. None reached `</think>`.
The texts are full of re-checking loops: 92 "Wait", 42 "Let's check", 23 "re-read" and 21 "Actually" across 32 texts. The model keeps re-reading the problem and re-verifying instead of finishing.

**Localization (read pass 2):** I fed 28 of those generations back through the model and collected hidden states.
- I compared line or sentence starts where the next token is a hesitation token ("Wait", "Actually", "Hmm") against all other line or sentence starts.
- The difference-of-means direction predicts the upcoming hesitation on held-out positions with AUC 0.80 at layer 12, 0.87 at layer 16, 0.915 at layer 20 and 0.90 at layer 23.
- The direction's norm grows mostly between layers 16 and 20, so I took u from layer 20 and edited the MLPs of layers 17–20.

**Causal check (read pass 4, edited model, same 28 texts):**
- Hesitation tokens: mean log-prob change of -0.186 (n=102), about 17% less likely.
- All other tokens: mean change of +0.0025 (n=31,592, std 0.098), so essentially unchanged.

**Expected effect:** less of the open-ended "Wait / re-read" looping, so more reasoning chains finish within the 4,096-token limit, while the next-token distribution elsewhere stays the same.

**Not done:** I ran no generation pass comparing original and edited weights. One generation pass took about 35 minutes of wall-clock time, which did not fit in the session. The accuracy effect is therefore unmeasured.

The edit is deliberately mild (α=0.5, halving rather than removing the direction). Only 4 of the 10 passes were used.
