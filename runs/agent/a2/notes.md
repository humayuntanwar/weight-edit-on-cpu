# Notes (attempt a2)
- Pass 1: `wsp gen --model original --source math --n 32 --seed 1` (max 1024 tokens). Took 2,324 s wall time (~39 min of the 60-min budget; the machine is shared).
- Result: 32/32 cut at 1024 tokens; 0 closed </think>; accuracy 0/32.
- Text stats: 1.9k-3.6k chars each; 0-8 "Wait" per text; almost no duplicated lines (max 4) -> no degenerate looping within 1024 tokens.
  The failures are long, still-progressing reasoning (planning bullet lists, re-checks), not repetition. At 1024 tokens there is
  no signal separating right from wrong reasoning, and no failure mechanism showed up that I could localize.
- Not enough time left for a paired original/edited pass (~40 min each at this load), so no edit could be verified.
- Decision: submit no edit (./edited empty).
