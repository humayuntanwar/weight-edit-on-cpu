"""Cross-check from the authors' own generations (no new sampling).

The logs repo (RikiyaT/lm_understand-logs, exp8_math/eval/<run>/parts/math500_*/math500_first_samples.jsonl.gz) holds
the full text of the first sample for 200 MATH-500 problems, generated with vLLM at a 32,768-token cap. Generation is
causal, so the first T tokens of such a sample are a sample with a T-token cap. This script cuts each text at T tokens
(Qwen3.5 tokenizer), grades the last \\boxed{} with the same grader as our runs, and also grades the full text
(checked against the authors' own first-sample grades).
usage: authors_truncated.py <exp8_math/eval dir> <tokenizer dir> <out.json>
"""
import glob, gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathgrade import grade
from score import wilson
from transformers import AutoTokenizer

ev, tokdir, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(tokdir)
res = {}
for run in ("unedited", "p10c", "p25c", "p50c"):
    rows, author = [], {}
    for d in sorted(glob.glob(f"{ev}/{run}/parts/math500_*")):
        for p in json.load(open(f"{d}/math500_per_problem.json")):
            author[p["id"]] = p["correct"][0]
        with gzip.open(f"{d}/math500_first_samples.jsonl.gz", "rt") as f:
            rows += [json.loads(l) for l in f]
    ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    R = {"problems": len(rows)}
    full = [grade(r["text"], r["answer"])["correct"] for r in rows]
    R["full_32768"] = {"correct": sum(full), "agree_with_author_grades": sum(int(a) == author[r["id"]] for a, r in zip(full, rows))}
    for T in (1024, 4096, 8192, 32768):
        k = cut = 0
        for r, x in zip(rows, ids):
            hit = len(x) >= T
            text = tok.decode(x[:T]) if hit else r["text"]
            k += grade(text, r["answer"])["correct"]; cut += hit
        n = len(rows)
        R[f"cap_{T}"] = {"n": n, "correct": k, "accuracy": k / n, "acc_ci95": wilson(k, n), "hit_cap": cut,
                         "hit_cap_share": cut / n}
    res[run] = R
    print(run, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'acc_ci95'}) for k, v in R.items()}))
json.dump(res, open(out, "w"), indent=1)
