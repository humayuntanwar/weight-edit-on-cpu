"""Fixed MATH-500 subset: 12 problems per difficulty level (60 total), chosen with random.Random(0)."""
import json, random, sys
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
rows = [dict(r) for r in ds]
rng = random.Random(0)
out = []
for lvl in range(1, 6):
    pool = sorted([r for r in rows if int(r["level"]) == lvl], key=lambda r: r["unique_id"])
    out += rng.sample(pool, 12)
# interleave levels so any prefix of the list is still roughly stratified
out = [out[l * 12 + i] for i in range(12) for l in range(5)]
with open(sys.argv[1], "w") as f:
    for i, r in enumerate(out):
        f.write(json.dumps({"idx": i, "unique_id": r["unique_id"], "level": int(r["level"]), "subject": r["subject"],
                            "problem": r["problem"], "answer": r["answer"]}) + "\n")
print(len(out), "problems written")
