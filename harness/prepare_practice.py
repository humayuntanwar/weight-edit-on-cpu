"""Practice problems for the agent: GSM8K train and MATH train only (no test split). Any MATH train problem whose
whitespace-normalized text equals a MATH-500 problem is dropped (as a guard; MATH-500 comes from the MATH test split)."""
import json, os, re, sys
from datasets import load_dataset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from mathgrade import last_boxed

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", s).strip()
m500 = {norm(r["problem"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
n = 0
with open(f"{out}/gsm8k_train.jsonl", "w") as f:
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        f.write(json.dumps({"id": f"gsm8k:{i}", "source": "gsm8k", "problem": r["question"], "answer": ans,
                            "solution": r["answer"]}) + "\n"); n += 1
print("gsm8k", n)
n = drop = 0
with open(f"{out}/math_train.jsonl", "w") as f:
    for i, r in enumerate(load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")):
        if norm(r["problem"]) in m500:
            drop += 1; continue
        ans = last_boxed(r["solution"])
        if ans is None:
            continue
        f.write(json.dumps({"id": f"math:{i}", "source": "math", "subject": r["type"], "level": r["level"],
                            "problem": r["problem"], "answer": ans, "solution": r["solution"]}) + "\n"); n += 1
print("math", n, "dropped as MATH-500 duplicates:", drop)
