"""Re-derive every number from the raw generations.

Reads runs/part1/*.jsonl[.gz] and runs/agent_eval/*.jsonl[.gz] (one row per generation), groups them by
(condition, cap, model) and writes runs/summary.json plus a markdown table to stdout.
Intervals: Wilson 95% for shares over generations; for paired edited-minus-base differences a 95% bootstrap over
problems (both samples of a problem stay together), 5,000 resamples, seed 0.
Grader: math-verify on the last \\boxed{} (see scripts/mathgrade.py); `correct` is stored in each row and re-checked
here with --regrade.
"""
import argparse, glob, gzip, json, math, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def read_rows(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def boot_diff(a, b, seed=0, B=5000):
    """a, b: dict idx -> list of 0/1 (same keys and sample sets). Returns mean diff (b - a) and 95% CI."""
    keys = sorted(a)
    da = {k: sum(b[k]) - sum(a[k]) for k in keys}
    nk = {k: len(a[k]) for k in keys}
    tot = sum(nk.values())
    md = sum(da.values()) / tot
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        bs.append(sum(da[k] for k in s) / sum(nk[k] for k in s))
    bs.sort()
    return md, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regrade", action="store_true", help="re-grade every response with math-verify")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "summary.json"))
    a = ap.parse_args()
    rows = read_rows(os.path.join(ROOT, "runs", "part1", "*.jsonl*")) + \
        read_rows(os.path.join(ROOT, "runs", "agent_eval", "*.jsonl*"))
    rows = [r for r in rows if "response" in r]
    if a.regrade:
        sys.path.insert(0, HERE)
        from mathgrade import grade
        changed = 0
        for r in rows:
            g = grade(r["response"], r["gold"])
            changed += g["correct"] != r["correct"]
            r["correct"] = g["correct"]
        print(f"regraded {len(rows)} rows, {changed} changed", file=sys.stderr)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cond"], r["cap"], r["model"])].append(r)
    out = {"groups": [], "paired_vs_base": []}
    for (cond, cap, model), rr in sorted(groups.items()):
        n = len(rr)
        k = sum(r["correct"] for r in rr)
        cut = sum(r["hit_cap"] for r in rr)
        fin = [r for r in rr if not r["hit_cap"]]
        out["groups"].append({
            "cond": cond, "cap": cap, "model": model, "generations": n,
            "problems": len({r["idx"] for r in rr}), "samples_per_problem": max(r["sample"] for r in rr) + 1,
            "correct": k, "accuracy": k / n, "acc_ci95": wilson(k, n),
            "hit_cap": cut, "hit_cap_share": cut / n, "hit_cap_ci95": wilson(cut, n),
            "mean_tokens": sum(r["tokens"] for r in rr) / n,
            "finished": len(fin), "acc_among_finished": (sum(r["correct"] for r in fin) / len(fin)) if fin else None,
            "no_box": sum(r["pred"] is None for r in rr),
            "correct_str_match": sum(r["correct_str"] for r in rr),
        })
    # paired comparisons against the base model on the same (problem, sample) pairs
    for (cond, cap, model), rr in sorted(groups.items()):
        if model == "base" or (cond, cap, "base") not in groups:
            continue
        base = {(r["idx"], r["sample"]): int(r["correct"]) for r in groups[(cond, cap, "base")]}
        mine = {(r["idx"], r["sample"]): int(r["correct"]) for r in rr}
        common = sorted(set(base) & set(mine))
        if not common:
            continue
        A, B = defaultdict(list), defaultdict(list)
        for key in common:
            A[key[0]].append(base[key]); B[key[0]].append(mine[key])
        md, lo, hi = boot_diff(A, B)
        bc = {(r["idx"], r["sample"]): int(r["hit_cap"]) for r in groups[(cond, cap, "base")]}
        mc = {(r["idx"], r["sample"]): int(r["hit_cap"]) for r in rr}
        out["paired_vs_base"].append({
            "cond": cond, "cap": cap, "model": model, "pairs": len(common), "problems": len(A),
            "base_acc": sum(base[c] for c in common) / len(common), "model_acc": sum(mine[c] for c in common) / len(common),
            "diff": md, "diff_ci95_bootstrap": [lo, hi],
            "wrong_to_right": sum(1 for c in common if mine[c] > base[c]),
            "right_to_wrong": sum(1 for c in common if mine[c] < base[c]),
            "base_hit_cap": sum(bc[c] for c in common) / len(common), "model_hit_cap": sum(mc[c] for c in common) / len(common)})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("| cond | cap | model | n (problems x samples) | accuracy [95% CI] | hit cap [95% CI] | mean tokens | acc. among finished |")
    print("|---|---|---|---|---|---|---|---|")
    for g in out["groups"]:
        af = "-" if g["acc_among_finished"] is None else f"{100*g['acc_among_finished']:.0f}% of {g['finished']}"
        print(f"| {g['cond']} | {g['cap']} | {g['model']} | {g['generations']} ({g['problems']}x{g['samples_per_problem']}) | "
              f"{100*g['accuracy']:.1f}% [{100*g['acc_ci95'][0]:.1f}, {100*g['acc_ci95'][1]:.1f}] | "
              f"{100*g['hit_cap_share']:.1f}% [{100*g['hit_cap_ci95'][0]:.1f}, {100*g['hit_cap_ci95'][1]:.1f}] | "
              f"{g['mean_tokens']:.0f} | {af} |")
    print()
    print("| cond | cap | model vs base | pairs | base acc | model acc | diff [95% bootstrap CI] | wrong->right | right->wrong |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in out["paired_vs_base"]:
        print(f"| {p['cond']} | {p['cap']} | {p['model']} | {p['pairs']} | {100*p['base_acc']:.1f}% | {100*p['model_acc']:.1f}% | "
              f"{100*p['diff']:+.1f} pts [{100*p['diff_ci95_bootstrap'][0]:+.1f}, {100*p['diff_ci95_bootstrap'][1]:+.1f}] | "
              f"{p['wrong_to_right']} | {p['right_to_wrong']} |")


if __name__ == "__main__":
    main()
