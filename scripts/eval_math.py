"""Run one model on the MATH-500 subset under one condition; append raw generations to a jsonl file (resumable).

usage: eval_math.py --model X.gguf --tag base --cond A --cap 4096 --samples 2 --n 60 --out runs/part1/A4k_base.jsonl
"""
import argparse, concurrent.futures as cf, gzip, hashlib, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathgrade import build_user_message, grade
from llserver import Server

# A: the paper's evaluation sampling. B: model card, thinking mode, text tasks. C: model card, non-thinking, text tasks.
CONDITIONS = {
    "A": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "B": {"thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "C": {"thinking": False, "temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "presence_penalty": 2.0},
}
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", "/workspace/dev-reels/2026-11-05-weight-space-programming/models/base")


def render(tok, problem, thinking):
    return tok.apply_chat_template([{"role": "user", "content": build_user_message(problem)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def seed_for(cond, idx, s):
    return int(hashlib.sha256(f"{cond}|{idx}|{s}".encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", required=True, choices=list(CONDITIONS)); ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=2); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--problems", default=os.path.join(os.path.dirname(__file__), "..", "runs", "math500_subset60.jsonl"))
    ap.add_argument("--out", required=True); ap.add_argument("--parallel", type=int, default=24)
    ap.add_argument("--port", type=int, default=8091); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    cfg = CONDITIONS[a.cond]
    probs = [json.loads(l) for l in open(a.problems)][: a.n]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["idx"], r["sample"]))
    jobs = [(p, s) for s in range(a.samples) for p in probs if (p["idx"], s) not in done]
    print(f"[{a.tag} {a.cond} cap={a.cap}] {len(jobs)} jobs to run ({len(done)} already done)", flush=True)
    if not jobs:
        return
    srv = Server(a.model, a.parallel, a.cap + 512, port=a.port, threads=a.threads, log=a.out + ".server.log")
    lock, t0, ntok = threading.Lock(), time.time(), [0, 0]
    fout = open(a.out, "a")

    def one(job):
        p, s = job
        prompt = render(tok, p["problem"], cfg["thinking"])
        seed = seed_for(a.cond, p["idx"], s)
        t = time.time()
        r = srv.complete(prompt, a.cap, cfg, seed)
        g = grade(r["content"], p["answer"])
        rec = {"model": a.tag, "cond": a.cond, "cap": a.cap, "idx": p["idx"], "unique_id": p["unique_id"],
               "level": p["level"], "sample": s, "seed": seed, "sampling": {k: v for k, v in cfg.items()},
               "tokens": r["tokens_predicted"], "prompt_tokens": r.get("tokens_evaluated"),
               "stop_type": r.get("stop_type"), "hit_cap": r["tokens_predicted"] >= a.cap,
               "has_think_end": "</think>" in r["content"], "gold": p["answer"], **g,
               "wall_s": round(time.time() - t, 2), "response": r["content"]}
        with lock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            ntok[0] += rec["tokens"]; ntok[1] += 1
            el = time.time() - t0
            print(f"[{a.tag} {a.cond}] {ntok[1]}/{len(jobs)} idx={p['idx']} s={s} tok={rec['tokens']} cap={rec['hit_cap']} "
                  f"ok={rec['correct']} | agg {ntok[0]/el:.0f} tok/s, {el/60:.1f} min", flush=True)

    try:
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            for f in cf.as_completed([ex.submit(one, j) for j in jobs]):
                f.result()
    finally:
        srv.stop(); fout.close()
    el = time.time() - t0
    with open(a.out + ".throughput.json", "a") as f:
        f.write(json.dumps({"tag": a.tag, "cond": a.cond, "cap": a.cap, "generations": ntok[1], "tokens": ntok[0],
                            "wall_s": round(el, 1), "agg_tok_per_s": round(ntok[0] / el, 1),
                            "parallel": a.parallel, "threads": a.threads}) + "\n")


if __name__ == "__main__":
    main()
