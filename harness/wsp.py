"""Weight-editing harness: a Python library over the base weights plus a CLI with a metered forward-pass budget.

Library (import wsp in your own Python scripts):
    wsp.NAMES                    all tensor names (model.language_model.*)
    wsp.load(name)               float32 torch tensor: your current edit if there is one, else the original
    wsp.load_original(name)      float32 torch tensor of the original weights
    wsp.save_edit(name, tensor)  store an edit (same shape); refused for the final norm, embeddings / lm_head and for
                                 edits that only multiply the whole tensor by a constant
    wsp.drop_edit(name), wsp.edited()
    wsp.tok                      the tokenizer (lazy)
CLI (each gen/read call that succeeds uses one forward pass):
    wsp status
    wsp gen  --model original|edited|both (--problems ids.txt | --source gsm8k|math|mix --n N --seed S [--level L])
             [--samples K] [--max-tokens T]
    wsp read --model original|edited --texts texts.jsonl [--hidden 3,15] [--mlp-in 15,19] [--mlp-out 15,19] [--logprobs]
"""
import argparse, concurrent.futures as cf, glob, hashlib, json, math, os, random, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HOME = os.environ.get("WSP_HOME", "/home/box/wsp_agent")
BASE = os.environ.get("WSP_BASE", os.path.join(HOME, "base"))
SCRATCH = os.environ.get("WSP_SCRATCH", os.getcwd())
STATE = os.environ.get("WSP_STATE", os.path.join(SCRATCH, ".wsp_state"))
PRACTICE = os.environ.get("WSP_PRACTICE", os.path.join(SCRATCH, "practice"))
EDIT_DIR = os.path.join(SCRATCH, "edited")
PASS_DIR = os.path.join(SCRATCH, "passes")
SHARD = os.path.join(BASE, "model.safetensors-00001-of-00001.safetensors")
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")
GEN_MAX, CAP_MAX = 32, 1024                          # answers per generation pass, new tokens per answer
READ_MAX_TEXTS, READ_MAX_TOK, READ_TOTAL_TOK, READ_MAX_LAYERS = 64, 2048, 32768, 8
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}

_st = None


def _safe():
    global _st
    if _st is None:
        from safetensors import safe_open
        _st = safe_open(SHARD, "pt")
    return _st




def _names():
    return sorted(k for k in _safe().keys() if k.startswith("model.language_model.") or k == "lm_head.weight")


def __getattr__(attr):   # lazy module attributes
    if attr == "NAMES":
        return _names()
    if attr == "tok":
        from transformers import AutoTokenizer
        globals()["tok"] = AutoTokenizer.from_pretrained(BASE)
        return globals()["tok"]
    raise AttributeError(attr)


def load_original(name):
    return _safe().get_tensor(name).float()


def load(name):
    p = os.path.join(EDIT_DIR, name + ".npy")
    if os.path.exists(p):
        import numpy as np, torch
        return torch.from_numpy(np.load(p))
    return load_original(name)


def save_edit(name, tensor):
    import numpy as np, torch
    if name in FROZEN:
        raise PermissionError(f"{name} is out of scope (final norm, embeddings and output layer are frozen)")
    orig = load_original(name)
    t = torch.as_tensor(tensor).detach().float().cpu()
    if tuple(t.shape) != tuple(orig.shape):
        raise ValueError(f"shape {tuple(t.shape)} != {tuple(orig.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("non-finite values")
    c = float((t * orig).sum() / (orig * orig).sum().clamp_min(1e-30))
    if abs(c - 1) > 1e-6 and float((t - c * orig).norm()) <= 1e-4 * float(orig.norm()):
        raise PermissionError("edits that only multiply a whole tensor by a constant are refused")
    os.makedirs(EDIT_DIR, exist_ok=True)
    np.save(os.path.join(EDIT_DIR, name + ".npy"), t.numpy())
    _log("save_edit", {"name": name, "rel_change": float((t - orig).norm() / orig.norm())})
    return f"saved {name}"


def drop_edit(name):
    p = os.path.join(EDIT_DIR, name + ".npy")
    if os.path.exists(p):
        os.remove(p)
        _log("drop_edit", {"name": name})


def edited():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(EDIT_DIR, "*.npy")))


# ------------------------------------------------------------------ budget and log (harness-owned files)
def _budget():
    with open(os.path.join(STATE, "budget.json")) as f:
        return json.load(f)


def _use_pass(kind, summary):
    b = _budget()
    b["used"] += 1
    b["passes"].append({"k": b["used"], "kind": kind, "time": time.strftime("%H:%M:%S")})
    with open(os.path.join(STATE, "budget.json"), "w") as f:
        json.dump(b, f, indent=1)
    return b["used"]


def _log(event, data):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, "calls.jsonl"), "a") as f:
        f.write(json.dumps({"t": time.time(), "event": event, **data}) + "\n")


def _check_budget():
    b = _budget()
    if b["used"] >= b["budget"]:
        sys.exit(f"BUDGET EXHAUSTED: {b['used']} of {b['budget']} forward passes used. Make your final choice.")
    return b


def _edit_hash():
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(EDIT_DIR, "*.npy"))):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:16]


def _gguf_for(model):
    if model == "original":
        return os.path.join(HOME, "cache", "original-q8.gguf")
    if not edited():
        sys.exit("no edits saved: use --model original")
    out = os.path.join(STATE, "build", f"edited_{_edit_hash()}-q8.gguf")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for old in glob.glob(os.path.join(STATE, "build", "*.gguf")):
            os.remove(old)
        from build_model import build
        build(BASE, EDIT_DIR, out, log=os.path.join(STATE, "build.log"))
    return out


# ------------------------------------------------------------------ generation pass
def _practice():
    rows = {}
    for f in ("gsm8k_train.jsonl", "math_train.jsonl"):
        for l in open(os.path.join(PRACTICE, f)):
            r = json.loads(l)
            rows[r["id"]] = r
    return rows


def _seed(pid, s):
    return int(hashlib.sha256(f"{pid}|{s}".encode()).hexdigest()[:8], 16)


def _run_gen(model, items, cap, out_path):
    from llserver import Server
    from mathgrade import build_user_message, grade
    tk = __getattr__("tok")
    gg = _gguf_for(model)
    srv = Server(gg, min(GEN_MAX, len(items)), cap + 700, port=int(os.environ.get("WSP_PORT", "8095")),
                 log=os.path.join(STATE, "server.log"), threads=int(os.environ.get("WSP_THREADS", "3")))
    res, lock = [], threading.Lock()

    def one(it):
        p, s = it
        prompt = tk.apply_chat_template([{"role": "user", "content": build_user_message(p["problem"])}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=True)
        r = srv.complete(prompt, cap, SAMPLING, _seed(p["id"], s))
        g = grade(r["content"], p["answer"])
        rec = {"id": p["id"], "source": p["source"], "sample": s, "model": model, "tokens": r["tokens_predicted"],
               "cut": r["tokens_predicted"] >= cap, "closed_think": "</think>" in r["content"], "gold": p["answer"],
               "pred": g["pred"], "correct": g["correct"], "text": r["content"]}
        with lock:
            res.append(rec)

    try:
        with cf.ThreadPoolExecutor(GEN_MAX) as ex:
            list(ex.map(one, items))
    finally:
        srv.stop()
    res.sort(key=lambda r: (r["id"], r["sample"]))
    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    return res


def _summ(res):
    n = len(res)
    acc = sum(r["correct"] for r in res) / n
    by = {}
    for src in sorted({r["source"] for r in res}):
        rr = [r for r in res if r["source"] == src]
        by[src] = {"n": len(rr), "accuracy": round(sum(r["correct"] for r in rr) / len(rr), 4)}
    fin = [r for r in res if not r["cut"]]
    return {"n": n, "accuracy": round(acc, 4), "stderr": round(math.sqrt(acc * (1 - acc) / n), 4), "by_source": by,
            "mean_tokens": round(sum(r["tokens"] for r in res) / n, 1),
            "share_cut_at_max_tokens": round(sum(r["cut"] for r in res) / n, 4),
            "failures": {"cut_at_max_tokens": sum(r["cut"] for r in res),
                         "finished_without_box": sum(1 for r in fin if r["pred"] is None),
                         "finished_wrong_answer": sum(1 for r in fin if r["pred"] is not None and not r["correct"])}}


def _tails(res, k=3):
    out = []
    for r in [r for r in res if r["cut"]][:k] + [r for r in res if not r["cut"] and not r["correct"]][:k]:
        out.append({"id": r["id"], "model": r["model"], "cut": r["cut"], "pred": r["pred"], "gold": r["gold"],
                    "tail": r["text"][-300:]})
    return out


def cmd_gen(a):
    b = _check_budget()
    rows = _practice()
    cap = min(a.max_tokens, CAP_MAX)
    if a.problems:
        ids = [l.strip() for l in open(a.problems) if l.strip()]
        bad = [i for i in ids if i not in rows]
        if bad:
            sys.exit(f"unknown practice ids: {bad[:5]}")
    else:
        pool = sorted(i for i, r in rows.items() if (a.source == "mix" or r["source"] == a.source)
                      and (a.level is None or r.get("level") == f"Level {a.level}"))
        ids = random.Random(a.seed).sample(pool, a.n)
    items = [(rows[i], s) for i in ids for s in range(a.samples)]
    per_model = GEN_MAX // 2 if a.model == "both" else GEN_MAX
    if len(items) > per_model:
        sys.exit(f"too many answers: {len(items)} > {per_model} per model for --model {a.model}")
    k = b["used"] + 1
    os.makedirs(PASS_DIR, exist_ok=True)
    t0 = time.time()
    out = {"pass": k, "type": "generate", "model": a.model, "max_tokens": cap, "problems": len(ids),
           "samples_per_problem": a.samples}
    if a.model == "both":
        ro = _run_gen("original", items, cap, os.path.join(PASS_DIR, f"gen_{k}_original.jsonl"))
        re_ = _run_gen("edited", items, cap, os.path.join(PASS_DIR, f"gen_{k}_edited.jsonl"))
        out["original"], out["edited"] = _summ(ro), _summ(re_)
        d = [int(e["correct"]) - int(o["correct"]) for o, e in zip(ro, re_)]
        md = sum(d) / len(d)
        sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, len(d) - 1)) / math.sqrt(len(d))
        out["paired"] = {"diff_edited_minus_original": round(md, 4), "stderr": round(sd, 4),
                         "wrong_to_right": d.count(1), "right_to_wrong": d.count(-1)}
        out["sample_tails"] = _tails(ro, 2) + _tails(re_, 2)
        files = [f"passes/gen_{k}_original.jsonl", f"passes/gen_{k}_edited.jsonl"]
    else:
        r = _run_gen(a.model, items, cap, os.path.join(PASS_DIR, f"gen_{k}.jsonl"))
        out[a.model] = _summ(r)
        out["sample_tails"] = _tails(r)
        files = [f"passes/gen_{k}.jsonl"]
    out["files"] = files
    out["edited_tensors"] = edited() if a.model != "original" else []
    out["wall_seconds"] = round(time.time() - t0, 1)
    used = _use_pass("generate", out)
    out["passes_used"] = f"{used} of {b['budget']}"
    _log("gen", {k2: v for k2, v in out.items() if k2 != "sample_tails"})
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ teacher-forced read pass
def cmd_read(a):
    import numpy as np, torch
    b = _check_budget()
    tk = __getattr__("tok")
    texts = [json.loads(l) for l in open(a.texts) if l.strip()]
    if len(texts) > READ_MAX_TEXTS:
        sys.exit(f"too many texts: {len(texts)} > {READ_MAX_TEXTS}")
    layers = {k: [int(x) for x in v.split(",")] if v else [] for k, v in
              (("hidden", a.hidden), ("mlp_in", a.mlp_in), ("mlp_out", a.mlp_out))}
    for k, v in layers.items():
        if len(v) > READ_MAX_LAYERS or any(not 0 <= x <= 24 for x in v):
            sys.exit(f"--{k}: at most {READ_MAX_LAYERS} layers in 0..24 (hidden) / 0..23 (mlp)")
    ids = []
    for t in texts:
        if "chat" in t:
            s = tk.apply_chat_template(t["chat"], tokenize=False, enable_thinking=t.get("enable_thinking", True))
        else:
            s = t["text"]
        x = tk(s, add_special_tokens=False)["input_ids"]
        if len(x) > READ_MAX_TOK:
            sys.exit(f"a text has {len(x)} tokens > {READ_MAX_TOK}")
        ids.append(x)
    if sum(map(len, ids)) > READ_TOTAL_TOK:
        sys.exit(f"total tokens {sum(map(len, ids))} > {READ_TOTAL_TOK}")
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("WSP_THREADS", "3")))
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).eval()
    if a.model == "edited":
        sd = dict(m.named_parameters())
        for name in edited():
            tgt = name.replace("model.language_model.", "model.")
            with torch.no_grad():
                sd[tgt].copy_(load(name).to(torch.bfloat16))
    caps = {}
    hooks = []
    for l in layers["mlp_in"]:
        hooks.append(m.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda mod, inp, l=l: caps.__setitem__(f"mlp_in_{l}", inp[0][0].float().numpy().astype(np.float16))))
    for l in layers["mlp_out"]:
        hooks.append(m.model.layers[l].mlp.register_forward_hook(
            lambda mod, inp, out, l=l: caps.__setitem__(f"mlp_out_{l}", out[0].float().numpy().astype(np.float16))))
    k = b["used"] + 1
    od = os.path.join(PASS_DIR, f"read_{k}")
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    meta = []
    with torch.no_grad():
        for i, x in enumerate(ids):
            caps.clear()
            o = m.model(torch.tensor([x]), output_hidden_states=bool(layers["hidden"]))
            arrs = dict(caps)
            for l in layers["hidden"]:
                arrs[f"hidden_{l}"] = o.hidden_states[l][0].float().numpy().astype(np.float16)
            if a.logprobs:
                h = o.last_hidden_state[0]
                lp = []
                for j in range(0, len(x) - 1, 256):
                    lg = m.lm_head(h[j:min(j + 256, len(x) - 1)]).float().log_softmax(-1)
                    tgt = torch.tensor(x[j + 1:j + 1 + lg.shape[0]])
                    lp.append(lg.gather(1, tgt[:, None])[:, 0])
                arrs["token_logprobs"] = torch.cat(lp).numpy()
            arrs["input_ids"] = np.array(x, dtype=np.int64)
            np.savez(os.path.join(od, f"text_{i:03d}.npz"), **arrs)
            meta.append({"i": i, "tokens": len(x), "arrays": sorted(arrs)})
    for h in hooks:
        h.remove()
    json.dump({"model": a.model, "layers": layers, "texts": meta,
               "notes": "hidden_l = residual stream after layer l (0 = embeddings; 24 = after the final norm); mlp_in_l = input of layer l's "
                        "down_proj (3584); mlp_out_l = output of layer l's MLP (1024); token_logprobs[j] = log p(token j+1)"},
              open(os.path.join(od, "meta.json"), "w"), indent=1)
    out = {"pass": k, "type": "teacher_forced", "model": a.model, "texts": len(ids), "tokens": sum(map(len, ids)),
           "folder": f"passes/read_{k}", "wall_seconds": round(time.time() - t0, 1),
           "edited_tensors": edited() if a.model == "edited" else []}
    used = _use_pass("teacher_forced", out)
    out["passes_used"] = f"{used} of {b['budget']}"
    _log("read", out)
    print(json.dumps(out, indent=1))


def cmd_status(a):
    b = _budget()
    print(json.dumps({"passes_used": b["used"], "budget": b["budget"], "edited_tensors": edited(),
                      "limits": {"generation_answers_per_pass": GEN_MAX, "both_answers_per_model": GEN_MAX // 2,
                                 "max_new_tokens": CAP_MAX, "read_texts": READ_MAX_TEXTS,
                                 "read_tokens_per_text": READ_MAX_TOK, "read_tokens_total": READ_TOTAL_TOK},
                      "sampling": SAMPLING}, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="wsp")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    g = sp.add_parser("gen")
    g.add_argument("--model", choices=["original", "edited", "both"], required=True)
    g.add_argument("--problems"); g.add_argument("--source", choices=["gsm8k", "math", "mix"], default="mix")
    g.add_argument("--n", type=int, default=32); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--level", type=int); g.add_argument("--samples", type=int, default=1)
    g.add_argument("--max-tokens", type=int, default=CAP_MAX)
    r = sp.add_parser("read")
    r.add_argument("--model", choices=["original", "edited"], required=True)
    r.add_argument("--texts", required=True)
    r.add_argument("--hidden", default=""); r.add_argument("--mlp-in", default=""); r.add_argument("--mlp-out", default="")
    r.add_argument("--logprobs", action="store_true")
    a = ap.parse_args()
    {"status": cmd_status, "gen": cmd_gen, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    main()
