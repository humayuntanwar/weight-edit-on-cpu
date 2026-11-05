"""Compare two safetensors checkpoints tensor by tensor (sha256 of raw bytes) and count changed entries."""
import hashlib, json, sys
from safetensors import safe_open
import torch

a_path, b_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"a": a_path, "b": b_path, "tensors_total": 0, "differing": []}
with safe_open(a_path, "pt") as fa, safe_open(b_path, "pt") as fb:
    ka, kb = set(fa.keys()), set(fb.keys())
    res["only_in_a"], res["only_in_b"] = sorted(ka - kb), sorted(kb - ka)
    n_params = 0
    for k in sorted(ka & kb):
        ta, tb = fa.get_tensor(k), fb.get_tensor(k)
        n_params += ta.numel()
        res["tensors_total"] += 1
        ha = hashlib.sha256(ta.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        hb = hashlib.sha256(tb.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if ha != hb or ta.shape != tb.shape:
            ne = (ta.view(torch.int16) != tb.view(torch.int16)) if ta.dtype == torch.bfloat16 else (ta != tb)
            cols = ne.any(dim=0).nonzero().flatten().tolist() if ne.dim() == 2 else []
            res["differing"].append({"name": k, "shape": list(ta.shape), "dtype": str(ta.dtype),
                                     "changed_entries": int(ne.sum()), "changed_columns": len(cols),
                                     "changed_column_ids": cols, "sha256_a": ha, "sha256_b": hb,
                                     "rel_fro_change": float((tb.float() - ta.float()).norm() / ta.float().norm())})
    res["params_total"] = n_params
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "differing"}, indent=1))
for d in res["differing"]:
    print(d["name"], d["shape"], "changed_entries", d["changed_entries"], "changed_columns", d["changed_columns"], "rel", round(d["rel_fro_change"], 4))
