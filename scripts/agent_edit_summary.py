"""Describe an agent's submitted edit against the base weights (after the bfloat16 cast used for evaluation) and keep a
compact copy of it in the repo: the changed entries (flat index + bfloat16 bits) in edit_diff.npz when that is small,
else the delta's leading singular vectors (approximate; noted in the manifest).
usage: agent_edit_summary.py <base_safetensors> <edit_dir> <out_dir>
"""
import hashlib, json, os, sys
import numpy as np, torch
from safetensors import safe_open

base, edit_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
man, arrays = {"tensors": []}, {}
with safe_open(base, "pt") as f:
    for fn in sorted(os.listdir(edit_dir)) if os.path.isdir(edit_dir) else []:
        if not fn.endswith(".npy"):
            continue
        name = fn[:-4]
        b = f.get_tensor(name)
        e = torch.from_numpy(np.load(os.path.join(edit_dir, fn))).to(torch.bfloat16)
        ne = (b.view(torch.int16) != e.view(torch.int16))
        d = e.float() - b.float()
        info = {"name": name, "shape": list(b.shape), "changed_entries": int(ne.sum()), "numel": b.numel(),
                "rel_fro_change": float(d.norm() / b.float().norm()),
                "edited_bf16_sha256": hashlib.sha256(e.view(torch.int16).numpy().tobytes()).hexdigest()}
        if b.dim() == 2:
            info["changed_rows"] = int(ne.any(1).sum()); info["changed_columns"] = int(ne.any(0).sum())
            s = torch.linalg.svdvals(d)
            info["delta_top_singular_values"] = [round(float(x), 5) for x in s[:5]]
            info["delta_energy_in_top1"] = float(s[0] ** 2 / (s ** 2).sum()) if s.sum() > 0 else None
        key = name.replace(".", "_")
        if ne.sum() <= 400_000:
            idx = ne.flatten().nonzero().flatten()
            arrays[key + "__idx"] = idx.numpy().astype(np.int32)
            arrays[key + "__bf16bits"] = e.view(torch.int16).flatten()[idx].numpy()
            info["stored"] = "exact changed entries"
        else:
            U, S, V = torch.linalg.svd(d, full_matrices=False)
            r = max(1, int((S > 1e-3 * S[0]).sum().clamp(max=8)))
            arrays[key + "__U"] = (U[:, :r] * S[:r]).numpy().astype(np.float32); arrays[key + "__V"] = V[:r].numpy().astype(np.float32)
            info["stored"] = f"rank-{r} factors of the delta (approximate, not bit exact)"
        man["tensors"].append(info)
man["changed_tensors"] = len(man["tensors"])
man["changed_entries_total"] = sum(t["changed_entries"] for t in man["tensors"])
if arrays:
    np.savez_compressed(os.path.join(out, "edit_diff.npz"), **arrays)
json.dump(man, open(os.path.join(out, "edit_manifest.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in man.items() if k != "tensors"}),
      [(t["name"].replace("model.language_model.", ""), t["changed_entries"], t.get("changed_columns")) for t in man["tensors"]])
