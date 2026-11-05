"""Build a Q8_0 GGUF of the base model with an edit applied.

An edit is a folder of <tensor name>.npy files (float32, full tensor). Each one is cast to bfloat16 and replaces the
base tensor (the same rule the authors used for their evaluation), the result goes through llama.cpp's
convert_hf_to_gguf.py (bf16) and llama-quantize (Q8_0), exactly like the unedited and paper-edited models.

usage: build_model.py <base_hf_dir> <edit_dir or 'none'> <out.gguf>
"""
import os, shutil, subprocess, sys, tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("WSP_TOOLS", "/home/box/wsp_agent/tools")
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/workspace/dev-reels/llamacpp/llama-b11870")
SHARD = "model.safetensors-00001-of-00001.safetensors"


def build(base_dir, edit_dir, out_gguf, log=None):
    logf = open(log, "a") if log else subprocess.DEVNULL
    tmp = tempfile.mkdtemp(prefix="wsp_build_", dir=os.path.dirname(os.path.abspath(out_gguf)))
    try:
        for f in os.listdir(base_dir):
            if f != SHARD and not f.startswith("."):
                os.symlink(os.path.join(os.path.abspath(base_dir), f), os.path.join(tmp, f))
        sd = load_file(os.path.join(base_dir, SHARD))
        changed = []
        if edit_dir and edit_dir != "none" and os.path.isdir(edit_dir):
            for f in sorted(os.listdir(edit_dir)):
                if not f.endswith(".npy"):
                    continue
                name = f[:-4]
                if name not in sd:
                    raise KeyError(f"unknown tensor in edit: {name}")
                t = torch.from_numpy(np.load(os.path.join(edit_dir, f))).to(torch.bfloat16)
                if tuple(t.shape) != tuple(sd[name].shape):
                    raise ValueError(f"shape mismatch for {name}")
                sd[name] = t.contiguous()
                changed.append(name)
        save_file(sd, os.path.join(tmp, SHARD), metadata={"format": "pt"})
        del sd
        bf16 = os.path.join(tmp, "model-bf16.gguf")
        subprocess.run([sys.executable, os.path.join(TOOLS, "convert_hf_to_gguf.py"), tmp, "--outtype", "bf16",
                        "--outfile", bf16], check=True, stdout=logf, stderr=subprocess.STDOUT)
        subprocess.run([os.path.join(LLAMA_DIR, "llama-quantize"), bf16, out_gguf, "Q8_0", os.environ.get("WSP_THREADS", "3")], check=True,
                       stdout=logf, stderr=subprocess.STDOUT)
        return changed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(build(sys.argv[1], sys.argv[2], sys.argv[3], log=sys.argv[3] + ".build.log"))
