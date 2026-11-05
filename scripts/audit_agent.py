"""Audit one agent attempt's transcript: every tool call, any network use, any path outside the scratch folder and
harness, any training code (optimizers, backward, autograd), and the forward-pass ledger.
usage: audit_agent.py <attempt dir> <out.json>
"""
import json, re, sys

A, out = sys.argv[1], sys.argv[2]
calls, init, result = [], None, None
for l in open(f"{A}/transcript.jsonl"):
    r = json.loads(l)
    if r.get("type") == "system" and r.get("subtype") == "init":
        init = {"model": r.get("model"), "tools": r.get("tools"), "mcp_servers": r.get("mcp_servers")}
    if r.get("type") == "result":
        result = {k: r.get(k) for k in ("duration_ms", "num_turns", "total_cost_usd", "is_error", "subtype")}
    if r.get("type") == "assistant":
        for c in r["message"]["content"]:
            if c["type"] == "tool_use":
                calls.append({"tool": c["name"], "input": c["input"]})
allowed = (A.rstrip("/") + "/work", "/home/box/wsp_agent/harness", "/home/box/wsp_agent/base")
text = json.dumps([c["input"] for c in calls])
paths = sorted(set(re.findall(r"(/(?:home|workspace|tmp|root|etc|usr|opt|var)[^\s\"'\;|)]*)", text)))
flags = {
    "network": sorted(set(re.findall(r"\b(curl|wget|requests\.get|urllib|http[s]?://\S+|pip install|huggingface|hf_hub)", text))),
    "paper_artifacts": sorted(set(re.findall(r"(lm_understand|RikiyaT|weight-space-programming|apply_edit|rikiyat)", text, re.I))),
    "training": sorted(set(re.findall(r"(\.backward\(|torch\.optim|requires_grad_?\(True|requires_grad=True|autograd|\.step\(\))", text))),
    "model_loading": sorted(set(re.findall(r"(AutoModel\w*|from_pretrained\(|llama-server|llama_cpp)", text))),
    "paths_outside_allowed": [p for p in paths if not p.startswith(allowed)],
}
res = {"init": init, "result": result, "tool_calls": len(calls),
       "tools_used": {t: sum(c["tool"] == t for c in calls) for t in sorted({c["tool"] for c in calls})},
       "flags": flags, "budget": json.load(open(f"{A}/state/budget.json"))}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1)[:3000])
