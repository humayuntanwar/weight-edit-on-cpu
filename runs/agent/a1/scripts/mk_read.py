import json, wsp
from collections import Counter
tk=wsp.tok
rows=[json.loads(l) for l in open('passes/gen_1.jsonl')]
out=open('read1.jsonl','w'); tot=0; c=Counter()
for r in rows:
    p=tk.apply_chat_template([{"role":"user","content":r['problem'] if 'problem' in r else ''}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    out.write(json.dumps({"text":r['text']})+"\n")
