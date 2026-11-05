import numpy as np, json, wsp, torch
from collections import Counter
tk=wsp.tok
D='passes/read_2/'
W={13428,13784}
L=[14,16,18,19,20,21,22,23]
E=wsp.load_original('model.language_model.embed_tokens.weight')
nw=wsp.load_original('model.language_model.norm.weight')
u=(E[13428]+E[13784])/2*(1+nw) if nw.mean()<0.5 else (E[13428]+E[13784])/2*nw
u=u-0  # direction
print('normw mean',nw.mean().item())
# collect
gstart=[]
prev=Counter(); 
data=[]
for i in range(32):
    z=np.load(D+f'text_{i:03d}.npz'); ids=z['input_ids']
    st=int(np.where(ids==tk.convert_tokens_to_ids('<think>'))[0][-1])+1
    data.append((ids,st))
    for j in range(st,len(ids)-1):
        if ids[j+1] in W: prev[ids[j]]+=1
print([(repr(tk.decode([t])),n) for t,n in prev.most_common(8)])
ctx=set(t for t,_ in prev.most_common(6))
res={}
for l in L:
    S_w=np.zeros(3584);n_w=0;S_c=np.zeros(3584);n_c=0
    for i in range(32):
        ids,st=data[i]; a=np.load(D+f'text_{i:03d}.npz')[f'mlp_in_{l}'].astype(np.float32)
        nxt=ids[1:]; 
        for j in range(st,len(ids)-1):
            if nxt[j] in W: S_w+=a[j]; n_w+=1
            elif ids[j] in ctx: S_c+=a[j]; n_c+=1
    dw=S_w/n_w-S_c/n_c
    Wd=wsp.load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight')
    dla=(u@Wd).numpy()
    sc=dw*dla
    res[l]=(dw,dla)
    o=np.argsort(-sc)[:6]
    print(l,n_w,n_c,'top',[(int(k),round(float(dw[k]),2),round(float(dla[k]),2)) for k in o], 'sum pos', round(float(sc[sc>0].sum()),1))
np.save('wait_scores.npy',{l:np.stack(res[l]) for l in L},allow_pickle=True)
