import numpy as np, wsp
dirs=np.load('wait_dirs.npy',allow_pickle=True).item()
S=np.load('wait_scores.npy',allow_pickle=True).item()
d=dirs[22]; d=d/np.linalg.norm(d)
allc=[]
for l in [16,18,19,20,21,22]:
    dw=S[l][0]
    Wd=wsp.load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    w=d@Wd
    c=dw*w
    o=np.argsort(-c)
    print(l,'total diff contrib',round(float(c.sum()),3),'pos',round(float(c[c>0].sum()),3),'top10 share',round(float(c[o[:10]].sum()),3),
          [(int(k),round(float(c[k]),3),round(float(dw[k]),2)) for k in o[:5]])
    for k in o[:20]: allc.append((float(c[k]),l,int(k),float(dw[k]),float(w[k])))
allc.sort(reverse=True)
np.save('cand.npy',np.array(allc))
print(allc[:15])
