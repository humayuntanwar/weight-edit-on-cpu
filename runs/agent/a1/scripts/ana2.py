import numpy as np, wsp
from sklearn.metrics import roc_auc_score
tk=wsp.tok
D='passes/read_2/'; W={13428,13784}
ctx={tk.encode(s)[0] for s in ['  ','$.','\n','   ','.',').']}
think=tk.convert_tokens_to_ids('<think>')
H=[10,14,17,20,22]
X={h:[] for h in H}; y=[]; tid=[]
for i in range(32):
    z=np.load(D+f'text_{i:03d}.npz'); ids=z['input_ids']
    st=int(np.where(ids==think)[0][-1])+1
    for j in range(st,len(ids)-1):
        if ids[j+1] in W or ids[j] in ctx:
            y.append(int(ids[j+1] in W)); tid.append(i)
            for h in H: X[h].append(z[f'hidden_{h}'][j].astype(np.float32))
y=np.array(y); tid=np.array(tid)
tr=tid<22; te=~tr
dirs={}
for h in H:
    A=np.stack(X[h]); d=A[tr&(y==1)].mean(0)-A[tr&(y==0)].mean(0)
    dirs[h]=d
    s=A@d/np.linalg.norm(d)
    print(h,'AUC test',round(roc_auc_score(y[te],s[te]),3),'|d|',round(float(np.linalg.norm(d)),2),'mean |h|',round(float(np.linalg.norm(A,axis=1).mean()),1), 'proj w/c', round(float(s[y==1].mean()),2), round(float(s[y==0].mean()),2))
# cosine between dirs
for a in H:
    print(a,[round(float(dirs[a]@dirs[b]/np.linalg.norm(dirs[a])/np.linalg.norm(dirs[b])),2) for b in H])
# full-data dirs
full={}
for h in H:
    A=np.stack(X[h]); full[h]=A[y==1].mean(0)-A[y==0].mean(0)
np.save('wait_dirs.npy',full,allow_pickle=True)
