import numpy as np,wsp,glob,torch
tok=wsp.tok
f=sorted(glob.glob('passes/read_2/text_*.npz'))
hes=set()
for w in ['Wait',' Wait','Actually',' Actually','Hmm',' Hmm']:
    ids=tok.encode(w,add_special_tokens=False)
    if len(ids)==1: hes.add(ids[0])
print('hes ids',hes)
think=tok.encode('<think>',add_special_tokens=False)[0]
H={L:([],[]) for L in [12,16,20,23]}; lp_h=[];lp_o=[]
for fn in f:
    z=dict(np.load(fn)); ids=z['input_ids']; st=int(np.where(ids==think)[0][-1])+2
    prevtxt=[t.replace('Ċ','\n') for t in tok.convert_ids_to_tokens(ids.tolist())]
    for p in range(st,len(ids)-1):
        nxt=ids[p+1]
        if '\n' not in prevtxt[p] and not prevtxt[p].strip().endswith(('.','*',':')): continue
        g=0 if nxt in hes else 1
        for L in H: H[L][g].append(z[f'hidden_{L}'][p].copy())
        (lp_h if g==0 else lp_o).append(z['token_logprobs'][p])
print('n hes',len(H[23][0]),'n other',len(H[23][1]),'mean lp hes',np.mean(lp_h))


from sklearn.metrics import roc_auc_score
for L in H:
    a=np.array(H[L][0]);b=np.array(H[L][1]); d=a.mean(0)-b.mean(0)
    n=len(a)//2; dd=a[:n].mean(0)-b[::2].mean(0)
    s=np.r_[a[n:]@dd,b[1::2]@dd]; y=np.r_[np.ones(len(a)-n),np.zeros(len(b[1::2]))]
    print(L,'heldout AUC %.3f'%roc_auc_score(y,s),'|d| %.2f'%np.linalg.norm(d))
    np.save(f'dir_{L}.npy',d)
