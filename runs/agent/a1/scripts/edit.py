import numpy as np, wsp, torch, sys
alpha=float(sys.argv[1]); L0,L1=int(sys.argv[2]),int(sys.argv[3])
dirs=np.load('wait_dirs.npy',allow_pickle=True).item()
d=dirs[20]/np.linalg.norm(dirs[20])+dirs[22]/np.linalg.norm(dirs[22]); d=torch.tensor(d/np.linalg.norm(d),dtype=torch.float32)
for n in list(wsp.edited()): wsp.drop_edit(n)
cnt=0
for l in range(L0,L1+1):
    outs=[f'model.language_model.layers.{l}.mlp.down_proj.weight',
          f'model.language_model.layers.{l}.self_attn.o_proj.weight' if l%4==3 else f'model.language_model.layers.{l}.linear_attn.out_proj.weight']
    for n in outs:
        W=wsp.load_original(n)
        W2=W-alpha*torch.outer(d,d@W)
        wsp.save_edit(n,W2); cnt+=W.numel()
print('edited',len(wsp.edited()),'tensors',cnt,'weights')
