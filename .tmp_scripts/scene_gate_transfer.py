import sys, json
from pathlib import Path
import numpy as np, torch
REPO=Path('/data/fangyuxuan/projects/gait')
for p in (str(REPO), str(REPO/'results_display/script')):
    if p not in sys.path: sys.path.insert(0,p)
from e6_gate_values_occlusion import build_loader, build_model
from ab_decomp import run_rows
from stage2_gate_pilot import run_split, feats, GateHead, win_lower_pa, spearman, IG_HEAD
from ig_contact_pilot import ContactHead
from anysole.types import FPS

dev=torch.device('cuda')
fold='fS5S6S11'
net,ckpt,tp=build_model('V8_Cv5_%s_s1'%fold,dev)
head=ContactHead(256).to(dev); head.load_state_dict(torch.load(IG_HEAD,map_location='cpu')); head.eval(); head.requires_grad_(False)
cfg='anysole/configs/v1_sceneocc_%s.yaml'%fold
loader=build_loader(cfg,'test',64,t_phys_v2=tp)
r=run_split(net,head,loader,dev,'test','scene')
loader2=build_loader(cfg,'test',64,t_phys_v2=tp)
rows=run_rows(net,loader2,dev,'test','scene')
n=len(r['g_star']); print('n',n,'rows',len(rows['e_vt']))
gate=GateHead(r['v_tok'].shape[1]).to(dev)
gate.load_state_dict(torch.load(REPO/'results/diagnostics/20261107_stage2_gate_pilot/gate_head.pt',map_location='cpu'))
gate.eval()
with torch.no_grad(): ghat=gate(torch.from_numpy(feats(r)).float().to(dev)).cpu().numpy()
print('ghat~g* spearman',spearman(ghat,r['g_star']))
# align rows order; same loader order
assert len(rows['p_v'])==n
mix=(1-ghat.reshape(-1,1,1,1))*rows['p_t']+ghat.reshape(-1,1,1,1)*rows['p_v']
e_gate=win_lower_pa(mix,rows['gt'])
print('VT %.2f  gate %.2f  gain %+.2f'%(np.mean(r['e_vt']),np.mean(e_gate),np.mean(r['e_vt']-e_gate)))
# by vision state
states=json.load(open(REPO/'results/diagnostics/20261107_e5_sceneocc_full/manifests/%s.json'%fold if False else '/dev/null')) if False else None
man={}
for p in (REPO/'results/diagnostics/20261107_e5_sceneocc_full/manifests').glob('S*.json'):
    m=json.load(open(p))
    for w in m['windows']: man[(m['session_id'],int(w['start']))]=w['state']
st=np.array([man.get((s,int(t)),'?') for s,t in zip(r['session'],r['start'])])
for s in ['clean','partial','full']:
    m=st==s
    if m.any(): print(' %-7s n=%3d g*=%.2f ghat=%.2f spearman=%+.3f VT=%.1f gate=%.1f gain=%+.1f'%(s,m.sum(),r['g_star'][m].mean(),ghat[m].mean(),spearman(ghat[m],r['g_star'][m]),r['e_vt'][m].mean(),e_gate[m].mean(),np.mean((r['e_vt']-e_gate)[m])))
