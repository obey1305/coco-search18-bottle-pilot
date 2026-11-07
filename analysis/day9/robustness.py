"""Paired 2x2 sensitivity: averaging-patch radius and alternative sampling seed."""
from pathlib import Path
import sys,csv,json,math
import numpy as np
from PIL import Image
from scipy.stats import rankdata
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'day5'));sys.path.insert(0,str(B/'day6'))
from colour_contrast import rgb_to_lab,convolve_same
from features import responses

def read(p):return list(csv.DictReader(p.open()))
def save(name,rows):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 original=read(B/'day7_semantic/chosen_vs_alternatives.csv');new=read(P/'seed_repeat/chosen_vs_alternatives.csv')
 ids={r['decision_id'] for r in original};assert ids=={r['decision_id'] for r in new}
 # Changing RNG must not change eligibility or pool sizes.
 oldlogs=read(B/'day7_semantic/decision_log.csv');newlogs=read(P/'seed_repeat/decision_log.csv');assert oldlogs==newlogs
 original_chosen={r['decision_id']:(r['x'],r['y']) for r in original if r['role']=='chosen'}
 assert original_chosen=={r['decision_id']:(r['x'],r['y']) for r in new if r['role']=='chosen'}
 overlap=[]
 for k in sorted(ids):
  a={(r['x'],r['y']) for r in original if r['decision_id']==k and r['role']!='chosen'};b={(r['x'],r['y']) for r in new if r['decision_id']==k and r['role']!='chosen'}
  assert len(a)==len(b)==8;overlap.append(dict(decision_id=k,shared_alternatives=len(a&b),changed_alternatives=8-len(a&b)))
 save('seed_overlap.csv',overlap)
 expanded=[];checks=[]
 for tr in json.loads((B/'day7_semantic/transforms.json').read_text()):
  name=f"{tr['image_id']:012d}.jpg";rgb=np.asarray(Image.open(B/'day4'/name).convert('RGB'),dtype=float)/255;lab=rgb_to_lab(rgb)
  linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);gray=np.einsum('...j,j->...',linear,np.array([.2126729,.7151522,.0721750]),optimize=False);resp=responses(gray)
  yy,xx=np.mgrid[-32:33,-32:33];sq=xx*xx+yy*yy;disk=sq<=8**2;ring=(sq>=16**2)&(sq<=32**2);diff=disk/disk.sum()-ring/ring.sum()
  small=dict(colour=np.hypot(convolve_same(lab[...,1],diff),convolve_same(lab[...,2],diff)),luminance=np.abs(convolve_same(lab[...,0],diff)),orientation=np.linalg.norm(np.stack([convolve_same(resp[...,k],diff) for k in range(4)],axis=-1),axis=-1))
  large=np.load(B/'day8'/f"{tr['image_id']}_features.npz");error=0
  for seed,rows in [(20261006,original),(20261007,new)]:
   for r in [r for r in rows if r['image']==name]:
    x,y=int(float(r['x'])+.5),int(float(r['y'])+.5);assert large['orientation_valid'][y,x]
    for radius,fields in [(16,large),(8,small)]:
     values={f:float(fields[f][y,x]) for f in ['colour','orientation','luminance']};assert all(math.isfinite(v) for v in values.values())
     expanded.append(dict(decision_id=r['decision_id'],image=name,subject=r['subject'],role=r['role'],x=r['x'],y=r['y'],gaze_distance=r['gaze_distance'],seed=seed,centre_radius=radius,**values))
    patch=lab[y-32:y+33,x-32:x+33];rp=resp[y-32:y+33,x-32:x+33]
    direct={'colour':np.linalg.norm(patch[disk,1:3].mean(axis=0)-patch[ring,1:3].mean(axis=0)),'luminance':abs(patch[disk,0].mean()-patch[ring,0].mean()),'orientation':np.linalg.norm(rp[disk].mean(axis=0)-rp[ring].mean(axis=0))}
    for f in direct:error=max(error,abs(direct[f]-small[f][y,x]))
  assert error<1e-9;checks.append(dict(image=name,small_patch_max_direct_error=error))
 save('candidate_features.csv',expanded);obs=[]
 scenarios=[(16,20261006,'Baseline'),(8,20261006,'Smaller patch'),(16,20261007,'New seed'),(8,20261007,'Smaller patch + new seed')]
 for radius,seed,label in scenarios:
  for key in sorted(ids):
   group=[r for r in expanded if r['decision_id']==key and r['seed']==seed and r['centre_radius']==radius];assert len(group)==9
   q=next(r for r in group if r['role']=='chosen');controls=[r for r in group if r['role']!='chosen']
   o=dict(scenario=label,centre_radius=radius,seed=seed,decision_id=key,image=q['image'],subject=q['subject'],distance=float(q['gaze_distance']))
   for f in ['colour','orientation','luminance']:o[f]=q[f]-math.fsum(r[f] for r in controls)/8
   obs.append(o)
 save('observations.csv',obs)
 baseline={r['decision_id']:r for r in obs if r['scenario']=='Baseline'}
 day8={r['decision_id']:r for r in read(B/'day8/observations.csv')}
 for k,r in baseline.items():
  for f in ['colour','orientation','luminance']:assert abs(r[f]-float(day8[k][f+'_difference']))<1e-12
 summary=[]
 for _,_,label in scenarios:
  rs=[r for r in obs if r['scenario']==label]
  for f in ['colour','orientation','luminance']:
   vals=np.array([r[f] for r in rs]);ref=np.array([baseline[r['decision_id']][f] for r in rs]);distance=np.array([r['distance'] for r in rs]);rho=float(np.corrcoef(rankdata(distance),rankdata(vals))[0,1])
   summary.append(dict(scenario=label,feature=f,n=len(rs),positive=int((vals>0).sum()),negative=int((vals<0).sum()),median_difference=float(np.median(vals)),sign_changes_vs_baseline=int((np.sign(vals)!=np.sign(ref)).sum()),descriptive_spearman_rho=rho))
 save('summary.csv',summary)
 lines=['# Day 9 — robustness and limitations','', 'Same 10 landings; 8 alternatives each. All results are descriptive.','', '| Setting | Feature | Positive / 10 | Median chosen-minus-control | Sign changes vs baseline | Distance rank association |','|---|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['scenario']} | {r['feature']} | {r['positive']}/10 | {r['median_difference']:.4f} | {r['sign_changes_vs_baseline']} | {r['descriptive_spearman_rho']:.3f} |")
 (P/'tables.md').write_text('\n'.join(lines)+'\n')
 (P/'verification.json').write_text(json.dumps(dict(checks=checks,baseline_reproduced=True,identical_decision_pools=True,observations=len(obs),candidate_rows=len(expanded),seed_alternatives_changed=sum(r['changed_alternatives'] for r in overlap),unseen_images_tested=0),indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()
