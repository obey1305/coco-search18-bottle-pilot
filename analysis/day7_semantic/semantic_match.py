"""Semantic + spatial matching of the existing 17-decision pilot; mask sensitivity."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent

def save(name,rows):
 if rows:
  with (ROOT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 transforms=json.loads((ROOT/'transforms.json').read_text());cache={};alignment=[]
 for tr in transforms:
  id=tr['image_id'];meta=json.loads((ROOT/f'{id}_annotation.json').read_text());cats={c['id']:c for c in meta['categories']};segs={s['id']:s for s in meta['annotation']['segments_info']}
  src=np.asarray(Image.open(ROOT/f'{id}_panoptic.png'),dtype=np.int32);ids=src[...,0]+256*src[...,1]+65536*src[...,2]
  assert ids.shape==(tr['original_height'],tr['original_width'])
  assert set(np.unique(ids))-{0}==set(segs)
  for sid,s in segs.items():assert int((ids==sid).sum())==s['area']
  # Integer segment IDs use nearest-neighbour, never interpolated RGB mask colours.
  resized=np.asarray(Image.fromarray(ids.astype('int32')).resize((tr['width'],tr['height']),Image.Resampling.NEAREST))
  canvas=np.zeros((1050,1680),dtype='int32');left,top=tr['left'],tr['top'];canvas[top:top+tr['height'],left:left+tr['width']]=resized
  category=np.zeros_like(canvas);area=np.zeros_like(canvas);thing=np.zeros_like(canvas,dtype=bool);target=np.zeros_like(canvas,dtype=bool);crowd=np.zeros_like(canvas,dtype=bool)
  for sid,s in segs.items():
   m=canvas==sid;cat=cats[s['category_id']];category[m]=s['category_id'];area[m]=m.sum();thing[m]=bool(cat['isthing']);crowd[m]=bool(s.get('iscrowd',0))
   if cat['name']=='bottle':target|=m
  assert target.any()
  distance=distance_transform_edt(~target)
  cache[f'{id:012d}.jpg']=dict(tr=tr,category=category,area=area,thing=thing,target=target,distance=distance,crowd=crowd,cats=cats)
  np.savez_compressed(ROOT/f'{id}_aligned_masks.npz',segment_id=canvas,category_id=category,target=target)
  alignment.append(dict(image_id=id,target_instances=sum(cats[s['category_id']]['name']=='bottle' for s in segs.values()),target_pixels=int(target.sum()),original_mask_area_verified=True,mean_absolute_rgb_error=tr['mean_absolute_rgb_error']))
  image=np.asarray(Image.open(BASE/'day4'/f'{id:012d}.jpg').convert('RGB'))/255
  palette=np.zeros((*canvas.shape,3));rng=np.random.default_rng(123)
  for sid in segs:palette[canvas==sid]=rng.uniform(.15,1,3)
  blend=image.copy();known=canvas>0;blend[known]=.65*image[known]+.35*palette[known]
  fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
  axs[0].imshow(image);axs[0].set_title('Released Search18 photograph')
  axs[1].imshow(blend);axs[1].contour(target,levels=[.5],colors=['cyan'],linewidths=1.8);axs[1].contour(distance<=16,levels=[.5],colors=['yellow'],linewidths=.8)
  axs[1].set_title('Aligned instance/stuff regions; cyan = bottles; yellow = 16 px exclusion')
  for ax in axs:ax.set(xlabel='x (stimulus pixels)',ylabel='y (stimulus pixels)',xlim=(-.5,1679.5),ylim=(1049.5,-.5))
  fig.suptitle(f'Image {id} • nearest-neighbour masks after verified photograph transform',fontsize=14)
  fig.savefig(ROOT/f'{id}_semantic_overlay.png',dpi=140);plt.close(fig)
 trials={(t['name'],t['subject']):t for t in json.loads((BASE/'audit/raw/coco_search18_fixations_TP_train_split1.json').read_text()) if t['task']=='bottle'}
 eligible=[r for r in csv.DictReader((BASE/'day4/transition_log.csv').open()) if r['status']=='retained'];outputs=[];flows=[]
 for dilation in [0,16,32]:
  for row in eligible:
   name=row['image'];sub=int(row['subject']);i=int(row['origin']);j=int(row['destination']);assert j==i+1
   t=trials[(name,sub)];m=cache[name];tr=m['tr'];g=np.array([t['X'][i],t['Y'][i]]);q=np.array([t['X'][j],t['Y'][j]]);history=np.column_stack((t['X'][:i+1],t['Y'][:i+1]));qx,qy=np.floor(q+.5).astype(int)
   key=f'train_split1|bottle|{name}|subject={sub}|origin={i}|destination={j}'
   record=dict(dilation_px=dilation,decision_id=key,image=name,subject=sub,origin=i,landing=j,category='',spatial_pool=0,semantic_pool=0,status='',reason='')
   def fail(reason):record.update(status='excluded',reason=reason);flows.append(record)
   margin=88;l=tr['left'];top=tr['top'];right=l+tr['width'];bottom=top+tr['height']
   if not(l+margin<=q[0]<right-margin and top+margin<=q[1]<bottom-margin):fail('chosen_support');continue
   past=np.floor(np.column_stack((t['X'][:j+1],t['Y'][:j+1]))+.5).astype(int)
   if np.any(m['distance'][past[:,1],past[:,0]]<=dilation):fail('target_mask_entry_at_or_before_landing');continue
   cat=int(m['category'][qy,qx]);record['category']=m['cats'][cat]['name'] if cat else 'void'
   if cat==0 or m['crowd'][qy,qx]:fail('chosen_void_or_crowd');continue
   # Guard uncertainty at a one-pixel label boundary for the observed chosen point.
   if np.any(m['category'][qy-1:qy+2,qx-1:qx+2]!=cat):fail('chosen_semantic_boundary_within_1px');continue
   yy,xx=np.meshgrid(np.arange(0,1050,8),np.arange(0,1680,8),indexing='ij');p=np.column_stack((xx.ravel(),yy.ravel()));px,py=p.T
   e=np.linalg.norm(p-g,axis=1);c=np.linalg.norm(p-[840,525],axis=1);td=m['distance'][py,px];prior=np.min(np.linalg.norm(p[:,None,:]-history,axis=2),axis=1)<=32
   qe=float(np.linalg.norm(q-g));qc=float(np.linalg.norm(q-[840,525]));qt=float(m['distance'][qy,qx]);qh=bool(np.min(np.linalg.norm(history-q,axis=1))<=32)
   mask=(px>=l+margin)&(px<right-margin)&(py>=top+margin)&(py<bottom-margin)&(td>dilation)&(e>=32)&(np.linalg.norm(p-q,axis=1)>=32)&(np.abs(e-qe)<=16)&(np.abs(c-qc)<=32)&(np.abs(td-qt)<=32)&(prior==qh)
   record['spatial_pool']=int(mask.sum())
   mask&=(m['category'][py,px]==cat)&(~m['crowd'][py,px])
   # Equal category plus factor-two mapped instance area for things; stuff has no area caliper.
   if m['thing'][qy,qx]:mask&=(m['area'][py,px]>=m['area'][qy,qx]/2)&(m['area'][py,px]<=m['area'][qy,qx]*2)
   pool=np.flatnonzero(mask);record['semantic_pool']=len(pool)
   if len(pool)<8:record.update(status='failed',reason='fewer_than_8_semantic_alternatives');flows.append(record);continue
   seed=(20261006+int(hashlib.sha256(key.encode()).hexdigest()[:16],16))%(2**64);pick=np.random.default_rng(seed).choice(pool,8,replace=False)
   assert len(set(pick))==8
   record.update(status='matched',reason='semantic_and_spatial');flows.append(record)
   for role,point in [('chosen',q)]+[(f'alternative_{k+1}',p[idx]) for k,idx in enumerate(pick)]:
    x,y=np.floor(point+.5).astype(int);de=float(np.linalg.norm(point-g));dc=float(np.linalg.norm(point-[840,525]));dt=float(m['distance'][y,x]);a=int(m['area'][y,x]);ca=int(m['category'][y,x]);assert ca==cat
    if role!='chosen':
     assert abs(de-qe)<=16+1e-9 and abs(dc-qc)<=32+1e-9 and abs(dt-qt)<=32+1e-9
     assert dt>dilation and np.linalg.norm(point-q)>=32 and de>=32
     if m['thing'][qy,qx]:assert .5<=a/m['area'][qy,qx]<=2
    outputs.append(dict(dilation_px=dilation,decision_id=key,image=name,subject=sub,role=role,x=float(point[0]),y=float(point[1]),category=m['cats'][ca]['name'],isthing=bool(m['thing'][y,x]),instance_area_pixels=a if m['thing'][y,x] else '',gaze_distance=de,centre_distance=dc,target_mask_distance=dt,delta_gaze=de-qe,delta_centre=dc-qc,delta_target=dt-qt))
 save('decision_log.csv',flows);save('candidates_all_dilations.csv',outputs);primary=[r for r in outputs if r['dilation_px']==16];save('chosen_vs_alternatives.csv',primary)
 summary=[]
 for d in [0,16,32]:
  fs=[r for r in flows if r['dilation_px']==d];summary.append(dict(dilation=d,attempted=len(fs),matched=sum(r['status']=='matched' for r in fs),failed=sum(r['status']=='failed' for r in fs),excluded=sum(r['status']=='excluded' for r in fs)))
 assert len(primary)==9*summary[1]['matched']
 (ROOT/'verification.json').write_text(json.dumps(dict(alignment=alignment,sensitivity=summary,primary_rows=len(primary),checks='Original PNG segment IDs and areas; nearest-neighbour ID preservation; calipers; target mask and history; category; thing area; eight unique alternatives; counts'),indent=2));print(json.dumps(summary,indent=2))
 lines=['# Semantic and spatial matching — primary 16 px target exclusion','', '| Image | Participant | Movement | Category | Spatial pool | Semantic pool | Status | Reason |','|---|---:|---|---|---:|---:|---|---|']
 for r in flows:
  if r['dilation_px']==16:lines.append(f"| {r['image']} | {r['subject']} | {r['origin']}→{r['landing']} | {r['category']} | {r['spatial_pool']} | {r['semantic_pool']} | {r['status']} | {r['reason']} |")
 if primary:
  lines+=['','## First matched set','', '| Role | Category | x | y | Gaze distance | Centre distance | Target-mask distance |','|---|---|---:|---:|---:|---:|---:|']
  for r in primary:
   if r['decision_id']==primary[0]['decision_id']:lines.append(f"| {r['role']} | {r['category']} | {r['x']:.1f} | {r['y']:.1f} | {r['gaze_distance']:.1f} | {r['centre_distance']:.1f} | {r['target_mask_distance']:.1f} |")
 (ROOT/'tables.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
