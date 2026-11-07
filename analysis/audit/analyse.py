from pathlib import Path
import json,csv,hashlib,math
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path(__file__).parent
files=[root/'raw'/('coco_search18_fixations_TP_'+s+'_split1.json') for s in ['train','validation']]
data=sum([json.loads(p.read_text()) for p in files],[])
def inside(x,y,b):
 a,c,w,h=b;return a<=x<=a+w and c<=y<=c+h
def transitions(t):
 n=min(len(t['X']),len(t['Y']));out=[]
 for k in range(1,n):
  x0,y0=t['X'][k-1],t['Y'][k-1];x,y=t['X'][k],t['Y'][k]
  if inside(x0,y0,t['bbox']) or inside(x,y,t['bbox']):break
  if all([0<=x0<1680,0<=x<1680,0<=y0<1050,0<=y<1050]):out.append((k,x,y,math.hypot(x-x0,y-y0)))
 return out
rows=[]
for cat in sorted(set(t['task'] for t in data)):
 d=[t for t in data if t['task']==cat];good=[t for t in d if t['correct']==1]
 ds=[v[3] for t in good for v in transitions(t)]
 rows.append(dict(task=cat,images=len(set(t['name'] for t in d)),trials=len(d),correct_trials=len(good),pretarget_nontarget_transitions=len(ds),median_distance_px=round(float(np.median(ds)),1)))
with open(root/'audit_counts.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
t=next(t for t in data if t['name']=='000000478726.jpg' and t['subject']==2 and t['task']=='bottle')
im=Image.open(root/'representative.jpg');print('image',im.size)
fig,ax=plt.subplots(figsize=(12,7.5));ax.imshow(im);ax.plot(t['X'],t['Y'],'-',color='#ffb000',lw=2)
for i,(x,y) in enumerate(zip(t['X'],t['Y'])):ax.text(x,y,str(i),ha='center',va='center',fontsize=12,bbox=dict(boxstyle='circle',fc='white',ec='#222222'))
x,y,w,h=t['bbox'];ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,color='#0072b2',lw=3));ax.set_title('Public COCO-Search18: bottle search, image 478726, participant 2\n0 = initial fixation; blue rectangle = released target bounding box',fontsize=13);ax.axis('off');fig.tight_layout();fig.savefig(root/'scanpath_example.png',dpi=150);plt.close(fig)
summary={'source':'http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-fixations-TP.zip','sha256':hashlib.sha256((root/'fixations.zip').read_bytes()).hexdigest(),'files':[p.name for p in files],'trials':len(data),'images':len(set(t['name'] for t in data)),'subjects':sorted(set(t['subject'] for t in data)),'categories':len(rows),'correct_trials':sum(t['correct']==1 for t in data),'pretarget_nontarget_transitions':sum(r['pretarget_nontarget_transitions'] for r in rows),'representative':t,'table':rows}
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
