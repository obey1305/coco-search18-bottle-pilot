"""Descriptive chosen-minus-mean-control features for the fixed semantic pilot."""
from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path.insert(0,str(BASE/'day5'));sys.path.insert(0,str(BASE/'day6'))
from colour_contrast import rgb_to_lab,calculate,kernels,direct
from features import feature_maps,gabor

def save(name,rows):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 source=BASE/'day7_semantic/chosen_vs_alternatives.csv';raw=list(csv.DictReader(source.open()));assert len(raw)==90
 transforms=json.loads((BASE/'day7_semantic/transforms.json').read_text());byimage={};expanded=[];checks=[]
 disk,ring=kernels()
 hashes={m['image']:m['image_sha256'] for m in json.loads((BASE/'day4/selection.json').read_text())}
 for tr in transforms:
  name=f"{tr['image_id']:012d}.jpg";path=BASE/'day4'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
  rgb=np.asarray(Image.open(path).convert('RGB'),dtype=float)/255;lab=rgb_to_lab(rgb)
  bounds=(tr['left'],tr['top'],tr['left']+tr['width'],tr['top']+tr['height'])
  colour,cm=calculate(lab,bounds)
  linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);gray=np.einsum('...j,j->...',linear,np.array([.2126729,.7151522,.0721750]),optimize=False)
  orient,lum,om,lm,responses=feature_maps(gray,lab[...,0],bounds)
  assert np.isfinite(colour[cm]).all() and np.isfinite(orient[om]).all() and np.isfinite(lum[lm]).all()
  np.savez_compressed(P/f"{tr['image_id']}_features.npz",colour=colour,orientation=orient,luminance=lum,colour_valid=cm,orientation_valid=om,luminance_valid=lm,content_bounds=bounds)
  rows=[r for r in raw if r['image']==name];maxerr=0.
  for row in rows:
   x,y=int(np.floor(float(row['x'])+.5)),int(np.floor(float(row['y'])+.5));assert cm[y,x] and om[y,x] and lm[y,x]
   _,_,dc=direct(lab,x,y);patch=responses[y-64:y+65,x-64:x+65];do=float(np.linalg.norm(patch[disk].mean(axis=0)-patch[ring].mean(axis=0)))
   light=lab[y-64:y+65,x-64:x+65,0];dl=float(abs(light[disk].mean()-light[ring].mean()))
   err=max(abs(dc-colour[y,x]),abs(do-orient[y,x]),abs(dl-lum[y,x]));assert err<1e-9;maxerr=max(maxerr,err)
   expanded.append(dict(row,sample_x=x,sample_y=y,colour=float(colour[y,x]),orientation=float(orient[y,x]),luminance=float(lum[y,x])))
  # A direct centre check on every image, including the image with no matched observations.
  x,y=840,525;assert om[y,x]
  _,_,v=direct(lab,x,y);assert abs(v-colour[y,x])<1e-9
  lp=lab[y-64:y+65,x-64:x+65,0];assert abs(abs(lp[disk].mean()-lp[ring].mean())-lum[y,x])<1e-9
  for k,a in enumerate([0,45,90,135]):assert abs(abs(np.sum(gray[y-24:y+25,x-24:x+25]*gabor(a)[::-1,::-1]))-responses[y,x,k])<1e-9
  op=responses[y-64:y+65,x-64:x+65];assert abs(np.linalg.norm(op[disk].mean(axis=0)-op[ring].mean(axis=0))-orient[y,x])<1e-9
  checks.append(dict(image=name,feature_rows=len(rows),max_direct_error=maxerr,valid_colour=int(cm.sum()),valid_orientation=int(om.sum()),valid_luminance=int(lm.sum()),centre_check='pass'))
  print('Computed and checked',name,flush=True)
 save('candidate_features.csv',expanded)
 observations=[]
 for key in dict.fromkeys(r['decision_id'] for r in expanded):
  group=[r for r in expanded if r['decision_id']==key];q=next(r for r in group if r['role']=='chosen');controls=[r for r in group if r['role']!='chosen'];assert len(group)==9 and len(controls)==8
  item=dict(decision_id=key,image=q['image'],subject=int(q['subject']),category=q['category'],pre_saccadic_distance=float(q['gaze_distance']),control_count=8)
  for feature in ['colour','orientation','luminance']:
   mean=float(np.mean([r[feature] for r in controls]));item[feature+'_chosen']=q[feature];item[feature+'_control_mean']=mean;item[feature+'_difference']=q[feature]-mean
  observations.append(item)
 assert len(observations)==10;save('observations.csv',observations)
 logs=list(csv.DictReader((BASE/'day7_semantic/decision_log.csv').open()));counts=[]
 for tr in transforms:
  name=f"{tr['image_id']:012d}.jpg";r=[v for v in logs if v['image']==name and v['dilation_px']=='16'];obs=[o for o in observations if o['image']==name]
  counts.append(dict(image=name,attempted=len(r),matched=len(obs),failed=sum(v['status']=='failed' for v in r),excluded=sum(v['status']=='excluded' for v in r),participants=len({v['subject'] for v in obs}),controls=8*len(obs)))
 save('counts.csv',counts)
 fig,axs=plt.subplots(1,3,figsize=(15,6));fig.subplots_adjust(left=.065,right=.98,bottom=.28,top=.78,wspace=.30)
 colours={'000000496294.jpg':'#b55b00','000000295317.jpg':'#0072b2'};markers={'000000496294.jpg':'D','000000295317.jpg':'o'}
 labels=['Colour difference (a*, b* distance)','Orientation difference (filter-response units)','Luminance difference (L* units)']
 for ax,feature,label in zip(axs,['colour','orientation','luminance'],labels):
  ax.axhline(0,color='#555555',lw=1,ls='--',zorder=1)
  for image,color in colours.items():
   items=[o for o in observations if o['image']==image]
   ax.scatter([o['pre_saccadic_distance'] for o in items],[o[feature+'_difference'] for o in items],s=60,marker=markers[image],color=color,edgecolors='white',linewidth=.6,label=f"Image {int(image[:-4])} (n={len(items)})",zorder=3)
  ax.set(xlabel='Pre-saccadic distance (pixels)',ylabel=label,xlim=(0,650))
  ax.set_title(feature.capitalize(),fontsize=13);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.suptitle('Chosen contrast minus mean matched-alternative contrast\n10 individual landings • 8 alternatives per landing • descriptive pilot',fontsize=16,y=.96)
 handles,legends=axs[0].get_legend_handles_labels();fig.legend(handles,legends,loc='lower center',bbox_to_anchor=(.5,.145),ncol=2,frameon=False)
 subjects=len({o['subject'] for o in observations})
 fig.text(.5,.11,f'All three images processed: 496294: 1 match; 522129: 0 matches (5 unlabelled, 2 failures); 295317: 9 matches.',ha='center',fontsize=10)
 fig.text(.5,.065,f'10 landings from {subjects} participants; repeated observations are not independent. Separate y-axis units. No fitted trend or significance test.',ha='center',fontsize=10)
 fig.text(.5,.025,'Above zero: chosen point has higher contrast. Below zero: chosen point has lower contrast than its alternatives’ mean.',ha='center',fontsize=10)
 # Ensure fixed plot limits include every original distance, without clipping any observation.
 assert all(0<=o['pre_saccadic_distance']<=650 for o in observations)
 fig.savefig(P/'descriptive_contrast.png',dpi=160);fig.savefig(P/'descriptive_contrast.pdf');plt.close(fig)
 report=dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),images_computed=3,observations=len(observations),participants=subjects,contributing_images=len({o['image'] for o in observations}),candidate_rows=len(expanded),checks=checks,positive_counts={f:sum(o[f+'_difference']>0 for o in observations) for f in ['colour','orientation','luminance']})
 (P/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
