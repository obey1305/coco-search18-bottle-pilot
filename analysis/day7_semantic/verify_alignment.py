from pathlib import Path
import numpy as np,json,math
from PIL import Image
O=Path(__file__).resolve().parent;P=O.parent;out=[]
for m in json.loads((P/'day4/selection.json').read_text()):
 id=int(m['image'][:-4]);orig=Image.open(O/f'{id}_original.jpg').convert('RGB');dst=np.array(Image.open(P/'day4'/m['image']).convert('RGB'),dtype=float);w,h=orig.size;s=min(1680/w,1050/h);results=[]
 for nw in set([math.floor(w*s),round(w*s),math.ceil(w*s)]):
  for nh in set([math.floor(h*s),round(h*s),math.ceil(h*s)]):
   for method in ['BILINEAR','BICUBIC','LANCZOS']:
    a=np.asarray(orig.resize((nw,nh),getattr(Image.Resampling,method)),dtype=float)
    for left in set([math.floor((1680-nw)/2),math.ceil((1680-nw)/2)]):
     for top in set([math.floor((1050-nh)/2),math.ceil((1050-nh)/2)]):
      err=np.mean(np.abs(a-dst[top:top+nh,left:left+nw]));results.append(dict(image_id=id,original_width=w,original_height=h,width=nw,height=nh,left=left,top=top,method=method,mean_absolute_rgb_error=float(err)))
 best=min(results,key=lambda q:q['mean_absolute_rgb_error']);out.append(best);print(best,flush=True)
 (O/f'{id}_alignment_candidates.json').write_text(json.dumps(sorted(results,key=lambda q:q['mean_absolute_rgb_error']),indent=2))
(O/'transforms.json').write_text(json.dumps(out,indent=2))
