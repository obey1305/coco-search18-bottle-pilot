"""Day 4: deterministic selection, image retrieval and descriptive QC (no modelling)."""
from pathlib import Path
import json, hashlib, csv, struct, urllib.request, zlib, math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'audit/raw/coco_search18_fixations_TP_train_split1.json'
data=json.loads(DATA.read_text())
key=lambda n:hashlib.sha256(('20261006|bottle|'+n).encode()).hexdigest()
names=sorted({t['name'] for t in data if t['task']=='bottle'},key=key)[:3]
url='http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip'
def request_range(a,b=None):
    spec=f'bytes=-{a}' if b is None else f'bytes={a}-{b}'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':spec}),timeout=45) as r:
        if r.status!=206: raise RuntimeError('Server ignored byte range')
        return r.read(),r.headers['Content-Range']
if any(not (ROOT/n).exists() for n in names):
    tail,cr=request_range(1000000); total=int(cr.split('/')[-1]); e=struct.unpack_from('<4s4H2LH',tail,tail.rfind(b'PK\x05\x06')); i=e[6]-(total-len(tail)); entries={}
    while tail[i:i+4]==b'PK\x01\x02':
        v=struct.unpack_from('<4s6H3L5H2L',tail,i); n,ex,co=v[10:13]; name=tail[i+46:i+46+n].decode(); entries[name]=(v[8],v[16],v[4],v[7]); i+=46+n+ex+co
    for name in names:
        if (ROOT/name).exists():continue
        cs,off,method,crc=entries['images/bottle/'+name]
        b,_=request_range(off,off+cs+1024); v=struct.unpack_from('<4s5H3L2H',b,0); start=30+v[-2]+v[-1]; raw=b[start:start+cs]; raw=zlib.decompress(raw,-15) if method==8 else raw
        assert zlib.crc32(raw)&0xffffffff==crc
        (ROOT/name).write_bytes(raw)

def save_csv(name,rows):
    with (ROOT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]; transitions=[]; observations=[]; selections=[]
for name in names:
    im=Image.open(ROOT/name).convert('RGB'); a=np.asarray(im); h,w=a.shape[:2]; assert (w,h)==(1680,1050)
    # Near-black full rows/columns identify padding; require subsequent visual review.
    rows=np.flatnonzero(a.max(axis=(1,2))>16); cols=np.flatnonzero(a.max(axis=(0,2))>16)
    l,r=int(cols[0]),int(cols[-1])+1; top,bot=int(rows[0]),int(rows[-1])+1
    trials=sorted([t for t in data if t['task']=='bottle' and t['name']==name],key=lambda t:t['subject'])
    boxes={tuple(t['bbox']) for t in trials}; assert len(boxes)==1
    bbox=next(iter(boxes)); bx,by,bw,bh=bbox
    assert bw>0 and bh>0 and 0<=bx<bx+bw<=w and 0<=by<by+bh<=h
    def inside(x,y):return bx<=x<=bx+bw and by<=y<=by+bh
    def problems(x,y,d):
        reasons=[]
        if not all(math.isfinite(v) for v in (x,y,d)):reasons.append('nonfinite')
        elif not (0<=x<w and 0<=y<h):reasons.append('outside_display')
        elif not (l<=x<r and top<=y<bot):reasons.append('padding')
        if math.isfinite(d) and not 50<=d<=1000:reasons.append('duration_outside_50_1000_ms')
        return reasons
    count={}
    for t in trials:
        x,y,d=t['X'],t['Y'],t['T']; assert len(x)==len(y)==len(d)==t['length']
        issues=[problems(*v) for v in zip(x,y,d)]
        hit=next((i for i in range(len(x)) if math.isfinite(x[i]) and math.isfinite(y[i]) and inside(x[i],y[i])),None)
        for i in range(len(x)):
            observations.append(dict(image=name,subject=t['subject'],fixation=i,x=x[i],y=y[i],duration_ms=d[i],first_target_entry=hit,status=';'.join(issues[i]) or 'valid_geometry_and_duration'))
        for k in range(1,len(x)):
            if t['correct']!=1:reason='incorrect_trial'
            elif hit is not None and k>=hit:reason='target_entry_or_later'
            elif any(s!='duration_outside_50_1000_ms' for s in issues[k-1]+issues[k]):reason='invalid_coordinate_endpoint'
            elif 'duration_outside_50_1000_ms' in issues[k-1]:reason='invalid_origin_duration'
            else:reason='retained'
            count[reason]=count.get(reason,0)+1
            transitions.append(dict(image=name,subject=t['subject'],origin=k-1,destination=k,status=reason))
    selections.append(dict(image=name,selection_sha256=key(name),image_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),width=w,height=h,content_left=l,content_top=top,content_right_exclusive=r,content_bottom_exclusive=bot,bbox=list(bbox)))
    summary.append(dict(image=name,trials=len(trials),fixations=sum(len(t['X']) for t in trials),adjacent_pairs=sum(count.values()),retained=count.get('retained',0),incorrect_trial=count.get('incorrect_trial',0),target_entry_or_later=count.get('target_entry_or_later',0),invalid_coordinate_endpoint=count.get('invalid_coordinate_endpoint',0),invalid_origin_duration=count.get('invalid_origin_duration',0)))
    # One fixed representative trial per image: smallest participant ID, irrespective of outcome.
    t=trials[0]; x,y=t['X'],t['Y']; hit=next((i for i in range(len(x)) if inside(x[i],y[i])),None)
    fig,ax=plt.subplots(figsize=(12,8)); ax.imshow(im);ax.add_patch(Rectangle((l,top),r-l,bot-top,fill=False,ec='#d89000',lw=2,ls='--'));ax.add_patch(Rectangle((bx,by),bw,bh,fill=False,ec='#0072b2',lw=2.5))
    for k in range(1,len(x)):
        if all(math.isfinite(v) for v in (x[k-1],y[k-1],x[k],y[k])):
            ax.plot(x[k-1:k+1],y[k-1:k+1],color='#eeeeee',lw=1.5,ls=':')
    for i,(xx,yy) in enumerate(zip(x,y)):
        if not math.isfinite(xx) or not math.isfinite(yy):continue
        ax.plot(xx,yy,'+',color='#ffbf00',markersize=7,markeredgewidth=1.5)
        offset=({2:(24,20),3:(-30,15),4:(-30,-18),5:(-25,38)}.get(i,(12,12)) if name=='000000295317.jpg' else (12,12))
        ax.annotate(str(i),(xx,yy),xytext=offset,textcoords='offset points',fontsize=10,color='#111111',arrowprops=dict(arrowstyle='-',color='white',lw=.8),bbox=dict(boxstyle='circle,pad=.2',fc='white',ec='#222222'))
    ax.set(xlim=(-.5,w-.5),ylim=(h-.5,-.5),xlabel='x (stimulus pixels)',ylabel='y (stimulus pixels)')
    ax.set_title(f'Bottle • {name} • participant {t["subject"]}\nInitial fixation: 0 | First target-box entry: {hit} | Correct response: {t["correct"]}',fontsize=13)
    ax.legend(handles=[Line2D([0],[0],color='#0072b2',label='Released target box'),Line2D([0],[0],color='#d89000',ls='--',label='Estimated image-content boundary')],loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,frameon=False)
    fig.text(.5,.02,'Gold crosses: exact fixations; numbered labels are offset. Dotted lines: recorded pairs, not eligibility.',ha='center',fontsize=10)
    fig.subplots_adjust(bottom=.19,top=.88);fig.savefig(ROOT/(name[:-4]+'_overlay.png'),dpi=150);plt.close(fig)
save_csv('exclusion_table.csv',summary);save_csv('transition_log.csv',transitions);save_csv('observation_log.csv',observations)
(ROOT/'selection.json').write_text(json.dumps(selections,indent=2))
print(json.dumps({'selection':selections,'exclusions':summary},indent=2))
