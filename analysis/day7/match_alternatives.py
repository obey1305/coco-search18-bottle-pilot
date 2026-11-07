"""Spatial matching only: all Day 4 eligible decisions, fixed calipers, no models."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent

def box_distance(p,box):
    x,y,w,h=box;p=np.asarray(p)
    dx=np.maximum(np.maximum(x-p[...,0],0),p[...,0]-(x+w))
    dy=np.maximum(np.maximum(y-p[...,1],0),p[...,1]-(y+h))
    return np.hypot(dx,dy)

def metrics(p,gaze,box,history):
    p=np.asarray(p);return np.linalg.norm(p-gaze,axis=-1),np.linalg.norm(p-np.array([840,525]),axis=-1),box_distance(p,box),np.min(np.linalg.norm(p[...,None,:]-history,axis=-1),axis=-1)<=32

def write(name,rows):
    with (ROOT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    data=json.loads((ROOT.parent/'audit/raw/coco_search18_fixations_TP_train_split1.json').read_text())
    metas={m['image']:m for m in json.loads((ROOT.parent/'day4/selection.json').read_text())}
    trials={(t['name'],t['subject']):t for t in data if t['task']=='bottle' and t['name'] in metas}
    eligible=[r for r in csv.DictReader((ROOT.parent/'day4/transition_log.csv').open()) if r['status']=='retained']
    assert len(eligible)==17
    candidates=[];flow=[];balances=[]
    for row in eligible:
        name=row['image'];subject=int(row['subject']);origin=int(row['origin']);dest=int(row['destination']);assert dest==origin+1
        t=trials[(name,subject)];meta=metas[name];box=t['bbox'];g=np.array([t['X'][origin],t['Y'][origin]]);chosen=np.array([t['X'][dest],t['Y'][dest]])
        # Prior to the landing: history includes the current origin, but no future fixations.
        history=np.array(list(zip(t['X'][:origin+1],t['Y'][:origin+1])))
        ce,cc,ct,ch=metrics(chosen,g,box,history)
        l,top,r,b=(meta[k] for k in ('content_left','content_top','content_right_exclusive','content_bottom_exclusive'))
        key=f'train_split1|bottle|{name}|subject={subject}|origin={origin}|destination={dest}'
        base=dict(decision_id=key,image=name,subject=subject,origin_index=origin,landing_index=dest,chosen_x=float(chosen[0]),chosen_y=float(chosen[1]),chosen_gaze_distance=float(ce),chosen_centre_distance=float(cc),chosen_target_box_distance=float(ct),chosen_previously_viewed=bool(ch),support_radius=88,eligible_pool=0,status='',reason='')
        # Radius 88 supports the complete Gabor + surround pipeline from Day 6.
        if not(l+88<=chosen[0]<r-88 and top+88<=chosen[1]<b-88):
            base.update(status='excluded',reason='chosen_patch_crosses_content_boundary');flow.append(base);continue
        assert ct>0 and t['correct']==1
        xx,yy=np.meshgrid(np.arange(0,1680,8),np.arange(0,1050,8));p=np.column_stack((xx.ravel(),yy.ravel()))
        mask=(p[:,0]>=l+88)&(p[:,0]<r-88)&(p[:,1]>=top+88)&(p[:,1]<b-88)
        e,c,td,h=metrics(p,g,box,history)
        mask&=(td>0)&(e>=32)&(np.linalg.norm(p-chosen,axis=1)>=32)
        mask&=(np.abs(e-ce)<=16)&(np.abs(c-cc)<=32)&(np.abs(td-ct)<=32)&(h==ch)
        pool=np.flatnonzero(mask);base['eligible_pool']=len(pool)
        if len(pool)<8:
            base.update(status='failed',reason='fewer_than_8_alternatives');flow.append(base);continue
        seed=(20261006+int(hashlib.sha256(key.encode()).hexdigest()[:16],16))%(2**64)
        ix=np.random.default_rng(seed).choice(pool,8,replace=False)
        assert len(set(ix.tolist()))==8
        base.update(status='matched',reason='spatial_matching_only');flow.append(base)
        for label,point in [('chosen',chosen)]+[(f'alternative_{i+1}',p[j]) for i,j in enumerate(ix)]:
            ee,ccc,tt,hh=metrics(point,g,box,history)
            candidates.append(dict(decision_id=key,image=name,subject=subject,origin_index=origin,landing_index=dest,role=label,chosen=int(label=='chosen'),x=float(point[0]),y=float(point[1]),gaze_distance=float(ee),centre_distance=float(ccc),target_box_distance=float(tt),previously_viewed=bool(hh),delta_gaze=float(ee-ce),delta_centre=float(ccc-cc),delta_target=float(tt-ct),semantic_category='unverified',matching='spatial matching only'))
            if label!='chosen':
                assert abs(ee-ce)<=16+1e-10 and abs(ccc-cc)<=32+1e-10 and abs(tt-ct)<=32+1e-10
                assert hh==ch and tt>0 and ee>=32 and np.linalg.norm(point-chosen)>=32
                assert l+88<=point[0]<r-88 and top+88<=point[1]<b-88
    write('decision_log.csv',flow)
    if candidates:write('chosen_vs_alternatives.csv',candidates)
    # Fixed descriptive distance bands, not estimated quantile thresholds.
    for low,high,label in [(0,100,'0–<100 px'),(100,300,'100–<300 px'),(300,float('inf'),'300+ px')]:
        subset=[f for f in flow if low<=f['chosen_gaze_distance']<high];ids={f['decision_id'] for f in subset};alts=[c for c in candidates if c['decision_id'] in ids and not c['chosen']]
        balances.append(dict(gaze_band=label,input_decisions=len(subset),matched=sum(f['status']=='matched' for f in subset),support_excluded=sum(f['status']=='excluded' for f in subset),pool_failed=sum(f['status']=='failed' for f in subset),alternatives=len(alts),mean_abs_delta_gaze=np.mean([abs(a['delta_gaze']) for a in alts]) if alts else '',mean_abs_delta_centre=np.mean([abs(a['delta_centre']) for a in alts]) if alts else '',mean_abs_delta_target=np.mean([abs(a['delta_target']) for a in alts]) if alts else ''))
    write('balance_by_distance.csv',balances)
    # Known rectangle distances: interior, side, diagonal.
    np.testing.assert_allclose(box_distance(np.array([[15,15],[25,15],[23,24]]),[10,10,10,10]),[0,5,5])
    matched=sum(f['status']=='matched' for f in flow);assert len(candidates)==9*matched
    report=dict(input_decisions=len(flow),matched=matched,support_excluded=sum(f['status']=='excluded' for f in flow),pool_failed=sum(f['status']=='failed' for f in flow),candidate_rows=len(candidates),status='spatial matching only',semantic_status='Public archive reachable; no downloaded, aligned and verified semantic masks in pilot',validation='PASS: box-distance reference cases; adjacency; eight unique alternatives; calipers; prior-view status; support; target-box exclusion; row reconciliation')
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    # Inspectable compact Markdown table for the first matched decision only.
    lines=['# Day 7 — spatial matching only','', '## Small chosen-versus-alternative table','']
    if candidates:
        first=candidates[0]['decision_id'];small=[c for c in candidates if c['decision_id']==first]
        lines+=['Decision: `'+first+'`','', '| Location | x | y | From gaze (px) | From centre (px) | From target box (px) |','|---|---:|---:|---:|---:|---:|']
        for c in small:lines.append(f"| {c['role']} | {c['x']:.1f} | {c['y']:.1f} | {c['gaze_distance']:.1f} | {c['centre_distance']:.1f} | {c['target_box_distance']:.1f} |")
    lines+=['','## Every attempted decision','', '| Image | Participant | Movement | Eligible pool | Status | Reason |','|---|---:|---|---:|---|---|']
    for f in flow:lines.append(f"| {f['image']} | {f['subject']} | {f['origin_index']}→{f['landing_index']} | {f['eligible_pool']} | {f['status']} | {f['reason']} |")
    (ROOT/'tables.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
