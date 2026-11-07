from pathlib import Path
import csv,json,math,hashlib,subprocess,sys
import numpy as np
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'chosen_vs_alternatives.csv').open())) if (P/'chosen_vs_alternatives.csv').exists() else []
trials={(t['name'],t['subject']):t for t in json.loads((P.parent/'audit/raw/coco_search18_fixations_TP_train_split1.json').read_text()) if t['task']=='bottle'}
sets={}
for r in rows:sets.setdefault(r['decision_id'],[]).append(r)
for key,group in sets.items():
 assert len(group)==9 and len({(r['x'],r['y']) for r in group})==9
 q=next(r for r in group if r['role']=='chosen');origin=int(key.split('origin=')[1].split('|')[0]);t=trials[(q['image'],int(q['subject']))]
 mask=np.load(P/f"{int(q['image'][:-4])}_aligned_masks.npz");ys,xs=np.nonzero(mask['target']);target_points=np.column_stack((xs,ys))
 for r in group:
  x,y=float(r['x']),float(r['y']);xi,yi=int(math.floor(x+.5)),int(math.floor(y+.5))
  assert mask['category_id'][yi,xi]==mask['category_id'][int(float(q['y'])+.5),int(float(q['x'])+.5)]
  direct=float(np.min(np.hypot(target_points[:,0]-xi,target_points[:,1]-yi)))
  assert abs(direct-float(r['target_mask_distance']))<1e-9 and direct>16
  assert abs(math.hypot(x-t['X'][origin],y-t['Y'][origin])-float(r['gaze_distance']))<1e-9
  assert abs(math.hypot(x-840,y-525)-float(r['centre_distance']))<1e-9
  if r['role']!='chosen':
   assert abs(float(r['delta_gaze']))<=16+1e-9 and abs(float(r['delta_centre']))<=32+1e-9 and abs(float(r['delta_target']))<=32+1e-9
   if r['isthing']=='True':assert .5<=int(r['instance_area_pixels'])/int(q['instance_area_pixels'])<=2
files=['decision_log.csv','candidates_all_dilations.csv']+(['chosen_vs_alternatives.csv'] if rows else [])
before={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files}
subprocess.run([sys.executable,'-W','error::RuntimeWarning',str(P/'semantic_match.py')],check=True,stdout=subprocess.DEVNULL)
assert before=={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files}
(P/'independent_checks.txt').write_text(f'PASS: {len(rows)} primary rows checked against nearest target-mask pixels by exhaustive distance calculation, category IDs, area ratios and geometry. Byte-identical CSVs on rerun.\n')
print((P/'independent_checks.txt').read_text())
