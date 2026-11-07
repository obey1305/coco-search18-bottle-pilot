"""Day 6: verified real Gabor orientation and Lab L* centre-surround maps."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'day5'))
from colour_contrast import rgb_to_lab, kernels, convolve_same
ANGLES=(0,45,90,135)

def gabor(theta):
    y,x=np.mgrid[-24:25,-24:25];t=np.deg2rad(theta)
    carrier=x*np.cos(t)+y*np.sin(t)
    k=np.exp(-(x*x+y*y)/(2*8**2))*np.cos(2*np.pi*carrier/16)
    k-=k.mean()  # finite kernel is exactly DC-balanced before normalization
    return k/np.linalg.norm(k)

def responses(gray):
    return np.stack([np.abs(convolve_same(gray,gabor(a))) for a in ANGLES],axis=-1)

def support(shape,bounds,margin):
    y,x=np.indices(shape);l,t,r,b=bounds
    return (x-margin>=l)&(x+margin<r)&(y-margin>=t)&(y+margin<b)

def feature_maps(gray,lightness,bounds):
    d,r=kernels();diff=d/d.sum()-r/r.sum()
    response=responses(gray)
    delta=np.stack([convolve_same(response[...,i],diff) for i in range(4)],axis=-1)
    orientation=np.linalg.norm(delta,axis=-1)
    luminance=np.abs(convolve_same(lightness,diff))
    om=support(gray.shape,bounds,88);lm=support(gray.shape,bounds,64)
    orientation[~om]=np.nan;luminance[~lm]=np.nan
    return orientation,luminance,om,lm,response

def verify_filters():
    y,x=np.indices((256,256)); vertical=.5+.5*np.cos(2*np.pi*x/16);horizontal=.5+.5*np.cos(2*np.pi*y/16)
    patterns=[vertical,horizontal];means=[]
    for pattern,expected in zip(patterns,(0,2)):
        resp=responses(pattern);m=resp[32:-32,32:-32].mean(axis=(0,1));means.append(m)
        assert int(m.argmax())==expected
        assert m[expected]>10*m[2-expected]
    np.testing.assert_allclose(means[0][0],means[1][2],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(gabor(0).T,gabor(90),atol=1e-14)
    for a in ANGLES:
        k=gabor(a);assert abs(k.sum())<1e-12;assert abs(np.linalg.norm(k)-1)<1e-12
    uniform=np.full((256,256),.5)
    o,l,om,lm,_=feature_maps(uniform,np.full_like(uniform,50),(0,0,256,256))
    assert np.nanmax(o)<1e-10 and np.nanmax(l)<1e-10
    # Known luminance step confined exactly to centre disk: |60 - 50| = 10.
    field=np.full_like(uniform,50);field[(x-128)**2+(y-128)**2<=16**2]=60
    _,l,_,_,_=feature_maps(uniform,field,(0,0,256,256));assert abs(l[128,128]-10)<1e-10
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,(p,m,title) in enumerate(zip(patterns,means,['Vertical stripes','Horizontal stripes'])):
        axs[row,0].imshow(p,cmap='gray',vmin=0,vmax=1);axs[row,0].set_title(title);axs[row,0].axis('off')
        axs[row,1].bar([str(a)+'°' for a in ANGLES],m,color='#0072b2');axs[row,1].set_ylim(0,max(np.max(v) for v in means)*1.2)
        axs[row,1].set(xlabel='Filter normal angle',ylabel='Mean absolute response',title='Strongest: '+str(ANGLES[int(m.argmax())])+'°')
    fig.suptitle('Synthetic checks before processing the photograph\n0° normal detects vertical stripes; 90° normal detects horizontal stripes')
    fig.savefig(ROOT/'filter_checks.png',dpi=150);plt.close(fig)
    return {'vertical_stripe_responses':means[0].tolist(),'horizontal_stripe_responses':means[1].tolist(),'status':'PASS: preferred orientation, transpose symmetry, kernel norm/DC, uniform fields, known L* step'}

def main():
    # Deliberately complete synthetic tests before loading the photograph.
    test=verify_filters();print('Synthetic filter checks PASS',flush=True)
    meta=json.loads((ROOT.parent/'day4/selection.json').read_text())[0]
    source=ROOT.parent/'day4'/meta['image'];assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['image_sha256']
    rgb=np.asarray(Image.open(source).convert('RGB'),dtype=float)/255
    lab=rgb_to_lab(rgb);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    gray=np.einsum('...j,j->...',linear,np.array([.2126729,.7151522,.0721750]),optimize=False)
    bounds=tuple(meta[k] for k in ['content_left','content_top','content_right_exclusive','content_bottom_exclusive'])
    o,l,om,lm,response=feature_maps(gray,lab[...,0],bounds)
    d,r=kernels();rows=[]
    for name,x,y in [('A',840,525),('B',400,500),('C',1050,350),('D',1480,350),('E',900,750)]:
        assert om[y,x] and lm[y,x]
        patch=response[y-64:y+65,x-64:x+65];cv=patch[d].mean(axis=0);sv=patch[r].mean(axis=0);od=float(np.linalg.norm(cv-sv))
        light=lab[y-64:y+65,x-64:x+65,0];ld=float(abs(light[d].mean()-light[r].mean()))
        assert abs(od-o[y,x])<1e-9 and abs(ld-l[y,x])<1e-9
        # Check FFT filter output against direct spatial convolution at all four angles.
        for i,a in enumerate(ANGLES):
            direct=abs(float(np.sum(gray[y-24:y+25,x-24:x+25]*gabor(a)[::-1,::-1])))
            assert abs(direct-response[y,x,i])<1e-10
        rows.append(dict(label=name,x=x,y=y,orientation=float(o[y,x]),orientation_direct=od,orientation_error=abs(od-o[y,x]),luminance=float(l[y,x]),luminance_direct=ld,luminance_error=abs(ld-l[y,x])))
    for mask,margin in [(om,88),(lm,64)]:
        left,top,right,bottom=bounds
        for x,y,expected in [(left+margin-1,525,False),(left+margin,525,True),(right-margin,525,False),(right-margin-1,525,True),(840,top+margin-1,False),(840,top+margin,True),(840,bottom-margin,False),(840,bottom-margin-1,True)]:assert bool(mask[y,x])==expected
        assert mask.sum()==(right-left-2*margin)*(bottom-top-2*margin)
    assert np.isnan(o[~om]).all() and np.isfinite(o[om]).all()
    assert np.isnan(l[~lm]).all() and np.isfinite(l[lm]).all()
    with (ROOT/'five_values.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(ROOT/'feature_maps.npz',orientation=o,luminance=l,orientation_valid=om,luminance_valid=lm,content_bounds=bounds)
    report=dict(image=meta['image'],image_sha256=meta['image_sha256'],synthetic_tests=test,orientation_valid_centres=int(om.sum()),luminance_valid_centres=int(lm.sum()),photograph_tests='PASS: five direct centre-surround checks per feature; 20 direct Gabor checks; 16 boundary cases; valid/excluded arrays',points=rows)
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    axs[0].imshow(rgb);axs[0].set_title('Original stimulus • fixed points A–E')
    cm=plt.get_cmap('magma').copy();cm.set_bad('#d5d9df')
    for ax,field,title,unit in [(axs[1],o,'Orientation centre–surround contrast','L2 difference of four mean absolute responses'),(axs[2],l,'Luminance centre–surround contrast','Absolute difference of mean L*')]:
        im=ax.imshow(field,cmap=cm,vmin=0,vmax=float(np.nanmax(field)));ax.set_title(title)
        fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.85,pad=.09,label=unit)
    for ax in axs:
        ax.set(xlabel='x (stimulus pixels)',ylabel='y (stimulus pixels)',xlim=(-.5,1679.5),ylim=(1049.5,-.5))
        for row in rows:
            x,y=row['x'],row['y'];ax.plot(x,y,'+',color='cyan',ms=6)
            ax.annotate(row['label'],(x,y),xytext=(5,5),textcoords='offset points',fontsize=9,bbox=dict(fc='white',ec='black',pad=1.5))
    fig.suptitle('Day 6 • bottle image 496294\nGrey = excluded; orientation requires 88 px of support, luminance 64 px • separate colour scales',fontsize=14)
    fig.savefig(ROOT/'feature_maps.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
