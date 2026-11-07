"""Reproducible Day 5 colour centre-surround computation in stimulus pixels."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parent

def rgb_to_lab(rgb):
    """sRGB [0,1] -> CIELAB, D65 2-degree white; no luminance in contrast."""
    rgb=np.asarray(rgb,dtype=np.float64)
    linear=np.where(rgb<=0.04045,rgb/12.92,((rgb+0.055)/1.055)**2.4)
    xyz=np.einsum('...j,ij->...i',linear,np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]),optimize=False)
    ratio=xyz/np.array([.95047,1.,1.08883]); delta=6/29
    f=np.where(ratio>delta**3,np.cbrt(ratio),ratio/(3*delta**2)+4/29)
    return np.stack((116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])),axis=-1)

def kernels():
    yy,xx=np.mgrid[-64:65,-64:65]; d=xx*xx+yy*yy
    return (d<=16**2),( (d>=32**2)&(d<=64**2) )

def convolve_same(a,k):
    """Linear FFT convolution, never circular wrapping; symmetric kernels."""
    shape=(a.shape[0]+k.shape[0]-1,a.shape[1]+k.shape[1]-1)
    full=np.fft.irfft2(np.fft.rfft2(a,s=shape)*np.fft.rfft2(k,s=shape),s=shape)
    oy,ox=k.shape[0]//2,k.shape[1]//2
    return full[oy:oy+a.shape[0],ox:ox+a.shape[1]]

def calculate(lab,bounds):
    disk,ring=kernels(); diff=disk/disk.sum()-ring/ring.sum()
    da=convolve_same(lab[...,1],diff);db=convolve_same(lab[...,2],diff)
    contrast=np.hypot(da,db)
    l,t,r,b=bounds; yy,xx=np.indices(contrast.shape)
    valid=(xx-64>=l)&(xx+64<r)&(yy-64>=t)&(yy+64<b)
    contrast[~valid]=np.nan
    return contrast,valid

def direct(lab,x,y):
    disk,ring=kernels();patch=lab[y-64:y+65,x-64:x+65,1:3]
    centre=patch[disk].mean(axis=0);surround=patch[ring].mean(axis=0)
    return centre,surround,float(np.linalg.norm(centre-surround))

def main():
    meta=json.loads((ROOT.parent/'day4/selection.json').read_text())[0]
    source=ROOT.parent/'day4'/meta['image']; raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==meta['image_sha256']
    rgb=np.array(Image.open(source).convert('RGB'))/255.
    lab=rgb_to_lab(rgb)
    bounds=tuple(meta[k] for k in ['content_left','content_top','content_right_exclusive','content_bottom_exclusive'])
    contrast,valid=calculate(lab,bounds)
    points=[('A',840,525),('B',400,500),('C',1050,350),('D',1480,350),('E',900,750)]
    rows=[]
    for label,x,y in points:
        assert valid[y,x]
        c,s,v=direct(lab,x,y);mapped=float(contrast[y,x]);error=abs(v-mapped)
        assert error<1e-9
        rows.append(dict(label=label,x_px=x,y_px=y,centre_a=c[0],centre_b=c[1],surround_a=s[0],surround_b=s[1],direct_contrast=v,map_contrast=mapped,absolute_error=error,status='valid'))
    with (ROOT/'five_values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(ROOT/'colour_contrast.npz',contrast=contrast,valid=valid,content_bounds=np.array(bounds))
    # Tests independent of the scene: known Lab endpoints, constant and synthetic fields.
    np.testing.assert_allclose(rgb_to_lab([0,0,0]),[0,0,0],atol=1e-10)
    np.testing.assert_allclose(rgb_to_lab([1,1,1]),[100,0,0],atol=2e-5)
    np.testing.assert_allclose(rgb_to_lab([1,0,0]),[53.2408,80.0925,67.2032],atol=1e-3)
    uniform=np.zeros((150,150,3));uniform[:]=[50,20,-10]
    u,uv=calculate(uniform,(0,0,150,150));assert np.nanmax(u)<1e-10
    yy,xx=np.indices((150,150));synthetic=uniform.copy();centre=(xx-75)**2+(yy-75)**2<=16**2
    synthetic[centre,1]+=3;synthetic[centre,2]+=4
    sm,sv=calculate(synthetic,(0,0,150,150));assert abs(sm[75,75]-5)<1e-10
    # Exact edge cases for the actual content rectangle, outer radius included.
    l,t,r,b=bounds
    checks=[(l+63,525,False),(l+64,525,True),(840,t+63,False),(840,t+64,True),(r-65,525,True),(r-64,525,False),(840,b-65,True),(840,b-64,False)]
    for x,y,expected in checks:assert bool(valid[y,x])==expected
    assert np.isnan(contrast[~valid]).all() and np.isfinite(contrast[valid]).all()
    assert int(valid.sum())==(r-l-128)*(b-t-128)
    report=dict(image=meta['image'],sha256=meta['image_sha256'],content_bounds=bounds,centre_pixels=int(kernels()[0].sum()),surround_pixels=int(kernels()[1].sum()),valid_centres=int(valid.sum()),excluded_centres=int((~valid).sum()),minimum=float(np.nanmin(contrast)),maximum=float(np.nanmax(contrast)),max_five_point_error=max(r['absolute_error'] for r in rows),tests='PASS: black/white/red conversion; uniform colour; synthetic 3-4-5 contrast; eight boundary cases; masked values; five direct comparisons',points=rows)
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(16,6.5),layout='constrained')
    axes[0].imshow(rgb);axes[0].set_title('Five fixed locations on the original stimulus')
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d5d9df')
    plot=axes[1].imshow(contrast,cmap=cmap,vmin=0,vmax=float(np.nanmax(contrast)))
    axes[1].set_title('Colour centre–surround contrast; grey = excluded')
    for ax in axes:
        ax.set(xlabel='x (stimulus pixels)',ylabel='y (stimulus pixels)',xlim=(-.5,1679.5),ylim=(1049.5,-.5))
        for label,x,y in points:
            ax.plot(x,y,'+',color='cyan',ms=7)
            ax.annotate(label,(x,y),xytext=(7,7),textcoords='offset points',color='black',fontsize=11,bbox=dict(fc='white',ec='black',pad=2))
    for _,x,y in points:
        for radius,style in [(16,'-'),(32,':'),(64,'--')]:axes[0].add_patch(Circle((x,y),radius,fill=False,ec='cyan',lw=.8,ls=style))
    fig.colorbar(plot,ax=axes[1],shrink=.72,label='Distance between mean (a*, b*) colours')
    fig.suptitle('Day 5 • bottle image 496294\nCentre radius 16 px; surround 32–64 px • sRGB → CIELAB (D65)',fontsize=14)
    fig.savefig(ROOT/'colour_contrast_map.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
