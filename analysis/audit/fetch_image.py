import struct,urllib.request,zlib
from pathlib import Path
u='http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip'
r=urllib.request.urlopen(urllib.request.Request(u,headers={'Range':'bytes=-1000000'}),timeout=30)
b=r.read();total=int(r.headers['Content-Range'].split('/')[-1]);
p=b.rfind(b'PK\x05\x06');e=struct.unpack_from('<4s4H2LH',b,p);size,off=e[5:7];start=total-len(b);i=off-start;entries=[]
while b[i:i+4]==b'PK\x01\x02':
 v=struct.unpack_from('<4s6H3L5H2L',b,i);n,ex,co=v[10:13];name=b[i+46:i+46+n].decode();entries.append((name,v[8],v[16],v[4]));i+=46+n+ex+co
print(len(entries));item=next(x for x in entries if x[0].endswith('000000478726.jpg'));print(item)
name,cs,offset,method=item;u='http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip';req=urllib.request.Request(u,headers={'Range':f'bytes={offset}-{offset+cs+500}'})
b=urllib.request.urlopen(req,timeout=30).read();v=struct.unpack_from('<4s5H3L2H',b,0);st=30+v[-2]+v[-1];data=b[st:st+cs];data=zlib.decompress(data,-15) if method==8 else data
Path('audit/representative.jpg').write_bytes(data);print(len(data))
