import sys, struct, gzip, urllib.request, numpy as np, os
from concurrent.futures import ThreadPoolExecutor
B="https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16"
X0,Y0,Z0,NX,NY,NZ = 6144,512,3200,512,512,16
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def chunk(lvl,cx,cy,cz):
    d=urllib.request.urlopen(f"{B}/s{lvl}/{cx}/{cy}/{cz}",timeout=60).read()
    mode,nd=struct.unpack(">HH",d[:4]); dims=struct.unpack(">"+"I"*nd,d[4:4+4*nd]); off=4+4*nd
    if mode==1: off+=4
    a=np.frombuffer(gzip.decompress(d[off:]),dtype=">u2").reshape(dims[2],dims[1],dims[0])
    return (cx,cy,cz),a
for lvl in range(4):
    x0,y0,z0=X0>>lvl,Y0>>lvl,Z0>>lvl; nx,ny,nz=NX>>lvl,NY>>lvl,max(1,NZ>>lvl)
    keys=[(cx,cy,cz) for cx in range(x0//64,(x0+nx-1)//64+1) for cy in range(y0//64,(y0+ny-1)//64+1) for cz in range(z0//64,(z0+nz-1)//64+1)]
    vol=np.zeros((nz,ny,nx),np.uint16)
    with ThreadPoolExecutor(8) as ex:
        for (cx,cy,cz),a in ex.map(lambda k:chunk(lvl,*k),keys):
            gx,gy,gz=cx*64,cy*64,cz*64
            xs,xe=max(gx,x0),min(gx+a.shape[2],x0+nx); ys,ye=max(gy,y0),min(gy+a.shape[1],y0+ny); zs,ze=max(gz,z0),min(gz+a.shape[0],z0+nz)
            vol[zs-z0:ze-z0,ys-y0:ye-y0,xs-x0:xe-x0]=a[zs-gz:ze-gz,ys-gy:ye-gy,xs-gx:xe-gx]
    np.save(f"{out}/s{lvl}.npy",vol.astype(np.uint16)); print(lvl,vol.shape,len(keys),"chunks",vol.min(),vol.max(),flush=True)
