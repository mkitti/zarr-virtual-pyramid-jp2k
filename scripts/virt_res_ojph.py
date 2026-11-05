"""Resolution-r virtual codestreams from an OpenJPH file (tile-part per resolution, TLM, no PLT)."""
import sys, struct, pathlib, numpy as np, imagecodecs as ic, refkv
from refkv.pb import Concat, Range, Source
SP=pathlib.Path(sys.argv[1]); LEV=3; W0=512
f=SP/"enc_ojph/p00.j2c"; d=f.read_bytes()
pos=2; segs={}
while d[pos:pos+2]!=b"\xff\x90":
    L=struct.unpack(">H",d[pos+2:pos+4])[0]; segs[d[pos:pos+2]]=(pos,L); pos+=2+L
tps=[]; p=pos                      # tile-parts: (SOT offset, Psot)
while p<len(d) and d[p:p+2]==b"\xff\x90":
    ps=struct.unpack(">I",d[p+6:p+10])[0]; tps.append((p,ps)); p+=ps
assert len(tps)==LEV+1 and d[p:p+2]==b"\xff\xd9"
tlm_pos,tlm_L=segs[b"\xff\x55"]; stlm=d[tlm_pos+5]; assert stlm==0x60   # Ttlm 16-bit, Ptlm 32-bit
seg=lambda m: bytearray(d[segs[m][0]:segs[m][0]+2+segs[m][1]])
def build(r, patch_tn=True):
    w=W0>>(LEV-r)
    siz=seg(b"\xff\x51")
    for off in (6,10,22,26): siz[off:off+4]=struct.pack(">I",w)
    cod=seg(b"\xff\x52"); cod[9]=r
    n=1+3*r; qcd=seg(b"\xff\x5c")[:5+n]; qcd[2:4]=struct.pack(">H",3+n)
    tlm=seg(b"\xff\x55"); tlm=tlm[:6+6*(r+1)]; tlm[2:4]=struct.pack(">H",4+6*(r+1))
    main=b"\xff\x4f"+bytes(siz)+bytes(seg(b"\xff\x50"))+bytes(cod)+bytes(qcd)+bytes(seg(b"\xff\x64"))+bytes(tlm)
    parts=[Range(data=main)]
    for i in range(r+1):
        sp,ps=tps[i]; sot=bytearray(d[sp:sp+12])
        if patch_tn: sot[11]=r+1
        parts+= [Range(data=bytes(sot)+b"\xff\x93"), Range(source=0,offset=sp+14,length=ps-14)]
    parts.append(Range(data=b"\xff\xd9"))
    return Concat(tuple(parts)), sum(tps[i][1] for i in range(r+1))
for patch in (True,False):
    store=refkv.ReferenceStore({f"r{r}":build(r,patch)[0] for r in range(LEV+1)},[Source(url=f.resolve().as_uri())])
    print("patch TNsot:",patch)
    for r in range(LEV+1):
        b=store.get(f"r{r}"); ref=np.fromfile(SP/f"enc/_r{r}.raw","<u2")   # ndic --resolution r decode of the ndic-encoded plane
        res=[]
        for n,fn in (("openjph",ic.htj2k_decode),("openjpeg",ic.jpeg2k_decode)):
            try: a=fn(b).squeeze(); res.append(f"{n} {a.shape} == ndic-encoded LL: {np.array_equal(a.ravel(),ref)}")
            except Exception as e: res.append(f"{n} ERR {e!r}"[:70])
        print(f"  r{r}: {len(b)} B ({100*len(b)/len(d):.1f}%)",*res,sep=" | ")
