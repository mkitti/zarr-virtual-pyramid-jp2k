import sys, struct, pathlib, subprocess, numpy as np
import refkv, imagecodecs as ic
from refkv.pb import Concat, Range, Source
SP=pathlib.Path(sys.argv[1]); NDIC=sys.argv[2]
f=SP/"enc/p00.j2c"; d=f.read_bytes()
# --- parse markers of the original (single tile, single tile-part) ---
pos=2; segs={}
while d[pos:pos+2]!=b"\xff\x90":
    m=d[pos:pos+2]; L=struct.unpack(">H",d[pos+2:pos+4])[0]; segs[m]=(pos,L); pos+=2+L
sot=pos; assert d[sot+2:sot+4]==b"\x00\x0a"
psot=struct.unpack(">I",d[sot+6:sot+10])[0]; assert psot+sot<=len(d)
p=sot+12; assert d[p:p+2]==b"\xff\x58"; plt_L=struct.unpack(">H",d[p+2:p+4])[0]
plt=d[p+5:p+2+plt_L]  # after Lplt(2), Zplt(1)
lens=[];v=0
for b in plt:
    v=(v<<7)|(b&0x7f)
    if not b&0x80: lens.append(v); v=0
sod=p+2+plt_L; assert d[sod:sod+2]==b"\xff\x93"; body=sod+2
assert body+sum(lens)==sot+psot, (body+sum(lens), sot+psot)
ends=np.cumsum(lens)+body        # absolute end of packet for each resolution
print("packets (res0..3) end offsets:",ends.tolist())
def varint(n):
    o=[n&0x7f]; n>>=7
    while n: o.append(0x80|(n&0x7f)); n>>=7
    return bytes(reversed(o))
def header(r):
    levels=r; w=512>>(3-r); keep=lens[:r+1]
    h=bytearray(d[:segs[b"\xff\x55"][0]])           # SOC..before TLM
    def seg(m): pos_,L=segs[m]; return bytearray(d[pos_:pos_+2+L])
    siz=seg(b"\xff\x51"); 
    for off in (6,10,22,26): siz[off:off+4]=struct.pack(">I",w)   # Xsiz,Ysiz,XTsiz,YTsiz
    cod=seg(b"\xff\x52"); cod[2+2+1+4]=levels                      # SPcod levels
    qcd=seg(b"\xff\x5c"); n=1+3*r; qcd=qcd[:2+3+n]; qcd[2:4]=struct.pack(">H",3+n)
    plt_new=b"".join(varint(x) for x in keep)
    pltseg=b"\xff\x58"+struct.pack(">H",3+len(plt_new))+b"\x00"+plt_new
    pkt_bytes=int(ends[r]-body); psot_new=12+len(pltseg)+2+pkt_bytes
    tlm=seg(b"\xff\x55"); tlm[-4:]=struct.pack(">I",psot_new)
    sotb=bytearray(d[sot:sot+12]); sotb[6:10]=struct.pack(">I",psot_new)
    # main header order in original: SIZ COD QCD CAP COM TLM
    out=b"\xff\x4f"+bytes(siz)+bytes(cod)+bytes(qcd)+d[segs[b"\xff\x50"][0]:segs[b"\xff\x50"][0]+2+segs[b"\xff\x50"][1]]+d[segs[b"\xff\x64"][0]:segs[b"\xff\x64"][0]+2+segs[b"\xff\x64"][1]]+bytes(tlm)+bytes(sotb)+pltseg+b"\xff\x93"
    return out,pkt_bytes
src=Source(url=f.resolve().as_uri())
mapping={}
for r in range(4):
    hdr,nb=header(r)
    mapping[f"r{r}"]=Concat((Range(data=hdr),Range(source=0,offset=body,length=nb),Range(data=b"\xff\xd9")))
store=refkv.ReferenceStore(mapping,[src])
ref=[np.load(SP/f"hela2_crop/s{i}.npy") for i in range(4)]
for r in range(4):
    b=store.get(f"r{r}"); (SP/f"enc/virt_r{r}.j2c").write_bytes(b)
    out=[]
    for n,fn in (("openjph",ic.htj2k_decode),("openjpeg",ic.jpeg2k_decode)):
        try: out.append((n,fn(b).squeeze()))
        except Exception as e: out.append((n,repr(e)[:120]))
    sub=subprocess.run([NDIC,"expand","-i",str(f),"-o",str(SP/f"enc/_r{r}.raw"),"--resolution",str(r)],capture_output=True,text=True)
    w=512>>(3-r); nd=np.fromfile(SP/f"enc/_r{r}.raw","<u2").reshape(w,w)
    res=[]
    for n,a in out:
        res.append(f"{n}: "+(a if isinstance(a,str) else f"{a.shape} == ndic res{r} decode: {np.array_equal(a,nd)}"))
    print(f"r{r}: {len(b)} bytes of {len(d)} ({100*len(b)/len(d):.1f}%) ->",*res,sep="\n   ")
