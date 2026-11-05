import sys, struct, pathlib, json, subprocess, numpy as np
import refkv, refkv.zarr as rz, zarr
from refkv.pb import Concat, Range, Source
SP=pathlib.Path(sys.argv[1]); NDIC=sys.argv[2]; LEV=3; NZ=16; W0=512

def varint(n):
    o=[n&0x7f]; n>>=7
    while n: o.append(0x80|(n&0x7f)); n>>=7
    return bytes(reversed(o))

def plane_concats(d, src_idx):
    """Per resolution r (0..LEV): Concat(literal rewritten header, source range of packets 0..r, literal EOC)."""
    pos=2; segs={}
    while d[pos:pos+2]!=b"\xff\x90":
        L=struct.unpack(">H",d[pos+2:pos+4])[0]; segs[d[pos:pos+2]]=(pos,L); pos+=2+L
    sot=pos; psot=struct.unpack(">I",d[sot+6:sot+10])[0]
    p=sot+12; assert d[p:p+2]==b"\xff\x58"; Lp=struct.unpack(">H",d[p+2:p+4])[0]
    lens=[];v=0
    for b in d[p+5:p+2+Lp]:
        v=(v<<7)|(b&0x7f)
        if not b&0x80: lens.append(v); v=0
    sod=p+2+Lp; assert d[sod:sod+2]==b"\xff\x93"; body=sod+2
    assert body+sum(lens)==sot+psot and len(lens)==LEV+1
    ends=np.cumsum(lens)+body
    seg=lambda m: bytearray(d[segs[m][0]:segs[m][0]+2+segs[m][1]])
    out={}
    for r in range(LEV+1):
        w=W0>>(LEV-r)
        siz=seg(b"\xff\x51")
        for off in (6,10,22,26): siz[off:off+4]=struct.pack(">I",w)
        cod=seg(b"\xff\x52"); cod[9]=r
        n=1+3*r; qcd=seg(b"\xff\x5c")[:5+n]; qcd[2:4]=struct.pack(">H",3+n)
        pn=b"".join(varint(x) for x in lens[:r+1])
        pltseg=b"\xff\x58"+struct.pack(">H",3+len(pn))+b"\x00"+pn
        nb=int(ends[r]-body); psot_new=12+len(pltseg)+2+nb
        tlm=seg(b"\xff\x55"); tlm[-4:]=struct.pack(">I",psot_new)
        sotb=bytearray(d[sot:sot+12]); sotb[6:10]=struct.pack(">I",psot_new)
        hdr=b"\xff\x4f"+bytes(siz)+bytes(cod)+bytes(qcd)+bytes(seg(b"\xff\x50"))+bytes(seg(b"\xff\x64"))+bytes(tlm)+bytes(sotb)+pltseg+b"\xff\x93"
        out[r]=(Concat((Range(data=hdr),Range(source=src_idx,offset=body,length=nb),Range(data=b"\xff\xd9"))), len(hdr)+nb+2, nb)
    return out

sources=[]; mapping={}; fetch={r:0 for r in range(LEV+1)}; full=0
for z in range(NZ):
    f=SP/f"enc/p{z:02d}.j2c"; sources.append(Source(url=f.resolve().as_uri()))
    cc=plane_concats(f.read_bytes(), z); full+=f.stat().st_size
    for r,(c,tot,nb) in cc.items():
        lvl=LEV-r; mapping[f"s{lvl}/c/{z}/0/0"]=c; fetch[r]+=tot
def arr_meta(w): return json.dumps({"zarr_format":3,"node_type":"array","shape":[NZ,w,w],"data_type":"uint16",
  "chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1,w,w]}},
  "chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},"fill_value":0,
  "codecs":[{"name":"imagecodecs_htj2k","configuration":{}}],"dimension_names":["z","y","x"]}).encode()
mapping["zarr.json"]=json.dumps({"zarr_format":3,"node_type":"group","attributes":{}}).encode()
for r in range(LEV+1): mapping[f"s{LEV-r}/zarr.json"]=arr_meta(W0>>(LEV-r))
store=refkv.ReferenceStore(mapping,sources)
store.save(str(SP/"hela2_pyramid.vzip"))
print("saved hela2_pyramid.vzip; keys:",len(mapping),"sources:",len(sources))

# --- read through zarr-python, from the saved archive ---
g=zarr.open_group(rz.open(str(SP/"hela2_pyramid.vzip")),mode="r")
print({k:(v.shape,v.chunks,v.dtype) for k,v in g.members()})
s0=np.load(SP/"hela2_crop/s0.npy")
assert np.array_equal(g["s0"][:],s0); print("s0 (full) == original u16 data for all 16 planes")
ok=True
for r in range(LEV):
    lvl=LEV-r; w=W0>>lvl; a=g[f"s{lvl}"][:]
    for z in range(NZ):
        pth=SP/f"enc/p{z:02d}.j2c"; raw=SP/f"enc/_zz.raw"
        subprocess.run([NDIC,"expand","-i",str(pth),"-o",str(raw),"--resolution",str(r)],check=True,capture_output=True)
        if not np.array_equal(np.fromfile(raw,"<u2").reshape(w,w),a[z]): ok=False; print("MISMATCH",lvl,z)
    print(f"s{lvl} {a.shape}: matches ndic --resolution {r} for all {NZ} planes: {ok}")
print("bytes touched per level (all 16 planes):",{f"s{LEV-r}":f"{v/full*100:.1f}% of {full}" for r,v in fetch.items()})
