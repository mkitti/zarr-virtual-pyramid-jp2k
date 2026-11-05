"""Step 6: stores whose chunks apply a quality-layer or band selection, read through zarr-python.

Store A (OpenJPEG classic, 4 layers): root = full quality; groups q1, q2, q3 = first 1..3 layers; each is an OME pyramid.
Store B (OpenJPH, 1 layer):           root = full;         group nohh = HH of each level's top resolution removed.
usage: pixi run python scripts/step6_variants.py <workdir>
"""
import sys, pathlib, subprocess, numpy as np, refkv, refkv.zarr as rz, zarr
from refkv.pb import Source
from ome_zarr_models.v05 import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, packets, util, dwt53

W = pathlib.Path(sys.argv[1]); NZ = 16; L = 3; HF = {"HL", "LH", "HH"}
orig = np.load(W / "hela2_crop/s0.npy"); opj = W / "enc_opj"
for z in range(NZ):                                           # encode the OpenJPEG planes once
    f = opj / f"p{z:02d}.j2k"
    if not f.exists():
        util.write_pgm16(opj / "_in.pgm", orig[z])
        subprocess.run(["opj_compress", "-i", str(opj / "_in.pgm"), "-o", str(f), "-n", "4", "-b", "64,64", "-p", "RPCL",
                        "-r", "32,16,8,1", "-TP", "R"], check=True, capture_output=True)

def src_bytes(c): return sum(p.length for p in c.parts if not p.data)

def build(folder, ext, codec, variants, vzip):
    """variants: {group path ('' = root): kwargs for packets.assemble, or a function n -> kwargs}"""
    sources, mapping, cost, tis = [], {}, {}, []
    for z in range(NZ):
        f = W / folder / f"p{z:02d}.{ext}"; sources.append(Source(url=f.resolve().as_uri()))
        tis.append(packets.TileIndex(j2kvirt.J2K(f.read_bytes()), 0))
    full = sum(len(t.j.d) for t in tis)
    for group, spec in variants.items():
        pre = group + "/" if group else ""
        mapping[pre + "zarr.json"] = util.ome_group_meta(L + 1, f"jrc_hela-2 crop, variant '{group or 'full'}'")
        for n in range(L + 1):
            w = 512 >> n; mapping[f"{pre}s{n}/zarr.json"] = util.array_meta((NZ, w, w), (1, w, w), codec)
            for z in range(NZ):
                c = packets.assemble(tis[z], z, r=L - n, **spec(n)); mapping[f"{pre}s{n}/c/{z}/0/0"] = c
                cost[(group, n)] = cost.get((group, n), 0) + src_bytes(c)
    out = W / vzip; refkv.ReferenceStore(mapping, sources).save(str(out))
    return zarr.open_group(rz.open(str(out)), mode="r"), cost, full

def table(cost, full, groups):
    print(f"  source bytes read, % of all {NZ} codestreams:   " + "".join(f"{'s' + str(n):>8}" for n in range(L + 1)))
    for g in groups: print(f"    {g or 'full':<8}" + " " * 28 + "".join(f"{100 * cost[(g, n)] / full:>7.1f}%" for n in range(L + 1)))

print("== A. quality groups from 4-layer classic J2K (OpenJPEG)")
variants = {"": lambda n: {}, **{f"q{K}": (lambda n, K=K: {"layers": K}) for K in (1, 2, 3)}}
g, cost, full = build("enc_opj", "j2k", "imagecodecs_jpeg2k", variants, "hela2_quality.vzip")
table(cost, full, variants)
print("  OME validation:", {k or "full": len(Image.from_zarr(g[k] if k else g).attributes.ome.multiscales[0].datasets) for k in variants}, "datasets each")
print("  full s0 equals the original data (all planes):", np.array_equal(g["s0"][:], orig))
def ref(z, K, n):
    o = opj / "_ref.pgm"; subprocess.run(["opj_decompress", "-i", str(opj / f"p{z:02d}.j2k"), "-o", str(o), "-r", str(n), "-l", str(K)], check=True, capture_output=True)
    return util.read_pgm16(o)
for K in (1, 2, 3):
    arrs = [g[f"q{K}/s{n}"][:] for n in range(L + 1)]
    ok = all(np.array_equal(arrs[n][z], ref(z, K, n)) for n in range(L + 1) for z in (0, 5, 10, 15))
    psnr = 10 * np.log10(65535 ** 2 / np.mean((arrs[0].astype(float) - orig) ** 2))
    print(f"  q{K}: s0 PSNR {psnr:.1f} dB, all four levels equal opj_decompress -r n -l {K} (planes 0, 5, 10, 15): {ok}")

print("== B. 'nohh' group from single-layer HTJ2K (OpenJPH): HH of each level's top resolution removed")
def nohh(n): return {"edit": (lambda r, px, py: HF - {"HH"} if r == L - n else None)} if n < L else {}
variants = {"": lambda n: {}, "nohh": nohh}
g, cost, full = build("enc_ojph", "j2c", "imagecodecs_htj2k", variants, "hela2_nohh.vzip")
table(cost, full, variants)
print("  OME validation:", {k or "full": len(Image.from_zarr(g[k] if k else g).attributes.ome.multiscales[0].datasets) for k in variants}, "datasets each")
ok = {n: True for n in range(L + 1)}
arrs = [g[f"nohh/s{n}"][:] for n in range(L + 1)]
for z in range(NZ):
    ll, det = dwt53.pyramid(orig[z], L)
    for n in range(L + 1):
        exp = dwt53.recon(ll, det, zero=(L - n, "HH") if n < L else None, level=n)
        ok[n] &= np.array_equal(arrs[n][z], exp)
print("  full s0 equals the original data:", np.array_equal(g["s0"][:], orig))
print("  nohh levels equal the numpy 5/3 reference (all 16 planes):", {f"s{n}": v for n, v in ok.items()})
print(f"  nohh s0 mean |diff| vs original: {np.abs(arrs[0].astype(int) - orig).mean():.1f} gray levels (16-bit data)")
