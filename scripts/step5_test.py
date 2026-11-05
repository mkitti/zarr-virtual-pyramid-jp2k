"""Step 5: (1) resolution + band/precinct edits combined, (2) multi-layer / multi-pass classic J2K from OpenJPEG.

References: a numpy 5/3 wavelet (dwt53.py) for band/precinct edits, and opj_decompress -l/-r for layer/resolution cuts.
usage: pixi run python scripts/step5_test.py <workdir>
"""
import sys, pathlib, subprocess, collections, numpy as np, refkv
from refkv.pb import Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, packets, util, dwt53

W = pathlib.Path(sys.argv[1]); orig = np.load(W / "hela2_crop/s0.npy")
def nbytes(c): return sum(len(p.data) if p.data else p.length for p in c.parts)
def get(c, f): return refkv.ReferenceStore({"k": c}, [Source(url=f.resolve().as_uri())]).get("k")
def verdict(b, exp, who=("openjph", "openjpeg")):
    out = util.decode_both(b); return {k: (not isinstance(v, str)) and v.shape == exp.shape and bool(np.array_equal(v, exp))
                                       for k, v in out.items() if k in who}
HF = {"HL", "LH", "HH"}

print("== 1a. resolution level + band drop (OpenJPH plane 0, whole image)")
f = W / "enc_ojph/p00.j2c"; j = j2kvirt.J2K(f.read_bytes()); ti = packets.TileIndex(j, 0); L = j.L
ll, det = dwt53.pyramid(orig[0], L); full = len(j.d)
for r, (dres, band) in [(2, (2, "HH")), (2, (1, "HL")), (1, (1, "LH"))]:
    c = packets.assemble(ti, 0, lambda rr, px, py: (HF - {band}) if rr == dres else None, r=r)
    exp = dwt53.recon(ll, det, zero=(dres, band), level=L - r); b = get(c, f)
    print(f"  s{L - r} {exp.shape}, drop {band} of res {dres}: {len(b)} B ({100 * len(b) / full:.1f}% of file) matches numpy reference: {verdict(b, exp)}")

print("== 1b. same for standalone tiles (OpenJPH 256x256 tiles, plane 0)")
f = W / "enc_tiles/p00.j2c"; j = j2kvirt.J2K(f.read_bytes())
for t, r, dres, band in [(0, 3, 3, "HL"), (0, 2, 2, "HH"), (3, 2, 1, "LH"), (2, 1, 1, "HL")]:
    ti = packets.TileIndex(j, t); tx, ty = t % 2, t // 2
    img = orig[0][ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256]; ll, det = dwt53.pyramid(img, L)
    c = packets.assemble(ti, 0, lambda rr, px, py: (HF - {band}) if rr == dres else None, r=r)
    exp = dwt53.recon(ll, det, zero=(dres, band), level=L - r); b = get(c, f)
    print(f"  tile {t} s{L - r} {exp.shape}, drop {band} of res {dres}: {len(b)} B; matches numpy reference: {verdict(b, exp)}")

print("== 1c. precinct x band geometry (tiles + precincts, tile 0): drop HH in 3 of the 4 finest precincts, keep precinct (1,0) whole")
f = W / "enc_tiles_precincts/p00.j2c"; j = j2kvirt.J2K(f.read_bytes()); ti = packets.TileIndex(j, 0)
img = orig[0][:256, :256]; ll, det = dwt53.pyramid(img, L)
def zero_hh_outside(res, band, a):
    if res != 3 or band != "HH": return a
    keep = np.zeros_like(a); keep[0:64, 64:128] = a[0:64, 64:128]; return keep        # precinct (1,0): x 64..127, y 0..63 of the HH band
c = packets.assemble(ti, 0, lambda rr, px, py: HF - {"HH"} if rr == 3 and (px, py) != (1, 0) else None)
b = get(c, f); print(f"  {len(b)} B of {sum(tp['end'] - tp['body'] for tp in j.tiles[0])}; matches numpy reference: {verdict(b, dwt53.recon(ll, det, zero=zero_hh_outside))}")

print("== 2. multi-layer, multi-pass classic J2K (OpenJPEG, 4 layers, RPCL, tile-part per resolution)")
out = W / "enc_opj"; out.mkdir(exist_ok=True); pgm = out / "p00.pgm"; util.write_pgm16(pgm, orig[0])
subprocess.run(["opj_compress", "-i", str(pgm), "-o", str(out / "p00.j2k"), "-n", "4", "-b", "64,64", "-p", "RPCL", "-r", "32,16,8,1", "-TP", "R"],
               check=True, capture_output=True)
f = out / "p00.j2k"; j = j2kvirt.J2K(f.read_bytes()); ti = packets.TileIndex(j, 0)
hist = collections.Counter(); per_layer = collections.Counter()
for p in ti.packets:
    per_layer[p["layer"]] += p["hdr"] + p["body"]
    for ob in p["bands"]: hist.update(cb["passes"] for cb in ob["cbs"] if cb)
print(f"  {j.layers} layers, {len(ti.packets)} packets, consumed exactly: {ti.consumed}; {sum(hist.values())} code-block contributions, "
      f"passes per contribution min {min(hist)} max {max(hist)}; bytes per layer {dict(sorted(per_layer.items()))} (file {len(j.d)} B)")
def opj_ref(k=None, red=0):
    o = out / "_ref.pgm"; cmd = ["opj_decompress", "-i", str(f), "-o", str(o), "-r", str(red)] + (["-l", str(k)] if k else [])
    subprocess.run(cmd, check=True, capture_output=True); return util.read_pgm16(o)
full_img = opj_ref(); print("  full decode is lossless:", np.array_equal(full_img, orig[0]))
ALL = {"LL", "HL", "LH", "HH"}
print("  identity header rewrite decodes to the original:", verdict(get(packets.assemble(ti, 0, lambda r, px, py: ALL), f), orig[0], ("openjpeg",)))
for K in (1, 2, 3):
    exp = opj_ref(K); b = get(packets.assemble(ti, 0, layers=K), f)
    psnr = 10 * np.log10(65535 ** 2 / max(np.mean((exp.astype(float) - orig[0]) ** 2), 1e-9))
    print(f"  first {K} layer(s): {len(b)} B ({100 * len(b) / len(j.d):.1f}%), PSNR {psnr:.1f} dB, equals opj_decompress -l {K}: {verdict(b, exp, ('openjpeg',))}")
for r, K in ((2, 2), (1, 3)):
    exp = opj_ref(K, L - r); b = get(packets.assemble(ti, 0, r=r, layers=K), f)
    print(f"  s{L - r} + first {K} layers: {len(b)} B, equals opj_decompress -r {L - r} -l {K}: {verdict(b, exp, ('openjpeg',))}")
ll, det = dwt53.pyramid(orig[0], L)
for dres, band in [(3, "HH"), (2, "HL"), (1, "LH")]:
    c = packets.assemble(ti, 0, lambda r, px, py: (HF - {band}) if r == dres else None); b = get(c, f)
    print(f"  drop {band} of res {dres} in all layers: {len(b)} B; matches numpy reference: {verdict(b, dwt53.recon(ll, det, zero=(dres, band)), ('openjpeg',))}")
exp = opj_ref(2, 1)
c = packets.assemble(ti, 0, lambda r, px, py: (HF - {'HH'}) if r == 2 else None, r=2, layers=2); b = get(c, f)
print(f"  s1 + 2 layers + HH of res 2 dropped: {len(b)} B; decodes: {not isinstance(util.decode_both(b)['openjpeg'], str)}; differs from layers-only reference: {not np.array_equal(util.decode_both(b)['openjpeg'], exp)}")

print("== 2b. OpenJPEG HT (-M 64) with layers: what do its code-blocks look like?")
try:
    subprocess.run(["opj_compress", "-i", str(pgm), "-o", str(out / "ht.j2k"), "-n", "4", "-b", "64,64", "-p", "RPCL", "-r", "32,16,8,1", "-TP", "R", "-M", "64"],
                   check=True, capture_output=True)
    jh = j2kvirt.J2K((out / "ht.j2k").read_bytes()); sty = dict(jh.main)[j2kvirt.COD][12]
    if sty != 0x40:
        print(f"  opj_compress -M 64 wrote a classic stream (code-block style {sty:#04x}): this OpenJPEG cannot encode HT, so no HT "
              "refinement-pass or multi-layer HT file is available from ndic, OpenJPH or OpenJPEG; that case stays untested")
    else:
        th = packets.TileIndex(jh, 0); h2 = collections.Counter()
        for p in th.packets:
            for ob in p["bands"]: h2.update(cb["passes"] for cb in ob["cbs"] if cb)
        print(f"  {jh.layers} layers, consumed exactly: {th.consumed}, passes per contribution {dict(h2)}, bad Scup {th.check_ht()[1]}")
except Exception as e: print("  not parsed:", repr(e)[:160])
