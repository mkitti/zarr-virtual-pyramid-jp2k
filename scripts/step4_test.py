"""Step 4: packet-header parsing -> per-subband / per-precinct / per-code-block byte ranges, and their use.

A. parse every packet of all test files; check exact consumption, PLT lengths, HT Scup validity
B. per-subband byte table (+ JSON code-block index)
C. precinct-level ROI: drop all finest-resolution precincts but one by replacing them with 1-byte empty packets
D. sub-band drops by rewriting packet headers, checked against an independent numpy 5/3 wavelet
usage: pixi run python scripts/step4_test.py <workdir>
"""
import sys, json, pathlib, collections, numpy as np, refkv
from refkv.pb import Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, packets, util

W = pathlib.Path(sys.argv[1]); orig = np.load(W / "hela2_crop/s0.npy"); NZ = 16
FILES = {"ndic (1 tile-part + PLT)": "enc", "OpenJPH (tile-part/res, TLM)": "enc_ojph",
         "OpenJPH 256x256 tiles": "enc_tiles", "OpenJPH 256x256 tiles + precincts": "enc_tiles_precincts"}
def nbytes(c): return sum(len(p.data) if p.data else p.length for p in c.parts)
def store_get(c, f): return refkv.ReferenceStore({"k": c}, [Source(url=f.resolve().as_uri())]).get("k")

print("== A. parse all packets")
for label, folder in FILES.items():
    npk = ncb = bad = pltbad = notdone = 0; hist = collections.Counter()
    for z in range(NZ):
        j = j2kvirt.J2K((W / folder / f"p{z:02d}.j2c").read_bytes())
        for isot in j.tiles:
            ti = packets.TileIndex(j, isot); notdone += not ti.consumed; npk += len(ti.packets)
            n, b = ti.check_ht(); ncb += n; bad += b
            for pkt in ti.packets:
                for ob in pkt["bands"]:
                    hist.update(cb["passes"] for cb in ob["cbs"] if cb)
            for k, tp in enumerate(j.tiles[isot]):
                lens = j2kvirt.plt_lengths(tp["segs"])
                if lens: pltbad += lens != [p["hdr"] + p["body"] for p in ti.packets if p["tp"] == k]
    print(f"  {label}: {npk} packets, {ncb} code-blocks; tile not consumed exactly: {notdone}; "
          f"bad Scup: {bad}; PLT mismatches: {pltbad}; coding passes per block {dict(hist)}")

print("== B. per-subband bytes, plane 0, OpenJPH file (single tile, no precincts)")
f = W / "enc_ojph/p00.j2c"; j = j2kvirt.J2K(f.read_bytes()); ti = packets.TileIndex(j, 0); rows = []
print(f"  {'res':>3} {'band':>4} {'blocks':>6} {'bytes':>8} {'% of file':>9}   (packet header bytes in last column)")
for pkt in ti.packets:
    for ob in pkt["bands"]:
        off, ln = ti.band_range(pkt, ob["name"]); n = sum(c is not None for c in ob["cbs"])
        print(f"  {pkt['r']:>3} {ob['name']:>4} {n:>6} {ln:>8} {100 * ln / len(j.d):>8.2f}%" + (f"   header {pkt['hdr']}" if ob["name"] in ("LL", "HL") else ""))
        for i, cb in enumerate(ob["cbs"]):
            if cb: rows.append(dict(res=pkt["r"], precinct=[pkt["px"], pkt["py"]], band=ob["name"], cb=[ob["geo"]["cx0"] + i % ob["ncbw"], ob["geo"]["cy0"] + i // ob["ncbw"]],
                                    passes=cb["passes"], zero_bitplanes=cb["zbp"], segment_lengths=cb["seglens"],
                                    file_offset=ti.file_off(pkt["off"] + pkt["hdr"] + cb["rel"]), length=cb["len"]))
(W / "codeblock_index_p00.json").write_text(json.dumps(rows)); print(f"  wrote {len(rows)} code-block rows to codeblock_index_p00.json")

print("== C. precinct ROI (tile 0 of the tiles+precincts file, finest resolution keeps only precinct (1,0) = x128..255, y0..127)")
f = W / "enc_tiles_precincts/p00.j2c"; j = j2kvirt.J2K(f.read_bytes()); ti = packets.TileIndex(j, 0)
c = packets.assemble(ti, 0, lambda p: "empty" if p["r"] == 3 and (p["px"], p["py"]) != (1, 0) else None)
b = store_get(c, f); res = util.decode_both(b); ref = orig[0][:256, :256]
tile_bytes = sum(tp["end"] - tp["body"] for tp in j.tiles[0])
print(f"  chunk = {len(b)} B of which {nbytes(c) - sum(len(p.data) for p in c.parts if p.data)} from the source; tile has {tile_bytes} B")
for name, a in res.items():
    if isinstance(a, str): print("  ", name, a); continue
    print(f"  {name}: ROI interior x132..255,y0..123 identical to original: {np.array_equal(a[0:124, 132:256], ref[0:124, 132:256])}; "
          f"dropped-precinct area x10..120,y140..250 differs (mean |diff| {np.abs(a[140:250, 10:120].astype(int) - ref[140:250, 10:120]).mean():.1f})")

print("== D. sub-band drops via rewritten packet headers (OpenJPH plane 0), vs numpy 5/3 reference")
def f1(x):
    xe, xo = x[..., 0::2], x[..., 1::2]; xen = np.concatenate([xe[..., 1:], xe[..., -1:]], -1)
    H = xo - ((xe + xen) >> 1); Hp = np.concatenate([H[..., :1], H[..., :-1]], -1); return xe + ((Hp + H + 2) >> 2), H
def i1(L, H):
    Hp = np.concatenate([H[..., :1], H[..., :-1]], -1); xe = L - ((Hp + H + 2) >> 2)
    xen = np.concatenate([xe[..., 1:], xe[..., -1:]], -1); out = np.empty(L.shape[:-1] + (2 * L.shape[-1],), np.int64)
    out[..., 0::2] = xe; out[..., 1::2] = H + ((xe + xen) >> 1); return out
def fwd2(a):
    Lv, Hv = f1(a.T); Lv, Hv = Lv.T, Hv.T; LL, HL = f1(Lv); LH, HH = f1(Hv); return LL, dict(HL=HL, LH=LH, HH=HH)
def inv2(LL, d): return i1(i1(LL, d["HL"]).T, i1(d["LH"], d["HH"]).T).T
def pyramid(img):
    ll, det = img.astype(np.int64) - 32768, []
    for _ in range(3): ll, d = fwd2(ll); det.append(d)
    return ll, det                                  # det[0] = level 1 = resolution 3
def recon(ll, det, zero=None):
    for lvl in (2, 1, 0):
        d = {k: (np.zeros_like(v) if zero == (3 - lvl, k) else v) for k, v in det[lvl].items()}; ll = inv2(ll, d)
    return np.clip(ll + 32768, 0, 65535).astype(np.uint16)
img = orig[0]; ll, det = pyramid(img)
print("  numpy 5/3 reference round-trips exactly:", np.array_equal(recon(ll, det), img),
      "| its LL1 equals OpenJPH's reduced decode (s1):", np.array_equal(np.clip(fwd2(img.astype(np.int64) - 32768)[0] + 32768, 0, 65535).astype(np.uint16), util.ojph_expand(W / "enc_ojph/p00.j2c", 1, W)))
f = W / "enc_ojph/p00.j2c"; j = j2kvirt.J2K(f.read_bytes()); ti = packets.TileIndex(j, 0); full = len(j.d)
same = sum(packets.write_header(p, {"LL", "HL", "LH", "HH"})[0] == ti.buf[p["off"]:p["off"] + p["hdr"]] for p in ti.packets)
print(f"  identity rewrite: {same}/{len(ti.packets)} packet headers byte-identical to the encoder's")
a = util.decode_both(store_get(packets.assemble(ti, 0, lambda p: {"LL", "HL", "LH", "HH"}), f))
print("  identity rewrite decodes to the original:", {k: np.array_equal(v, img) for k, v in a.items()})
for res_, band in [(3, "HH"), (3, "HL"), (3, "LH"), (2, "HH"), (1, "HL")]:
    c = packets.assemble(ti, 0, lambda p: ({"HL", "LH", "HH"} - {band}) if p["r"] == res_ else None)
    exp = recon(ll, det, zero=(res_, band)); out = util.decode_both(store_get(c, f))
    ok = {k: (not isinstance(v, str)) and np.array_equal(v, exp) for k, v in out.items()}
    drop = sum(len(x.data) if x.data else x.length for x in packets.assemble(ti, 0).parts) - nbytes(c)
    print(f"  drop {band} of resolution {res_}: file shrinks by {drop} B ({100 * drop / full:.1f}%); matches numpy reference: {ok}; differs from original: {not np.array_equal(exp, img)}")
