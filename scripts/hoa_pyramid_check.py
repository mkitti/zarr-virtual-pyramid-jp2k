"""Human Organ Atlas slices: (1) survey the codestream headers of every slice; (2) test resolution-prefix extraction on one slice.

(1) reads the first 600 bytes of each chunk of the refkv archive (HTTP range requests) and tabulates the coding parameters;
(2) rewrites one slice to each lower resolution (j2kvirt/packets, any layout) and compares OpenJPEG's decode of the result with
`opj_decompress -r n` of the original.
usage: pixi run python scripts/hoa_pyramid_check.py <archive.vzip> <slice.j2k bare codestream> <slice index in the archive> <workdir>
"""
import sys, struct, collections, pathlib, subprocess, concurrent.futures as cf, numpy as np, refkv, imagecodecs as ic
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, packets, util

ARCH, SLICE, IDX, WORK = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
u16 = lambda d, o: struct.unpack(">H", d[o:o + 2])[0]; u32 = lambda d, o: struct.unpack(">I", d[o:o + 4])[0]


def header_signature(head):
    pos = 0
    while head[pos + 4:pos + 8] != b"jp2c":
        pos += u32(head, pos)                                   # skip signature, ftyp, jp2h boxes
    pos += 8; assert head[pos:pos + 2] == b"\xff\x4f"; pos += 2; sig = {}; markers = []
    while head[pos:pos + 2] != b"\xff\x90":
        m = u16(head, pos); L = u16(head, pos + 2); body = head[pos + 4:pos + 2 + L]; markers.append(f"{m:04X}")
        if m == 0xFF51: sig.update(size=(u32(body, 2), u32(body, 6)), tile=(u32(body, 18), u32(body, 22)), comps=u16(body, 34), bits=(body[36] & 0x7f) + 1)
        if m == 0xFF52:
            sig.update(scod=body[0], prog=body[1], layers=u16(body, 2), mct=body[4], levels=body[5], cblk=((body[6] & 15) + 2, (body[7] & 15) + 2), style=body[8], wavelet=body[9])
        if m == 0xFF5C: sig.update(qstyle=body[0] & 31, guard=body[0] >> 5, qentries=(len(body) - 1) // (1 if body[0] & 31 == 0 else 2))
        pos += 2 + L
    sig["markers"] = tuple(markers); return tuple(sorted(sig.items()))


store = refkv.open(ARCH); keys = sorted((k for k in [f"s0/c/{z}/0/0" for z in range(1612)]), key=lambda k: int(k.split("/")[2]))
with cf.ThreadPoolExecutor(16) as ex: sigs = list(ex.map(lambda k: header_signature(bytes(store.get(k, refkv.RangeRequest(0, 600)))), keys))
count = collections.Counter(sigs)
print(f"== 1. coding parameters of all {len(keys)} slices: {len(count)} distinct signature(s)")
for s, n in count.most_common():
    d = dict(s); print(f"  {n} slices: image {d['size']}, tile {d['tile']}, {d['comps']} comp {d['bits']}-bit, {d['levels']} levels, {d['layers']} layer(s), "
                       f"progression {['LRCP','RLCP','RPCL','PCRL','CPRL'][d['prog']]}, precincts {'DEFINED' if d['scod'] & 1 else 'none (maximal)'}, SOP {bool(d['scod'] & 2)}, EPH {bool(d['scod'] & 4)}, "
                       f"code-blocks {2 ** d['cblk'][0]}x{2 ** d['cblk'][1]}, style {d['style']:#04x}, {'9/7 irreversible' if d['wavelet'] == 0 else '5/3 reversible'}, "
                       f"QCD style {d['qstyle']} ({d['qentries']} entries, {d['guard']} guard bits), main-header markers {d['markers']}")

print(f"== 2. resolution-prefix extraction on slice {IDX}")
cs = SLICE.read_bytes(); j = j2kvirt.J2K(cs); ti = packets.TileIndex(j, 0); L = j.L
print(f"  {len(ti.packets)} packets in {len(j.tiles[0])} tile-part(s), consumed exactly: {ti.consumed}; per resolution (header+body bytes, cumulative share of the file):")
cum = 0
for p in ti.packets:
    cum += p["hdr"] + p["body"]; print(f"    r{p['r']} ({-(-j.X >> (L - p['r']))}x{-(-j.Y >> (L - p['r']))}): {p['hdr'] + p['body']:>7} B, cumulative {100 * cum / len(cs):5.1f}%")
for n in range(0, L):                                            # Zarr level n = resolution L - n; OpenJPEG -r n drops n levels
    r = L - n; c = packets.assemble(ti, 0, r=r); b = b"".join(p.data if p.data else cs[p.offset:p.offset + p.length] for p in c.parts)
    mine = np.asarray(ic.jpeg2k_decode(b)); (WORK / "_in.j2k").write_bytes(cs)
    subprocess.run(["opj_decompress", "-i", str(WORK / "_in.j2k"), "-o", str(WORK / "_ref.pgm"), "-r", str(n)], check=True, capture_output=True)
    ref = util.read_pgm16(WORK / "_ref.pgm")
    print(f"  s{n} {mine.shape}: {len(b):>7} B ({100 * len(b) / len(cs):5.1f}% of the slice); equals opj_decompress -r {n}: {np.array_equal(mine, ref)}"
          + ("" if np.array_equal(mine, ref) else f" (max |diff| {np.abs(mine.astype(int) - ref).max()})"))
