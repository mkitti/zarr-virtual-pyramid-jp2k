"""Virtual isotropic pyramid of a Human Organ Atlas zip of JPEG 2000 slices, as a Zarr v3 / OME-Zarr 0.5 archive (refkv), no pixels copied.

Level sN keeps every 2^N-th slice (nearest, no filtering) at the resolution reduced N times (the LL band: the codestream's packets of
resolutions 0..L-N behind a rewritten header, see j2kvirt/packets). s0 references each .jp2 file whole. Only each even slice's first
~64 KiB is read to find the packet boundaries.

usage: pixi run python scripts/hoa_pyramid_to_vzip.py <zip URL> <out.vzip> [--voxel-um 100.32] [--verify] [--work DIR]
"""
import sys, json, struct, pathlib, subprocess, concurrent.futures as cf, numpy as np
import refkv, refkv.zarr as rz, zarr
from refkv.pb import Range, Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hoazip, j2kvirt, packets, util

URL, OUT = sys.argv[1], sys.argv[2]
arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
VOXEL_UM, VERIFY, WORK = float(arg("--voxel-um", 100.32)), "--verify" in sys.argv, pathlib.Path(arg("--work", "work/hoa/pyramid"))
SIZE, LM, entries = hoazip.list_slices(URL); Z = len(entries); fetch = hoazip.make_fetch(URL)
print(f"{Z} slices in the zip")


def prepare(z):
    """Packet index of slice z up to its second-to-last resolution, from a prefix of the file (retrying with more bytes if the headers need them)."""
    _, off, n = entries[z]
    for want in (1 << 16, 1 << 17, n):
        head = fetch(off, min(want, n)); cs_off = hoazip.codestream_offset(head); cs = head[cs_off:]
        d = bytearray(cs) + bytes(n - cs_off - len(cs)); d[-2:] = b"\xff\xd9"      # pad: only the header bytes in the prefix are read
        try:
            j = j2kvirt.J2K(bytes(d)); ti = packets.TileIndex(j, 0, max_res=j.L - 1)
        except (AssertionError, IndexError, struct.error):
            continue
        last = ti.packets[-1]
        if ti.file_off(last["off"]) + last["hdr"] <= len(cs): return z, ti, cs_off
    raise RuntimeError(f"cannot index slice {z}")


with cf.ThreadPoolExecutor(16) as ex: prepared = {z: (ti, cs_off) for z, ti, cs_off in ex.map(prepare, range(0, Z, 2))}
j0 = prepared[0][0].j; L, W0, H0 = j0.L, j0.X, j0.Y
assert all(t.j.L == L and t.j.X == W0 and t.j.Y == H0 and t.layers == 1 for t, _ in prepared.values()), "slices differ in levels, size or layers"
print(f"indexed {len(prepared)} slices ({H0} x {W0}, {L} decomposition levels)")
ceil_div = lambda a, n: -(-a >> n)
mapping, shapes = {}, {}
for z, (_, off, n) in enumerate(entries): mapping[f"s0/c/{z}/0/0"] = Range(source=0, offset=off, length=n)
for n in range(1, L + 1):
    for zp in range(ceil_div(Z, n)):
        z = zp << n; ti, cs_off = prepared[z]
        mapping[f"s{n}/c/{zp}/0/0"] = packets.assemble(ti, 0, r=L - n, src_offset=entries[z][1] + cs_off)
for n in range(L + 1):
    shapes[n] = (ceil_div(Z, n), ceil_div(H0, n), ceil_div(W0, n)); mapping[f"s{n}/zarr.json"] = util.array_meta(shapes[n], (1, shapes[n][1], shapes[n][2]), "imagecodecs_jpeg2k")
ome = {"version": "0.5", "multiscales": [{
    "name": "LADAF-2021-17 heart, 100.32 um (HiP-CT), virtual JPEG 2000 pyramid",
    "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
    "datasets": [{"path": f"s{n}", "coordinateTransformations": [{"type": "scale", "scale": [VOXEL_UM * 2 ** n] * 3}]} for n in range(L + 1)]}]}
mapping["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome, **util.HOA_ATTRIBUTION}}).encode()
pins = {"size": SIZE, **({"modified_not_after": LM} if LM else {})}
refkv.ReferenceStore(mapping, [Source(url=URL, **pins)]).save(OUT)
print(f"saved {OUT} ({pathlib.Path(OUT).stat().st_size} B): {len(mapping)} keys; levels " + ", ".join(f"s{n} {shapes[n]}" for n in range(L + 1)))

if VERIFY:
    WORK.mkdir(parents=True, exist_ok=True); stats = refkv.Stats(); g = zarr.open_group(rz.open(OUT, stats=stats), mode="r")
    from ome_zarr_models.v05 import Image
    print("OME-Zarr validation: OK,", len(Image.from_zarr(g).attributes.ome.multiscales[0].datasets), "datasets")
    print("level shape                 slices checked        bytes/slice fetched   equals opj_decompress -r N")
    for n in range(L + 1):
        arr = g[f"s{n}"]; nz = shapes[n][0]; picks = sorted({0, nz // 2, nz - 1}); ok = []; fetched = []
        for zp in picks:
            z = zp << n; _, off, size = entries[z]; (WORK / "slice.jp2").write_bytes(fetch(off, size))
            cmd = ["opj_decompress", "-i", str(WORK / "slice.jp2"), "-o", str(WORK / "ref.pgm")] + (["-r", str(n)] if n else [])
            subprocess.run(cmd, check=True, capture_output=True); ref = util.read_pgm16(WORK / "ref.pgm")
            stats.reset(); got = arr[zp]; fetched.append(stats.external_bytes); ok.append(bool(np.array_equal(got, ref)) and got.shape == ref.shape)
        print(f"  s{n}  {str(shapes[n]):<20} z' {picks!s:<22} {int(np.mean(fetched)):>9} B           {all(ok)} ({sum(ok)}/{len(ok)})")
