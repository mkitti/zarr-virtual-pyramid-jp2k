"""Measure what zarr-python actually fetches from the remote archives, with refkv.Stats, against the computed shares.

Opens the archives under https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/ and reads arrays; refkv.Stats
counts the requests it made and the bytes they returned (archive reads vs. reads of the codestreams). The computed
share is the sum of the source range lengths in the references, built from the local copy of the codestreams.
usage: pixi run python scripts/measure_io.py <upload/hela2 folder>
"""
import sys, pathlib, numpy as np, refkv, refkv.zarr as rz, zarr
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, packets

ROOT = pathlib.Path(sys.argv[1]); NZ = 16; L = 3; HF = {"HL", "LH", "HH"}
URL = "https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/"
def src_bytes(c): return sum(p.length for p in c.parts if not p.data)


def computed(folder, ext, spec):
    """(codestream bytes, {n: source bytes the references of level n point at}) from the local files."""
    tis = [packets.TileIndex(j2kvirt.J2K((ROOT / folder / f"p{z:02d}.{ext}").read_bytes()), 0) for z in range(NZ)]
    return sum(len(t.j.d) for t in tis), {n: sum(src_bytes(packets.assemble(t, 0, r=L - n, **spec(n))) for t in tis) for n in range(L + 1)}


def measure(vzip, path, full, comp, planes=None):
    st = refkv.Stats(); g = zarr.open_group(rz.open(URL + vzip, stats=st), mode="r"); open_bytes, open_req = st.archive_bytes, st.archive_requests
    arr = g[path]; st.reset(); a = arr[:] if planes is None else arr[planes]
    print(f"  {vzip[6:-5]:<14} {path:<8} {'16 planes' if planes is None else 'plane 0':<9} source {st.external_bytes:>9} B = {100 * st.external_bytes / full:5.1f}% "
          f"(computed {100 * comp / full:5.1f}%), {st.external_requests:>3} source requests; archive reads {st.archive_bytes} B in "
          f"{st.archive_requests} requests (opening: {open_bytes} B in {open_req})")


print("== pyramid (OpenJPH files, HTJ2K)")
full, c = computed("enc_ojph", "j2c", lambda n: {}); print(f"  codestreams total {full} B")
for n in (3, 2, 1, 0): measure("hela2_pyramid_ojph.vzip", f"s{n}", full, c[n])
measure("hela2_pyramid_ojph.vzip", "s0", full, c[0] // NZ, planes=0)
print("== quality groups (OpenJPEG files, 4 layers, classic J2K)")
full, c0 = computed("enc_opj", "j2k", lambda n: {}); print(f"  codestreams total {full} B")
for K in (1, 2, 3): measure("hela2_quality.vzip", f"q{K}/s0", full, computed("enc_opj", "j2k", lambda n, K=K: {"layers": K})[1][0])
for n in (1, 0): measure("hela2_quality.vzip", f"s{n}", full, c0[n])
print("== nohh group (OpenJPH files)")
full, cn = computed("enc_ojph", "j2c", lambda n: {"edit": (lambda r, px, py: HF - {"HH"} if r == L - n else None)} if n < L else {})
for n in (1, 0): measure("hela2_nohh.vzip", f"nohh/s{n}", full, cn[n])
