"""Open a single-tile, single-layer, resolution-major JP2 (no PLT needed) as a virtual multiscale OME-Zarr 0.5 pyramid.

One refkv archive (.vzip) holds, per level sN (s0 = full), ONE chunk = the whole level, written as a Concat of rewritten
header bytes + the packet byte range of resolutions 0..L-N of the original .jp2 (no re-encoding, no copy). The packets
are located by parsing the packet headers (scripts/packets.py), because the file has no PLT/precincts to read them from.
Then verifies each level through zarr-python against `opj_decompress -r N`.

usage: pixi run python -P scripts/make_jp2_omezarr.py <file.jp2> <output .vzip> [--um 8.01] [--verify]
"""
import sys, json, pathlib, subprocess, tempfile, numpy as np, zarr, refkv, refkv.zarr as rz, imagecodecs
from refkv.pb import Source, Concat, Range
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, packets

src_file = pathlib.Path(sys.argv[1]).resolve(); OUT = pathlib.Path(sys.argv[2])
um = float(sys.argv[sys.argv.index("--um") + 1]) if "--um" in sys.argv else 1.0
d = src_file.read_bytes(); cs = d.index(b"\xff\x4f\xff\x51")                  # SOC+SIZ: start of the raw codestream (after the JP2 boxes)
j = j2kvirt.J2K(memoryview(d)[cs:]); (isot,) = j.tiles; ti = packets.TileIndex(j, isot)
assert ti.consumed, "packet parse did not consume the codestream"
L = j.L; mapping = {}; nread = {}


def shifted(c):
    """packets.assemble() offsets are relative to the codestream; refkv ranges address the whole file."""
    return Concat(tuple(p if p.data is not None else Range(source=p.source, offset=p.offset + cs, length=p.length) for p in c.parts))


for n in range(L + 1):
    c = shifted(packets.assemble(ti, 0, r=L - n)); mapping[f"s{n}/c/0/0"] = c
    nread[n] = sum(p.length for p in c.parts if p.data is None)
    h, w = (-(-v // 2 ** n) for v in (j.Y, j.X))
    mapping[f"s{n}/zarr.json"] = json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": [h, w], "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [h, w]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0,
        "codecs": [{"name": "imagecodecs_jpeg2k", "configuration": {}}], "dimension_names": ["y", "x"]}).encode()
ome = {"version": "0.5", "multiscales": [{"name": src_file.stem,
       "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "yx"],
       "datasets": [{"path": f"s{n}", "coordinateTransformations": [{"type": "scale", "scale": [um * 2 ** n] * 2}]} for n in range(L + 1)]}]}
mapping["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}).encode()
refkv.ReferenceStore(mapping, [Source(url=src_file.as_uri())]).save(str(OUT))
print(f"wrote {OUT} ({OUT.stat().st_size} B): {j.X} x {j.Y}, {L} decomposition levels, {len(d)} B source")
print("bytes read per level:", ", ".join(f"s{n}={100 * nread[n] / (len(d) - cs):.2f}%" for n in range(L + 1)))

if "--verify" in sys.argv:
    g = zarr.open_group(rz.open(str(OUT)), mode="r")
    from ome_zarr_models.v05 import Image
    Image.from_zarr(g); print("OME-Zarr 0.5 validation: OK")
    with tempfile.TemporaryDirectory() as t:
        for n in range(L + 1):
            a = g[f"s{n}"][:]
            if n == 0: ref = imagecodecs.jpeg2k_decode(d)
            else:
                o = pathlib.Path(t) / f"r{n}.tif"
                subprocess.run(["opj_decompress", "-i", str(src_file), "-o", str(o), "-r", str(n)], check=True, capture_output=True)
                ref = imagecodecs.tiff_decode(o.read_bytes())
            print(f"  s{n} {a.shape}: equals opj_decompress -r {n}: {np.array_equal(a, ref)}")
