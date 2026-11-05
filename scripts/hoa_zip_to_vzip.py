"""Reference the JPEG 2000 slices of a Human Organ Atlas zip as a Zarr v3 / OME-Zarr 0.5 array, without copying pixels (one scale).

The zip stores every `.jp2` uncompressed, so each Zarr chunk (one slice, shape (1, H, W), codec `imagecodecs_jpeg2k`) is
one byte range of the zip. Only the zip's central directory and each entry's local header are read.

usage: pixi run python scripts/hoa_zip_to_vzip.py <zip URL> <out.vzip> [--voxel-um 100.32]
"""
import sys, json, struct, pathlib
import refkv
from refkv.pb import Range, Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hoazip, util

URL, OUT = sys.argv[1], sys.argv[2]
VOXEL_UM = float(sys.argv[sys.argv.index("--voxel-um") + 1]) if "--voxel-um" in sys.argv else 100.32
SIZE, LM, entries = hoazip.list_slices(URL)
print(f"{len(entries)} entries, all starting with a JP2 signature at their computed data offsets")
ihdr = hoazip.make_fetch(URL)(entries[0][1], 64); p = ihdr.find(b"ihdr"); h, w, nc = struct.unpack(">IIH", ihdr[p + 4:p + 14]); bpc = ihdr[p + 14] + 1
assert nc == 1 and bpc == 16, (nc, bpc)
print(f"slices: {len(entries)} x {h} x {w}, 1 component, {bpc}-bit")
pins = {"size": SIZE, **({"modified_not_after": LM} if LM else {})}
mapping = {f"s0/c/{z}/0/0": Range(source=0, offset=off, length=n) for z, (_, off, n) in enumerate(entries)}
mapping["s0/zarr.json"] = util.array_meta((len(entries), h, w), (1, h, w), "imagecodecs_jpeg2k")
ome = {"version": "0.5", "multiscales": [{
    "name": "LADAF-2021-17 heart, 100.32 um (HiP-CT), virtual JPEG 2000 slices",
    "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
    "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [VOXEL_UM] * 3}]}]}]}
mapping["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome, **util.HOA_ATTRIBUTION}}).encode()
refkv.ReferenceStore(mapping, [Source(url=URL, **pins)]).save(OUT)
print(f"saved {OUT}: {len(mapping)} keys, source pinned {pins}")
