"""Write the 16-plane jrc_hela-2 crop as a one-scale OME-Zarr 0.5 dataset with the nd-image-codecs `nd-lift-ht` series:
one chunk for the whole volume, a lossless 5/3 lift along z (nd_lift) then HTJ2K planes with a coefficient-plane index (htj2k).

Then checks that a fresh process opens it with plain zarr-python, that it is lossless, and that the metadata validates.
usage: pixi run python scripts/make_ndic_omezarr.py <workdir with hela2_crop/> <enc_ndic folder> <output .ome.zarr>
"""
import sys, json, shutil, pathlib, subprocess, numpy as np, zarr
from nd_image_codecs import codec_series

W, ENC, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
orig = np.load(W / "hela2_crop/s0.npy"); NZ, NY, NX = orig.shape
if OUT.exists(): shutil.rmtree(OUT)
series = codec_series(["z", "y", "x"], [NZ, NY, NX], "uint16", "nd-lift-ht", xy_levels=3, reversible=True)
filters = [c for c in series if c["name"] == "nd_lift"]; serializer = next(c for c in series if c["name"] == "htj2k")
ome = {"version": "0.5", "multiscales": [{
    "name": "jrc_hela-2 em/fibsem-uint16 crop (nd-image-codecs nd-lift-ht)",
    "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
    "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [5.24, 4.0, 4.0]}]}]}]}
g = zarr.create_group(str(OUT), zarr_format=3, attributes={"ome": ome})
a = g.create_array("0", shape=orig.shape, chunks=(NZ, NY, NX), dtype="uint16", filters=filters, serializer=serializer,
                   compressors=None, dimension_names=("z", "y", "x"))
a[:] = orig
files = sorted(f for f in (OUT / "0").rglob("*") if f.is_file() and f.name != "zarr.json")
size = sum(f.stat().st_size for f in files); per_plane = sum(p.stat().st_size for p in ENC.glob("p[0-9][0-9].j2c"))
print(f"wrote {OUT}\n  codecs: {[c['name'] for c in series]}; chunk files: {[str(f.relative_to(OUT)) for f in files]}, {size} B "
      f"(raw {orig.nbytes} B; the 16 independent per-plane ndic codestreams: {per_plane} B, {100 * size / per_plane:.1f}%)")
print("  array metadata codecs:", json.dumps(json.loads((OUT / "0/zarr.json").read_text())["codecs"]))

# 1. a fresh process that never imports nd_image_codecs must open it with plain zarr-python (codec found via entry points)
code = ("import sys, numpy as np, zarr; g = zarr.open_group(sys.argv[1], mode='r'); a = g['0'][:]; "
        "print('  fresh process:', a.shape, a.dtype, 'lossless:', np.array_equal(a, np.load(sys.argv[2])), "
        "'| nd_image_codecs loaded by zarr via its entry point:', 'nd_image_codecs' in sys.modules)")
subprocess.run([sys.executable, "-I", "-c", code, str(OUT), str(W / "hela2_crop/s0.npy")], check=True)
# 2. a single plane (the whole chunk is decoded: the codec's partial decode needs a store with range reads)
print("  plane 7 equals the original:", np.array_equal(zarr.open_group(str(OUT), mode="r")["0"][7], orig[7]))
# 3. OME-Zarr 0.5 metadata
from ome_zarr_models.v05 import Image
ms = Image.from_zarr(zarr.open_group(str(OUT), mode="r")).attributes.ome.multiscales[0]
print(f"  OME-Zarr 0.5 validation: OK, {len(ms.datasets)} dataset(s), axes {[x.name for x in ms.axes]}")
