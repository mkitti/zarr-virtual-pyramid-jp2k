"""Verify an archive made by ndic_pyramid_to_vzip.py on sample chunks (read through zarr-python over HTTP).

For a chunk (iz, 0, ix): (a) the referenced s1 chunk equals the original array's; (b) s3 and s4 (no inverse lifting) equal the exact decode of their
reduced planes by an independent decoder (OpenJPH via imagecodecs); (c) s2 stays within a few gray levels of the xy low-pass of the z low-pass
(lifting level 1) of the decoded s1 chunk, which the integer lifting steps do not commute with exactly; (d) all levels share the chunk's mean.
usage: pixi run python scripts/ndic_pyramid_verify.py <archive.vzip> <input .ome.zarr URL> [iz,ix ...]
"""
import sys, json, struct, shutil, pathlib, numpy as np, zarr, refkv, refkv.zarr as rz, imagecodecs as ic
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import dwt53
from ome_zarr_models.v05 import Image

ARCH, BASE = sys.argv[1], sys.argv[2].rstrip("/")
picks = [tuple(map(int, a.split(","))) for a in sys.argv[3:]] or [(0, 0), (97, 3), (193, 6)]
stats = refkv.Stats(); arch = zarr.open_group(rz.open(ARCH, stats=stats), mode="r"); orig = zarr.open_group(BASE, mode="r")["s1"]
print("OME-Zarr validation:", [d.path for d in Image.from_zarr(arch).attributes.ome.multiscales[0].datasets]); store = refkv.open(ARCH)
CZ, CY, CX = orig.chunks; Z, Y, X = orig.shape; tmp = pathlib.Path("work/h89/verify_lift.zarr")
cfg = lambda lv: {"version": "0.1", "transforms": [{"axis": "z", "dimension": 0, "kind": "lift53", "levels": lv, "group": 0}]}


def lifted(vol, lv):
    shutil.rmtree(tmp, ignore_errors=True)
    a = zarr.create_array(str(tmp), shape=vol.shape, chunks=vol.shape, dtype="uint16", filters=[{"name": "nd_lift", "configuration": cfg(lv)}],
                          serializer={"name": "bytes", "configuration": {"endian": "little"}}, compressors=None, fill_value=0, zarr_format=3)
    a[:] = vol; return np.frombuffer((tmp / "c/0/0/0").read_bytes(), "<i4").reshape(vol.shape)


for iz, ix in picks:
    print(f"== chunk z{iz} x{ix}")
    z0, x0 = iz * CZ, ix * CX; V = orig[z0:z0 + CZ, :, x0:x0 + CX]
    # (a)
    S1 = arch["s1"][z0:z0 + CZ, :, x0:x0 + CX]; print(f"  s1 {S1.shape} via the archive equals the original array: {np.array_equal(S1, V)}")
    Vp = np.zeros((CZ, CY, CX), np.uint16); Vp[:V.shape[0], :, :V.shape[2]] = V; mean1 = Vp[:V.shape[0], :, :V.shape[2]].mean()
    for n, name in ((2, "s2"), (3, "s3"), (4, "s4")):
        a = arch[name]; zc, cy, cx = a.chunks
        try: out = a[iz * zc:(iz + 1) * zc, :, ix * cx:(ix + 1) * cx]
        except Exception as e: print(f"  {name}: DECODE FAILED ({type(e).__name__}: {str(e)[:90]})"); continue
        key = f"{name}/c/{iz}/0/{ix}"; blob = bytes(store.get(key)); dims = struct.unpack("<III", blob[8:20]); r = blob[6]; stride = 12 + 4 * (r + 1)
        ent = [struct.unpack("<QI", blob[20 + i * stride:32 + i * stride]) for i in range(dims[0])]
        planes = np.stack([np.asarray(ic.htj2k_decode(blob[o:o + l])).astype(np.int64) for o, l in ent])     # OpenJPH
        msg = f"  {name} {out.shape} (chunk {a.chunks}): container {len(blob)} B, {dims[0]} planes of {dims[1]}x{dims[2]} at r{r}"
        zz, xx = out.shape[0], out.shape[2]
        if name == "s2":
            ref = np.stack([dwt53.fwd2(c.astype(np.int64))[0] for c in lifted(Vp, 1)[:8]])[:zz, :, :xx]; d = np.abs(out.astype(int) - ref)
            msg += f"; vs xy-LL of z-lift approx1: max |diff| {d.max()}, mean {d.mean():.2f}"
        else:
            msg += f"; equals OpenJPH's decode of the reduced planes: {np.array_equal(out, planes[:zz, :, :xx])}"
        print(msg + f"; mean {out.mean():.1f} vs s1 chunk mean {mean1:.1f}")
print("source bytes fetched while verifying:", stats.external_bytes)
