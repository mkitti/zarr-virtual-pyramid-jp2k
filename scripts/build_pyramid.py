"""Step 1: all 16 planes -> refkv pyramid store (s0 = full resolution), for each encoder's files.

usage: pixi run python scripts/build_pyramid.py <workdir>
  <workdir>/enc/pNN.j2c       ndic files   (one tile-part + PLT)
  <workdir>/enc_ojph/pNN.j2c  OpenJPH files (tile-part per resolution, TLM)
"""
import sys, pathlib, numpy as np, refkv, refkv.zarr as rz, zarr
from refkv.pb import Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, util

W = pathlib.Path(sys.argv[1]); NZ = 16
orig = np.load(W / "hela2_crop/s0.npy")
arrays = {}
for enc, folder in (("ndic", "enc"), ("ojph", "enc_ojph")):
    sources, mapping, read = [], {}, {}
    full = 0
    for z in range(NZ):
        f = W / folder / f"p{z:02d}.j2c"; d = f.read_bytes(); full += len(d)
        sources.append(Source(url=f.resolve().as_uri())); j = j2kvirt.J2K(d)
        for n in range(j.L + 1):                                # n = Zarr level, s0 = full
            c, nread = j.chunk(z, j.L - n); mapping[f"s{n}/c/{z}/0/0"] = c; read[n] = read.get(n, 0) + nread
    L = j.L; mapping["zarr.json"] = util.ome_group_meta(L + 1, "jrc_hela-2 em/fibsem-uint16 crop (virtual HTJ2K pyramid)")
    for n in range(L + 1):
        w = 512 >> n; mapping[f"s{n}/zarr.json"] = util.array_meta((NZ, w, w), (1, w, w))
    out = W / f"hela2_pyramid_{enc}.vzip"; refkv.ReferenceStore(mapping, sources).save(str(out))
    g = zarr.open_group(rz.open(str(out)), mode="r")
    arrays[enc] = [g[f"s{n}"][:] for n in range(L + 1)]
    print(f"[{enc}] {out.name}: {len(mapping)} keys; source bytes read per level "
          + ", ".join(f"s{n}={100 * read[n] / full:.1f}%" for n in range(L + 1)))
    print(f"[{enc}] s0 equals the original u16 data: {np.array_equal(arrays[enc][0], orig)}")
    from ome_zarr_models.v05 import Image          # validates the multiscales metadata against the member arrays
    img = Image.from_zarr(g); ms = img.attributes.ome.multiscales[0]
    print(f"[{enc}] OME-Zarr 0.5 validation (ome-zarr-models {__import__('ome_zarr_models').__version__}): OK, "
          f"{len(ms.datasets)} datasets, axes {[a.name for a in ms.axes]}")
for n in range(1, L + 1):
    print(f"s{n} {arrays['ndic'][n].shape}: identical between the ndic- and OpenJPH-encoded pyramids "
          f"(all {NZ} planes): {np.array_equal(arrays['ndic'][n], arrays['ojph'][n])}")
