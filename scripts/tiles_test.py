"""Step 3: tiled (and tiled + precincts) OpenJPH files -> one Zarr chunk per tile per level.

Each chunk is a standalone single-tile codestream rebuilt from that tile's tile-parts, so reading one
spatial chunk at one level touches only that tile's resolution prefix.
Reference: OpenJPH's own reduced decode (`ojph_expand -skip_res n`) of the full tiled file.
usage: pixi run python scripts/tiles_test.py <workdir>
"""
import sys, pathlib, numpy as np, refkv, refkv.zarr as rz, zarr
from refkv.pb import Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, util

W = pathlib.Path(sys.argv[1]); NZ = 16; TS = 256
orig = np.load(W / "hela2_crop/s0.npy")
variants = {"tiles": [], "tiles+precincts": ["-precincts", "{32,32},{64,64},{128,128},{128,128}"]}
for name, extra in variants.items():
    folder = W / ("enc_" + name.replace("+", "_")); folder.mkdir(exist_ok=True)
    sources, mapping, read, full = [], {}, {}, 0
    for z in range(NZ):
        f = folder / f"p{z:02d}.j2c"
        util.ojph_compress(orig[z], f, "-num_decomps", 3, "-reversible", "true", "-prog_order", "RPCL", "-tileparts", "R",
                           "-tlm_marker", "true", "-tile_size", f"{{{TS},{TS}}}", *extra, work=folder)
        d = f.read_bytes(); full += len(d); j = j2kvirt.J2K(d); sources.append(Source(url=f.resolve().as_uri()))
        for n in range(j.L + 1):
            for t in range(len(j.tiles)):
                c, nread = j.chunk(z, j.L - n, tile=t)
                mapping[f"s{n}/c/{z}/{t // j.ntx}/{t % j.ntx}"] = c; read[(n, t)] = read.get((n, t), 0) + nread
    L = j.L; mapping["zarr.json"] = util.GROUP_META
    for n in range(L + 1):
        w = 512 >> n; mapping[f"s{n}/zarr.json"] = util.array_meta((NZ, w, w), (1, TS >> n, TS >> n))
    out = W / f"hela2_pyramid_{name.replace('+', '_')}.vzip"; refkv.ReferenceStore(mapping, sources).save(str(out))
    g = zarr.open_group(rz.open(str(out)), mode="r")
    print(f"== {name}: {len(j.tiles)} tiles/plane, {len(mapping)} keys, {full} B of codestreams; "
          f"tile-parts/tile {len(j.tiles[0])}, COD precincts {'yes' if dict(j.main)[j2kvirt.COD][4] & 1 else 'no'}")
    arr = [g[f"s{n}"][:] for n in range(L + 1)]
    print(f"  s0 equals the original u16 data: {np.array_equal(arr[0], orig)}")
    for n in range(1, L + 1):
        ok = all(np.array_equal(arr[n][z], util.ojph_expand(W / folder.name / f"p{z:02d}.j2c", n, folder)) for z in (0, 7, 15))
        print(f"  s{n} {arr[n].shape} chunk {g[f's{n}'].chunks[1:]}: equals ojph_expand -skip_res {n} of the tiled file (planes 0, 7, 15): {ok}")
    for n in range(L + 1):
        # read[(n, t)] sums over all NZ planes, so compare with the same sum of codestream bytes per tile
        t_bytes = np.mean([read[(n, t)] for t in range(len(j.tiles))]) / (full / len(j.tiles))
        print(f"  one s{n} chunk reads {t_bytes * 100:.1f}% of its tile's bytes ({100 * t_bytes / len(j.tiles):.1f}% of the plane's file)")
    # chunk standalone validity under both decoders
    b = refkv.open(str(out)).get(f"s2/c/3/1/0")
    r = util.decode_both(b); print("  standalone chunk s2/c/3/1/0:", {k: (v if isinstance(v, str) else v.shape) for k, v in r.items()},
        "openjph==openjpeg:", all(not isinstance(v, str) for v in r.values()) and np.array_equal(*r.values()))
