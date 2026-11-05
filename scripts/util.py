"""Shared helpers: PGM I/O, OpenJPH CLI wrappers, Zarr v3 metadata for refkv pyramids."""
import json, subprocess, pathlib, numpy as np, imagecodecs as ic


def write_pgm16(path, a):
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n65535\n" % (a.shape[1], a.shape[0])); f.write(a.astype(">u2").tobytes())


def read_pgm16(path):
    d = pathlib.Path(path).read_bytes(); i = 0; toks = []
    while len(toks) < 4:                                   # P5 width height maxval, with optional # comments
        while d[i:i + 1].isspace(): i += 1
        if d[i:i + 1] == b"#":
            while d[i:i + 1] != b"\n": i += 1
            continue
        j = i
        while not d[j:j + 1].isspace(): j += 1
        toks.append(d[i:j]); i = j
    w, h = int(toks[1]), int(toks[2]); data = d[i + 1:]    # one whitespace byte separates the header from the data
    return np.frombuffer(data, dtype=">u2")[: w * h].reshape(h, w).astype(np.uint16)


def ojph_compress(plane, out, *opts, work):
    pgm = pathlib.Path(work) / "_in.pgm"; write_pgm16(pgm, plane)
    subprocess.run(["ojph_compress", "-i", str(pgm), "-o", str(out), *map(str, opts)], check=True, capture_output=True)
    pgm.unlink()


def ojph_expand(j2c, n, work):
    """OpenJPH's own decode with n resolutions skipped (reference for the reduced image)."""
    out = pathlib.Path(work) / "_ref.pgm"
    subprocess.run(["ojph_expand", "-i", str(j2c), "-o", str(out), "-skip_res", str(n)], check=True, capture_output=True)
    return read_pgm16(out)


def decode_both(b):
    res = {}
    for name, fn in (("openjph", ic.htj2k_decode), ("openjpeg", ic.jpeg2k_decode)):
        try: res[name] = np.asarray(fn(b)).squeeze()
        except Exception as e: res[name] = f"ERR {e!r}"[:80]
    return res


def array_meta(shape, chunks, codec="imagecodecs_htj2k"):
    return json.dumps({"zarr_format": 3, "node_type": "array", "shape": list(shape), "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0,
        "codecs": [{"name": codec, "configuration": {}}], "dimension_names": ["z", "y", "x"]}).encode()


GROUP_META = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode()


def ome_group_meta(levels, name, z_nm=5.24, xy_nm=4.0):
    """OME-NGFF 0.5 multiscales group. Only y and x are halved per level (z planes are separate codestreams).
    No translation: a 5/3 low-pass sample n sits at the position of full-resolution sample 2n (point-aligned)."""
    ome = {"version": "0.5", "multiscales": [{
        "name": name,
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": f"s{n}", "coordinateTransformations": [{"type": "scale", "scale": [z_nm, xy_nm * 2 ** n, xy_nm * 2 ** n]}]}
                     for n in range(levels)]}]}
    return json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}).encode()
