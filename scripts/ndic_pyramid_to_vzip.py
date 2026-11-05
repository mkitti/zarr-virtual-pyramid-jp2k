"""Virtual pyramid over an nd-image-codecs OME-Zarr (nd_lift along z, then htj2k planes), as one refkv archive; no pixels copied.

The input array (e.g. `s1`) is referenced in place, chunk objects whole. Further levels are built from the same chunk objects:
  * `htj2k` planes are independent RPCL codestreams, so xy is reduced by keeping the packets of the lower resolutions behind a
    rewritten header and rebuilding the container's header and plane index (as `j2kvirt` does for single codestreams);
  * `nd_lift` stores z as [approx_L | detail_L | ... | detail_1]. The first planes are therefore a z-downsampled volume:
    level +1 uses planes 0..m/2-1 with one fewer inverse lifting level, level +2 the approx band (no lifting), and further levels
    every second plane of it (point sampling), while xy is reduced one wavelet level per step, down to the codestreams' coarsest.
All numbers (shapes, scales, the levels) are derived from the input's metadata.

usage: pixi run python scripts/ndic_pyramid_to_vzip.py <input .ome.zarr URL> <out.vzip> [--array s1] [--verify]
"""
import sys, re, json, struct, datetime, threading, http.client, pathlib, urllib.request, urllib.parse, concurrent.futures as cf
import numpy as np, refkv
from refkv.pb import Concat, Range, Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, util

BASE = sys.argv[1].rstrip("/"); OUT = sys.argv[2]
arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
ARRAY, VERIFY = arg("--array", "s1"), "--verify" in sys.argv
get = lambda u: urllib.request.urlopen(urllib.request.Request(u), timeout=120).read()
group = json.loads(get(BASE + "/zarr.json")); meta = json.loads(get(f"{BASE}/{ARRAY}/zarr.json"))
(Z, Y, X), (CZ, CY, CX) = meta["shape"], meta["chunk_grid"]["configuration"]["chunk_shape"]
codecs = meta["codecs"]
assert [c["name"] for c in codecs] == ["nd_lift", "htj2k"], "input must be nd_lift -> htj2k"
lift = codecs[0]["configuration"]; XYL = codecs[1]["configuration"]["xy_levels"]
assert len(lift["transforms"]) == 1 and lift["transforms"][0]["kind"] == "lift53" and lift["transforms"][0]["dimension"] == 0 and lift["transforms"][0]["group"] == 0
ZL = lift["transforms"][0]["levels"]; assert CZ % (1 << ZL) == 0 and CY < (1 << 30)
grid = (-(-Z // CZ), -(-Y // CY), -(-X // CX))
print(f"{ARRAY}: shape {(Z, Y, X)}, chunks {(CZ, CY, CX)} -> grid {grid} = {np.prod(grid)} chunks; nd_lift lift53 x{ZL} over z, htj2k xy_levels {XYL}")

# chunk objects: sizes and times from the bucket listing (the same host serves the S3 API at its root)
u = urllib.parse.urlparse(BASE); bucket, _, prefix = u.path.lstrip("/").partition("/"); objs = {}; token = ""
while True:                                                           # S3 ListObjectsV2 (this server ignores the v1 `marker`)
    xml = get(f"{u.scheme}://{u.netloc}/{bucket}/?list-type=2&prefix={urllib.parse.quote(prefix + '/' + ARRAY + '/')}&max-keys=1000"
              + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")).decode()
    for m in re.finditer(r"<Contents>(.*?)</Contents>", xml, flags=re.S):
        k = re.search(r"<Key>([^<]*)</Key>", m.group(1)).group(1)
        size = int(re.search(r"<Size>(\d+)</Size>", m.group(1)).group(1)); when = re.search(r"<LastModified>([^<]*)</LastModified>", m.group(1)).group(1)
        objs[k[len(prefix) + 1:]] = (size, int(datetime.datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp()))
    nxt = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
    if "<IsTruncated>true</IsTruncated>" not in xml or nxt is None: break
    token = nxt.group(1)
keys = [f"{ARRAY}/c/{iz}/{iy}/{ix}" for iz in range(grid[0]) for iy in range(grid[1]) for ix in range(grid[2])]
missing = [k for k in keys if k not in objs]; assert not missing, f"{len(missing)} chunk objects missing from the listing, e.g. {missing[:2]}"
print(f"{len(keys)} chunk objects, {sum(objs[k][0] for k in keys) / 1e9:.2f} GB")

# levels: (name, planes of the chunk to keep, xy resolution r, nd_lift filter)
LIFT_CFG = lambda lv: {"version": "0.1", "transforms": [{**lift["transforms"][0], "levels": lv}]}
IDENT = {"version": "0.1", "transforms": []}
levels = []                                                            # one entry per derived level
m = CZ
for k in range(1, XYL + 1):
    r = XYL - k                                                       # xy resolution kept
    if k <= ZL: m //= 2; planes = list(range(m)); cfg = LIFT_CFG(ZL - k) if ZL - k > 0 else IDENT
    else: planes = planes[::2]; m = len(planes); cfg = IDENT
    levels.append(dict(n=k, r=r, planes=planes, cfg=cfg))
for L in levels: print(f"  s{int(ARRAY[1:]) + L['n']}: z planes {L['planes'][:4]}{'...' if len(L['planes']) > 4 else ''} ({len(L['planes'])} per chunk), xy resolution r{L['r']}, nd_lift {'levels ' + str(L['cfg']['transforms'][0]['levels']) if L['cfg']['transforms'] else 'none'}")


def plane_parts(src_id, chunk_obj_off, plane_head, plane_len, r):
    """Concat parts of one plane reduced to resolution r, source offsets shifted into the chunk object, plus its container-index data."""
    d = bytearray(plane_head) + bytes(plane_len - len(plane_head)); d[-2:] = b"\xff\xd9"
    j = j2kvirt.J2K(bytes(d)); c, _ = j.chunk(0, r)                    # single tile-part with a PLT: j2kvirt's ndic layout
    lens = j2kvirt.plt_lengths(j.tiles[next(iter(j.tiles))][0]["segs"])
    parts = [p if p.data else Range(source=src_id, offset=chunk_obj_off + p.offset, length=p.length) for p in c.parts]
    total = sum(len(p.data) if p.data else p.length for p in parts); head = len(parts[0].data) + len(parts[1].data)
    return parts, total, [head + sum(lens[:rr + 1]) for rr in range(r + 1)]


_local = threading.local(); _done = [0]; _host = urllib.parse.urlparse(BASE)


def rng(key, a, n):
    """Exactly n bytes of an object from offset a, on a keep-alive connection per thread."""
    for attempt in range(4):
        try:
            if not hasattr(_local, "conn"): _local.conn = http.client.HTTPSConnection(_host.netloc, timeout=120)
            _local.conn.request("GET", f"{_host.path}/{key}", headers={"Range": f"bytes={a}-{a + n - 1}"})
            r = _local.conn.getresponse(); data = r.read()
            assert r.status == 206, r.status; return data
        except (http.client.HTTPException, OSError, AssertionError):
            _local.conn.close(); del _local.conn
    raise RuntimeError(f"cannot read {key}")


def derive(args):
    src_id, key = args
    head = rng(key, 0, 8 + 4 * 3 + CZ * (12 + 4 * (XYL + 1)))
    assert head[:4] == b"ndht" and head[4] == 1 and head[6] == XYL and head[7] == 3, "unexpected container"
    dims = struct.unpack("<III", head[8:20]); stride = 12 + 4 * (XYL + 1)
    ent = [struct.unpack("<QI", head[20 + i * stride:32 + i * stride]) for i in range(dims[0])]
    need = sorted({p for L in levels for p in L["planes"]}); heads = {p: rng(key, ent[p][0], min(768, ent[p][1])) for p in need}   # each plane's header, once
    out = {}
    for L in levels:
        r, planes = L["r"], L["planes"]; n = XYL - r; ys, xs = -(-dims[1] >> n), -(-dims[2] >> n)
        entries, body, off = [], [], 8 + 12 + len(planes) * (12 + 4 * (r + 1))
        for p in planes:
            o, ln = ent[p]; parts, total, pre = plane_parts(src_id, o, heads[p], ln, r)
            entries.append(struct.pack("<QI", off, total) + struct.pack("<" + "I" * (r + 1), *pre)); body += parts; off += total
        header = b"ndht" + bytes([1, 1, r, 3]) + struct.pack("<III", len(planes), ys, xs) + b"".join(entries)
        out[L["n"]] = Concat((Range(data=header), *body))
    _done[0] += 1
    if _done[0] % 100 == 0: print(f"  {_done[0]} / {len(keys)} chunks", flush=True)
    return key, out


with cf.ThreadPoolExecutor(24) as ex: derived = dict(ex.map(derive, [(i, k) for i, k in enumerate(keys)]))
print(f"derived {len(derived) * len(levels)} chunks ({len(levels)} extra levels)")
srcs = [Source(url=f"{BASE}/{k}", size=objs[k][0], modified_not_after=objs[k][1]) for k in keys]
mapping = {k: Range(source=i, offset=0, length=objs[k][0]) for i, k in enumerate(keys)}
mapping[f"{ARRAY}/zarr.json"] = get(f"{BASE}/{ARRAY}/zarr.json")
n0 = int(ARRAY[1:]); ds = group["attributes"]["ome"]["multiscales"][0]["datasets"][0]; sc = ds["coordinateTransformations"]
datasets = [ds]; shapes = {0: (Z, Y, X)}
for L in levels:
    n = L["n"]; zc = len(L["planes"]); cy, cx = -(-CY >> (XYL - L["r"])), -(-CX >> (XYL - L["r"]))
    shape = (-(-Z // (CZ // zc)), -(-Y >> n), -(-X >> n)); shapes[n] = shape; name = f"s{n0 + n}"
    am = json.loads(mapping[f"{ARRAY}/zarr.json"]); am["shape"] = list(shape); am["chunk_grid"]["configuration"]["chunk_shape"] = [zc, cy, cx]
    am["codecs"] = [{"name": "nd_lift", "configuration": L["cfg"]}, {"name": "htj2k", "configuration": {**codecs[1]["configuration"], "xy_levels": L["r"]}}]
    mapping[f"{name}/zarr.json"] = json.dumps(am).encode()
    for k, d in derived.items(): mapping[f"{name}/c/{k.split('/c/')[1]}"] = d[n]
    scale = [s * 2 ** n for s in sc[0]["scale"]]; datasets.append({"path": name, "coordinateTransformations": [{"type": "scale", "scale": scale}, *sc[1:]]})
grp = json.loads(json.dumps(group)); grp["attributes"]["ome"]["multiscales"][0]["datasets"] = datasets
grp["attributes"]["ome"]["multiscales"][0]["name"] = group["attributes"]["ome"]["multiscales"][0]["name"] + " + virtual levels"
mapping["zarr.json"] = json.dumps(grp).encode()
refkv.ReferenceStore(mapping, srcs).save(OUT)
print(f"saved {OUT} ({pathlib.Path(OUT).stat().st_size} B): {len(mapping)} keys; " + ", ".join(f"s{n0 + n} {shapes[n]}" for n in sorted(shapes)))
