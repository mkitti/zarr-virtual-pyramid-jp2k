"""Reference the JPEG 2000 slices of a Human Organ Atlas zip as a Zarr v3 / OME-Zarr 0.5 array, without copying pixels.

The zip stores every `.jp2` uncompressed, so each Zarr chunk (one slice, shape (1, H, W), codec `imagecodecs_jpeg2k`) is
one byte range of the zip. Only the zip's central directory and each entry's local header are read.

usage: pixi run python scripts/hoa_zip_to_vzip.py <zip URL> <out.vzip> [--voxel-um 100.32]
"""
import io, sys, re, zipfile, struct, email.utils, urllib.request, concurrent.futures as cf
import numpy as np, refkv
from refkv.pb import Range, Source
import json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import util

URL, OUT = sys.argv[1], sys.argv[2]
VOXEL_UM = float(sys.argv[sys.argv.index("--voxel-um") + 1]) if "--voxel-um" in sys.argv else 100.32


def fetch(a, n):
    """Exactly n bytes from offset a (the S3 proxy honours the end of a Range; some servers stream to the end, so read n and close)."""
    with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={a}-{a + n - 1}"}), timeout=120) as r:
        assert r.status == 206, r.status
        return r.read(n)


with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
    SIZE, LM = int(r.headers["Content-Length"]), r.headers.get("Last-Modified")


class RangeFile(io.RawIOBase):
    """Read-only seekable view of the zip over HTTP Range requests, cached in 64 KiB blocks."""
    def __init__(self): self.pos, self.cache, self.requests = 0, {}, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0): self.pos = off if whence == 0 else self.pos + off if whence == 1 else SIZE + off; return self.pos
    def read(self, n=-1):
        n = SIZE - self.pos if n is None or n < 0 else min(n, SIZE - self.pos); out = []
        while n > 0:
            i, o = divmod(self.pos, 1 << 16)
            if i not in self.cache: self.cache[i] = fetch(i << 16, min(1 << 16, SIZE - (i << 16))); self.requests += 1
            part = self.cache[i][o:o + n]; out.append(part); self.pos += len(part); n -= len(part)
        return b"".join(out)
    def readinto(self, b): d = self.read(len(b)); b[:len(d)] = d; return len(d)


rf = RangeFile(); infos = [i for i in zipfile.ZipFile(rf).infolist() if not i.is_dir()]
print(f"{len(infos)} entries; central directory read in {rf.requests} range requests")
assert all(i.compress_type == 0 and i.filename.lower().endswith(".jp2") for i in infos), "entries must be stored .jp2 files"
names = [i.filename for i in infos]; idx = [int(re.search(r"_(\d+)\.jp2$", n).group(1)) for n in names]
assert idx == list(range(len(infos))), "slice numbers are not 0..N-1 in order"


def locate(info):
    """(data offset, first bytes of the data): reads the local header (its extra field can differ from the central one) and 12 data bytes."""
    head = fetch(info.header_offset, 30 + len(info.filename.encode()) + 12 + 64)
    assert head[:4] == b"PK\x03\x04"
    nlen, xlen = struct.unpack("<HH", head[26:30]); start = 30 + nlen + xlen
    if start + 12 > len(head): head = fetch(info.header_offset, start + 12)
    return info.header_offset + start, head[start:start + 12]


with cf.ThreadPoolExecutor(16) as ex: located = list(ex.map(locate, infos))
JP2_SIG = bytes.fromhex("0000000c6a5020200d0a870a")
bad = [i.filename for i, (_, sig) in zip(infos, located) if sig != JP2_SIG]
assert not bad, f"{len(bad)} entries do not start with a JP2 signature, e.g. {bad[:2]}"
print("all", len(infos), "entries start with the JP2 signature at the computed data offset")

# image shape and data type from the first slice's JP2 header (ihdr): height, width, components, bit depth
ihdr = fetch(located[0][0], 64); p = ihdr.find(b"ihdr"); h, w, nc = struct.unpack(">IIH", ihdr[p + 4:p + 14]); bpc = ihdr[p + 14] + 1
assert nc == 1 and bpc == 16, (nc, bpc)
print(f"slices: {len(infos)} x {h} x {w}, 1 component, {bpc}-bit")
sizes = [i.file_size for i in infos]
pins = {"size": SIZE}
if LM: pins["modified_not_after"] = int(email.utils.parsedate_to_datetime(LM).timestamp())
src = Source(url=URL, **pins)
mapping = {f"s0/c/{z}/0/0": Range(source=0, offset=off, length=sizes[z]) for z, (off, _) in enumerate(located)}
mapping["s0/zarr.json"] = util.array_meta((len(infos), h, w), (1, h, w), "imagecodecs_jpeg2k")
ome = {"version": "0.5", "multiscales": [{
    "name": "LADAF-2021-17 heart, 100.32 um (HiP-CT), virtual JPEG 2000 slices",
    "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
    "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [VOXEL_UM] * 3}]}]}]}
mapping["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome,
    "license": "CC-BY-4.0", "source": "https://doi.org/10.15151/ESRF-DC-1773966241",
    "attribution": "Walsh, C. L. et al. (2024). Complete scan at 25.08 um of the heart of the body donor LADAF-2021-17 (Version 1) [Dataset]. "
                   "European Synchrotron Radiation Facility. https://doi.org/10.15151/ESRF-DC-1773966241. CC BY 4.0. Unchanged slices, referenced in place."}}).encode()
store = refkv.ReferenceStore(mapping, [src]); store.save(OUT)
print(f"saved {OUT}: {len(mapping)} keys, source {URL.rsplit('/', 1)[-1]} pinned {pins}")
