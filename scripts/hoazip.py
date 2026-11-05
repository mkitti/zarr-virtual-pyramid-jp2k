"""Read a zip of stored (uncompressed) .jp2 slices over HTTP Range requests: entry list and the byte range of each file."""
import io, re, struct, zipfile, email.utils, urllib.request, concurrent.futures as cf

JP2_SIGNATURE = bytes.fromhex("0000000c6a5020200d0a870a")


def make_fetch(url):
    def fetch(a, n):
        """Exactly n bytes from offset a. Some servers stream to the end of the file whatever the Range end says, so read n and close."""
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + n - 1}"}), timeout=120) as r:
            assert r.status == 206, r.status
            return r.read(n)
    return fetch


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file over Range requests, cached in 64 KiB blocks."""
    def __init__(self, fetch, size): self.fetch, self.size, self.pos, self.cache, self.requests = fetch, size, 0, {}, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0): self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos); out = []
        while n > 0:
            i, o = divmod(self.pos, 1 << 16)
            if i not in self.cache: self.cache[i] = self.fetch(i << 16, min(1 << 16, self.size - (i << 16))); self.requests += 1
            part = self.cache[i][o:o + n]; out.append(part); self.pos += len(part); n -= len(part)
        return b"".join(out)
    def readinto(self, b): d = self.read(len(b)); b[:len(d)] = d; return len(d)


def list_slices(url):
    """Returns (size, last_modified epoch or None, [(name, data_offset, file_size)]) for a zip of stored .jp2 files, checking that every
    entry starts with a JP2 signature at its computed data offset and that the slice numbers run 0..N-1."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size, lm = int(r.headers["Content-Length"]), r.headers.get("Last-Modified")
    fetch = make_fetch(url)
    infos = [i for i in zipfile.ZipFile(RangeFile(fetch, size)).infolist() if not i.is_dir()]
    assert all(i.compress_type == 0 and i.filename.lower().endswith(".jp2") for i in infos), "entries must be stored .jp2 files"
    assert [int(re.search(r"_(\d+)\.jp2$", i.filename).group(1)) for i in infos] == list(range(len(infos))), "slice numbers are not 0..N-1 in order"

    def locate(info):
        head = fetch(info.header_offset, 30 + len(info.filename.encode()) + 12 + 64)
        assert head[:4] == b"PK\x03\x04"
        nlen, xlen = struct.unpack("<HH", head[26:30]); start = 30 + nlen + xlen
        if start + 12 > len(head): head = fetch(info.header_offset, start + 12)
        assert head[start:start + 12] == JP2_SIGNATURE, f"{info.filename} does not start with a JP2 signature"
        return info.filename, info.header_offset + start, info.file_size
    with cf.ThreadPoolExecutor(16) as ex: entries = list(ex.map(locate, infos))
    return size, (int(email.utils.parsedate_to_datetime(lm).timestamp()) if lm else None), entries


def codestream_offset(jp2_head):
    """Offset of the codestream (the payload of the jp2c box) inside a JP2 file, from its first bytes."""
    pos = 0
    while True:
        length, typ = struct.unpack(">I4s", jp2_head[pos:pos + 8])
        if typ == b"jp2c": return pos + 8
        assert length > 8, "box of unsupported length"; pos += length
