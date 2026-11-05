"""Virtual reduced-resolution JPEG 2000 / HTJ2K codestreams as refkv Concats.

A codestream with L decomposition levels holds resolutions r = 0 (smallest) .. L (full).
Zarr level n (s0 = full) is resolution r = L - n, i.e. the image reduced n times.

Supported input layouts (single quality layer, one component, resolution-major progression
such as RPCL or RLCP, no COC/QCC):
  * per tile, one tile-part with a PLT marker listing one packet per resolution (ndic), or
  * per tile, one tile-part per resolution (OpenJPH `-tileparts R`); PLT/TLM not needed.
Tiles are optional; `chunk(..., tile=t)` makes a standalone single-tile codestream.
"""
import struct
from refkv.pb import Concat, Range

SOC, SIZ, COD, QCD, TLM, PLT, SOT, SOD, EOC = (bytes.fromhex(x) for x in
    ("ff4f", "ff51", "ff52", "ff5c", "ff55", "ff58", "ff90", "ff93", "ffd9"))


def u16(d, o): return struct.unpack(">H", d[o:o + 2])[0]
def u32(d, o): return struct.unpack(">I", d[o:o + 4])[0]
def ceil_shift(a, n): return -((-a) >> n)


def varint(n):
    o = [n & 0x7f]; n >>= 7
    while n:
        o.append(0x80 | (n & 0x7f)); n >>= 7
    return bytes(reversed(o))


def plt_lengths(segs):
    lens, v = [], 0
    for m, b in segs:
        if m != PLT:
            continue
        for x in b[5:]:                       # after marker(2), Lplt(2), Zplt(1)
            v = (v << 7) | (x & 0x7f)
            if not x & 0x80:
                lens.append(v); v = 0
    return lens


class J2K:
    def __init__(self, d):
        self.d = d; pos = 2; self.main = []
        while d[pos:pos + 2] != SOT:
            L = u16(d, pos + 2); self.main.append((d[pos:pos + 2], bytes(d[pos:pos + 2 + L]))); pos += 2 + L
        seg = dict(self.main)
        siz = seg[SIZ]
        self.X, self.Y, self.XT, self.YT = u32(siz, 6), u32(siz, 10), u32(siz, 22), u32(siz, 26)
        assert u32(siz, 14) == u32(siz, 18) == u32(siz, 30) == u32(siz, 34) == 0, "image/tile offsets"
        assert u16(siz, 38) == 1, "one component only"
        cod = seg[COD]; self.L = cod[9]; assert u16(cod, 6) == 1, "one layer only"
        assert cod[5] in (0, 1, 2), "progression must be LRCP/RLCP/RPCL"   # LRCP == RLCP for 1 layer
        self.ntx = -(-self.X // self.XT)
        self.nty = -(-self.Y // self.YT)
        self.tiles = {}
        while pos < len(d) and d[pos:pos + 2] == SOT:
            isot, psot, tp = u16(d, pos + 4), u32(d, pos + 6), d[pos + 10]
            assert psot > 0
            q = pos + 12; segs = []
            while d[q:q + 2] != SOD:
                L = u16(d, q + 2); segs.append((d[q:q + 2], bytes(d[q:q + 2 + L]))); q += 2 + L
            self.tiles.setdefault(isot, []).append(dict(tp=tp, segs=segs, body=q + 2, end=pos + psot))
            pos += psot
        assert d[pos:pos + 2] == EOC
        for t in self.tiles.values():
            t.sort(key=lambda x: x["tp"])

    def units(self, isot, r):
        """Tile-part descriptors [(segments, source offset, length)] holding resolutions 0..r."""
        tps = self.tiles[isot]
        if len(tps) == self.L + 1:                      # one tile-part per resolution
            return [(tp["segs"], tp["body"], tp["end"] - tp["body"]) for tp in tps[:r + 1]]
        assert len(tps) == 1, "unsupported tile-part layout"
        tp = tps[0]; lens = plt_lengths(tp["segs"])
        assert len(lens) == self.L + 1, "PLT must list one packet per resolution (maximal precincts)"
        pn = b"".join(varint(x) for x in lens[:r + 1])
        plt = PLT + struct.pack(">H", 3 + len(pn)) + b"\x00" + pn
        segs = [(PLT, plt)] + [(m, b) for m, b in tp["segs"] if m != PLT]
        return [(segs, tp["body"], sum(lens[:r + 1]))]

    # -- header patches -------------------------------------------------------------------
    def _siz(self, b, X, Y, XT, YT):
        b = bytearray(b)
        for off, v in ((6, X), (10, Y), (22, XT), (26, YT)):
            b[off:off + 4] = struct.pack(">I", v)
        return bytes(b)

    def _cod(self, b, r):
        b = bytearray(b); b[9] = r
        if b[4] & 1:                                    # precinct sizes, lowest resolution first
            b = b[:14 + r + 1]; b[2:4] = struct.pack(">H", len(b) - 2)
        return bytes(b)

    def _qcd(self, b, r):
        style = b[4] & 0x1f
        if style == 1:
            raise NotImplementedError("scalar-derived QCD: exponents depend on the level count")
        esz = 1 if style == 0 else 2
        assert (len(b) - 5) // esz == 1 + 3 * self.L
        b = bytearray(b[:5 + (1 + 3 * r) * esz]); b[2:4] = struct.pack(">H", len(b) - 2)
        return bytes(b)

    def chunk(self, src, r, tile=None):
        """Concat for resolution r (0..L). tile=None: the whole (single-tile) image;
        tile=t: a standalone codestream of tile t. Returns (Concat, bytes read from the source)."""
        n = self.L - r
        if tile is None:
            assert len(self.tiles) == 1, "multi-tile image: pass tile="
            isot = next(iter(self.tiles)); new_isot = isot
            X, Y, XT, YT = (ceil_shift(v, n) for v in (self.X, self.Y, self.XT, self.YT))
        else:
            isot = tile; new_isot = 0
            tx, ty = tile % self.ntx, tile // self.ntx
            tw = min(self.XT, self.X - tx * self.XT); th = min(self.YT, self.Y - ty * self.YT)
            X = XT = ceil_shift(tw, n); Y = YT = ceil_shift(th, n)
        main = SOC
        for m, b in self.main:
            if m == SIZ: b = self._siz(b, X, Y, XT, YT)
            elif m == COD: b = self._cod(b, r)
            elif m == QCD: b = self._qcd(b, r)
            elif m == TLM: continue                      # optional; lengths would need rewriting
            main += b
        parts = [Range(data=main)]; units = self.units(isot, r); nread = 0
        for i, (segs, off, ln) in enumerate(units):
            sb = b"".join(b for _, b in segs)
            psot = 12 + len(sb) + 2 + ln
            parts += [Range(data=SOT + struct.pack(">HHIBB", 10, new_isot, psot, i, len(units)) + sb + SOD),
                      Range(source=src, offset=off, length=ln)]
            nread += ln
        parts.append(Range(data=EOC))
        return Concat(tuple(parts)), nread
