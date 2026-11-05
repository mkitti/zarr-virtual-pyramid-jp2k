"""Packet-header parsing / rewriting for single-layer, single-component J2K and HTJ2K codestreams.

Inside a packet, the body is band-major: all code-blocks of the first band, then the next band, ...
(LL for resolution 0; HL, LH, HH otherwise), each in raster order. So within one packet every
sub-band's data is ONE contiguous byte range, and a packet is [header | band | band | band].
Dropping a band therefore needs only a new (literal) header plus source ranges for the kept bands.

Supported: LRCP/RLCP/RPCL with one layer and one component (packet order = r, then precinct raster),
default code-block style or HT (code-block style 0x40), optional SOP/EPH (parsing only).
"""
import struct
from j2kvirt import J2K, COD, PLT, SOT, SOD, EOC, u16, u32, varint
from refkv.pb import Concat, Range

INF = 1 << 30
def cdiv(a, b): return -((-a) // b)
def flog2(n): return n.bit_length() - 1
BANDS = {False: [("LL", 0, 0)], True: [("HL", 1, 0), ("LH", 0, 1), ("HH", 1, 1)]}


class BitReader:
    def __init__(self, buf, pos): self.buf, self.pos, self.cur, self.n = buf, pos, 0, 0
    def bit(self):
        if self.n == 0:
            prev = self.cur; self.cur = self.buf[self.pos]; self.pos += 1; self.n = 7 if prev == 0xff else 8
        self.n -= 1
        return (self.cur >> self.n) & 1
    def bits(self, k):
        v = 0
        for _ in range(k): v = (v << 1) | self.bit()
        return v
    def align(self):
        self.n = 0
        if self.cur == 0xff: self.pos += 1          # stuffed byte after a final 0xFF
        self.cur = 0


class BitWriter:
    def __init__(self): self.out, self.cur, self.n, self.cap = bytearray(), 0, 0, 8
    def put(self, b):
        self.cur = (self.cur << 1) | b; self.n += 1
        if self.n == self.cap:
            self.out.append(self.cur); self.cap = 7 if self.cur == 0xff else 8; self.cur = 0; self.n = 0
    def bits(self, v, k):
        for i in range(k - 1, -1, -1): self.put((v >> i) & 1)
    def flush(self):
        if self.n: self.out.append(self.cur << (self.cap - self.n)); self.cur = self.n = 0
        if self.out and self.out[-1] == 0xff: self.out.append(0)
        return bytes(self.out)


def _dims(w, h):
    lv = [(w, h)]
    while lv[-1] != (1, 1): lv.append(((lv[-1][0] + 1) // 2, (lv[-1][1] + 1) // 2))
    return lv


class TagTree:
    """Decoder (leaf values unknown) and encoder (leaf values given) over a w x h grid."""
    def __init__(self, w, h, leaves=None):
        self.lv = _dims(w, h); self.low = [[0] * (a * b) for a, b in self.lv]
        self.known = [[False] * (a * b) for a, b in self.lv]
        self.val = [[INF] * (a * b) for a, b in self.lv]
        if leaves is not None:
            self.val[0] = list(leaves)
            for k in range(1, len(self.lv)):
                pw = self.lv[k][0]; cw, ch = self.lv[k - 1]
                for y in range(ch):
                    for x in range(cw):
                        i = (y >> 1) * pw + (x >> 1); self.val[k][i] = min(self.val[k][i], self.val[k - 1][y * cw + x])
    def decode(self, br, x, y, thr):
        low = 0
        for k in range(len(self.lv) - 1, -1, -1):
            i = (y >> k) * self.lv[k][0] + (x >> k)
            if low > self.low[k][i]: self.low[k][i] = low
            else: low = self.low[k][i]
            while low < thr and low < self.val[k][i]:
                if br.bit(): self.val[k][i] = low
                else: low += 1
            self.low[k][i] = low
        return self.val[0][y * self.lv[0][0] + x] < thr
    def encode(self, bw, x, y, thr):
        low = 0
        for k in range(len(self.lv) - 1, -1, -1):
            i = (y >> k) * self.lv[k][0] + (x >> k)
            if low > self.low[k][i]: self.low[k][i] = low
            else: low = self.low[k][i]
            while low < thr:
                if low >= self.val[k][i]:
                    if not self.known[k][i]: bw.put(1); self.known[k][i] = True
                    break
                bw.put(0); low += 1
            self.low[k][i] = low


def read_passes(br):
    if not br.bit(): return 1
    if not br.bit(): return 2
    v = br.bits(2)
    if v != 3: return 3 + v
    v = br.bits(5)
    return 6 + v if v != 31 else 37 + br.bits(7)


def write_passes(bw, n):
    if n == 1: bw.put(0)
    elif n == 2: bw.bits(0b10, 2)
    elif n <= 5: bw.bits(0b11, 2); bw.bits(n - 3, 2)
    elif n <= 36: bw.bits(0b1111, 4); bw.bits(n - 6, 5)
    else: bw.bits(0x1ff, 9); bw.bits(n - 37, 7)


def seg_passes(passes, ht):
    if not ht: return [passes]
    assert passes <= 3, "HT code-blocks with more than one HT set are not supported"
    return [1] if passes == 1 else [1, passes - 1]


def geometry(j, isot):
    """Per resolution: precinct grid and, per precinct, the bands with their code-block grids."""
    cod = dict(j.main)[COD]; L = j.L
    scod, xcb, ycb, sty = cod[4], (cod[10] & 15) + 2, (cod[11] & 15) + 2, cod[12]
    assert sty in (0, 0x40), "unsupported code-block style"
    pp = [(b & 15, b >> 4) for b in cod[14:14 + L + 1]] if scod & 1 else [(15, 15)] * (L + 1)
    tx, ty = isot % j.ntx, isot // j.ntx
    tcx0, tcy0 = tx * j.XT, ty * j.YT; tcx1, tcy1 = min(tcx0 + j.XT, j.X), min(tcy0 + j.YT, j.Y)
    res = []
    for r in range(L + 1):
        nl = L - r; ppx, ppy = pp[r]
        rx0, rx1, ry0, ry1 = cdiv(tcx0, 1 << nl), cdiv(tcx1, 1 << nl), cdiv(tcy0, 1 << nl), cdiv(tcy1, 1 << nl)
        npw = cdiv(rx1, 1 << ppx) - (rx0 >> ppx) if rx1 > rx0 else 0
        nph = cdiv(ry1, 1 << ppy) - (ry0 >> ppy) if ry1 > ry0 else 0
        precs = []
        for p in range(npw * nph):
            px, py = p % npw, p // npw; bands = []
            for name, xo, yo in BANDS[r > 0]:
                nb = L - r + 1 if r > 0 else L
                h = (1 << (nb - 1)) if r > 0 else 0
                bx0, bx1 = cdiv(tcx0 - h * xo, 1 << nb), cdiv(tcx1 - h * xo, 1 << nb)
                by0, by1 = cdiv(tcy0 - h * yo, 1 << nb), cdiv(tcy1 - h * yo, 1 << nb)
                qx0 = ((rx0 >> ppx) + px) << ppx; qy0 = ((ry0 >> ppy) + py) << ppy
                qx1, qy1 = qx0 + (1 << ppx), qy0 + (1 << ppy)
                if r > 0: qx0, qx1, qy0, qy1 = qx0 // 2, qx1 // 2, qy0 // 2, qy1 // 2
                x0, x1, y0, y1 = max(bx0, qx0), min(bx1, qx1), max(by0, qy0), min(by1, qy1)
                cxs = min(xcb, ppx - 1 if r > 0 else ppx); cys = min(ycb, ppy - 1 if r > 0 else ppy)
                ok = x1 > x0 and y1 > y0
                bands.append(dict(name=name, xcbp=cxs, ycbp=cys, cx0=x0 >> cxs, cy0=y0 >> cys, bx=(x0, x1), by=(y0, y1),
                                  ncbw=cdiv(x1, 1 << cxs) - (x0 >> cxs) if ok else 0,
                                  ncbh=cdiv(y1, 1 << cys) - (y0 >> cys) if ok else 0))
            precs.append(dict(px=px, py=py, bands=bands))
        res.append(dict(r=r, npw=npw, nph=nph, precincts=precs))
    return res


class TileIndex:
    """Every packet and code-block of one tile, with stream and file offsets."""
    def __init__(self, j, isot):
        self.j, self.isot = j, isot; d = j.d; cod = dict(j.main)[COD]
        scod, ht = cod[4], cod[12] == 0x40
        tps = j.tiles[isot]; self.pieces = []; chunks = []; s = 0
        for tp in tps:
            self.pieces.append((s, tp["body"], tp["end"] - tp["body"])); chunks.append(d[tp["body"]:tp["end"]]); s += tp["end"] - tp["body"]
        self.buf = buf = b"".join(chunks); self.geo = geometry(j, isot); self.packets = []; pos = 0
        for rg in self.geo:
            for pr in rg["precincts"]:
                start = pos
                if scod & 2: assert buf[pos:pos + 2] == b"\xff\x91"; pos += 6
                states = [(TagTree(b["ncbw"], b["ncbh"]) if b["ncbw"] else None, TagTree(b["ncbw"], b["ncbh"]) if b["ncbw"] else None)
                          for b in pr["bands"]]
                br = BitReader(buf, pos); bands = [dict(name=b["name"], ncbw=b["ncbw"], ncbh=b["ncbh"], geo=b,
                                                        cbs=[None] * (b["ncbw"] * b["ncbh"])) for b in pr["bands"]]
                if br.bit():
                    for ob, (incl, zbp) in zip(bands, states):
                        for jy in range(ob["ncbh"]):
                            for ix in range(ob["ncbw"]):
                                if not incl.decode(br, ix, jy, 1): continue
                                i = 1
                                while not zbp.decode(br, ix, jy, i): i += 1
                                passes = read_passes(br); lb = 3
                                while br.bit(): lb += 1
                                sp = seg_passes(passes, ht)
                                ob["cbs"][jy * ob["ncbw"] + ix] = dict(passes=passes, zbp=i - 1, lblock=lb, segp=sp,
                                                                       seglens=[br.bits(lb + flog2(p)) for p in sp])
                br.align(); pos = br.pos
                if scod & 4: assert buf[pos:pos + 2] == b"\xff\x92"; pos += 2
                hdr_end = pos
                for ob in bands:
                    ob["rel"] = pos - hdr_end
                    for cb in ob["cbs"]:
                        if cb: cb["rel"] = pos - hdr_end; cb["len"] = sum(cb["seglens"]); pos += cb["len"]
                    ob["len"] = pos - hdr_end - ob["rel"]
                tpi = max(i for i, pc in enumerate(self.pieces) if pc[0] <= start)
                assert pos <= self.pieces[tpi][0] + self.pieces[tpi][2], "packet crosses a tile-part"
                self.packets.append(dict(r=rg["r"], px=pr["px"], py=pr["py"], off=start, hdr=hdr_end - start, body=pos - hdr_end,
                                         bands=bands, tp=tpi))
        self.consumed = pos == len(buf)

    def file_off(self, stream_off):
        # half-open: an offset at a tile-part boundary belongs to the NEXT tile-part
        for s, f, n in self.pieces:
            if s <= stream_off < s + n: return f + stream_off - s
        s, f, n = self.pieces[-1]
        if stream_off == s + n: return f + n
        raise ValueError(stream_off)

    def band_range(self, pkt, name):
        """(file offset, length) of one band's data inside one packet."""
        b = next(b for b in pkt["bands"] if b["name"] == name)
        return self.file_off(pkt["off"] + pkt["hdr"] + b["rel"]), b["len"]

    def check_ht(self):
        """Every HT cleanup segment must end with a valid Scup (an independent check of the segment boundaries)."""
        bad = n = 0
        for pkt in self.packets:
            for b in pkt["bands"]:
                for cb in b["cbs"]:
                    if not cb: continue
                    o = pkt["off"] + pkt["hdr"] + cb["rel"]; l1 = cb["seglens"][0]; seg = self.buf[o:o + l1]; n += 1
                    scup = (seg[-1] << 4) + (seg[-2] & 15) if l1 >= 2 else 0
                    if not (2 <= scup <= min(l1, 4079)): bad += 1
        return n, bad


def write_header(pkt, keep):
    """New packet header that includes only the bands named in `keep`. Returns (bytes, [(band, rel, len)])."""
    bands = pkt["bands"]
    inc = [[(cb is not None and ob["name"] in keep) for cb in ob["cbs"]] for ob in bands]
    if not any(any(x) for x in inc): return b"\x00", []
    bw = BitWriter(); bw.put(1)
    for ob, flags in zip(bands, inc):
        if not ob["ncbw"]: continue
        w = ob["ncbw"]
        it = TagTree(w, ob["ncbh"], [0 if f else 1 for f in flags])
        zt = TagTree(w, ob["ncbh"], [ob["cbs"][i]["zbp"] if f else INF for i, f in enumerate(flags)])
        for i, f in enumerate(flags):
            x, y = i % w, i // w
            it.encode(bw, x, y, 1)
            if not f: continue
            cb = ob["cbs"][i]; zt.encode(bw, x, y, 1000); write_passes(bw, cb["passes"])
            lb = max(3, max(l.bit_length() - flog2(p) for l, p in zip(cb["seglens"], cb["segp"])))
            for _ in range(lb - 3): bw.put(1)
            bw.put(0)
            for l, p in zip(cb["seglens"], cb["segp"]): bw.bits(l, lb + flog2(p))
    return bw.flush(), [(ob["name"], ob["rel"], ob["len"]) for ob in bands if ob["name"] in keep and ob["len"]]


def assemble(ti, src, edit=lambda pkt: None):
    """Concat of the full-resolution codestream of tile `ti` as a standalone single-tile file, with per-packet edits.
    edit(pkt) -> None (keep) | 'empty' | set of band names to keep."""
    j = ti.j; cod = dict(j.main)[COD]; assert cod[4] & 6 == 0, "SOP/EPH not handled"
    parts = [j.chunk(src, j.L, tile=ti.isot if len(j.tiles) > 1 else None)[0].parts[0]]
    for k, tp in enumerate(j.tiles[ti.isot]):
        plt = any(m == PLT for m, _ in tp["segs"]); pk = [p for p in ti.packets if p["tp"] == k]
        body, lens = [], []
        for p in pk:
            e = edit(p); base = ti.file_off(p["off"])
            if e is None: ps = [Range(source=src, offset=base, length=p["hdr"] + p["body"])]
            elif e == "empty": ps = [Range(data=b"\x00")]
            else:
                hdr, ranges = write_header(p, e); ps = [Range(data=hdr)] + [
                    Range(source=src, offset=base + p["hdr"] + rel, length=ln) for _, rel, ln in ranges]
            body += ps; lens.append(sum(len(x.data) if x.data else x.length for x in ps))
        segs = [b for m, b in tp["segs"] if m != PLT]
        if plt:
            pn = b"".join(varint(x) for x in lens); segs.insert(0, PLT + struct.pack(">H", 3 + len(pn)) + b"\x00" + pn)
        sb = b"".join(segs); psot = 12 + len(sb) + 2 + sum(lens)
        parts.append(Range(data=SOT + struct.pack(">HHIBB", 10, 0 if len(j.tiles) > 1 else ti.isot, psot, k, len(j.tiles[ti.isot])) + sb + SOD))
        parts += body
    parts.append(Range(data=EOC))
    return Concat(tuple(parts))
