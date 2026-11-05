"""Describe how a JPEG 2000 file (JP2 boxes or a raw codestream) is encoded: boxes, SIZ, CAP, COD/COC, QCD/QCC, markers.
usage: pixi run python scripts/inspect_j2k.py <file.jp2|.j2k|.j2c|.jph>
"""
import sys, struct, collections, pathlib

MARKERS = {0xFF4F: "SOC", 0xFF51: "SIZ", 0xFF52: "COD", 0xFF53: "COC", 0xFF55: "TLM", 0xFF57: "PLM", 0xFF58: "PLT", 0xFF50: "CAP",
           0xFF59: "CPF", 0xFF5C: "QCD", 0xFF5D: "QCC", 0xFF5E: "RGN", 0xFF5F: "POC", 0xFF60: "PPM", 0xFF61: "PPT", 0xFF63: "CRG",
           0xFF64: "COM", 0xFF90: "SOT", 0xFF91: "SOP", 0xFF92: "EPH", 0xFF93: "SOD", 0xFFD9: "EOC"}
PROG = ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]
def u16(d, o): return struct.unpack(">H", d[o:o + 2])[0]
def u32(d, o): return struct.unpack(">I", d[o:o + 4])[0]


def boxes(d, pos=0, end=None, depth=0):
    end = len(d) if end is None else end; out = []
    while pos + 8 <= end:
        ln, typ = u32(d, pos), d[pos + 4:pos + 8].decode("latin1"); hdr = 8
        if ln == 1: ln = struct.unpack(">Q", d[pos + 8:pos + 16])[0]; hdr = 16
        if ln == 0: ln = end - pos
        out.append((depth, typ, pos, ln, hdr))
        if typ in ("jp2h", "res "): out += boxes(d, pos + hdr, pos + ln, depth + 1)
        pos += ln
    return out


def cod_style(sp, scod, ncomp_hdr=False):
    lv, xcb, ycb, sty, tr = sp[0], (sp[1] & 15) + 2, (sp[2] & 15) + 2, sp[3], sp[4]
    s = f"{lv} decomposition levels, code-blocks {1 << xcb}x{1 << ycb}, {'5/3 reversible' if tr == 1 else '9/7 irreversible' if tr == 0 else f'transform {tr}'}"
    flags = [n for b, n in ((1, "bypass"), (2, "reset"), (4, "termall"), (8, "vcausal"), (16, "predictable"), (32, "segsym"), (64, "HT"), (128, "mixed HT")) if sty & b]
    s += ", code-block style " + (f"0x{sty:02x} ({', '.join(flags)})" if sty else "0x00 (default Part 1)")
    if scod & 1:
        pp = sp[5:5 + lv + 1]; s += ", precincts (low->high res) " + " ".join(f"{1 << (p & 15)}x{1 << (p >> 4)}" for p in pp)
    else: s += ", maximal precincts"
    return s


def main(path):
    d = pathlib.Path(path).read_bytes(); cs = 0
    print(f"{path}: {len(d)} B")
    if d[4:8] == b"jP  ":
        print("JP2 boxes:")
        for depth, typ, pos, ln, hdr in boxes(d):
            extra = ""; body = d[pos + hdr:pos + ln]
            if typ == "ftyp": extra = f" brand {body[:4].decode('latin1')!r}, compatible {[body[i:i + 4].decode('latin1') for i in range(8, len(body), 4)]}"
            if typ == "ihdr": extra = f" {u32(body, 4)} x {u32(body, 0)} px (w x h), {u16(body, 8)} component(s), {(body[10] & 0x7f) + 1} bit {'signed' if body[10] & 0x80 else 'unsigned'}, compression {body[11]}"
            if typ == "colr": extra = f" method {body[0]}, " + (f"enumerated colourspace {u32(body, 3)}" if body[0] == 1 else f"ICC profile {len(body) - 3} B")
            if typ == "jp2c": cs = pos + hdr
            print("  " * (depth + 1) + f"{typ!r} {ln} B" + extra)
    d = d[cs:]
    pos = 2; assert d[:2] == b"\xff\x4f", "no SOC"; counts = collections.Counter(); main_hdr = True; tiles = set(); tparts = 0; sop = eph = 0
    print("codestream main header:")
    while pos < len(d) - 1:
        m = u16(d, pos); name = MARKERS.get(m, f"0x{m:04X}")
        if m == 0xFFD9: break
        if m == 0xFF93:                                         # SOD: skip the tile-part body using the SOT's Psot
            pos = tp_end; continue
        L = u16(d, pos + 2); body = d[pos + 4:pos + 2 + L]; counts[name] += 1
        if m == 0xFF90:
            main_hdr = False; isot, psot, tpn, ntp = u16(body, 0), u32(body, 2), body[6], body[7]; tiles.add(isot); tparts += 1; tp_end = pos + psot if psot else len(d) - 2
        elif m == 0xFF51:
            rs, X, Y, XO, YO, XT, YT, XTO, YTO, C = u16(body, 0), u32(body, 2), u32(body, 6), u32(body, 10), u32(body, 14), u32(body, 18), u32(body, 22), u32(body, 26), u32(body, 30), u16(body, 34)
            comps = [(((body[36 + 3 * i]) & 0x7f) + 1, "signed" if body[36 + 3 * i] & 0x80 else "unsigned", body[37 + 3 * i], body[38 + 3 * i]) for i in range(C)]
            print(f"  SIZ: image {X - XO} x {Y - YO} (w x h), {C} component(s) {comps[:4]} (bits, sign, dx, dy); tile {XT} x {YT} ({-(-X // XT) * -(-Y // YT) if XT else '?'} tiles); Rsiz 0x{rs:04x}")
        elif m == 0xFF50: print(f"  CAP: Pcap 0x{u32(body, 0):08x} " + ("(Part 15 / HTJ2K)" if u32(body, 0) & (1 << 17) else "") + f", Ccap {[hex(u16(body, 4 + 2 * i)) for i in range((len(body) - 4) // 2)]}")
        elif m == 0xFF52:
            scod = body[0]; print(f"  COD: progression {PROG[body[1]]}, {u16(body, 2)} layer(s), MCT {body[4]}, SOP {'yes' if scod & 2 else 'no'}, EPH {'yes' if scod & 4 else 'no'}, " + cod_style(body[5:], scod))
        elif m == 0xFF53: print(f"  COC (component {body[0]}): " + cod_style(body[2:], body[1]))
        elif m in (0xFF5C, 0xFF5D):
            off = 0 if m == 0xFF5C else 1; sq = body[off]; st = sq & 0x1f; gb = sq >> 5; esz = 1 if st == 0 else 2
            n = (len(body) - off - 1) // esz; print(f"  {name}: {['no quantization (reversible)', 'scalar derived', 'scalar expounded'][st]}, {gb} guard bits, {n} subband entr{'y' if n == 1 else 'ies'}")
        elif m == 0xFF64: print(f"  COM: {body[2:].decode('latin1', 'replace')[:100]!r}")
        elif m == 0xFF55: print(f"  TLM: {(len(body) - 2) // ((2 if (body[1] >> 4) & 3 == 2 else (body[1] >> 4) & 3) + (4 if body[1] & 0x40 else 2))} tile-part length entries")
        pos += 2 + L
    print(f"  marker counts: {dict(counts)}; {len(tiles)} tile(s), {tparts} tile-part(s)")


main(sys.argv[1])
