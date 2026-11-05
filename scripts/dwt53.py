"""Independent reference: reversible 5/3 wavelet (JPEG 2000 Part 1, vertical then horizontal), numpy only.

pyramid() gives LL_L and per-level detail bands; det[0] is decomposition level 1, i.e. resolution L.
recon() inverts down to a chosen reduction level, optionally zeroing selected bands/regions first.
"""
import numpy as np


def f1(x):
    xe, xo = x[..., 0::2], x[..., 1::2]; xen = np.concatenate([xe[..., 1:], xe[..., -1:]], -1)
    H = xo - ((xe + xen) >> 1); Hp = np.concatenate([H[..., :1], H[..., :-1]], -1); return xe + ((Hp + H + 2) >> 2), H


def i1(L, H):
    Hp = np.concatenate([H[..., :1], H[..., :-1]], -1); xe = L - ((Hp + H + 2) >> 2)
    xen = np.concatenate([xe[..., 1:], xe[..., -1:]], -1); out = np.empty(L.shape[:-1] + (2 * L.shape[-1],), np.int64)
    out[..., 0::2] = xe; out[..., 1::2] = H + ((xe + xen) >> 1); return out


def fwd2(a):
    Lv, Hv = f1(a.T); Lv, Hv = Lv.T, Hv.T; LL, HL = f1(Lv); LH, HH = f1(Hv); return LL, dict(HL=HL, LH=LH, HH=HH)


def inv2(LL, d): return i1(i1(LL, d["HL"]).T, i1(d["LH"], d["HH"]).T).T


def pyramid(img, L=3, shift=32768):
    ll, det = img.astype(np.int64) - shift, []
    for _ in range(L): ll, d = fwd2(ll); det.append(d)
    return ll, det


def recon(ll, det, zero=None, level=0, shift=32768, bits=16):
    """Image at reduction `level` (0 = full). zero = (resolution, band) | callable(resolution, band, array) -> array."""
    L = len(det)
    for lvl in range(L - 1, level - 1, -1):
        res = L - lvl; d = {}
        for k, v in det[lvl].items():
            if callable(zero): d[k] = zero(res, k, v)
            else: d[k] = np.zeros_like(v) if zero == (res, k) else v
        ll = inv2(ll, d)
    return np.clip(ll + shift, 0, (1 << bits) - 1).astype(np.uint16)
