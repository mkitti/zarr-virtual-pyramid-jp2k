"""Step 2: irreversible 9/7 (lossy) HTJ2K -> virtual reduced resolutions.

Reference for each level n is OpenJPH's own reduced decode (`ojph_expand -skip_res n`) of the full file.
usage: pixi run python scripts/lossy_test.py <workdir> [qstep ...]
"""
import sys, pathlib, tempfile, numpy as np, refkv
from refkv.pb import Source
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import j2kvirt, util

W = pathlib.Path(sys.argv[1]); qsteps = [float(x) for x in sys.argv[2:]] or [0.0005, 0.004]
orig = np.load(W / "hela2_crop/s0.npy"); out = W / "enc_lossy"; out.mkdir(exist_ok=True)
for q in qsteps:
    for z in (0, 7):
        f = out / f"q{q}_p{z:02d}.j2c"
        util.ojph_compress(orig[z], f, "-num_decomps", 3, "-reversible", "false", "-qstep", q, "-prog_order", "RPCL",
                           "-tileparts", "R", "-tlm_marker", "true", work=out)
        d = f.read_bytes(); j = j2kvirt.J2K(d)
        qcd = dict(j.main)[j2kvirt.QCD]
        rec = util.ojph_expand(f, 0, out)
        psnr = 10 * np.log10(65535 ** 2 / np.mean((rec.astype(float) - orig[z]) ** 2))
        print(f"qstep={q} plane {z}: {len(d)} B ({len(d) / orig[z].nbytes * 100:.1f}% of raw), QCD style {qcd[4] & 0x1f}, PSNR {psnr:.1f} dB")
        try:
            store = refkv.ReferenceStore({f"n{n}": j.chunk(0, j.L - n)[0] for n in range(j.L + 1)},
                                         [Source(url=f.resolve().as_uri())])
        except NotImplementedError as e:
            print("  not supported:", e); continue
        for n in range(j.L + 1):
            b = store.get(f"n{n}"); ref = util.ojph_expand(f, n, out)
            for name, a in util.decode_both(b).items():
                if isinstance(a, str): print(f"  s{n} {name}: {a}"); continue
                same = a.shape == ref.shape and np.array_equal(a, ref)
                diff = "" if same else f" (max |diff| {np.abs(a.astype(int) - ref).max() if a.shape == ref.shape else 'shape ' + str(a.shape)})"
                print(f"  s{n} {a.shape} {name} == ojph_expand -skip_res {n}: {same}{diff}")
