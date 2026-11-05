import sys, subprocess, pathlib, numpy as np
SP=pathlib.Path(sys.argv[1]); out=SP/"enc_ojph"; out.mkdir(exist_ok=True)
s0=np.load(SP/"hela2_crop/s0.npy")
for z in range(s0.shape[0]):
    pgm=out/f"p{z:02d}.pgm"
    with open(pgm,"wb") as f:
        f.write(b"P5\n512 512\n65535\n"); f.write(s0[z].astype(">u2").tobytes())
    subprocess.run(["ojph_compress","-i",str(pgm),"-o",str(out/f"p{z:02d}.j2c"),"-num_decomps","3","-reversible","true",
                    "-prog_order","RPCL","-tileparts","R","-tlm_marker","true"],check=True,capture_output=True)
    pgm.unlink()
print("encoded",s0.shape[0],"planes ->",sum(p.stat().st_size for p in out.glob("*.j2c")),"bytes")
