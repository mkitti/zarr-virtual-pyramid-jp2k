import subprocess, numpy as np, sys, os
N=sys.argv[1]; D=sys.argv[2]
s=[np.load(f"{D}/s{i}.npy") for i in range(4)]
def run(*a): return subprocess.run([N,*a],check=True,capture_output=True,text=True).stdout
tot=0
for z in range(16):
    p=f"{D}/../enc/p{z:02d}"
    np.ascontiguousarray(s[0][z]).astype("<u2").tofile(p+".raw")
    run("compress","-i",p+".raw","-o",p+".j2c","--raw-size","512x512","--raw-dtype","u16","--levels","3")
    tot+=os.path.getsize(p+".j2c")
    # lossless check
    run("expand","-i",p+".j2c","-o",p+".dec.raw")
    assert np.array_equal(np.fromfile(p+".dec.raw","<u2").reshape(512,512),s[0][z]), z
print("16 planes lossless OK; raw",16*512*512*2,"bytes -> j2c",tot)
for z in (0,1):
    p=f"{D}/../enc/p{z:02d}"
    for r in range(4):  # resolution r  <-> pyramid level 3-r
        lvl=3-r; w=512>>lvl
        run("expand","-i",p+".j2c","-o",p+f".r{r}.raw","--resolution",str(r))
        a=np.fromfile(p+f".r{r}.raw","<u2")
        if a.size!=w*w: print("size mismatch",r,a.size,w*w); continue
        a=a.reshape(w,w).astype(float)
        ref=s[lvl][z>>lvl].astype(float)
        k=1<<lvl; mean=s[0][z].astype(float).reshape(w,k,w,k).mean((1,3))
        print(f"plane{z} res{r}->s{lvl} {w}x{w}: MAE vs OO s{lvl}={abs(a-ref).mean():.1f} (OO range {ref.min():.0f}-{ref.max():.0f}); MAE of in-plane mean vs OO={abs(mean-ref).mean():.1f}; MAE LL vs in-plane mean={abs(a-mean).mean():.1f}")
