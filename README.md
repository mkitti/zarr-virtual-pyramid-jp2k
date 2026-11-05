# Virtual scale pyramids over (HT)JPEG 2000 for Zarr

Experiments in serving a multiscale pyramid, and quality/band variants of it, from **one set of JPEG 2000 /
HTJ2K codestreams** without re-encoding: each Zarr chunk is a [refkv](https://github.com/d-v-b/refkv) reference
made by concatenating literal (rewritten) header bytes with byte ranges of the original file.

Test data: a 512 x 512 x 16 uint16 crop of OpenOrganelle `jrc_hela-2` (`em/fibsem-uint16`, s0..s3), each z-plane
encoded as its own 3-level reversible 5/3 codestream.

## Findings

**Where the bytes are**
- The packet is the smallest contiguous unit (one per layer, resolution, precinct). With RPCL/RLCP and one layer,
  resolutions 0..r are a contiguous byte prefix, so **Zarr level `s(n)` = JPEG 2000 resolution `L - n`**
  (`s0` is full resolution).
- Inside a packet the body is band-major (HL, LH, HH; LL at resolution 0), so **one sub-band of one precinct is one
  contiguous range**. Finding it needs the packet header (tag trees, pass counts, Lblock, HT segment lengths);
  `scripts/packets.py` parses it.
- TLM/PLT only give tile-part/packet boundaries, not bands. OpenJPH writes TLM but no PLT; ndic writes PLT.

**Concatenation**
- refkv references can be a list of parts (inline literal bytes and source ranges, from several URLs). kerchunk /
  fsspec `ReferenceFileSystem` only support `[url]` and `[url, offset, length]`.
- A valid reduced stream for level n is: rewritten main header (SIZ dims, COD levels = r and precinct list trimmed,
  QCD trimmed to the first 1+3r subbands, TLM dropped), tile-part headers (Psot, PLT recomputed), the packet range for
  resolutions 0..r, and a literal `FFD9`.

**What was verified** (OpenJPH 0.32 and OpenJPEG 2.5.4 as decoders, plus independent references)
| Case | Result |
|---|---|
| ndic- and OpenJPH-encoded planes, all 16, levels s0-s3 through zarr-python | s0 lossless; s1-s3 identical between encoders; level reads 100 / 26 / 7 / 1.8 % of the bytes |
| Lossy 9/7 (expounded QCD) | matches `ojph_expand -skip_res n` exactly under OpenJPH; OpenJPEG within 1 gray level (decoder rounding, also on the original file) |
| 256x256 tiles, one standalone chunk per tile per level, with and without precincts | matches OpenJPH's reduced decode |
| Packet-header parser, 4 file layouts | every packet consumed exactly; PLT lengths match; all HT cleanup segments end in a valid Scup |
| Drop one sub-band (any resolution) by rewriting packet headers | matches an independent numpy 5/3 wavelet exactly; 24.6 % of the file for the finest band |
| Drop precincts (empty 1-byte packets) / band-by-precinct | ROI interior identical to the original; band/precinct geometry matches numpy |
| 4-layer classic J2K, layers 1..3, with reduced resolution | equals `opj_decompress -l K -r n`; references cover 4.0 / 8.1 / 16.2 % of the bytes (transferred: see below) |
| Quality groups (`q1`-`q3`) and `nohh` group as OME-Zarr 0.5 pyramids | validate with `ome-zarr-models`; read through zarr-python |

**Limits and untested**
- HT code-blocks with refinement passes or split across layers: no available encoder produces them (ndic and OpenJPH
  write cleanup passes only, OpenJPEG 2.5.4 cannot encode HT). The parser asserts on them; the HT segment rule there is
  from reading the spec, not tested.
- One component, reversible 5/3 or expounded-QCD 9/7; scalar-derived QCD, PCRL/CPRL, SOP/EPH rebuilding unsupported.
- The wavelet low-pass pyramid is not OpenOrganelle's 3D-mean pyramid (mean abs diff roughly 3-4 % of the range at s3);
  z is not downsampled; no translation is written (a 5/3 LL sample sits at full-resolution sample 2n).
- Byte shares in the tables are computed from range lengths. `scripts/measure_io.py` measures what zarr-python
  fetches from the S3 archives (`refkv.Stats`): pyramid reads match (s3/s2/s1/s0 = 1.8 / 6.9 / 26.3 / 100 %) and so does
  `nohh/s0` (75.4 %), but quality-layer reads transfer more than the references cover: `q1/q2/q3` at s0 fetch
  9.0 / 28.7 / 36.0 % instead of 4.0 / 8.1 / 16.1 %, because refkv merges ranges less than 64 KiB apart (`MERGE_GAP`) and
  a plane's early layers sit between larger later-layer packets. With `MERGE_GAP` of 4096 or 0 the transfer equals the
  references, in 64 requests instead of 16-32. No wall-clock benchmark has been done.
- ndic (`nd-image-codecs`) is an independent HT implementation and strict: single tile, reversible 5/3, PLT required.

## Layout

| File | Purpose |
|---|---|
| `scripts/j2kvirt.py` | marker parsing; reduced-resolution codestreams as refkv `Concat`s |
| `scripts/packets.py` | packet-header parser / writer; per-band, per-precinct, per-layer ranges; `assemble()` |
| `scripts/dwt53.py` | independent numpy reversible 5/3 reference |
| `scripts/util.py` | PGM I/O, OpenJPH wrappers, Zarr v3 / OME-NGFF 0.5 metadata |
| `scripts/fetch_crop.py` | download and crop `jrc_hela-2` from the public COSEM bucket |
| `scripts/encode_cmp.py`, `encode_ojph.py` | encode planes with ndic / OpenJPH |
| `scripts/build_pyramid.py` | step 1: pyramid stores for both encoders, OME validation |
| `scripts/lossy_test.py`, `tiles_test.py` | lossy 9/7; tiles and precincts |
| `scripts/step4_test.py`, `step5_test.py` | parser, band/precinct/layer edits, references |
| `scripts/step6_variants.py` | quality-layer and band-drop variant stores |

## Running

Environment: [pixi](https://pixi.sh) (`pixi.toml`). `PYTHONPATH` in `pixi.toml` points at a local refkv checkout
(with its compiled `refkv._core`); edit it for your machine. `imagecodecs-zarr` is installed from PyPI (not on
conda-forge). The ndic encoder is built from [fideus-labs/nd-image-codecs](https://github.com/fideus-labs/nd-image-codecs)
(`cargo build -p ndic-cli`). Scripts take a work directory holding `hela2_crop/`, `enc/` (ndic), `enc_ojph/`, etc.

```
pixi run python scripts/fetch_crop.py <work>/hela2_crop
pixi run python scripts/build_pyramid.py <work>
pixi run python scripts/step6_variants.py <work>
```

The encoded codestreams used here are public at
[https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/](https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/)
(`enc_ojph`, `enc_opj`, `enc_tiles_precincts`, `enc_ndic`, with a
[`MANIFEST.sha256`](https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/MANIFEST.sha256)).

## Reading the remote archives with zarr-python

Archives (refkv `.vzip` stores whose chunks are byte-range references into the codestreams above), under
`https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/`:

| Archive | Contents |
|---|---|
| [`hela2_pyramid_ojph.vzip`](https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/hela2_pyramid_ojph.vzip) | OME-Zarr pyramid `s0`..`s3` from the OpenJPH files |
| [`hela2_pyramid_ndic.vzip`](https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/hela2_pyramid_ndic.vzip) | the same pyramid from the ndic files |
| [`hela2_quality.vzip`](https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/hela2_quality.vzip) | full quality at the root, plus groups `q1`, `q2`, `q3` (first 1-3 quality layers) |
| [`hela2_nohh.vzip`](https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/hela2_nohh.vzip) | full quality at the root, plus group `nohh` (finest HH band removed) |

```python
import zarr
import refkv.zarr as rz

url = "https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/hela2_pyramid_ojph.vzip"
g = zarr.open_group(rz.open(url), mode="r")

g["s3"].shape                     # (16, 64, 64)    smallest level, reads about 2% of the bytes
g["s0"].shape                     # (16, 512, 512)  full resolution, lossless
plane = g["s1"][0]                # one z-plane at half resolution (256 x 256, uint16)
```

Quality and band-drop variants are groups inside the other archives:

```python
base = "https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/archives/"
q = zarr.open_group(rz.open(base + "hela2_quality.vzip"), mode="r")
q["q1/s0"][0]     # first quality layer only, PSNR 39 dB; references 4% of the bytes, 9% transferred (see Limits)
q["s0"][0]        # full quality, lossless
n = zarr.open_group(rz.open(base + "hela2_nohh.vzip"), mode="r")
n["nohh/s0"][0]   # finest HH band removed, about 75% of the bytes
```

Needs `refkv` importable (see `pixi.toml`; plain `zarr.open("https://....vzip")` does not work, since zarr-python has
no vzip store), plus `imagecodecs-zarr` for the `imagecodecs_htj2k` / `imagecodecs_jpeg2k` codecs and `ml_dtypes`.
The archives were built with `--base-url https://s3.janelia.org/scicompsoft/kittisopikulm/hela2/` (sources pinned by
size and last-modified time). Not tested: xarray, ome-zarr-py, or any viewer (Neuroglancer has no JPEG 2000 decoder).

## License

3-clause BSD, copyright Howard Hughes Medical Institute (the [Janelia open-source license](https://www.janelia.org/open-science/software-licensing)). See `LICENSE`.
