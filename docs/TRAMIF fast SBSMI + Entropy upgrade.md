# TRAMIF: Fast SBSMI + Entropy (experimental, lossless-semantics upgrade)

**Source:** Refactor of the uploaded TRAMIF distributed V2 generation implementation. These modules were **not pushed** to GitHub and **not run against private BODMAS binaries**. All included tests are synthetic. The archive includes **copies of the original V2 Entropy reference modules** for independent comparison, so tests work even if you have only the current group GitHub main branch. Integrating optimized imports into the distributed V2 generator still requires the earlier V2 patch. It leaves the existing V1 group generator intact.

## 1. Contents

```
TRAMIF_fast_views_patch/
├── src/preprocessing/
│   ├── sbsmi_vectorized.py          # vectorized 6-bit SBSMI
│   ├── entropy_core.py             # V2 reference dependency, copy only if absent
│   ├── entropy_stream.py           # V2 reference dependency, copy only if absent
│   ├── entropy_image_stream.py     # V2 reference dependency, copy only if absent
│   ├── entropy_stream_fast.py       # batched, streaming 256-byte entropy windows
│   └── entropy_image_stream_fast.py # same V2 mask-aware fractional-area resizing
├── scripts/
│   └── compare_fast_views.py        # original V2 vs optimized; file or ZIP
├── tests/
│   └── test_fast_views.py           # synthetic regression suite
└── README.md
```

The **reference implementations are not modified**. Copies of the three Entropy reference modules are bundled for convenient installation; if your V2 repo already contains these files, compare them before overwriting:

- `src/preprocessing/implementation_sbsmi.py`
- `src/preprocessing/entropy_stream.py`
- `src/preprocessing/entropy_core.py`
- `src/preprocessing/entropy_image_stream.py`

Only the `src/preprocessing/implementation_sbsmi.py` reference is required from the existing group GitHub to run these **fast-view regression tests**. To activate the fast imports in the **unified distributed V2 generator**, first install the earlier distributed V2 patch (including `src/preprocessing/raw_byte_stream.py` and `scripts/generate_multiview_dataset_v2.py`).

## 2. Install and run tests (Windows PowerShell, repository root)

Create a fresh branch; do not overwrite the original reference modules:

```powershell
git switch main
git pull
git switch -c feature/fast-sbsmi-entropy
```

Copy the supplied `src/`, `scripts/`, and `tests/` files into matching repo folders; **do not blindly overwrite any already modified V2 reference modules**. If the group has not yet installed V2, the archive provides the three necessary Entropy reference files. Run:

```powershell
python -m pytest -q tests/test_fast_views.py
# Only after installing the separate distributed V2 patch:
python -m pytest -q tests/test_distributed_multiview_v2.py
```

**Actual tests executed in this environment:** 224 new tests passed, plus the 12 existing V2 integration tests after activating fast imports (236 total). These are synthetic tests, not real-malware benchmarks or model accuracy measurements.

## 3. Compare before switching: same binary, original vs optimized

**Extracted file** (choose a small **training-period** sample initially):

```powershell
python -m scripts.compare_fast_views `
  --file "E:\BODMAS_GW\altered\YOUR_SHA.exe" `
  --report "reports\speed_compare_train.json"
```

**ZIP member**, without extraction:

```powershell
python -m scripts.compare_fast_views `
  --zip "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --member "altered/YOUR_SHA.exe" `
  --report "reports\speed_compare_train.json"
```

Replace `YOUR_SHA` with an actual **original BODMAS sample identifier** from a training manifest. The printed `disarmed_content_sha256` can differ from the original sample ID; this is expected for disarmed BODMAS binaries.

The command reads the *same source content* independently through four operations (old and fast SBSMI; old and fast Entropy), fingerprinting the data each implementation sees to rule out input mismatch. It reports:

- SBSMI: number of different **uint8** pixels. **Required: zero**.
- Entropy: max and mean absolute difference between pre-save **float64 [0,1]** arrays. Required `np.allclose(ref, fast, atol=1e-6, rtol=0)`; synthetic tests pass a much tighter `1e-12` check.
- Entropy: number of differing stored float32 values, and difference in *optional* quantized preview pixels. The floating-point reductions may differ in their least-significant bits; these diagnostics do **not** automatically mean the entropy equation has changed.
- Separate function times, excluding model inference, ZIP indexing, and image-saving overhead.

**This comparison is against your friend's / distributed V2 reference.** The old GitHub **V1** Raw-byte and Entropy use a *different* resize operator (`torch.nn.functional.interpolate(mode='area')`) and may substantially differ from V2 geometric fractional-area resizing. Do not require V1==V2, or claim the optimization changed those semantics: the documented V2 construction already changed them.

### Check the generated files as well

The fast SBSMI must remain exactly the same 64x64 uint8 image after PNG save-load; Entropy V2 must stay a 64x64 float32 NPY in [0,1] after NPY save-load. The included roundtrip test checks this. If you already generated a V2 image for the same SHA and preprocessing version, load it and compare it with a new optimized output (SBSMI `np.array_equal`; Entropy `np.allclose` with explicit atol) **after verifying source identity**. Do not rely only on screenshots or image visualization.

Before any full build, run this comparison on a small sample of **historical train files** spanning tiny, medium, and large binaries, and try a few different `--chunk-size` values. Do not use April–September future-test data to choose a faster or altered implementation.

## 4. Activate fast imports in the distributed V2 generator

In `scripts/generate_multiview_dataset_v2.py`, change **only** these two imports:

```diff
-from src.preprocessing.entropy_image_stream import stream_to_entropy_image
+from src.preprocessing.entropy_image_stream_fast import stream_to_entropy_image_fast as stream_to_entropy_image
 from src.preprocessing.raw_byte_stream import stream_to_raw_byte
-from src.preprocessing.implementation_sbsmi import stream_to_sbsmi
+from src.preprocessing.sbsmi_vectorized import stream_to_sbsmi_fast as stream_to_sbsmi
```

Add the **three new modules** to `SOURCE_MODULES` alongside its existing entries:

```python
    "src/preprocessing/entropy_stream_fast.py",
    "src/preprocessing/entropy_image_stream_fast.py",
    "src/preprocessing/sbsmi_vectorized.py",
```

**Do not delete the old reference modules.** The fast entropy module imports the old reference for nonstandard window settings; SBSMI falls back to the old reference when `bit_num != 6`.

Use a **new output directory**, e.g. `E:\TRAMIF_jobs\entropy_fast_v2`, even when the output is numerically equivalent. The generator's source-code fingerprint will change, so it must not mix cache records from old and new preprocessing implementations. Future evaluation still requires the frozen historical configuration and the same primary experimental protocol.

The three-laptop distribution, split/month folders, output dtypes, per-file reports, ZIP/folder input, SHA-based matching, resume checks, and merge procedure do **not** change.

## 5. Actual local synthetic speed test (not a real-BODMAS or end-to-end benchmark)

Five timed repetitions per function; median times for a 1 MiB NumPy-generated binary, in-memory stream; same execution environment:

| View | Original V2 (median) | Fast prototype (median) | Speedup |
|---|---:|---:|---:|
| Entropy | 230.02 ms | 14.57 ms | approx. 15.8x |
| SBSMI | 428.13 ms | 3.75 ms | approx. 114.2x |

Do **not** claim 15.8x/114.2x in the paper or as real-machine generation throughput until tested with actual BODMAS workloads and a reproducible benchmark. ZIP decompression, source hashing, filesystem, logging, and image I/O can dominate end-to-end time.

## 6. Research notes

- SBSMI: exact 6-bit MSB-first states, high-order zero tail, transition row probabilities, integer-floor `floor(255 * P)` output. No augmentation or interpretation change. Framework **Sections 4.3–4.4**.
- Entropy: exact 256-byte window, stride 128, overlapping normalized Shannon entropies averaged at each byte offset, one final window reaching EOF. The V2 fractional-area resize remains unchanged. Framework **Sections 4.2–4.4**.
- A speedup is not an accuracy finding; **no model evaluation has been run** as part of this patch.
- If a primary model's preprocessing has already been frozen, do not silently replace it. Prove matching output, record code change and configuration hash, or run as a separately versioned experiment (§7).
- The future test remains **Apr–Sep 2020**, frozen; do not use it for optimization decisions (§7 / §11.2).
- Respect restricted BODMAS access conditions; do not publish/distribute malware-derived samples or byte arrays without authorization.
