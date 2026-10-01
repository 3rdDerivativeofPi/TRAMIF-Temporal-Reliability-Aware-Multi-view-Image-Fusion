# Manual: `generate_sbsmi_10_per_family.py`

This document explains how to use and understand the small real-data SBSMI pilot script.

The script is meant to do one simple job:

```text
10 malware families
x
10 real BODMAS binaries per family
=
100 requested SBSMI images
```

Important:

> `100 images` is the target of the pilot, not a measured result until the script actually finishes successfully on the real BODMAS archive.

---

# 1. What this script is for

The unit tests in `test_sbsmi.py` use tiny synthetic bytes and fake ZIP files.

Those tests answer:

> "Does our SBSMI code behave correctly on controlled examples?"

This script answers a different question:

> "Can our SBSMI implementation actually read real BODMAS binaries from the real ZIP and generate images for a small real sample?"

So this is a **real-data smoke test** before large-scale generation.

---

# 2. What the script does

The script follows this pipeline:

```text
bodmas_metadata.csv
        +
BODMAS ZIP archive
        |
        v
find samples from the 10 chosen families
        |
        v
check which binaries really exist in the ZIP
        |
        v
select 10 samples from each family
        |
        v
save selection manifest
        |
        v
call generate_sbsmi_dataset()
        |
        v
generate 100 requested SBSMI PNGs
        |
        v
save generation log
        |
        v
print success / skipped / failed summary
```

The script does **not** implement SBSMI again.

It reuses:

```python
generate_sbsmi_dataset()
```

from the existing SBSMI implementation.

That is intentional: there should be only one SBSMI algorithm in the project.

---

# 3. The 10 families

The pilot uses these families:

```text
ceeinject
drolnux
gandcrab
mira
musecador
sfone
sillyp2p
small
upatre
wabot
```

The goal is:

```text
10 samples x 10 families = 100 samples
```

---

# 4. Where to put the script

Recommended location:

```text
scripts/
    generate_sbsmi_10_per_family.py
```

Example repository layout:

```text
project/
|
+-- src/
|   +-- preprocessing/
|       +-- implementation_sbsmi.py
|
+-- scripts/
|   +-- generate_sbsmi_10_per_family.py
|
+-- data/
|   +-- processed/
|
+-- tests/
```

---

# 5. What you MUST edit before running

At the top of the script, find:

```python
ZIP_PATH = Path(
    r"E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip"
)

METADATA_PATH = Path(
    r"E:\BODMAS_GW\bodmas_metadata.csv"
)
```

Change these to the actual paths on your computer.

Example:

```python
ZIP_PATH = Path(
    r"D:\BODMAS\BODMAS_disarmed_malware_binaries.zip"
)

METADATA_PATH = Path(
    r"D:\BODMAS\bodmas_metadata.csv"
)
```

Do not copy somebody else's drive letter blindly.

---

# 6. Import path: check this once

The script imports:

```python
from src.preprocessing.implementation_sbsmi import (
    generate_sbsmi_dataset,
)
```

This is correct only if your function is actually inside:

```text
src/preprocessing/implementation_sbsmi.py
```

If your project instead uses:

```text
src/preprocessing/sbsmi.py
```

change the import to:

```python
from src.preprocessing.sbsmi import (
    generate_sbsmi_dataset,
)
```

Do not keep both implementations.

---

# 7. Output location

The script uses:

```python
OUTPUT_ROOT = Path(
    "data/processed/"
    "malcsbsv_reproduction/"
    "sbsmi_pilot_10_per_family"
)
```

This means the generated PNGs should go under:

```text
data/
└── processed/
    └── malcsbsv_reproduction/
        └── sbsmi_pilot_10_per_family/
```

Inside that folder, images are organized by family.

Example:

```text
sbsmi_pilot_10_per_family/
|
+-- mira/
|   +-- <sha1>.png
|   +-- <sha2>.png
|   +-- ...
|
+-- small/
|   +-- <sha>.png
|
+-- wabot/
    +-- <sha>.png
```

---

# 8. Why the script also creates a manifest

The script creates:

```text
sbsmi_pilot_10_per_family_manifest.csv
```

This records exactly which 100 samples were selected.

Columns:

```text
family
sha
path
```

Example row:

```text
mira,abc123...,altered/abc123....exe
```

Why keep this?

Because later we need to know:

> "Which exact files did we use in the pilot?"

Without a manifest, everybody may accidentally select a different set.

---

# 9. Why the script also creates a generation log

The script creates:

```text
sbsmi_pilot_10_per_family_log.csv
```

This records what happened to each selected sample.

Useful columns:

```text
family
sha
path
output_path
status
error_type
error_message
```

Possible status values:

```text
success
skipped
failed
```

Meaning:

```text
success
= image was generated now

skipped
= image already existed and overwrite=False

failed
= something went wrong for this sample
```

---

# 10. `RANDOM_SEED = 42`

The script contains:

```python
RANDOM_SEED = 42
```

This is used for reproducible random selection.

It means:

> if two teammates use the same metadata, ZIP contents, family list, and seed, they should select the same samples.

Why sort before sampling?

Because filesystem or CSV ordering should not accidentally change the random result.

The script sorts candidates by SHA first, then samples with the fixed seed.

Do not casually change the seed after one teammate has already generated the pilot.

---

# 11. `find_available_candidates()`

This function answers:

> "Which samples from our 10 families actually have binaries inside the ZIP?"

It creates a dictionary like:

```python
{
    "mira": [
        {...},
        {...},
    ],

    "small": [
        {...},
        {...},
    ],
}
```

For each metadata row, it checks:

1. Is the family one of our 10?
2. Does the row have a SHA?
3. Does the expected ZIP member exist?

Only then is the sample accepted as a candidate.

---

# 12. Why we check the ZIP before selecting samples

Metadata may mention a sample, but that does not automatically guarantee the expected member path is available in the archive you currently have.

So we do not:

```text
metadata
-> immediately select sample
```

We do:

```text
metadata
-> construct expected ZIP path
-> confirm ZIP member exists
-> candidate
```

This prevents selecting samples that cannot actually be generated.

---

# 13. The expected ZIP path

The script currently assumes BODMAS members look like:

```text
altered/<sha>.exe
```

So if:

```text
sha = abc123
```

it checks for:

```text
altered/abc123.exe
```

If your archive uses a different internal structure, this line must be changed.

The relevant code is:

```python
member_path = (
    f"altered/{sha}.exe"
)
```

Do not change it unless you have checked the actual ZIP member names.

---

# 14. Does the script extract the whole ZIP?

No.

It does **not** unzip the whole archive.

First it reads the ZIP directory:

```python
zip_file.namelist()
```

Later, `generate_sbsmi_dataset()` opens only the selected ZIP members.

Conceptually:

```text
huge ZIP
 |
 +-- selected file 1 -> stream -> SBSMI
 |
 +-- selected file 2 -> stream -> SBSMI
 |
 +-- ...
```

No full extraction is required.

---

# 15. `choose_samples()`

This function selects:

```text
10 samples per family
```

It first prints how many valid candidates are available.

Example:

```text
mira        : 532 available
small       : 1842 available
...
```

Those numbers are examples only.

Then it checks:

```python
if available_count < SAMPLES_PER_FAMILY:
```

If a family has fewer than 10 usable files, the script stops.

Why?

Because otherwise we would silently end up with an incomplete pilot such as:

```text
10
10
10
7
10
...
```

and somebody might incorrectly think every family had 10 samples.

---

# 16. `SAMPLES_PER_FAMILY`

The setting is:

```python
SAMPLES_PER_FAMILY = 10
```

That is why the intended total is:

```text
10 families x 10 = 100
```

If you later want a different smoke test, for example:

```python
SAMPLES_PER_FAMILY = 5
```

then the target becomes:

```text
10 x 5 = 50
```

For this current team task, keep it at 10.

---

# 17. `save_selection_manifest()`

This function writes the selected sample list to CSV.

It should run **before** SBSMI generation.

That is useful because even if generation later crashes, we still know what was selected.

---

# 18. `generate_sbsmi_dataset()`

This is where the actual image generation happens.

The pilot calls something like:

```python
records = generate_sbsmi_dataset(
    source_path=ZIP_PATH,
    output_root=OUTPUT_ROOT,
    source_mode="zip",
    samples=selected,
    bit_num=6,
    chunk_size=65536,
    overwrite=False,
    max_files=100,
)
```

Meaning:

```text
source_path
= real BODMAS ZIP

source_mode
= ZIP mode

samples
= our chosen 100 samples

bit_num
= 6-bit SBSMI

chunk_size
= read 64 KB at a time

overwrite=False
= do not regenerate existing PNGs

max_files=100
= safety limit
```

---

# 19. Why `max_files=100` is useful here

For this pilot, we expect:

```text
exactly 100 selected samples
```

So:

```python
max_files=100
```

acts as a simple safety check.

If somebody accidentally passes:

```text
5000 samples
```

the bulk generator should stop before processing them.

This is not part of the SBSMI algorithm.

It is only a job-size safeguard.

---

# 20. `overwrite=False`

The script should normally use:

```python
overwrite=False
```

If a PNG already exists:

```text
status = skipped
```

This is useful if generation was interrupted.

You can rerun the script and already-generated outputs do not need to be recreated.

Do not change to:

```python
overwrite=True
```

unless you intentionally want to regenerate the images.

---

# 21. `save_generation_log()`

After generation, this function writes the returned status records to CSV.

This is important because:

```text
100 requested
```

does not automatically mean:

```text
100 successfully generated
```

The log lets us inspect exactly what happened.

---

# 22. `print_summary()`

This gives a quick family-by-family summary.

Example format:

```text
mira         ready=10/10  success=10  skipped=0  failed=0
small        ready=10/10  success=9   skipped=1  failed=0
wabot        ready=9/10   success=9   skipped=0  failed=1
```

`ready` means:

```text
success + skipped
```

because a skipped PNG already exists and is therefore still available.

---

# 23. Why `skipped` counts as ready

Suppose:

```text
sample A
```

was generated yesterday.

Today the script runs again with:

```python
overwrite=False
```

The bulk generator sees the existing PNG and returns:

```text
skipped
```

That does not mean failure.

It means:

> "This output already exists, so I did not regenerate it."

Therefore:

```text
ready = success + skipped
```

---

# 24. Final completeness check

At the end, the script checks every family.

It expects:

```text
10 ready images per family
```

If any family has fewer than 10:

```text
pilot incomplete
```

and the script raises an error.

This prevents this situation:

```text
99 images generated
1 failed
```

from being casually reported as:

```text
"pilot succeeded"
```

---

# 25. How to run it

Open a terminal at the **repository root**.

Example:

```text
D:\TRAMIF-Temporal-Reliability-Aware-Multi-view-Image-Fusion>
```

Then run:

```powershell
python -m scripts.generate_sbsmi_10_per_family
```

Prefer this over:

```powershell
python scripts/generate_sbsmi_10_per_family.py
```

because running with `-m` usually avoids Python import-root problems with `src`.

---

# 26. Before running: checklist

Check these first:

- [ ] Python environment is activated
- [ ] `numpy` installed
- [ ] `Pillow` installed
- [ ] `pytest` installed if you want to run tests first
- [ ] `ZIP_PATH` points to the real BODMAS ZIP
- [ ] `METADATA_PATH` points to the real metadata CSV
- [ ] SBSMI import path is correct
- [ ] enough disk space exists for 100 PNGs and logs
- [ ] `SAMPLES_PER_FAMILY = 10`
- [ ] `RANDOM_SEED` has not been randomly changed
- [ ] output directory is acceptable

---

# 27. Recommended order of execution

Do this:

```text
1. Run SBSMI unit tests
2. Run generate_sbsmi_10_per_family.py
3. Read terminal summary
4. Check generation log
5. Confirm 10 family folders
6. Confirm 10 usable PNGs per family
7. Open a few PNGs manually
8. Only then consider larger generation
```

---

# 28. Example expected terminal flow

A successful run should look roughly like:

```text
Finding real binaries for the 10 reproduction families...

Selecting 10 samples from each family...

ceeinject   : ... available
drolnux     : ... available
...
wabot       : ... available

Selected 100 real binaries.

Selection manifest:
...

Generating SBSMI images directly from the ZIP...

ZIP mode: 100 selected sample(s).

============================================================
SBSMI PILOT SUMMARY
============================================================

ceeinject    ready=10/10 ...
drolnux      ready=10/10 ...
...
wabot        ready=10/10 ...

Pilot complete: 10 usable SBSMI images for each of the 10 families.
```

The actual candidate counts and success counts are not known until the script is run.

---

# 29. Common problem: `ModuleNotFoundError: No module named 'src'`

Usually caused by running the script from the wrong location or using:

```powershell
python scripts/generate_sbsmi_10_per_family.py
```

Try from the repo root:

```powershell
python -m scripts.generate_sbsmi_10_per_family
```

Also check that the import points to the real SBSMI module.

---

# 30. Common problem: ZIP file not found

If you see:

```text
FileNotFoundError
```

check:

```python
ZIP_PATH
```

Do not assume another teammate has the same drive letter.

---

# 31. Common problem: metadata file not found

Check:

```python
METADATA_PATH
```

and verify the filename.

Example:

```text
bodmas_metadata.csv
```

---

# 32. Common problem: family has fewer than 10 candidates

Example:

```text
RuntimeError:
mira only has 7 usable binaries; need 10.
```

Do **not** immediately lower the required count just to make the script pass.

First inspect:

```text
metadata
ZIP member naming
family spelling
SHA field
archive path
```

The problem may be path matching rather than the real sample count.

---

# 33. Common problem: every sample fails

If all 100 fail, likely causes include:

```text
wrong ZIP internal path
wrong SBSMI import
wrong archive
corrupted ZIP
output permission problem
```

Check the first few:

```text
error_type
error_message
```

in the generation log.

Do not blindly change the SBSMI algorithm.

---

# 34. Common problem: some samples are `skipped`

This is usually fine.

It means the PNG already exists and:

```python
overwrite=False
```

was used.

If you intended a clean fresh run, either:

- use a new output folder, or
- deliberately use `overwrite=True`

Do not delete large directories casually just to start over.

---

# 35. Common problem: selected sample count is not 100

The script calculates:

```text
len(FAMILIES) x SAMPLES_PER_FAMILY
```

With:

```text
10 x 10
```

the expected total is:

```text
100
```

If the selected list does not contain exactly 100 entries, the script stops.

That is intentional.

---

# 36. What files should exist after a successful pilot

You should have:

```text
data/processed/malcsbsv_reproduction/
|
+-- sbsmi_pilot_10_per_family_manifest.csv
|
+-- sbsmi_pilot_10_per_family_log.csv
|
+-- sbsmi_pilot_10_per_family/
    |
    +-- ceeinject/
    |   +-- 10 PNGs
    |
    +-- drolnux/
    |   +-- 10 PNGs
    |
    +-- gandcrab/
    |   +-- 10 PNGs
    |
    +-- mira/
    |   +-- 10 PNGs
    |
    +-- musecador/
    |   +-- 10 PNGs
    |
    +-- sfone/
    |   +-- 10 PNGs
    |
    +-- sillyp2p/
    |   +-- 10 PNGs
    |
    +-- small/
    |   +-- 10 PNGs
    |
    +-- upatre/
    |   +-- 10 PNGs
    |
    +-- wabot/
        +-- 10 PNGs
```

Again: that is the intended successful output structure, not a result we can claim before execution.

---

# 37. What this script proves if it succeeds

A successful pilot supports the claim that:

```text
our current pipeline can:
- read real selected BODMAS members from the ZIP
- pass them through the existing SBSMI implementation
- save real SBSMI PNG outputs
- organize outputs by family
- record the exact selection
- record failures/successes
```

It does **not** prove:

```text
classification accuracy
paper reproduction accuracy
temporal robustness
model performance
```

Those require later experiments.

---

# 38. What this script does NOT do

It does not:

- train MalSBSLCNet;
- calculate accuracy;
- calculate macro F1;
- select the final paper-matched reproduction cohort;
- perform temporal train/validation/test splitting;
- tune models;
- modify malware binaries;
- extract the full ZIP.

It is only a small real-data SBSMI generation pilot.

---

# 39. Short version for teammates

If you do not want to read the whole document:

```text
1. Edit ZIP_PATH
2. Edit METADATA_PATH
3. Check the SBSMI import
4. Keep 10 families and 10 samples each
5. Run from repo root:

   python -m scripts.generate_sbsmi_10_per_family

6. Expect 100 requested samples
7. Check the manifest
8. Check the log
9. Confirm 10 usable PNGs in each family
10. If anything fails, read error_type/error_message before changing code
```

---

# 40. One-screen mental model

```text
find_available_candidates()
    =
metadata + ZIP
-> real usable candidates

choose_samples()
    =
candidates
-> deterministic 10 per family

save_selection_manifest()
    =
remember exactly what we selected

generate_sbsmi_dataset()
    =
real ZIP members
-> SBSMI PNGs

save_generation_log()
    =
remember what succeeded / skipped / failed

print_summary()
    =
quick human-readable check
```

That is the entire script.
