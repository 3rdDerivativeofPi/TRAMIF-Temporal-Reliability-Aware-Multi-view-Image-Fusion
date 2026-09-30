# SBSMI Implementation Guide

This document explains the beginner-friendly SBSMI implementation in:

```text
src/preprocessing/sbsmi.py
tests/preprocessing/test_sbsmi.py
```

The code keeps important comments next to the tricky operations. This guide explains the overall logic so teammates do not have to understand the whole file at once.

The implementation follows the SBSMI construction used by Zhang et al. [1] and the project preprocessing controls in Sections 4.3-4.4 of the research framework.

---

## 1. What SBSMI does

SBSMI means **Short Bit Sequence Markov Image**.

It does not draw the original malware bytes directly. Instead, it asks:

> Which short bit pattern tends to follow which other short bit pattern?

For the main project setting:

```text
bit_num = 6
```

A 6-bit value has:

```text
2^6 = 64
```

possible states:

```text
000000 = 0
000001 = 1
...
111111 = 63
```

That gives a:

```text
64 x 64
```

transition image.

Pixel `[i, j]` represents the probability that state `j` appears immediately after state `i`.

---

## 2. Whole pipeline

```text
binary file / ZIP entry
        |
        v
iterate_lbit_states()
        |
        | states such as 53, 38, 12, ...
        v
stream_to_sbsmi()
        |
        | count transitions
        | normalize each row
        | floor(P * 255)
        v
64 x 64 uint8 matrix
        |
        v
validate_sbsmi()
        |
        v
save_sbsmi()
        |
        v
PNG
```

For many files, `generate_sbsmi_dataset()` repeats this pipeline.

---

## 3. Function summary

### `_check_bit_num()`

Checks that the short-bit state length is valid.

The main experiment uses:

```text
bit_num = 6
```

The implementation allows values from 1 to 8 so short-bit ablations remain possible.

### `_check_chunk_size()`

Checks that each `.read()` asks for a positive number of bytes.

Default:

```text
65536 bytes = 64 KB
```

Changing chunk size must not change the final SBSMI.

### `_relative_path()`

Makes sure bulk sample paths stay relative to the chosen input folder or ZIP root.

---

## 4. `iterate_lbit_states()`

Purpose:

```text
raw bytes
    ->
continuous bit stream
    ->
non-overlapping short-bit states
```

Example:

```text
11010110 01101100
```

with:

```text
bit_num = 6
```

becomes:

```text
110101 | 100110 | 1100
```

which becomes:

```text
53, 38, 12
```

The function uses `yield`, so it produces states one by one instead of storing the entire state sequence in memory.

---

## 5. Why `bit_buffer` is an integer

A beginner might expect:

```python
bit_buffer = []
```

and then `.append()` bits.

That is possible, but it would require repeatedly converting:

```text
byte
-> list of 8 bits
-> list slicing
-> integer state
```

Instead, the code stores unused bits inside one integer.

Example:

```text
old unused bits: 10
new byte:        01101100
```

The code:

```python
bit_buffer = (bit_buffer << 8) | byte
```

conceptually gives:

```text
1001101100
```

`bit_count` is also needed because Python integers do not preserve leading zeroes.

For example:

```text
10
0010
```

both have integer value `2`, so `bit_count` tells us whether the intended bit pattern has 2 bits or 4 bits.

---

## 6. Why chunk boundaries must not reset the buffer

The file is one continuous bitstream.

A state may cross the boundary between two chunks.

Example:

```text
chunk A ends:   ...101
chunk B begins: 011010...
```

Those bits still belong next to each other.

Therefore this would be wrong:

```python
while True:
    chunk = stream.read(chunk_size)
    bit_buffer = 0
    bit_count = 0
```

The real implementation creates `bit_buffer` and `bit_count` before the read loop.

The chunk-independence tests verify that different chunk sizes produce identical states and identical SBSMIs.

---

## 7. Why the final tail is NOT left-shifted

This is one of the easiest source-compatibility mistakes.

Suppose:

```text
bit_num = 6
```

and the file ends with:

```text
101
```

The source paper states that missing **higher-order bits** are treated as zero.

So the correct interpretation is:

```text
000101 = 5
```

Therefore the implementation does:

```python
yield bit_buffer
```

It does **not** do:

```python
yield bit_buffer << (bit_num - bit_count)
```

because that would produce:

```text
101000 = 40
```

That places zeroes on the low-order side and changes the representation.

The test `test_iterate_lbit_states_tail_is_not_left_shifted()` protects this rule.

---

## 8. `stream_to_sbsmi()`

This is the central SBSMI function.

It accepts any readable binary stream, regardless of whether the bytes came from:

```text
a normal file
```

or:

```text
a ZIP member
```

This gives the project one SBSMI implementation instead of maintaining separate file and ZIP algorithms.

---

## 9. Transition counting

Suppose the state sequence is:

```text
5 -> 12 -> 5 -> 20
```

Transitions are:

```text
5 -> 12
12 -> 5
5 -> 20
```

So the count matrix updates:

```text
counts[5, 12] += 1
counts[12, 5] += 1
counts[5, 20] += 1
```

For `bit_num = 6`, the count matrix is `64 x 64`.

---

## 10. Why counts use `uint64`

Do not use `uint8` for transition counts.

`uint8` can only store:

```text
0..255
```

A real binary may contain the same transition far more than 255 times.

The count matrix therefore uses:

```python
np.uint64
```

---

## 11. Why normalization is row-wise

For each source state `i`, SBSMI computes:

```text
P(next=j | current=i)
```

Example:

```text
5 -> 12   once
5 -> 20   once
```

Then:

```text
row total for state 5 = 2
P(12 | 5) = 1/2
P(20 | 5) = 1/2
```

The whole matrix is not normalized as one global distribution.

Each row is normalized separately.

---

## 12. Why `floor()` is used instead of `round()`

The source-compatible grayscale rule is:

```text
pixel = floor(P * 255)
```

Example:

```text
P = 0.5
0.5 * 255 = 127.5
floor(127.5) = 127
```

Using `round()` could produce `128`, so it would not exactly match the specified quantization.

---

## 13. No Laplace smoothing in the default path

The default reproduction pipeline is:

```text
transition counts
    ->
row normalization
```

No `+1` is added to transition counts.

Laplace smoothing is a separate experiment in the source paper and should remain an ablation rather than silently changing the default SBSMI.

---

## 14. `file_to_sbsmi()`

Normal-file wrapper:

```text
open(path, "rb")
    ->
stream_to_sbsmi()
```

It does not duplicate the SBSMI algorithm.

---

## 15. `zip_entry_to_sbsmi()`

ZIP wrapper:

```text
ZipFile.open(member)
    ->
stream_to_sbsmi()
```

The whole ZIP is not extracted.

Only the requested member is streamed.

---

## 16. `validate_sbsmi()`

For the main setting `bit_num = 6`, the expected output is:

```text
shape = (64, 64)
dtype = uint8
value range = 0..255
```

Invalid output should fail clearly instead of silently entering the dataset.

---

## 17. `save_sbsmi()`

This function:

```text
validate
    ->
create parent folder if needed
    ->
save PNG with Pillow
```

Install Pillow with:

```powershell
python -m pip install Pillow
```

but import it as:

```python
from PIL import Image
```

---

## 18. `generate_sbsmi_dataset()`

This is the bulk/orchestration function.

It supports two modes.

### Folder mode

```python
source_mode="folder"
```

If `samples=None`, every file under the source folder is discovered recursively.

Example:

```text
raw/
    a.bin
    nested/
        b.bin
```

becomes:

```text
processed/
    a.png
    nested/
        b.png
```

You can also provide `samples=[...]` to process only selected files.

### ZIP mode

```python
source_mode="zip"
```

ZIP mode requires an explicit `samples` list.

Example:

```python
samples = [
    {
        "path": "folder/file.bin",
        "sha": "abc123...",
        "family": "mira",
    }
]
```

Output:

```text
processed/
    mira/
        abc123....png
```

The ZIP is opened once and selected members are streamed one by one.

The archive is never fully extracted.

---

## 19. Why ZIP mode requires an explicit sample list

This is deliberate.

The BODMAS archive is large, and the research protocol uses explicit cohorts/manifests.

We do not want:

```text
run ZIP mode
```

to accidentally mean:

```text
process every file in the archive
```

The caller must provide the intended selected samples.

For the primary temporal experiment, sample selection must remain consistent with the fixed protocol rather than being changed after observing future-test results.

---

## 20. Why `assert zip_file is not None` exists

The variable is typed as:

```python
ZipFile | None
```

because:

```text
folder mode -> None
ZIP mode    -> ZipFile
```

Inside the ZIP branch, we already know the archive must be open.

The line:

```python
assert zip_file is not None
```

makes that fact explicit to Pylance and removes the type warning before calling `zip_entry_to_sbsmi()`.

---

## 21. Failure handling

One bad file should not stop a large bulk job.

Each requested sample receives a status:

```text
success
skipped
failed
```

For failures, the returned record contains:

```text
error_type
error_message
```

Failures are recorded instead of silently disappearing from the experiment.

---

## 22. What the tests check

The tests use synthetic bytes only.

They are implementation tests, not malware-classification results.

They check:

```text
known state splitting
source-compatible tail handling
zero-valued tail
chunk-size independence
known transition probabilities
floor(P * 255)
empty-input handling
normal file == stream
ZIP entry == stream
shape/dtype validation
PNG save/load round trip
folder bulk mode
ZIP bulk mode
failure recording
```

---

## 23. How to run the tests

From the repository root:

```powershell
python -m pytest tests/preprocessing/test_sbsmi.py -v
```

Prefer this over running the test file directly with `python test_sbsmi.py`, because `python -m pytest` keeps the repository root in the expected import context.

---

## 24. Recommended repository layout

```text
src/
    preprocessing/
        sbsmi.py

tests/
    preprocessing/
        test_sbsmi.py

docs/
    SBSMI_IMPLEMENTATION_GUIDE.md
```

Generated images should go under something like:

```text
data/processed/
```

and should normally be ignored by Git.

---

## 25. What comes next

After unit tests pass:

```text
synthetic tests
    ->
one real BODMAS ZIP member
    ->
small real-binary sample across reproduction families
    ->
bulk SBSMI generation
```

Do not jump directly to the full archive before confirming the small real-binary test.

---

## 26. Temporal-protocol reminder

This module is deterministic preprocessing infrastructure.

For the primary experiment:

```text
train      = Aug 2019 - Jan 2020
validation = Feb 2020 - Mar 2020
future     = Apr 2020 - Sep 2020
```

Future-test outcomes must not be used to change preprocessing rules or select a different primary cohort.

The limited-label adaptation track is separate from the frozen primary result.

---

## Reference

[1] Zhang, J., Guo, C., Shen, G., Ping, Y., Cui, Y., Chen, Y. *A lightweight malware classification method based on short bit sequence visualization*. Engineering Applications of Artificial Intelligence 181, 115559 (2026).
