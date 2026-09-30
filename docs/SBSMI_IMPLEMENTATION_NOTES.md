# SBSMI implementation notes

This document explains the clean SBSMI implementation in `src/preprocessing/sbsmi.py`.

Source basis:

- Zhang et al. [1], SBSMI construction in Section 4.1 and Algorithm 1.
- Project framework Section 4.3: source-compatible SBSMI construction.
- Project framework Section 4.4: version and freeze bit order, tail handling, quantization, and preprocessing rules.
- Project framework Sections 7 and 11.2: future-test data must not be used to change the frozen preprocessing or selection rules.

The code itself is intentionally short. The details live here instead.

---

## 1. Pipeline

The main SBSMI path is:

```text
binary bytes
    |
    v
continuous bit stream
    |
    v
non-overlapping l-bit states
    |
    v
count adjacent state transitions
    |
    v
row-wise transition probabilities
    |
    v
floor(probability * 255)
    |
    v
uint8 SBSMI
```

The project default is:

```text
bit_num = 6
number of states = 2^6 = 64
SBSMI size = 64 x 64
```

The code supports `bit_num` from 1 to 8. The main project setting remains `6`.

---

## 2. Why `bit_buffer` is an integer instead of a list

We could store bits like this:

```python
[1, 0, 0, 1, 1, 0]
```

but then every byte would need to be converted to eight list elements, repeatedly sliced, and converted back into an integer state.

Instead, the implementation stores the same bit pattern inside a normal Python integer.

For example:

```text
unused bits: 10
new byte:    01101100
```

The code:

```python
bit_buffer = (bit_buffer << 8) | byte
```

produces a buffer representing:

```text
1001101100
```

`bit_count` is also required because integers do not preserve leading zeroes.

For example, both of these have integer value `2`:

```text
10
0010
```

So:

```text
bit_buffer = 2
bit_count = 2
```

means `10`, while:

```text
bit_buffer = 2
bit_count = 4
```

means `0010`.

---

## 3. Why the buffer must survive chunk boundaries

The file is read in chunks for memory efficiency.

A 6-bit state does not necessarily end at a byte boundary or chunk boundary.

For example, one read may end with:

```text
...101
```

and the next read may begin with:

```text
011010...
```

Those pieces belong to the same continuous file bitstream.

Therefore `bit_buffer` and `bit_count` are created before the read loop and are not reset after every chunk.

The test `test_iterate_lbit_states_is_chunk_independent` checks this by generating states with different chunk sizes and requiring identical output.

---

## 4. MSB-first extraction

Bytes are processed from their most-significant bit toward their least-significant bit.

For:

```text
11010110
```

and `bit_num = 6`, the first state is:

```text
110101
```

which is integer `53`.

The leftover bits are:

```text
10
```

They remain in the buffer and are joined with bits from the next byte.

---

## 5. Why the final partial state is NOT left-shifted

This is one of the easiest places to accidentally change the source method.

The source paper states that when the final subsequence contains fewer than `l` bits, it is directly converted into an integer and the missing higher-order bits are treated as zero.

Example with `bit_num = 5`:

```text
remaining bits:
101
```

The source-compatible interpretation is:

```text
00101
```

which is integer:

```text
5
```

Therefore the implementation does:

```python
yield bit_buffer
```

It does NOT do:

```python
yield bit_buffer << (bit_num - bit_count)
```

because that would produce:

```text
10100
```

which is integer `20`.

That would put zeroes on the low-order side instead of treating the missing high-order bits as zero.

The unit test `test_iterate_lbit_states_tail_is_not_left_shifted` protects this rule.

---

## 6. Transition counting

If the state sequence is:

```text
5 -> 12 -> 5 -> 20
```

the counted transitions are:

```text
5  -> 12
12 -> 5
5  -> 20
```

For `bit_num = 6`, `counts` is a `64 x 64` matrix.

`counts[i, j]` means:

> how many times state `i` is immediately followed by state `j`.

The count matrix uses `uint64` because transition counts can easily exceed `255` in real binaries.

Using `uint8` for counts could overflow.

---

## 7. Why normalization is row-wise

SBSMI uses conditional transition probabilities.

For state `i`:

```text
P(j | i)
=
count(i -> j)
/
all transitions leaving i
```

Example:

```text
5 -> 12    once
5 -> 20    once
```

Then:

```text
P(12 | 5) = 1/2
P(20 | 5) = 1/2
```

The entire matrix is not normalized as one big distribution.

Each source-state row is normalized separately.

Rows with no outgoing transitions remain all zero.

---

## 8. Why the code uses `floor`, not `round`

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

Using `round()` could produce `128`, which would no longer match the specified quantization.

The test `test_stream_to_sbsmi_known_transitions_and_floor` checks this.

---

## 9. No Laplace smoothing in the default reproduction path

The source paper evaluates Laplace smoothing separately.

For the default `l = 6` reproduction path, the implementation keeps the direct transition-count normalization:

```text
counts
    |
    v
row normalization
    |
    v
probabilities
```

No `+1` or other smoothing is added.

If smoothing is tested later, treat it as a separate ablation rather than silently changing the baseline representation.

---

## 10. Why `stream_to_sbsmi()` is the single core implementation

There are two ways we need to obtain bytes:

```text
normal file
    |
    v
open(..., "rb")
```

and:

```text
ZIP entry
    |
    v
ZipFile.open(...)
```

Both produce readable binary streams.

Therefore both wrappers call the same function:

```text
stream_to_sbsmi()
```

This avoids having separate SBSMI algorithms for normal files and ZIP entries.

Otherwise one implementation could accidentally drift from the other, for example by using a different tail rule.

---

## 11. Folder mode versus ZIP mode

`generate_sbsmi_dataset()` supports two modes.

### Folder mode

```python
source_mode="folder"
```

If `samples=None`, every file under the folder is processed recursively.

Output structure is preserved when no metadata is supplied.

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

A `samples` list may also be supplied to process only selected relative paths.

### ZIP mode

```python
source_mode="zip"
```

ZIP mode requires an explicit `samples` list.

Example:

```python
samples = [
    {
        "path": "some/path/file.bin",
        "sha": "abc...",
        "family": "mira",
    }
]
```

The output becomes:

```text
output/
    mira/
        abc....png
```

The ZIP is opened once and each requested member is streamed individually. The whole archive is not extracted.

ZIP mode deliberately requires an explicit sample list so that a large archive is not accidentally processed in full and so that the intended experimental cohort remains explicit.

For the temporal study, this matters because sample selection and chronological splits must follow the prespecified protocol rather than being changed after inspecting future-test results.

---

## 12. Failure handling

Bulk generation catches errors per sample and records:

```text
path
sha
family
output_path
status
error_type
error_message
```

A failed file is not silently skipped.

The caller can save the returned records as CSV or JSON.

Example statuses:

```text
success
skipped
failed
```

`skipped` means the output file already existed and `overwrite=False`.

---

## 13. What the tests cover

The tests use synthetic bytes only. They do not report malware-classification results.

They check:

- known state splitting;
- source-compatible tail handling;
- zero-valued tails;
- independence from chunk size;
- exact transition probabilities;
- `floor(P * 255)`;
- empty-input failure;
- normal-file and stream equivalence;
- ZIP-entry and stream equivalence;
- SBSMI shape and dtype validation;
- PNG save/load round trip;
- folder dataset generation;
- ZIP dataset generation;
- failure logging for a missing ZIP member.

After these unit tests pass, the next step is to run the same implementation on a small permitted real-binary sample before starting bulk generation.

---

## 14. Recommended repository placement

```text
src/
    preprocessing/
        sbsmi.py

tests/
    preprocessing/
        test_sbsmi.py
```

Run from the repository root:

```powershell
python -m pytest tests/preprocessing/test_sbsmi.py -v
```

Do not run the test file by its filesystem path with `python test_sbsmi.py`, because that can change Python's import root and lead to `ModuleNotFoundError` for `src`.

---

## 15. Protocol note for the temporal experiment

This module only performs deterministic preprocessing.

It should not decide which temporal samples belong to training, validation, or future testing.

For the primary project experiment, the caller must preserve the fixed chronology and cohort rules from the project framework. Future-test outcomes must not be used to change these preprocessing rules or select a different cohort.

---

## Reference

[1] Zhang, J., Guo, C., Shen, G., Ping, Y., Cui, Y., Chen, Y. *A lightweight malware classification method based on short bit sequence visualization*. Engineering Applications of Artificial Intelligence 181, 115559 (2026).
