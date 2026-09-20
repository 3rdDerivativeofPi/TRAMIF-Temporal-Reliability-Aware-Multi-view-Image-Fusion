# **Day 03 Deterministic Binary Image Preprocessing**

Date: 20 September 2026

## **Purpose**

Implement and verify the deterministic preprocessing pipeline for the raw-byte and entropy image representations used in the proposed temporal-reliability malware-family classification framework.

Raw BODMAS malware binaries were still unavailable during Day 3. Therefore, preprocessing was implemented and tested using synthetic byte sequences rather than real malware samples.

This work concerns representation construction only. It contains no malware-classification performance results.

The implementation follows the preprocessing requirements in Sections 4.1, 4.2, and 4.4 of the research framework.

## **Implemented files**

The Day-3 preprocessing implementation introduced the following components:

- `configs/preprocessing_v1.yaml`
- `src/preprocessing/common.py`
- `src/preprocessing/raw_byte.py`
- `src/preprocessing/entropy.py`
- `tests/preprocessing/test_common.py`
- `tests/preprocessing/test_raw_byte.py`
- `tests/preprocessing/test_entropy.py`

A preprocessing configuration was created so that representation-level choices can be versioned rather than being left implicit in the source code.

## **Preprocessing configuration**

The initial preprocessing configuration records the following main settings.

### **Common output**

- Output image size: 64 × 64
- Output representation type: floating-point values in `[0, 1]`
- Resize method: area interpolation
- Padding contribution during resize: disabled

### **Raw-byte view**

- Row width: 256 bytes
- Byte range: 0–255
- Normalization: divide byte values by 255
- Final partial row represented using a validity mask
- Empty-file policy: preprocessing error

### **Entropy view**

- Row width: 256
- Window size: 256 bytes
- Stride: 128 bytes
- Entropy normalization: divide Shannon entropy by 8 bits
- Overlapping window estimates combined by arithmetic mean
- Files shorter than 256 bytes use one partial window
- A trailing partial window is included where required
- Final partial row represented using the same validity-mask mechanism
- Empty-file policy: preprocessing error

The configuration also contains placeholders for SBSMI-specific settings that have not yet been fixed.

## **Mask-aware area resizing**

A shared mask-aware area-resize utility was implemented.

The purpose of the mask is to distinguish between:

- a genuine byte-derived value of zero, and
- a zero inserted only to pad the final row into a rectangular array.

Without a validity mask, zero padding would incorrectly influence the area average during resizing.

The implemented procedure resizes both:

- the value array multiplied by the validity mask, and
- the validity mask itself.

The final value is obtained from the ratio of these two resized quantities where the resized validity weight is non-zero.

Conceptually:

`resized(image × mask) / resized(mask)`

Locations with no valid contribution are assigned zero.

This follows the requirement in Section 4.1 that padding must not contribute to area averages.

## **Raw-byte representation**

The raw-byte representation preserves original file-offset order.

For a byte at offset `i`, its initial image coordinates are:

`row = floor(i / 256)`

`column = i mod 256`

For example:

- byte 0 → row 0, column 0
- byte 255 → row 0, column 255
- byte 256 → row 1, column 0

Each byte is normalized using:

`normalized value = byte / 255`

The resulting variable-height width-256 representation is then resized to 64 × 64 using the mask-aware area-resize implementation.

The resulting preprocessing function therefore follows:

`file bytes -> width-256 layout -> fixed [0,1] scaling -> validity mask -> mask-aware resize -> 64 × 64 image`

No dataset-derived normalization statistics are used.

This is consistent with the fixed-range preprocessing specified in Sections 4.1 and 4.4.

## **Raw-byte edge handling**

The implementation explicitly handles final partial rows.

For example, a 257-byte input produces:

- one complete 256-byte row;
- one second row containing one valid byte;
- 255 invalid padding positions in the second row.

Padding positions are represented in the rectangular array but excluded from resize averaging through the validity mask.

An empty byte string raises a preprocessing error rather than producing an artificial all-zero image.

This empty-file policy is an implementation decision and is recorded as part of preprocessing version `v1`.

## **Entropy representation**

The entropy view computes local Shannon entropy over the binary byte stream.

For each entropy window:

`H = -Σ p(b) log2 p(b)`

where `p(b)` is the empirical probability of byte value `b` within the window.

Only byte values that occur in the window contribute to the numerical calculation, avoiding evaluation of `log2(0)`.

Since byte entropy has a maximum of 8 bits, entropy is normalized as:

`H_normalized = H / 8`

The resulting entropy values lie in the interval `[0, 1]`.

This follows the construction specified in Section 4.2.

## **Entropy windowing**

The implemented entropy parameters are:

- Window size: 256 bytes
- Stride: 128 bytes

Because the stride is smaller than the window size, adjacent entropy windows overlap.

For example:

- Window 0 covers offsets 0–255
- Window 1 covers offsets 128–383

Offsets covered by multiple windows receive the arithmetic mean of their window entropy estimates.

The implementation maintains, for each byte offset:

- the sum of entropy values assigned to that offset;
- the number of entropy windows covering that offset.

The final per-offset value is:

`entropy_sum / entropy_count`

A runtime check verifies that no byte offset is left without an entropy estimate.

## **Short-file entropy handling**

Files shorter than one complete 256-byte window are handled using a single partial window beginning at offset zero.

For example, a 100-byte input uses one entropy window containing the 100 valid bytes.

No artificial padding contributes to the entropy calculation.

## **Trailing entropy window handling**

When bytes remain after the final complete entropy window, a trailing partial window begins at the next normal stride position.

For example, a 300-byte file produces entropy windows beginning at:

- offset 0
- offset 128

The second window uses the remaining valid bytes through offset 299.

No additional window beginning at offset 256 is created under preprocessing version `v1`.

This short-file and trailing-window behavior is documented because Section 4.2 requires those conventions to be specified explicitly.

## **Entropy image construction**

After entropy has been mapped back to individual byte offsets, the per-offset entropy vector is arranged using the same width-256 spatial layout as the raw-byte representation.

The same validity-mask mechanism is applied to the final partial row.

The entropy image construction therefore follows:

`file bytes -> entropy windows -> normalized entropy -> per-offset averaging -> width-256 layout -> validity mask -> mask-aware resize -> 64 × 64 image`

Using the same layout and resize logic across the raw-byte and entropy views helps ensure that differences between the representations are caused by the information encoded rather than inconsistent handling of padding or output geometry.

## **Implementation verification**

The preprocessing implementation was exercised using synthetic byte sequences with pytest.

These are measured implementation-test results from the Day-3 development session. They are not BODMAS measurements and are not malware-classification performance results.

The test environment recorded in the execution log was:

- Python: 3.11.16
- pytest: 9.1.1
- platform: Windows
- Conda environment: `malwaredetector`

### **Import-path issue**

The initial attempt to execute the preprocessing tests failed during collection with:

`ModuleNotFoundError: No module named 'src'`

The project was subsequently executed using:

`python -m pytest ...`

with the repository root and `pytest.ini` recognized correctly.

This resolved the module-discovery problem.

### **Mask-aware resize tests**

The shared mask-aware resize implementation was tested with two synthetic cases.

Measured result:

- Tests collected: 2
- Tests passed: 2
- Tests failed: 0

The tests confirmed that:

- padded positions do not reduce the value of valid observations;
- valid zero-valued observations remain part of the average.

The synthetic checks produced the expected values:

- `[1.0, padding, padding, padding]` -> `1.0`
- `[1.0, valid 0.0]` -> `0.5`

These tests verify the masking behavior required by the raw-byte and entropy resize pipeline.

### **Raw-byte preprocessing tests**

The raw-byte test file contained seven tests covering:

- exactly 256 bytes producing one complete row;
- byte offset 256 beginning the second row;
- invalid padding positions in a partial final row;
- final 64 × 64 output shape;
- output values remaining in the `[0, 1]` range;
- deterministic repeated execution;
- explicit failure on an empty input.

An intermediate test run produced:

- 5 passed
- 2 failed

Both failures were caused by debugging statements in the test code attempting to execute:

`print(mask)`

inside tests where `mask` had not been defined.

The resulting exception was:

`NameError: name 'mask' is not defined`

This was a test-code issue rather than a failure of the raw-byte preprocessing implementation.

After correcting the test code, the raw-byte tests were rerun.

Final measured result:

- Tests collected: 7
- Tests passed: 7
- Tests failed: 0

The final run therefore verified the intended raw-byte layout, masking, fixed-range normalization, output shape, deterministic behavior, and empty-input handling on the synthetic fixtures.

### **Entropy preprocessing tests**

The entropy preprocessing test file contained ten tests covering:

- constant-byte input producing zero entropy;
- all 256 byte values producing maximum normalized entropy;
- short-file handling;
- trailing partial-window handling;
- exact full-window handling;
- averaging across overlapping entropy windows;
- final 64 × 64 output shape;
- output values remaining in the `[0, 1]` range;
- deterministic repeated execution;
- explicit failure on an empty input.

Final measured result:

- Tests collected: 10
- Tests passed: 10
- Tests failed: 0

The entropy tests therefore verified the implemented 256-byte window, 128-byte stride, normalized Shannon entropy, overlap averaging, partial-window policy, layout, and deterministic output on the synthetic fixtures.

## **Measured Day-3 test summary**

The final successful test-file runs produced:

| Component              | Passed | Failed |
| ---------------------- | -----: | -----: |
| Mask-aware resize      |      2 |      0 |
| Raw-byte preprocessing |      7 |      0 |
| Entropy preprocessing  |     10 |      0 |
| **Total**              | **19** |  **0** |

The total above combines the three final successful test-file executions.

These results establish that the implemented preprocessing logic passes the current synthetic unit-test suite.

They do not establish correctness on real BODMAS binaries, classification accuracy, temporal robustness, or agreement with the SBSMI source method.

## **Development issues encountered**

Two implementation/testing issues were identified and resolved during Day 3.

First, pytest initially failed to import the project's `src` package when the test was invoked directly.

Running the tests through:

`python -m pytest`

with the repository pytest configuration resolved the import-path issue.

Second, two raw-byte tests initially failed because debugging code referenced an undefined `mask` variable.

This did not indicate an error in the preprocessing output. The test code was corrected and the complete raw-byte test file subsequently passed.

Recording these intermediate failures is useful for reproducibility and avoids presenting the final successful run as though no implementation debugging was required.

## **Determinism**

The preprocessing functions were designed so that identical byte sequences produce identical representations under the same preprocessing version and software environment.

No stochastic augmentation is applied.

In particular, no image flips or rotations are used because those operations do not have an established semantic interpretation for the binary representations in the primary experiment.

This follows the implementation controls specified in Section 4.4.

## **SBSMI status**

SBSMI was not finalized as part of this implementation stage.

The planned SBSMI representation uses non-overlapping six-bit states and a 64 × 64 transition image, but source compatibility still requires explicit verification of:

- bit ordering;
- handling of an incomplete final six-bit block;
- source-compatible integer quantization.

These settings remain unset rather than being guessed.

The project therefore does not yet claim that an SBSMI implementation reproduces the Zhang et al. construction.

This follows the source-compatibility requirement in Section 4.3.

## **BODMAS binary-access status**

Raw BODMAS malware binaries were still unavailable during Day 3.

Consequently:

- no real BODMAS binary was processed;
- no BODMAS raw-byte image was generated;
- no BODMAS entropy image was generated;
- no classifier was trained;
- no classification metric was measured.

The measured results reported in this document are exclusively synthetic preprocessing unit-test results.

## **Protocol relevance**

The preprocessing implementation is intended to remain identical across:

- training: Aug 2019–Jan 2020;
- validation: Feb–Mar 2020;
- future test: Apr–Sep 2020.

Future-test samples must not receive a different preprocessing pipeline.

Any later change to row width, entropy parameters, resize behavior, masking, or related construction choices must either:

- occur before protocol freeze and create a new documented preprocessing version, or
- be treated as an explicitly identified ablation.

The future-test period must not be used to choose such changes.

## **Generated artifacts**

Day 3 produced the preprocessing implementation and its associated configuration and tests.

Primary artifacts include:

- `configs/preprocessing_v1.yaml`
- `src/preprocessing/common.py`
- `src/preprocessing/raw_byte.py`
- `src/preprocessing/entropy.py`
- `tests/preprocessing/test_common.py`
- `tests/preprocessing/test_raw_byte.py`
- `tests/preprocessing/test_entropy.py`

If an environment snapshot was created after implementation, the corresponding environment files should be versioned alongside this work.

## **Current limitations**

The current preprocessing implementation has two main limitations.

First, raw BODMAS malware binaries remain unavailable, so the raw-byte and entropy transforms have only been exercised using synthetic inputs.

Second, SBSMI source conventions have not yet been fully verified and therefore remain outside the completed source-compatible preprocessing pipeline.

Neither limitation affects the synthetic verification of the raw-byte and entropy construction logic, but both must be resolved before the complete three-view BODMAS experiment can be run.

## **Status**

Day-3 deterministic raw-byte and entropy preprocessing: complete.

Measured implementation verification:

- mask-aware resize: 2/2 tests passed;
- raw-byte preprocessing: 7/7 tests passed after correcting a test-code debugging error;
- entropy preprocessing: 10/10 tests passed;
- total across the three final test-file runs: 19/19 tests passed.

Completed work:

1. versioned preprocessing settings were defined;
2. mask-aware area resizing was implemented and tested;
3. deterministic raw-byte image construction was implemented and tested;
4. deterministic entropy image construction was implemented and tested;
5. short-file and trailing-window behavior was explicitly defined;
6. overlap averaging was explicitly tested;
7. output shape, range, empty-input behavior, and deterministic execution were tested;
8. SBSMI source-specific choices were deliberately left pending rather than guessed.

No malware-classification experiment was performed during Day 3.

Next planned work:

1. verify the SBSMI construction conventions used by Zhang et al.;
2. implement and unit-test the source-compatible SBSMI representation;
3. continue monitoring the BODMAS raw-binary access request;
4. retain the completed raw-byte and entropy preprocessing as preprocessing version `v1` unless a documented pre-freeze correction is required.
