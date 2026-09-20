# **Day 04 SBSMI Reproduction**

Date: 21 September 2026

## **Purpose**

Implement the Short Bit Sequence Markov Image (SBSMI) representation used by
Zhang et al. as the third image view in the temporal-reliability
malware-family classification framework.

The original authors do not provide a publicly available reference
implementation.

Therefore, the Day-4 implementation follows the SBSMI pseudocode given in the
paper and validates the mathematical construction using hand-checkable
synthetic examples.

This work concerns representation construction only.

It contains no malware-classification performance results.

The implementation follows the SBSMI requirements in Sections 4.3 and 4.4 of
the research framework. :contentReference[oaicite:0]{index=0}

## **Source algorithm**

The source paper describes SBSMI construction using a bit-level reader over the
input PE file.

The algorithm:

1. reads the input file as a bitstream;
2. divides the bitstream into consecutive short-bit states;
3. converts each state into an integer index;
4. counts transitions between consecutive states;
5. normalizes the transition matrix row-wise;
6. scales transition probabilities into the grayscale range;
7. converts the matrix into an image.

For the primary configuration:

`l = 6`

Therefore:

`2^6 = 64`

possible short-bit states exist.

The resulting transition matrix has shape:

`64 × 64`

and directly forms the SBSMI representation.

## **Transition construction**

Let the short-bit state sequence be:

`s_0, s_1, ..., s_n`

For each pair of consecutive states:

`s_t -> s_(t+1)`

the corresponding transition count is incremented.

For example, the synthetic state sequence:

`0 -> 1 -> 0 -> 1`

produces:

`m[0,1] = 2`

and:

`m[1,0] = 1`

with all other unobserved transitions remaining zero.

This behavior follows the transition update shown in the source pseudocode:

`m[i,j] = m[i,j] + 1`

## **Row-wise normalization**

After transition counting, the matrix is normalized by row.

For a source state `i`, the transition probability to destination state `j`
is therefore:

`P(j | i) = m[i,j] / sum_j m[i,j]`

for rows with at least one outgoing transition.

Rows with no observed outgoing transitions remain zero in the current
implementation.

The source pseudocode describes this step as row-wise L1 normalization.

## **Intensity scaling**

After row normalization, the source algorithm multiplies the probability
matrix by:

`255`

Therefore a transition probability of:

`1.0`

corresponds to an intensity of:

`255`

before final image conversion.

The SBSMI therefore represents transition structure rather than spatial byte
layout.

This distinguishes it from the raw-byte and entropy views implemented on Day 3.

## **Implementation structure**

The SBSMI implementation was added under:

`src/preprocessing/sbsmi.py`

The implementation is divided into separate stages so that each part of the
source algorithm can be tested independently.

The processing pipeline is:

`bytes -> bitstream -> 6-bit states -> transition counts -> row normalization -> intensity scaling -> 64 × 64 SBSMI`

The main implementation components include:

- conversion from file bytes to a bitstream;
- conversion from groups of six bits to state indices;
- construction of the 64 × 64 transition-count matrix;
- row-wise probability normalization;
- grayscale scaling.

Separating the stages makes it possible to verify transition behavior without
depending on a real malware binary.

## **Synthetic implementation tests**

The SBSMI implementation was tested using synthetic inputs with known expected
state sequences and transition matrices.

These tests are implementation checks only.

They are not measurements on BODMAS and are not classifier-performance
results.

The test file was:

`tests/preprocessing/test_sbsmi.py`

The measured test run used:

- Python 3.11.16
- pytest 9.1.1
- Windows
- Conda environment: `malwaredetector`

The command executed was:

`python -m pytest tests/preprocessing/test_sbsmi.py -v`

## **Measured test results**

Five SBSMI unit tests were executed.

| Test                                          | Result |
| --------------------------------------------- | ------ |
| Known six-bit state conversion                | Passed |
| Known transition counts                       | Passed |
| Row normalization                             | Passed |
| Row normalization with multiple target states | Passed |
| Incomplete-state handling                     | Passed |

Final measured result:

- Tests collected: 5
- Tests passed: 5
- Tests failed: 0

The complete SBSMI synthetic unit-test suite therefore passed:

`5 / 5`

These results verify the current implementation logic against the synthetic
cases encoded in the test suite.

They do not establish classification accuracy or byte-for-byte equivalence
with an unavailable author implementation.

## **Known-state conversion test**

The state-conversion test uses explicitly defined bit groups whose decimal
values are known in advance.

For example:

`000000 -> 0`

`000001 -> 1`

`111111 -> 63`

The test verifies that the six-bit state conversion produces the expected
integer indices.

## **Transition-count test**

A synthetic state sequence is used to verify transition direction and counts.

For the sequence:

`0 -> 1 -> 0 -> 1`

the implementation must produce:

`m[0,1] = 2`

`m[1,0] = 1`

with a total of three observed transitions.

This directly verifies the core Markov counting operation.

## **Normalization tests**

The row-normalization tests verify both deterministic and mixed outgoing
transitions.

Where a state has only one observed destination, the resulting transition
probability is:

`1.0`

A separate synthetic case verifies that multiple outgoing transitions are
normalized according to their relative counts.

For example, a source state with two transitions to state 1 and one transition
to state 2 should produce:

`P(1 | source) = 2/3`

`P(2 | source) = 1/3`

## **Incomplete-state handling**

The current implementation discards an incomplete final short-bit state.

For example, if one complete six-bit state is followed by only three remaining
bits, the three-bit tail is not converted into an additional state.

The synthetic test verifies this implemented behavior.

However, the exact behavior of the source paper's `ReadBits` operation when
fewer than six bits remain is not fully specified by the available pseudocode.

The current behavior should therefore be treated as a documented implementation
interpretation rather than a confirmed property of an unavailable reference
implementation.

## **Remaining source ambiguities**

The source pseudocode defines the mathematical structure of SBSMI clearly, but
some low-level implementation details remain insufficiently specified.

These include:

- exact bit ordering inside each byte;
- exact incomplete-tail semantics of the source `ReadBits` implementation;
- exact numeric conversion performed by `MatrixConvImage`;
- exact rounding or casting behavior after multiplication by 255.

These choices must remain explicitly documented.

The project must not claim exact reproduction of an unavailable implementation
where the source paper does not expose enough detail to verify the convention.

This follows the reproducibility controls in Section 4.4. :contentReference[oaicite:1]{index=1}

## **Source-code availability limitation**

No official SBSMI implementation from Zhang et al. was available for direct
output comparison.

Therefore, the reproduction strategy used on Day 4 was:

`paper pseudocode -> independent implementation -> synthetic hand-checkable tests`

This is weaker than direct agreement with an official source implementation but
still allows the explicitly described mathematical behavior to be tested.

If official code later becomes available, direct output equivalence should be
checked before changing the current reproduction status.

## **BODMAS binary-access status**

Raw BODMAS malware binaries were still unavailable during Day 4.

Therefore:

- no real BODMAS SBSMI was generated;
- no SBSMI classifier was trained;
- no classification result was measured;
- no future-test data was used.

The measured results reported here are exclusively synthetic preprocessing
unit-test results.

## **Generated artifacts**

Day 4 produced the SBSMI implementation and tests.

Primary artifacts include:

- `src/preprocessing/sbsmi.py`
- `tests/preprocessing/test_sbsmi.py`

The SBSMI settings are also documented in:

- `configs/preprocessing_v1.yaml`

Source-specific assumptions should remain visible in the configuration and
documentation rather than being hidden inside implementation code.

## **Current status**

Day-4 SBSMI mathematical-core implementation: complete.

Measured implementation verification:

- six-bit state conversion: passed;
- transition counting: passed;
- row normalization: passed;
- mixed-target normalization: passed;
- implemented incomplete-tail handling: passed;
- total: 5/5 synthetic tests passed.

The implementation reproduces the explicit mathematical operations in the
available SBSMI pseudocode.

Exact source compatibility remains provisional because no official reference
implementation is available and several low-level conventions remain
under-specified.

No malware-classification experiment was performed during Day 4.

## **Next planned work**

1. retain the current SBSMI implementation as the documented reproduction of
   the available pseudocode;
2. resolve any remaining bit-order or image-conversion details if additional
   source material becomes available;
3. prepare the common 51-family class mapping derived from training data only;
4. begin the source-compatible SBSMI classifier architecture and shared model
   interface;
5. continue monitoring the BODMAS raw-binary access request.
