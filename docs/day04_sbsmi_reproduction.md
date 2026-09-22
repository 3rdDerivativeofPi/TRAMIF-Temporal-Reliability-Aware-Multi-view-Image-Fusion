# Day 04 SBSMI Reproduction

Date: 21 September 2026

Source-compatibility correction: 22 September 2026

## Purpose

Implement the Short Bit Sequence Markov Image (SBSMI) representation used by
Zhang et al. as the third image view in the temporal-reliability
malware-family classification framework.

The original authors do not provide a publicly available reference
implementation.

Therefore, the Day-4 implementation follows the SBSMI description and
pseudocode given in the paper and validates the mathematical construction
using hand-checkable synthetic examples.

This work concerns representation construction only.

It contains no malware-classification performance results.

The implementation follows the SBSMI requirements in Sections 4.3 and 4.4 of
the research framework.

## Source algorithm

The source paper describes SBSMI construction using a bit-level representation
of the input executable.

The algorithm:

1. reads the input file as a binary sequence;
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

`64 x 64`

and directly forms the SBSMI representation.

## State construction

The binary sequence is divided into non-overlapping subsequences of length
`l`, starting from the first bit.

For the primary configuration:

`l = 6`

Each complete six-bit subsequence is converted into an integer state in the
range:

`0 ... 63`

For example:

`000000 -> 0`

`000001 -> 1`

`111111 -> 63`

The exact ordering of bits within each input byte remains unresolved from the
reviewed paper text.

The current implementation uses MSB-first ordering as a documented development
assumption.

This assumption must not yet be described as source-verified.

## Incomplete final state

The original Day-4 implementation discarded a final subsequence containing
fewer than six bits.

A later source audit found that this interpretation was inconsistent with
Section 4.1 of Zhang et al.

The source paper states that if the final subsequence contains fewer than
`l` bits, the subsequence is still converted into an integer state, with the
missing higher-order bits implicitly treated as `0`.

For example, for a six-bit state length:

`111 -> 000111 -> 7`

Therefore, the source-compatible implementation retains the incomplete final
subsequence rather than discarding it.

The implementation and corresponding unit test were corrected on
22 September 2026.

## Transition construction

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

This behavior follows the transition definition in Section 4.1 of the source
paper.

The row represents the preceding state and the column represents the
immediately following state.

## Row-wise normalization

After transition counting, the matrix is normalized by row.

For a source state `i`, the transition probability to destination state `j`
is:

`P(j | i) = m[i,j] / sum_j m[i,j]`

for rows with at least one outgoing transition.

Rows with no observed outgoing transitions remain zero.

The resulting matrix therefore contains transition probabilities in the range:

`0 <= P <= 1`

## Intensity scaling and quantization

The source paper specifies that transition probabilities are mapped to the
standard grayscale range `[0, 255]`.

Section 4.1 defines the pixel value as:

`floor(P * 255)`

The result is an 8-bit grayscale image.

For example:

`P = 1.0`

produces:

`floor(1.0 * 255) = 255`

and:

`P = 0.5`

produces:

`floor(0.5 * 255) = floor(127.5) = 127`

Therefore, source-compatible SBSMI quantization uses floor rather than
round-to-nearest behavior.

The resulting image values are integers in the range:

`0 ... 255`

with output represented as 8-bit grayscale values.

## Implementation structure

The SBSMI implementation is located at:

`src/preprocessing/sbsmi.py`

The implementation is divided into separate stages so that each part of the
source algorithm can be tested independently.

The processing pipeline is:

`bytes -> bitstream -> 6-bit states -> transition counts -> row normalization -> intensity scaling -> integer quantization -> 64 x 64 SBSMI`

The main implementation components include:

- conversion from file bytes to a bitstream;
- conversion from groups of six bits to state indices;
- preservation of an incomplete final state using zero-filled missing
  higher-order bits;
- construction of the 64 x 64 transition-count matrix;
- row-wise probability normalization;
- grayscale intensity scaling;
- source-compatible `floor(P * 255)` quantization.

Separating the stages makes it possible to verify transition behavior without
depending on a real malware binary.

## Original Day-4 synthetic implementation tests

The original SBSMI implementation was tested using synthetic inputs with known
expected state sequences and transition matrices.

These tests were implementation checks only.

They were not measurements on BODMAS and were not classifier-performance
results.

The test file was:

`tests/preprocessing/test_sbsmi.py`

The original Day-4 measured test run used:

- Python 3.11.16
- pytest 9.1.1
- Windows
- Conda environment: `malwaredetector`

The command executed was:

`python -m pytest tests/preprocessing/test_sbsmi.py -v`

## Original Day-4 measured test results

Five SBSMI unit tests were executed during the original Day-4 run.

| Test | Result |
| --- | --- |
| Known six-bit state conversion | Passed |
| Known transition counts | Passed |
| Row normalization | Passed |
| Row normalization with multiple target states | Passed |
| Original implemented incomplete-state handling | Passed |

Original measured result:

- Tests collected: 5
- Tests passed: 5
- Tests failed: 0

The original synthetic unit-test suite therefore produced:

`5 / 5`

This remains a valid record of the software test suite as it existed on
21 September 2026.

However, the result does not establish that every expectation encoded in that
test suite was source-compatible.

In particular, the original incomplete-state test encoded a discard-tail
interpretation that was later found to conflict with Section 4.1 of the source
paper.

## Known-state conversion test

The state-conversion test uses explicitly defined bit groups whose decimal
values are known in advance.

For example:

`000000 -> 0`

`000001 -> 1`

`111111 -> 63`

The test verifies that six-bit state conversion produces the expected integer
indices.

## Transition-count test

A synthetic state sequence is used to verify transition direction and counts.

For the sequence:

`0 -> 1 -> 0 -> 1`

the implementation must produce:

`m[0,1] = 2`

`m[1,0] = 1`

with a total of three observed transitions.

This directly verifies the core Markov counting operation.

## Normalization tests

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

## Source-compatibility correction

A later source audit on 22 September 2026 identified two issues in the original
Day-4 SBSMI reproduction.

### Incomplete final state

The original implementation discarded an incomplete final short-bit state.

Section 4.1 of Zhang et al. instead specifies that an incomplete final
subsequence is retained and converted into a state with missing higher-order
bits treated as `0`.

The implementation and corresponding unit test were corrected accordingly.

### Integer quantization

The original implementation left the final source-compatible integer
quantization unresolved.

Section 4.1 of Zhang et al. explicitly defines the grayscale pixel value as:

`floor(P * 255)`

and states that this conversion produces an 8-bit grayscale image.

This quantization rule was subsequently implemented and tested.

### Regression verification after correction

After applying the incomplete-tail and quantization corrections, the
repository-wide test suite was executed on 22 September 2026.

The measured result was:

`25 passed`

The correction run used:

- Python 3.11.9
- pytest 9.1.1
- Windows
- project-local Python virtual environment

This is a software regression-test result only.

It is not a BODMAS result, classifier-performance result, temporal-drift
result, or evidence that the proposed research method improves malware-family
classification.

The original Day-4 `5/5` result remains documented above because it was the
actual measured result of the earlier test suite.

The later `25 passed` result records the state of the repository after the
source-compatibility correction.

## Remaining source ambiguity

The source audit resolved the incomplete-tail convention and grayscale
quantization rule.

One low-level convention remains unresolved from the reviewed paper text:

- exact bit ordering within each input byte.

The current implementation uses MSB-first ordering.

This must remain documented as a development assumption rather than a
source-verified convention.

No change to bit ordering was made as part of the source-compatibility
correction.

## Source-code availability limitation

No official SBSMI implementation from Zhang et al. was available for direct
output comparison during this reproduction work.

Therefore, the reproduction strategy remains:

`paper description -> independent implementation -> synthetic hand-checkable tests`

The source paper provides direct evidence for the state construction,
incomplete-tail convention, transition probabilities, and grayscale
quantization described above.

However, because byte-level bit ordering remains insufficiently specified and
no official implementation is available for direct output comparison, the
project should not claim complete byte-for-byte equivalence with an author
implementation.

If official code or additional source material later becomes available,
direct output equivalence should be checked before upgrading that status.

## BODMAS binary-access status

Raw BODMAS malware binaries were still unavailable during the Day-4
reproduction work.

Therefore, during Day 4:

- no real BODMAS SBSMI was generated;
- no SBSMI classifier was trained;
- no classification result was measured;
- no future-test data was used.

The results documented for Day 4 are exclusively synthetic preprocessing
unit-test results.

The later source-compatibility correction likewise does not constitute a
malware-classification experiment.

## Generated artifacts

Day 4 produced the initial SBSMI implementation and tests.

Primary artifacts include:

- `src/preprocessing/sbsmi.py`
- `tests/preprocessing/test_sbsmi.py`

The SBSMI settings are documented in:

- `configs/preprocessing_v1.yaml`

Source-specific evidence and unresolved assumptions are documented in:

- `docs/source_notes_malcsbsv.md`

The source-compatibility correction updates these artifacts while preserving
the original Day-4 measured test record.

## Current status

The SBSMI mathematical core currently includes:

- six-bit state construction;
- 64 x 64 transition counting;
- row-wise transition-probability normalization;
- zero-valued rows for states without outgoing transitions;
- source-compatible incomplete-tail handling;
- source-compatible `floor(P * 255)` grayscale quantization.

The repository-wide regression suite executed after the correction produced:

`25 passed`

This is a software verification result only.

Exact SBSMI source compatibility remains provisional with respect to the
unresolved byte-level bit-order convention.

No malware-family classification experiment has yet been established by these
preprocessing tests.

## Next planned work

1. retain the corrected SBSMI implementation and source-compatibility notes;
2. keep the current MSB-first bit-order behavior explicitly marked as
   unresolved until additional source evidence is available;
3. avoid changing SBSMI preprocessing conventions after protocol freeze without
   documenting the change as a protocol deviation;
4. continue development of the common training-only family mapping and
   source-compatible SBSMI classifier;
5. continue monitoring BODMAS raw-binary access;
6. keep future classifier, drift, fusion, and temporal-test results separate
   from preprocessing software-test results.