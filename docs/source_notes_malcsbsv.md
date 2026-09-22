# MalCSBSV / SBSMI Source-Compatibility Notes

Source:

Zhang et al., "A lightweight malware classification method based on
short bit sequence visualization" [1].

## Binary conversion

Byte-to-bit convention:

[TO VERIFY]

Evidence:

The reviewed paper text does not explicitly specify the ordering of bits
within each byte.

## State construction

State length:

6 bits

Number of states:

64

Segmentation:

Non-overlapping short-bit subsequences.

Evidence:

Source paper Section 4.1.

## Bit order

Convention:

[TO VERIFY]

Current implementation assumption:

MSB-first within each byte.

Evidence:

The reviewed paper text does not explicitly specify MSB-first or LSB-first.
Therefore MSB-first must remain documented as a development assumption.

## Incomplete final block

Convention:

Keep the incomplete final subsequence. Missing higher-order bits are
treated as 0 before converting the subsequence to its integer state.

Example:

A 3-bit final subsequence `111` with state length 6 is interpreted as
`000111`, producing state 7.

Evidence:

Source paper Section 4.1 states that when the final subsequence contains
fewer than l bits, it is still converted to an integer state and the
missing higher-order bits are treated as 0.

## Transition matrix

Matrix shape:

64 x 64

Transition direction:

The row represents the preceding state and the column represents the
immediately following state.

Row normalization:

Transition counts are normalized by the total outgoing transitions from
the source state.

Zero-outgoing-transition rows:

Remain zero.

Evidence:

Source paper Section 4.1 and the transition-probability definition.

## Quantization

Source-compatible quantization:

`floor(P * 255)`

where `P` is the row-normalized transition probability.

Output dtype/range:

8-bit grayscale integer values in the range 0 to 255.

Example:

`P = 0.5`

`floor(0.5 * 255) = floor(127.5) = 127`

Evidence:

Source paper Section 4.1 defines the SBSMI grayscale pixel value using
`floor(P * 255)`.

## Implementation status

Verified and implemented:

- 6-bit states
- 64 x 64 transition matrix
- non-overlapping state segmentation
- source-to-next-state transition direction
- row normalization
- zero-outgoing-transition rows
- incomplete final-state zero filling
- `floor(P * 255)` quantization to 8-bit grayscale

Still unresolved:

- byte-to-bit ordering within each byte

The current implementation uses MSB-first as a development assumption.
Do not mark SBSMI fully source-compatible until the bit-order convention
has been independently verified or explicitly retained as an unresolved
implementation assumption.

## MalSBSLCNet architecture

Figure 4 specifies an initial 3 × 3 convolution producing 16 channels,
followed by BatchNorm and ReLU.

The feature extractor then contains six BaseBlocks:

| Block | Stride | Input channels | Output channels |
| --- | ---: | ---: | ---: |
| 1 | 2 | 16 | 32 |
| 2 | 1 | 32 | 32 |
| 3 | 2 | 32 | 64 |
| 4 | 1 | 64 | 64 |
| 5 | 2 | 64 | 128 |
| 6 | 1 | 128 | 128 |

The stride-1 BaseBlock splits the channels into two branches. One branch is
retained while the other applies pointwise convolution, depthwise convolution,
and a second pointwise convolution. The two branches are concatenated and
channel shuffled.

The stride-2 BaseBlock uses two transformed branches. One branch performs
depthwise downsampling followed by pointwise convolution. The other performs
pointwise convolution, depthwise downsampling, and a second pointwise
convolution. Their outputs are concatenated and channel shuffled.

The classifier is specified as:

AdaptiveAvgPool -> Flatten -> BatchNorm -> Dropout(0.4) -> Linear.

The exact AdaptiveAvgPool output size should be verified from Appendix A before
claiming exact source compatibility.