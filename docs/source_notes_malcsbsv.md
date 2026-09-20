# MalCSBSV / SBSMI Source-Compatibility Notes

Source:
Zhang et al., "A lightweight malware classification method based on
short bit sequence visualization" [1].

## Binary conversion

Byte-to-bit convention:
[TO VERIFY]

Evidence:
[paper section/page/equation]

## State construction

State length:
6 bits

Number of states:
64

Segmentation:
non-overlapping

Evidence:
[paper section/page/equation]

## Bit order

Convention:
[TO VERIFY]

Example:
[TO VERIFY]

Evidence:
[paper section/page/equation]

## Incomplete final block

Convention:
[TO VERIFY: discard / pad / other]

Evidence:
[paper section/page/equation]

## Transition matrix

Matrix shape:
64 x 64

Transition direction:
[TO VERIFY]

Row normalization:
[TO VERIFY]

Zero-outgoing-transition rows:
[TO VERIFY]

Evidence:
[paper section/page/equation]

## Quantization

Source-compatible quantization:
[TO VERIFY]

Output dtype/range:
[TO VERIFY]

Evidence:
[paper section/page/equation]

## Implementation status

Do not mark SBSMI source-compatible until every required convention above
has either been verified from the source or explicitly documented as
unresolved.