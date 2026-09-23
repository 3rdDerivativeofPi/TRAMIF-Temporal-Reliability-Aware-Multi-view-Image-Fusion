# **Day 06 MalSBSLCNet Architecture Reproduction**

Date: 23 September 2026

Updated: 23 September 2026

## **Purpose**

Analyze and reproduce the architecture of MalSBSLCNet, the lightweight CNN
used by Zhang et al. for classifying 64 × 64 Short Bit Sequence Markov Images
(SBSMIs).

The goal of Day 6 was to extract the source architecture accurately before
finalizing a PyTorch implementation.

This follows the project framework requirement to begin from a
source-compatible SBSMI classifier before allocating comparable capacity to
the raw-byte and entropy branches.

No malware-classification experiment was performed during Day 6.

No classification metric is reported in this document.

## **Source material inspected**

The source paper provides:

- the overall MalSBSLCNet architecture;
- the structure of the BaseBlock for stride 1;
- the structure of the BaseBlock for stride 2;
- the use of channel shuffle;
- the sequence of channel dimensions in the feature extractor;
- the high-level classifier structure;
- the optimizer and loss used for training.

The paper also states that the detailed architecture is provided separately
in Appendix A.

The publicly visible architecture figure and method description were used as
the basis for the Day-6 reproduction work.

## **MalSBSLCNet overview**

MalSBSLCNet is designed for the compact 64 × 64 SBSMI representation.

The model consists of two major components:

1. a lightweight feature-extraction backbone;
2. a classifier head.

The feature extractor begins with a standard 3 × 3 convolution producing
16 channels, followed by Batch Normalization and ReLU.

This is followed by six BaseBlocks.

The source architecture gives the following sequence:

| Block | Stride | Input channels | Output channels |
| --- | ---: | ---: | ---: |
| Initial convolution | 1 | 1 | 16 |
| BaseBlock 1 | 2 | 16 | 32 |
| BaseBlock 2 | 1 | 32 | 32 |
| BaseBlock 3 | 2 | 32 | 64 |
| BaseBlock 4 | 1 | 64 | 64 |
| BaseBlock 5 | 2 | 64 | 128 |
| BaseBlock 6 | 1 | 128 | 128 |

For a 64 × 64 SBSMI input, the resulting spatial progression is expected to
follow:

`1 × 64 × 64`

-> initial 3 × 3 convolution

`16 × 64 × 64`

-> BaseBlock, stride 2

`32 × 32 × 32`

-> BaseBlock, stride 1

`32 × 32 × 32`

-> BaseBlock, stride 2

`64 × 16 × 16`

-> BaseBlock, stride 1

`64 × 16 × 16`

-> BaseBlock, stride 2

`128 × 8 × 8`

-> BaseBlock, stride 1

`128 × 8 × 8`

## **Classifier dimensions verified from Appendix A**

Appendix A resolves the previously uncertain Adaptive Average Pooling
configuration.

The final BaseBlock produces:

`128 × 8 × 8`

The source classifier then applies Adaptive Average Pooling to produce:

`128 × 4 × 4`

After flattening:

`128 × 4 × 4 = 2048`

features remain.

The classifier therefore follows:

`AdaptiveAvgPool2d((4,4))`

-> `Flatten(2048)`

-> `BatchNorm1d(2048)`

-> `Dropout(0.4)`

-> `Linear`

The source configuration uses nine output classes.

Appendix A reports 18,441 parameters for the source final linear layer,
which agrees with:

`2048 × 9 + 9 = 18,441`

The present project instead uses the training-derived 51-family class space.
Therefore, only the output dimension of the final linear classification layer
is adapted from the source architecture.

This adaptation will change the total project-model parameter count and must
be reported separately from the source paper's reported complexity.

## **Stride-1 BaseBlock**

The stride-1 BaseBlock follows a ShuffleNetV2-style split-transform-concatenate
structure.

The input channels are divided into two branches.

The first branch acts as an identity path.

The second branch applies:

1. 1 × 1 pointwise convolution;
2. Batch Normalization;
3. ReLU;
4. 3 × 3 depthwise convolution;
5. Batch Normalization;
6. 1 × 1 pointwise convolution;
7. Batch Normalization;
8. ReLU.

The two branches are concatenated.

A channel-shuffle operation is then applied to permit information exchange
between the two channel groups.

For stride 1, the input and output channel counts remain equal.

## **Stride-2 BaseBlock**

The stride-2 BaseBlock uses two transformed branches.

The first branch applies:

1. 3 × 3 depthwise convolution with stride 2;
2. Batch Normalization;
3. 1 × 1 pointwise convolution;
4. Batch Normalization;
5. ReLU.

The second branch applies:

1. 1 × 1 pointwise convolution;
2. Batch Normalization;
3. ReLU;
4. 3 × 3 depthwise convolution with stride 2;
5. Batch Normalization;
6. 1 × 1 pointwise convolution;
7. Batch Normalization;
8. ReLU.

The two outputs are concatenated and channel shuffled.

The stride-2 block therefore performs both spatial downsampling and channel
expansion.

## **Channel shuffle**

The source network uses channel shuffle to allow information exchange between
the feature branches.

Conceptually, the operation reorganizes channels from grouped form:

`A A A A | B B B B`

into an interleaved arrangement such as:

`A B A B A B A B`

The operation changes channel ordering but does not alter the tensor shape.

The source paper presents this mechanism as a low-cost way to reduce the
channel independence introduced by grouped or branch-based computation.

## **Classifier structure**

The architecture figure specifies the classifier as:

`AdaptiveAvgPool`

-> `Flatten`

-> `BatchNorm`

-> `Dropout(0.4)`

-> `Linear`

-> class logits

The source text confirms a dropout rate of:

`0.4`

The use of Adaptive Average Pooling reduces the spatial dimensions before the
final linear layer and therefore reduces the number of fully connected
parameters.

## **Reported source complexity**

Zhang et al. report approximately:

- 0.054 million parameters;
- 9.22 million FLOPs;

for their MalSBSLCNet configuration operating on 64 × 64 SBSMI inputs.

These values are source-paper results.

They are not measurements of the implementation in this project.

The project implementation must report its own measured parameter count after
the classifier architecture has been finalized.

Furthermore, the final linear layer in this project will depend on the
training-derived known-family class count.

The current provisional known-family space contains 51 classes.

Therefore, exact parameter equality with the source configuration should not
be assumed without accounting for the different classification head.

## **Training procedure described by the source**

The source paper states that MalSBSLCNet is trained:

- from scratch;
- using the Adam optimizer;
- using cross-entropy loss.

The inspected source material does not justify adding undocumented training
choices such as:

- learning-rate schedulers;
- weight decay;
- early stopping;
- alternative losses.

Such choices should only be introduced if supported by the source,
the project framework, or a separately documented experimental decision.

## **PyTorch implementation strategy**

The project retains ordinary PyTorch modules for the scientific model
architecture.

The intended structure is:

`preprocessing`

-> `PyTorch model`

-> `BranchOutput(logits, embedding)`

-> `training framework`

This separation keeps the model definition independent of the experiment
orchestration layer.

The existing shared interface in:

`src/models/branch.py`

returns:

- classification logits;
- an intermediate embedding.

The logits will later be used for calibration and probability-level fusion.

The embedding will later be used for historical representation-drift
analysis.

This design supports the MMD-based drift analysis required by Section 5 of
the project framework.

## **PyTorch Lightning integration decision**

PyTorch Lightning is planned as the experiment and training orchestration
layer.

The current project design does not require rewriting the preprocessing or
model architecture code to use Lightning.

The intended separation is:

`MalSBSLCNet / RawByteNet / EntropyNet`

-> ordinary PyTorch modules

wrapped by:

`LightningMalwareClassifier`

for training, validation, metric logging, and checkpoint orchestration.

The Lightning implementation will follow the current stable PyTorch Lightning
documentation.

Lightning is treated as an implementation framework rather than part of the
scientific model definition.

The temporal protocol remains controlled by the project data split and
experiment configuration.

## **Temporal-protocol constraint**

PyTorch Lightning must not alter the chronological experimental protocol.

The primary protocol remains:

- training: August 2019 through January 2020;
- validation: February through March 2020;
- frozen future test: April through September 2020.

Validation callbacks, checkpoint selection, hyperparameter selection, or
other Lightning features must not use the April-September future-test labels.

This follows Section 7 of the project framework.

## **Implementation status**

During Day 6, the architecture was analyzed from the source paper and the
required model components were identified.

The following components are sufficiently specified by the inspected source:

- initial 3 × 3 convolution;
- channel progression;
- six BaseBlocks;
- stride-1 BaseBlock structure;
- stride-2 BaseBlock structure;
- depthwise convolution usage;
- channel shuffle;
- Batch Normalization placement at the architectural level;
- ReLU activation usage;
- Adaptive Average Pooling as the classifier entry point;
- Flatten;
- Batch Normalization;
- Dropout with rate 0.4;
- final linear classifier;
- Adam optimizer;
- cross-entropy loss;
- training from scratch.

## **Measured results**

No MalSBSLCNet training result was produced during Day 6.

No classification metric was measured.

No parameter-count result is reported as a project finding.

No FLOP count was measured for the project implementation.

Any values such as 0.054M parameters or 9.22M FLOPs in this document refer
only to values reported by the source paper.

## **Current project state**

Completed before or during Day 6:

- deterministic raw-byte preprocessing;
- deterministic entropy preprocessing;
- SBSMI mathematical-core reproduction;
- synthetic SBSMI unit tests;
- training-derived 51-family label space;
- shared model branch interface;
- MalSBSLCNet architecture analysis;
- BaseBlock structure extraction;
- channel-shuffle structure extraction;
- training-framework design decision.

Pending:

- confirmation of the exact MalSBSLCNet classifier dimensions;
- final MalSBSLCNet implementation;
- MalSBSLCNet tensor-shape tests;
- project-specific parameter-count measurement;
- project-specific FLOP measurement;
- Lightning training-wrapper implementation and verification, if not already
  completed;
- real BODMAS binary processing after access is granted.

## **Day-6 conclusion**

Day 6 established the source-supported structure required to reproduce
MalSBSLCNet without inventing undocumented architectural details.

The feature-extraction backbone and BaseBlock design are now sufficiently
understood for implementation.

Appendix A subsequently resolved the outstanding classifier-dimensionality ambiguity. The complete MalSBSLCNet architecture can therefore now be implemented from the available source specification, with the final output layer adapted from the source nine-class configuration to the project's 51-class known-family space.

## **Next planned work**

Once the outstanding architecture detail is resolved:

1. verify tensor dimensions using synthetic 64 × 64 SBSMI inputs;
2. measure the actual trainable parameter count;
3. compare the measured architecture size with the source configuration while
   accounting for the 51-class output head;
4. add the shared PyTorch Lightning training wrapper;
5. verify Adam and cross-entropy configuration;
6. retain all temporal split logic outside the model;
7. proceed toward single-view model training only when the required binary
   data become available.