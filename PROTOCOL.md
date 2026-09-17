# Experimental Protocol

## Project

Lightweight Malware Family Classification under Temporal Drift by
Fusing Multiple Binary Image Views Based on Historical Reliability

## Primary task

Known-family malware classification under chronological distribution
shift.

The eligible family set K will be determined using training data only.

## Chronology

### Training
August 2019 - January 2020

Allowed:
- Fit preprocessing-related historical parameters where applicable
- Train CNN encoders and classification heads
- Construct historical development episodes

### Validation
February 2020 - March 2020

Allowed:
- Temperature scaling
- Historical reliability estimation
- Hyperparameter selection
- Reliability-rule / coefficient selection

### Future test
April 2020 - September 2020

Allowed:
- Frozen inference
- Prespecified evaluation
- Post-test diagnostic analysis

Not allowed:
- Model training
- Hyperparameter tuning
- Temperature fitting
- Fusion-weight adjustment
- Test-label use for model selection
- Test-batch adaptation in the primary experiment

## Temporal variable

Use the dataset observation timestamp.

Do not use PE compilation timestamps to define the chronological split.

Exact BODMAS metadata field:
TBD after dataset audit.

## Primary evaluation

Six monthly future evaluations:
- April 2020
- May 2020
- June 2020
- July 2020
- August 2020
- September 2020

Primary metrics:
- Mean monthly macro F1
- Worst-month macro F1

Additional metrics follow Section 11.4 of the framework.

## Protocol status

Status: DEVELOPMENT

The protocol is not yet frozen.

Freeze is planned before any primary April-September evaluation.