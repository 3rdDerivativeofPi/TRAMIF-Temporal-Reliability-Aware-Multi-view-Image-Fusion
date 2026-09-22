# **Day 05 Shared Label Space and Model Interface**

Date: 22 September 2026

## **Purpose**

Define a deterministic known-family class space and a shared model interface
before implementing the three image-classification branches.

No classifier was trained and no malware-classification performance result was
measured during this work.

## **Training-derived family space**

The provisional eligible-family set was derived using training-period support
only.

The previously measured support audit identified:

- 395 families represented during training;
- 51 families satisfying the provisional eligibility rule.

The rule requires:

- at least 50 unique training files;
- representation in at least 3 training months.

The 51 eligible families were sorted deterministically and assigned contiguous
class IDs from 0 through 50.

The resulting mapping is stored in:

`data/manifests/bodmas_eligible_families_v0.csv`

Validation and future-test performance were not used to select or order the
classes.

## **Label-space utility**

A reusable label-space loader was implemented in:

`src/data/labels.py`

The loader verifies:

- unique family names;
- unique class IDs;
- contiguous IDs beginning at zero;
- deterministic family-to-ID and ID-to-family mappings.

All future model branches are intended to use this same class ordering.

## **Shared branch interface**

A common branch interface was implemented in:

`src/models/branch.py`

Each image-classification branch is required to return:

- classification logits over the shared known-family set;
- a compact encoder embedding.

The logits will later support calibration and probability-level fusion.

The embeddings will later support historical representation-drift analysis
using fixed encoders.

## **Verification**

```
=========================================================== test session starts ===========================================================
configfile: pytest.ini
collected 6 items                                                                                                                          


tests/data/test_labels.py::test_label_space_has_51_classes PASSED                                                                    [ 16%]
tests/data/test_labels.py::test_class_ids_are_contiguous PASSED                                                                      [ 33%]
tests/data/test_labels.py::test_family_mapping_is_bidirectional PASSED                                                               [ 50%]
tests/data/test_labels.py::test_family_order_is_deterministic PASSED                                                                 [ 66%]
tests/models/test_branch.py::test_branch_rejects_invalid_class_count PASSED                                                          [ 83%]
tests/models/test_branch.py::test_branch_output_shapes PASSED                                                                        [100%]

============================================================ 6 passed in 1.74s ============================================================
```

These are software implementation checks only and are not
malware-classification performance results.

## **Status**

Day-5 shared label-space and model-interface work: complete.

Next planned work:

1. reproduce the lightweight SBSMI classifier architecture;
2. test model tensor shapes and parameter counts using synthetic inputs;
3. use the same branch interface for the raw-byte and entropy models;
4. continue waiting for restricted BODMAS binary access.