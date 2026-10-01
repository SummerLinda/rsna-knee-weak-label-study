# Knee MRI classification: a reproducible weak-label pilot

**RSNA Knee Abnormality Detection · Kaggle project · August–October 2026 (2026-08 to 2026-10)**  
**Project summary updated: 1 October 2026**

I built a frozen ConvNeXt-Tiny baseline, reproduced and retrained credited public high-scoring pipelines, and investigated whether conflicting report-derived training labels should receive less weight. A controlled pilot found that one label-cleaning intervention improved performance against report-derived labels but reduced performance on a small expert-labeled diagnostic subset, so I did **not** adopt that intervention. I then conducted a controlled 2×2 inference ablation on eight models retrained under the reproduced public pipeline, testing MRI volume coverage and slice-order reversal. The selected inference strategy increases volumetric coverage while avoiding slice-order reversal.

## What each result means

| Submission | Method and contribution | Public leaderboard AUC (approximately) | Interpretation |
| --- | --- | ---: | --- |
| Baseline | My ConvNeXt-Tiny frozen features, slice pooling and average predictions from five classification heads | 0.782 | My engineering baseline; not a clinical estimate. |
| Public-solution reproduction A | Reproduction and retraining of a credited public high-scoring ensemble | 0.941 | Original architecture and fusion design belong to the public authors. |
| Public-solution reproduction B | Reproduction and retraining of another credited public high-scoring ensemble | 0.943 | Original architecture and fusion design belong to the public authors. |

These submissions differ in representation, preprocessing and fusion, so their leaderboard differences cannot identify the effect of any single component or establish an original model improvement. The paired experiments below are **separate** from the reproduced public ensembles and their hidden-test scores.

## The question I tested

Two public report readers disagreed explicitly on 198 ACL training entries. I compared the same frozen 4,608-dimensional image features, classification head, minibatch sequence and 12-epoch training budget across three matched seeds. The candidate set only the ACL training loss of those 198 entries to zero; the paired control retained it. I evaluated identical eligible cases under each label source.

| ACL AUC, candidate minus paired control | Seed 4723 | Seed 9151 | Seed 2038 |
| --- | ---: | ---: | ---: |
| Held Steven explicit report labels | +0.00074 | +0.00137 | −0.00017 |
| Held Pilkwang explicit report labels | +0.00522 | +0.00582 | +0.00436 |
| 46 expert-labeled diagnostic studies | −0.00780 | −0.00975 | −0.00390 |

The 46 expert cases include 19 ACL positive and 27 negative. This sample is small and **not independent external validation**: public label developers may have accessed related expert labels. The Pilkwang label reader also defined the training disagreements, so its holdout is not an independent adjudicator of the intervention. The key finding is therefore not “label cleaning improved the model,” but that an apparent improvement on report-derived labels did **not** transfer to the expert-labeled diagnostic subset. For that reason, the intervention was not adopted.

Read the [full methods report](docs/weak_label_validation.md) for the partition logic, other pilots, limitations and attribution.

## Latest experiment: 2×2 inference ablation

This independently conducted experiment held **eight retrained model weights fixed** and changed only two inference factors:

1. **Volume coverage:** 24 three-slice windows versus all available windows (at most 42).
2. **Slice-order reversal:** off versus on.

Slice-order reversal changes the medial-to-lateral ordering of sagittal MRI slices and can therefore exchange laterality-related anatomical context. This was tested explicitly rather than assumed to be a safe augmentation.

Evaluation used **4,349 OOF report-labeled studies with five fold teachers** and **58 expert-labeled studies with the eight-model mean**, with paired bootstrap 95% confidence intervals for candidate-minus-A differences.

| Arm | Volume coverage | Slice-order reversal | OOF macro-AUC | Expert58 macro-AUC | Δ OOF vs A | Δ Expert58 vs A |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | 24 windows | Off | 0.8908 | 0.9038 | 0 (reference) | 0 (reference) |
| B | All available, ≤42 windows | Off | 0.8942 | 0.9076 | +0.0035* | +0.0038 |
| C | 24 windows | On | 0.8922 | 0.9044 | +0.0014 | +0.0006 |
| D | All available, ≤42 windows | On | 0.8947 | 0.9071 | +0.0040* | +0.0033 |

| Paired contrast | OOF Δ macro-AUC, 95% CI | Expert58 Δ macro-AUC, 95% CI |
| --- | --- | --- |
| B − A | [+0.0025, +0.0044] | [−0.0008, +0.0088] |
| C − A | [+0.0008, +0.0021] | [−0.0032, +0.0045] |
| D − A | [+0.0028, +0.0051] | [−0.0008, +0.0085] |

*AUC endpoints and paired differences are independently rounded to four decimals in the experiment logs. The logged B−A and D−A OOF differences (+0.0035 and +0.0040) are retained; subtracting displayed endpoints gives +0.0034 and +0.0039.

### Interpretation and decision

- **Increasing MRI volume coverage was the main source of improvement.** B increased OOF macro-AUC from 0.8908 to 0.8942 (+0.0035, 95% CI [+0.0025, +0.0044]), with all 12 targets moving in the positive direction.
- On the **58-study expert-labeled subset**, B increased macro-AUC from 0.9038 to 0.9076 (+0.0038). The confidence interval [−0.0008, +0.0088] crosses zero, so this is directionally consistent with OOF but not statistically conclusive.
- **Slice-order reversal added little.** C improved OOF by only +0.0014 and Expert58 by +0.0006. When combined with full coverage, D added only +0.0005 OOF over B and slightly reduced expert macro-AUC at displayed precision.
- Because reversing sagittal slice order exchanges medial/lateral anatomical context and some laterality-related labels declined in the expert-set analysis, the reversal strategy was not adopted.

**Selected inference strategy:** increase volumetric coverage to all available windows (≤42) and **do not use slice-order reversal**.

This is a controlled inference result, not a claim that the selected strategy has already improved the public leaderboard.

## Current results summary

- **My baseline:** ConvNeXt-Tiny frozen-feature multi-label classifier, Public AUC ≈ **0.782**.
- **Credited public-solution reproductions:** Public AUC ≈ **0.941–0.943**. These scores come from reproduced public methods, not an original architecture claim.
- **ACL label audit:** removing conflicting ACL labels improved report-label validation by roughly **0.004–0.006** in one report-label evaluation, but reduced AUC by roughly **0.004–0.010** on the expert-labeled diagnostic subset; the change was therefore rejected.
- **2×2 inference ablation:** full-volume inference improved OOF macro-AUC by **+0.0035** with a 95% CI entirely above zero, while the expert subset showed a similar +0.0038 directional gain but with a CI crossing zero.
- **Final inference decision:** improve volume coverage; reject slice-order reversal TTA.

## Contents and reproduction

| Path | Purpose |
| --- | --- |
| `docs/weak_label_validation.md` | English methods report and numerical results. |
| `scripts/prepare_rsna_report_masks.py` | Align public report readers and create report-group partitions. |
| `scripts/rsna_label_quality_gate.py` | Check study alignment, label counts and split integrity. |
| `scripts/rsna_cpu_mask_pilot.py` | Shared frozen-feature training and unknown-label pilot. |
| `scripts/rsna_cpu_disagreement_pilot.py` | Matched ACL exclusion and broader contradiction-weight experiments. |
| `scripts/rsna_cpu_v4_source_pilot.py` | Exploratory alternative report-label source check. |
| `results/*.json` | Aggregate quality-gate and paired run records, including seeds, eligible counts, input hashes and AUC intervals. |

Install Python dependencies with `pip install -r requirements.txt`. Obtain the competition training table, public report-label files and the original frozen feature cache through their respective sources and terms. The loader checks the cache's pinned SHA256; this repository does not contain MRI scans, labels, model weights or identifiable study-level data. Run the report's [reproduction commands](docs/weak_label_validation.md#reproduction-guide) from `scripts/`, using paths to your locally obtained inputs. The guide reproduces the **CPU proxy**, not the public leaderboard ensembles.

## Scope and attribution

The reproduced high-scoring ensembles, their original architecture choices and their original fusion designs are the work of the credited public authors. The report readers are third-party label sources, not expert ground truth I produced. My project contributions are the baseline implementation and submission, reproduction and retraining checks, report-label alignment and audit, paired CPU pilot, controlled 2×2 inference ablation, and interpretation of the mixed findings.

This is an unreviewed project report. It makes no claim of a prize, publication, patent, independent clinical validation or original authorship of the reproduced public ensemble methods.
