# Knee MRI classification: a reproducible weak-label pilot

**RSNA Knee Abnormality Detection · Kaggle project · August–October 2026 (2026-08 to 2026-10)**  
**Project summary updated: 1 October 2026**

I built a frozen ConvNeXt-Tiny baseline, reproduced two credited public ensemble notebooks, and investigated whether conflicting report-derived training labels should receive less weight. The controlled pilot found a small improvement against one report reader's labels but a decline on a small expert-labeled diagnostic sample. I did **not** submit the pilot as a new model. I then conducted a controlled 2×2 inference ablation to test MRI volume coverage and slice-order reversal, selecting full-volume inference without reverse TTA for subsequent inference work.

## What each result means

| Submission | Method and contribution | Public leaderboard AUC (approximately) | Interpretation |
| --- | --- | ---: | --- |
| V1 | My ConvNeXt-Tiny frozen features, slice pooling and average predictions from five classification heads | 0.782 | My engineering baseline; not a clinical estimate. |
| V2 | Credited reproduction of **prvsiyan V27** public ensemble | 0.941 | Original architecture, training and fusion belong to the public authors. |
| V3 | Credited reproduction of **Mattia Angeli V39** public ensemble | 0.943 | Public rank 339 as recorded on 2026-09-26; contest ongoing. |

V1, V2 and V3 differ in representation, preprocessing and fusion, so their leaderboard differences cannot identify the effect of any single component or establish an original model improvement. The paired experiment below is **separate** from V3 and its hidden-test score.

## The question I tested

Two public report readers disagreed explicitly on 198 ACL training entries. I compared the same frozen 4,608-dimensional image features, classification head, minibatch sequence and 12-epoch training budget across three matched seeds. The candidate set only the ACL training loss of those 198 entries to zero; the paired control retained it. I evaluated identical eligible cases under each label source.

| ACL AUC, candidate minus paired control | Seed 4723 | Seed 9151 | Seed 2038 |
| --- | ---: | ---: | ---: |
| Held Steven explicit report labels | +0.00074 | +0.00137 | −0.00017 |
| Held Pilkwang explicit report labels | +0.00522 | +0.00582 | +0.00436 |
| 46 expert-labeled diagnostic studies | −0.00780 | −0.00975 | −0.00390 |

The 46 expert cases include 19 ACL positive and 27 negative. This sample is small and **not independent external validation**: public label developers may have accessed related expert labels. The Pilkwang label reader also defined the training disagreements, so its holdout is not an independent adjudicator of the intervention. Bootstrap intervals and the broader half-weight pilot do not justify a significant or clinical gain. These are exploratory observations about how the choice of evaluation labels affects an apparent improvement.

Read the [full methods report](docs/weak_label_validation.md) for the partition logic, other pilots, limitations and attribution. Source attribution and links to the three Kaggle submissions appear at the end of that report.

## Latest experiment: 2×2 inference ablation

This independently conducted experiment held **eight trained models and their weights fixed** and changed only two inference factors: 24 three-slice windows versus all available windows (at most 42), and reverse TTA off versus on. Reverse TTA reverses slice order; anatomical laterality must remain meaningful. Evaluation used **4,349 OOF report-labeled studies with five fold teachers** and **58 expert-labeled studies with the eight-model mean**, with paired bootstrap 95% confidence intervals for candidate-minus-A differences (1,000 OOF resamples; 2,000 expert-set resamples). These cohorts differ from the earlier CPU pilot's report holdout and Expert46 sample.

| Arm | Volume coverage | Reverse TTA | OOF macro-AUC | Expert58 macro-AUC | Δ OOF vs A | Δ Expert58 vs A |
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

*AUC endpoints and paired differences are independently rounded to four decimals in the Kaggle logs. The logged B−A and D−A OOF differences (+0.0035 and +0.0040) are retained; subtracting displayed endpoints gives +0.0034 and +0.0039.

**Interpretation and decision.** Increasing volume coverage improved OOF AUC across all 12 targets. B's expert-set change points in the same direction, but its confidence interval crosses zero, so it does not establish a statistically conclusive expert-set gain. C's gains are small; adding reversal to B (D−B) adds only +0.0005 OOF and reduces expert macro-AUC by 0.0005 at displayed precision, with some laterality-related targets declining in the reported experiment. **Adopt B: full-volume inference (≤42 windows), without reverse TTA.** This is the selected inference strategy, not a claim that it has already improved the public leaderboard.

The experiment took approximately 33 minutes. Numerical results were checked against the [Kaggle experiment logs, version 6 (353762673)](https://www.kaggle.com/code/zijiacheng521/knee-mri-training-the-twelve-finding-model?scriptVersionId=353762673); the Notebook is private, so that source requires authorized access. This repository does not yet contain its raw predictions, bootstrap outputs or runnable GPU ablation notebook. The existing scripts reproduce the earlier CPU pilots only. Expert58 remains a small diagnostic sample and should not be presented as independent clinical validation.

**Leaderboard status:** the best confirmed public score recorded here remains approximately **0.943 from the credited V39/V3 reproduction**. The new fusion version's leaderboard result and B's effect on a formal submission are unconfirmed; neither is assigned a score here.

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

Install Python dependencies with `pip install -r requirements.txt`. Obtain the competition training table, public report-label files and the original frozen feature cache through their respective sources and terms. The loader checks the cache's pinned SHA256; this repository does not contain MRI scans, labels, model weights or identifiable study-level data. Run the report's [reproduction commands](docs/weak_label_validation.md#reproduction-guide) from `scripts/`, using paths to your locally obtained inputs. The guide reproduces the **CPU proxy**, not the public ensemble submissions.

## Scope and attribution

The V2/V3 ensembles, checkpoints and original experiments are the work of their credited public authors. The report readers are third-party label sources, not expert ground truth I produced. My project contributions are the V1 baseline and submissions, reproduction checks, report-label alignment and audit, paired CPU pilot, controlled 2×2 inference ablation, and interpretation of the mixed findings. ChatGPT and Codex assisted code development, debugging and writing; the linked scripts and report provide the inspectable record. This is an unreviewed project report, with no prize, publication, patent or claim of clinical validity.
