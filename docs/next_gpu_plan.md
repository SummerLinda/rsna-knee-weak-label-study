# Next GPU plan: minimal path beyond the current 0.943 reproduction

## Decision already made

The 2x2 inference ablation is closed. Do not tune 30/36/40-window variants and do not continue reverse-TTA experiments.

Selected inference condition:

- use every contiguous 3-slice window available in the preprocessed volume
- 44 stored slices -> 42 windows
- no slice-order reversal TTA

Evidence:
- OOF macro-AUC: 0.8908 -> 0.8942 (+0.0035; 95% CI [+0.0025, +0.0044])
- Expert58 macro-AUC: 0.9038 -> 0.9076 (+0.0038; 95% CI [-0.0008, +0.0088])
- all 12 OOF targets moved positively
- reverse TTA added little and showed laterality-related risk

## Gate 0: submission safety

Before another formal Kaggle submission:

1. Run CUDA inference in FP16.
2. Log processed-study count.
3. Log successful new-branch count.
4. Log branch failures and fallback count.
5. Log total forward calls and wall-clock inference time.
6. Treat any fallback as visible evidence; do not allow a silent fallback to make a new submission indistinguishable from the old reproduced ensemble.
7. Validate every 12-label prediction for shape and finite values before fusion.

Helper implementation: `scripts/inference_full_volume_guard.py`.

## GPU experiment 1: full-volume B in the formal inference pipeline

Change only the window coverage in the already trained branch:

- old: 24 three-slice windows per study
- new: all contiguous windows, up to 42 for a 44-slice volume
- reverse TTA: off
- weights: unchanged
- labels: unchanged
- ensemble weights: unchanged

Success criteria:
- no fallback studies
- runtime safely below the competition execution limit
- valid submission file
- leaderboard result recorded as an external check, not as the criterion that established B

Do not re-tune B based on the public leaderboard.

## GPU experiment 2: one structurally different model

Only after experiment 1 is operationally safe, train one complementary branch rather than another near-duplicate checkpoint.

Preferred first candidate:
- ConvNeXt-V2 family or a medically pretrained CNN representation, subject to available Kaggle weights and runtime

Use the same folds and evaluation protocol so its OOF predictions are directly comparable with the current ensemble.

Required outputs:
- OOF predictions for all eligible training studies
- per-label AUC
- macro-AUC
- Expert58 predictions
- model checkpoint provenance and training configuration

## Analysis before any blend submission

A complementary model is useful if it adds independent information, not merely if its standalone AUC is high.

Compare the new branch with the current ensemble using:
- prediction Pearson correlation
- prediction Spearman correlation
- residual/error correlation where defined
- per-label disagreement
- OOF macro-AUC after fixed candidate blend weights

Pre-specify blend weights:
- 5%
- 10%
- 15%
- 20%

Do not select weights from the public leaderboard.

## GPU experiment 3: one selected blend

Submit only the blend selected from OOF evidence and checked for direction on Expert58.

A blend advances only if:
- OOF macro-AUC improves over the existing ensemble
- improvement is not driven by one pathological label
- Expert58 does not show a strong contradictory direction
- runtime remains within the competition limit
- new branch success/fallback logging confirms that the intended branch actually ran

## Stop conditions

Do not spend GPU budget on:
- additional LLM label extraction
- reverse slice-order TTA
- 30/36/40-window micro-tuning
- wholesale 288/336 retraining of the existing eight models
- more same-family checkpoints solely to increase model count

The next research question is whether greater volume coverage plus a genuinely complementary representation can improve generalization beyond the reproduced public ensemble.
