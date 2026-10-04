# Next GPU plan — current B submission already exists

## Evidence checked on 2026-10-04

The private V3 notebook's Version 5 (script version 353790527) already uses
CN2 full-volume inference and a fixed 0.25 fusion weight. Its displayed Public
Score is 0.943, matching the displayed earlier 0.943 score. This does not
establish an improvement beyond the precision shown by Kaggle.

The saved sample run took 474.8 seconds on T4 x2. Its CN2 log covers three
sample studies, eight model members, zero reported failures, shape (8,3,12),
complete-study rate 1.000 and approximately 73 seconds for the CN2 branch.
Three training preprocessing comparisons report zero mean absolute difference
and correlation 1.000. These are sample checks, not full-cohort guarantees.
The hidden rerun's per-study failures and duration have not been verified.

The source enumerates every center between the first and last acquired slot.
With all 44 slots present this is 42 windows. Missing edge slots shorten the
span; internal padding remains, following the existing corpus recipe.
Full-volume here means this preprocessed stack, not every native DICOM slice.
The stale comment mentioning 24 windows does not describe the executed loop.

Precision is branch-specific: CN2 explicitly halves model/input tensors;
public Raptor uses FP16 autocast with a nonfinite FP32 retry; A5's configured
autocast is BF16. Do not change the entire ensemble to FP16 as an incidental fix.
Tensor policy and kernel-level numerical behaviour are distinct.

## Next action: prepare and verify the strict CN2 audit

The current CN2 release gate permits up to 5% incomplete studies and can use
the parent ranks for failed cases. Its `strict` label does not mean zero case
fallback. Correct this before drawing conclusions from another submission.

`scripts/patch_cn2_strict_audit.py` patches a locally exported V5 notebook to a
new output file. It rejects missing/duplicated patch targets and reapplication.
It embeds the updated helper, preserves the trained models, stack sampling,
interpolation, non-reversal policy, and 0.25 rank fusion, and clears old outputs.
It does not run CUDA, upload a notebook, or submit predictions.

```bash
python scripts/patch_cn2_strict_audit.py private_v5.ipynb private_audit_draft.ipynb
python -m unittest discover -s scripts -p 'test_inference_full_volume_guard.py' -v
```

The new release gate requires every expected study/model pair exactly once,
all 12 probabilities finite and in [0,1], exact model member names, input/label
order, the expected mask-derived centers and CUDA FP16 tensors before forward.
It saves per-study preparation/forward times, model precision and raw
predictions. CPU preparation is bounded to four pending studies. The pre-CN2
CSV is removed before inference so a CN2 failure cannot leave it as the final
submission file. The model loader retains its existing strict loading and
checkpoint hash logging; observed hashes are not a new pinned-hash manifest.

This patch audits CN2 only. Parent branches still contain decode handling,
neutral fills, precision retries and dense-grid fallbacks; these must be
counted separately before claiming zero fallback for the whole ensemble.
Early model-loading/preflight failures remain fatal but may occur before the
CN2 per-study receipt is created. The older parent `COMPLETE` receipt precedes
CN2 and is not proof of final fused completion.

Validation completed: the patch applied and compiled against the actual V5
export; only the CN2 cell source changed, metadata was preserved and outputs
were cleared. Nine CPU release-gate regressions passed. No CUDA execution of
the patched notebook has been performed; CUDA dtype/device checks require a
future private smoke run. Keep the existing submitted V5 intact. The prepared
draft has not been uploaded to Kaggle or submitted.

## Model work after audit readiness

Do not repeat B solely to recover the same leaderboard score. First inventory
available predictions and exact fold assignments. Verify that every proposed
OOF row excludes that study from model training, including teacher training,
pseudo-label generation and hyperparameter selection. Full-data teacher or
student predictions are not automatically OOF.

Then run one bounded pilot of a structurally different model, keeping the
same eligible studies, folds, label definitions and macro-AUC calculation.
Pin architecture, pretrained-weight source and seed before training; choose
the concrete model after checking attached assets and compute requirements.
Do not describe the existing CN2 checkpoint as ConvNeXt merely from its name:
the inspected loader uses a CoAtNet RMLP-2 architecture.

For any candidate, compare paired per-label AUC, macro-AUC and prediction
complementarity on a common valid evaluation cohort. Use paired study-level
bootstrap confidence intervals. The 58-case expert set has been used for
selection and is developmental, not independent external validation.

Only if a candidate offers reproducible complementary benefit, preselect one
blend using valid out-of-sample predictions and a documented small weight
grid before consulting the Public LB. Submit at most that selected blend.
If comparable OOF predictions do not exist, resolve that gap before selecting
weights; do not pretend a full-data teacher ensemble supplies valid OOF.

## Evidence and stop rules

The previously written `2x2` and `OOF improvement` descriptions are not
established by this notebook/source audit. Keep historical ablation numbers
separate until predictions, cohort identity and protocol are reverified.

- No further CN2 reversal TTA or 30/36/40-window sweep.
- Preserve the reproduced parent's existing TTA while auditing it.
- No new LLM relabelling round or wholesale 288/336 ensemble retraining.
- No extra same-family members merely to increase ensemble size.
- Stop if valid evaluation alignment or a reproducible benefit is absent.
- Treat a tied displayed Public LB score as a tie, not a measured gain.
