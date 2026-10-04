"""Utilities for full-volume knee MRI inference and runtime auditing.

Designed for the RSNA Knee Abnormality Detection project.

The module intentionally does not depend on a particular competition notebook.
It provides:
- all contiguous 3-slice windows (44 slices -> 42 windows)
- lightweight inference counters
- explicit fallback accounting
- FP16 dtype checks for CUDA inference

Use these helpers inside the private Kaggle submission notebook rather than
silently falling back to a previous branch without recording it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import List, Sequence, Tuple

import numpy as np


def all_three_slice_windows(num_slices: int) -> List[Tuple[int, int, int]]:
    """Return every contiguous 3-slice window.

    Examples
    --------
    44 slices -> 42 windows: (0,1,2), ..., (41,42,43)
    """
    if num_slices < 3:
        raise ValueError(f"Need at least 3 slices, got {num_slices}")
    return [(i, i + 1, i + 2) for i in range(num_slices - 2)]


def full_volume_window_count(num_slices: int) -> int:
    """Number of contiguous 3-slice windows available."""
    return max(0, num_slices - 2)


@dataclass
class InferenceAudit:
    studies_attempted: int = 0
    studies_succeeded: int = 0
    studies_fallback: int = 0
    branch_failures: int = 0
    forward_calls: int = 0
    elapsed_seconds: float = 0.0

    def record_success(self, forward_calls: int) -> None:
        self.studies_attempted += 1
        self.studies_succeeded += 1
        self.forward_calls += int(forward_calls)

    def record_fallback(self, forward_calls: int = 0) -> None:
        self.studies_attempted += 1
        self.studies_fallback += 1
        self.branch_failures += 1
        self.forward_calls += int(forward_calls)

    @property
    def fallback_rate(self) -> float:
        if self.studies_attempted == 0:
            return 0.0
        return self.studies_fallback / self.studies_attempted

    def summary(self) -> dict:
        out = asdict(self)
        out["fallback_rate"] = self.fallback_rate
        out["mean_seconds_per_study"] = (
            self.elapsed_seconds / self.studies_attempted
            if self.studies_attempted
            else 0.0
        )
        return out


class AuditTimer:
    """Context manager that accumulates wall-clock inference time."""

    def __init__(self, audit: InferenceAudit):
        self.audit = audit
        self._start = None

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is not None:
            self.audit.elapsed_seconds += perf_counter() - self._start


def assert_prediction_vector(pred: Sequence[float], n_labels: int = 12) -> np.ndarray:
    """Validate one study prediction before it enters an ensemble."""
    arr = np.asarray(pred, dtype=np.float32)
    if arr.shape != (n_labels,):
        raise ValueError(f"Expected shape ({n_labels},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Prediction contains NaN or Inf")
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError("Prediction is outside [0,1]")
    return arr


def assert_no_silent_fallback(audit: InferenceAudit) -> None:
    """Fail loudly if any study used a fallback branch.

    For exploratory runs you may choose to log instead. For a final submission,
    failing loudly is safer than unknowingly submitting the unchanged baseline.
    """
    if audit.studies_fallback:
        raise RuntimeError(
            f"Detected {audit.studies_fallback}/{audit.studies_attempted} "
            f"fallback studies ({audit.fallback_rate:.2%})."
        )


def recommended_torch_dtype():
    """Return the CN2 branch dtype, not a policy for the entire ensemble.

    Imported lazily so the repository's CPU-only utilities remain usable without
    torch installed.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the intended Kaggle inference run.")
    return torch.float16


def cn2_full_volume_centers(mask: Sequence[int]) -> List[int]:
    """Preserve V5's acquired-span recipe, including internal padding.

    Full 44-slot presence gives 42 windows. Missing edge slots shorten the
    span; internal absent slots remain in triplets to preserve preprocessing.
    This is not a filter for three individually acquired neighbouring slices.
    """
    arr = np.asarray(mask)
    if arr.ndim != 1 or len(arr) != 44 or not np.isin(arr, [0, 1]).all():
        raise ValueError("Expected a binary 44-slot acquisition mask")
    valid = np.flatnonzero(arr)
    if len(valid) < 3:
        raise ValueError("Fewer than three acquired slots")
    return list(range(int(valid[0]) + 1, int(valid[-1])))


def assert_cn2_fp16(model, inputs) -> dict:
    """Check explicit half-precision CN2 tensors immediately before forward.

    Call outside autocast. This verifies tensor policy, not kernel arithmetic.
    It intentionally does not change the parent ensemble's BF16/FP32 policies.
    """
    import torch

    if inputs.device.type != "cuda" or inputs.dtype != torch.float16:
        raise RuntimeError("CN2 input must be CUDA FP16")
    tensors = [*model.parameters(), *model.buffers()]
    floating = [t for t in tensors if t.is_floating_point()]
    if not floating or model.training:
        raise RuntimeError("CN2 must be an evaluated model with floating tensors")
    if any(t.device != inputs.device for t in tensors):
        raise RuntimeError("CN2 model/input device mismatch")
    if any(t.dtype != torch.float16 for t in floating):
        raise RuntimeError("CN2 floating model tensors must be FP16")
    return {"device": str(inputs.device), "dtype": str(inputs.dtype)}


class StrictCohortAudit:
    """Main-thread ledger: every expected study/model pair must succeed once.

    Record forwards only after validating their prediction. A failed call is
    counted separately. Do not use InferenceAudit's aggregate-only counters as
    the release gate. This ledger proves recorded coverage, not model identity;
    the loader must also verify checkpoint names/hashes and label order.
    """

    def __init__(self, study_ids, model_ids, n_labels=12):
        self.study_ids = tuple(map(str, study_ids))
        self.model_ids = tuple(map(str, model_ids))
        for values in (self.study_ids, self.model_ids):
            if not values or len(values) != len(set(values)):
                raise ValueError("Expected nonempty, unique study/model identities")
        if n_labels < 1:
            raise ValueError("Expected a positive label count")
        self.n_labels = n_labels
        self.records = {}
        self.failures = []
        self.prepared = {}
        self.precision = {}
        self._started = perf_counter()

    def record_preparation(self, uid, mask, centers, seconds):
        uid = str(uid)
        if uid not in self.study_ids or uid in self.prepared:
            raise ValueError("Unexpected or duplicate study preparation")
        expected = cn2_full_volume_centers(mask)
        if list(centers) != expected:
            raise ValueError("CN2 full-volume centers differ from the V5 recipe")
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid preparation time")
        self.prepared[uid] = {"windows": len(expected),
                              "acquired_slots": int(np.asarray(mask).sum()),
                              "preparation_seconds": float(seconds)}

    def record_forward(self, uid, model_id, prediction, seconds):
        key = (str(uid), str(model_id))
        if key[0] not in self.prepared or key[1] not in self.model_ids:
            raise ValueError("Unprepared study or unexpected model")
        if key in self.records:
            raise ValueError("Duplicate study/model forward")
        arr = assert_prediction_vector(prediction, self.n_labels)
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid forward time")
        self.records[key] = float(seconds)
        return arr

    def record_failure(self, uid, error, model_id=None):
        if str(uid) not in self.study_ids:
            raise ValueError("Unexpected failed study")
        if model_id is not None and str(model_id) not in self.model_ids:
            raise ValueError("Unexpected failed model")
        self.failures.append({"uid": str(uid), "model": model_id,
                              "error": str(error)[:500]})

    def assert_complete(self, predictions, prediction_ids, prediction_models):
        if tuple(map(str, prediction_ids)) != self.study_ids:
            raise RuntimeError("CN2 prediction study order/coverage mismatch")
        if tuple(map(str, prediction_models)) != self.model_ids:
            raise RuntimeError("CN2 prediction model order/coverage mismatch")
        expected = {(u, m) for u in self.study_ids for m in self.model_ids}
        if self.failures or set(self.records) != expected:
            raise RuntimeError("CN2 requires all study/model forwards; fallback forbidden")
        arr = np.asarray(predictions)
        shape = (len(self.model_ids), len(self.study_ids), self.n_labels)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise RuntimeError("CN2 raw predictions have invalid shape or NaN/Inf")
        if np.any((arr < 0) | (arr > 1)):
            raise RuntimeError("CN2 raw predictions outside [0,1]")

    def summary(self):
        complete = sum(all((u, m) in self.records for m in self.model_ids)
                       for u in self.study_ids)
        return {"studies_expected": len(self.study_ids),
                "studies_prepared": len(self.prepared),
                "studies_succeeded": complete,
                "forward_calls_succeeded": len(self.records),
                "forward_calls_expected": len(self.study_ids)*len(self.model_ids),
                "failures": self.failures,
                "fallback_policy": "forbidden",
                "precision_by_model": self.precision,
                "elapsed_seconds": perf_counter()-self._started,
                "studies": [{"uid": u, **self.prepared.get(u, {}),
                             "forward_seconds": sum(self.records.get((u, m), 0.)
                                                    for m in self.model_ids)}
                            for u in self.study_ids]}


def print_audit(audit: InferenceAudit) -> None:
    s = audit.summary()
    print("=== INFERENCE AUDIT ===")
    print(f"studies_attempted: {s['studies_attempted']}")
    print(f"studies_succeeded: {s['studies_succeeded']}")
    print(f"studies_fallback:  {s['studies_fallback']}")
    print(f"branch_failures:   {s['branch_failures']}")
    print(f"forward_calls:     {s['forward_calls']}")
    print(f"elapsed_seconds:   {s['elapsed_seconds']:.1f}")
    print(f"fallback_rate:     {s['fallback_rate']:.4%}")
    print(f"sec_per_study:     {s['mean_seconds_per_study']:.3f}")


if __name__ == "__main__":
    # Sanity check for the current preprocessed volume size.
    w = all_three_slice_windows(44)
    assert len(w) == 42
    assert w[0] == (0, 1, 2)
    assert w[-1] == (41, 42, 43)
    print("44-slice sanity check passed: 42 contiguous 3-slice windows.")
