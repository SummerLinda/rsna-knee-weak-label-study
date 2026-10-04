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
from typing import Iterable, List, Sequence, Tuple

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
    """Return torch.float16 for CUDA competition inference.

    Imported lazily so the repository's CPU-only utilities remain usable without
    torch installed.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the intended Kaggle inference run.")
    return torch.float16


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
