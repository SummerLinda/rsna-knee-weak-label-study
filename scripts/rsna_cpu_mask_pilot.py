"""Paired, CPU-only frozen-MRI-feature screen for the RSNA unknown-label mask.

Reads torch's ZIP storage directly after SHA256 verification. Never unpickles
the downloaded checkpoint or treats report-derived labels as expert truth.
This is a screening experiment, not a competition submission model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from prepare_rsna_report_masks import TARGETS, UID


EXPECTED_FEATURE_SHA = "7b2d7e864205dfe71b222050d354e96712c3902a9e84aafea5fc30019d845139"
FEATURE_SHAPE = (4395, 4608)


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    return digest.hexdigest()


def frozen_features(path: Path) -> tuple[list[str], np.ndarray]:
    if sha256(path) != EXPECTED_FEATURE_SHA:
        raise ValueError("Frozen feature cache does not match the audited source")
    with zipfile.ZipFile(path) as archive:
        prefix = "fold0_train_frozen_features/"
        ops = list(pickletools.genops(archive.read(prefix + "data.pkl")))
        uids = [arg for op, arg, _ in ops if op.name == "BINUNICODE"
                and isinstance(arg, str) and arg.startswith("1.2.826.")]
        raw = archive.read(prefix + "data/0")
    if len(uids) != FEATURE_SHAPE[0] or len(set(uids)) != len(uids):
        raise ValueError("UID count/order could not be established safely")
    if len(raw) != np.prod(FEATURE_SHAPE) * 4:
        raise ValueError("Unexpected feature tensor size")
    x = np.frombuffer(raw, dtype="<f4").reshape(FEATURE_SHAPE).copy()
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite cached feature")
    return uids, x


def score(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    per = {}
    for j, target in enumerate(TARGETS):
        keep = mask[:, j] & np.isfinite(y[:, j])
        actual = y[keep, j] > .5
        per[target] = dict(n=int(keep.sum()), pos=int(actual.sum()),
                           auc=(float(roc_auc_score(actual, pred[keep, j]))
                                if np.unique(actual).size == 2 else None))
    au = [v["auc"] for v in per.values() if v["auc"] is not None]
    return {"macro_auc": float(np.mean(au)) if au else None,
            "n_targets": len(au), "per_target": per}


def group_bootstrap_delta(y: np.ndarray, control: np.ndarray, candidate: np.ndarray,
                          mask: np.ndarray, groups: np.ndarray,
                          draws: int = 500) -> dict:
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(20260925)
    diffs = []
    for _ in range(draws):
        chosen = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[i] for i in chosen])
        a = score(y[idx], control[idx], mask[idx])
        b = score(y[idx], candidate[idx], mask[idx])
        if a["n_targets"] == b["n_targets"] == len(TARGETS):
            diffs.append(b["macro_auc"]-a["macro_auc"])
    if len(diffs) < draws * .8:
        return {"valid_draws":len(diffs), "requested_draws":draws,
                "ci95":None, "reason":"too many single-class bootstrap draws"}
    return {"valid_draws":len(diffs),"requested_draws":draws,
            "ci95":np.quantile(diffs,[.025,.975]).tolist(),
            "fraction_positive":float(np.mean(np.array(diffs)>0))}


def fit_pair(x: np.ndarray, y: np.ndarray, explicit: np.ndarray,
             epochs: int, seed: int,
             candidate_targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Same initialization and deterministic batches for both arms. 12 independent
    # sigmoid heads on one frozen ConvNeXt embedding; no GPU/backbone fine-tuning.
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    init = rng.normal(0, .002, size=(dim, len(TARGETS))).astype("float32")
    weights = [init.copy(), init.copy()]
    biases = [np.zeros(len(TARGETS), dtype="float32") for _ in range(2)]
    ms = [np.zeros_like(init) for _ in range(2)]
    vs = [np.zeros_like(init) for _ in range(2)]
    mb = [np.zeros(len(TARGETS), dtype="float32") for _ in range(2)]
    vb = [np.zeros(len(TARGETS), dtype="float32") for _ in range(2)]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            idx = order[start:start+256]
            xx = x[idx]
            yy = y[idx]
            step += 1
            for arm, mask in enumerate((np.ones_like(yy), explicit[idx])):
                actual_target = yy if arm == 0 or candidate_targets is None else candidate_targets[idx]
                logits = xx @ weights[arm] + biases[arm]
                grad_z = (expit(logits) - actual_target) * mask
                grad_z /= np.maximum(mask.sum(axis=0, keepdims=True), 1)
                gw = xx.T @ grad_z + .001 * weights[arm]
                gb = grad_z.sum(axis=0)
                ms[arm] = .9 * ms[arm] + .1 * gw
                vs[arm] = .999 * vs[arm] + .001 * gw * gw
                mb[arm] = .9 * mb[arm] + .1 * gb
                vb[arm] = .999 * vb[arm] + .001 * gb * gb
                weights[arm] -= .0003 * (ms[arm] / (1-.9**step)) / (np.sqrt(vs[arm] / (1-.999**step)) + 1e-8)
                biases[arm] -= .0003 * (mb[arm] / (1-.9**step)) / (np.sqrt(vb[arm] / (1-.999**step)) + 1e-8)
        print(f"epoch {epoch+1}/{epochs} complete", flush=True)
    return weights[0], biases[0], weights[1], biases[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=4723)
    a = p.parse_args()
    if not 1 <= a.epochs <= 20:
        raise ValueError("Pilot epoch bound is 1..20")
    begun = time.perf_counter()
    ids, x = frozen_features(a.features)
    manifest = pd.read_csv(a.manifest, dtype={UID: str}).set_index(UID)
    official = pd.read_csv(a.train_csv, dtype={UID: str}).set_index(UID)
    if not manifest.index.is_unique or not official.index.is_unique:
        raise ValueError("Duplicate study UID")
    if set(manifest.index) != set(official.index):
        raise ValueError("Manifest IDs differ from train.csv")
    m = manifest.reindex(ids)
    if m.partition.isna().any():
        raise ValueError("Feature UID missing from manifest")
    train = m.partition.eq("train").to_numpy()
    held = m.partition.eq("heldout").to_numpy()
    if (int(train.sum()), int(held.sum())) != (3476, 869):
        raise ValueError("Locked split size changed or missing cache features")
    if set(m.loc[train, "report_group_sha256"]) & set(m.loc[held, "report_group_sha256"]):
        raise ValueError("Report group leakage")
    y = np.column_stack([m[f"{t}__soft_target"].to_numpy("float32") for t in TARGETS])
    mask = np.column_stack([m[f"{t}__explicit_mask"].to_numpy(bool) for t in TARGETS])
    if not (np.isfinite(y).all() and np.logical_and(0<=y,y<=1).all()):
        raise ValueError("Invalid report targets")
    # The source head uses a LayerNorm over the 4,608 frozen dimensions.
    x -= x.mean(axis=1, keepdims=True)
    x /= np.sqrt((x*x).mean(axis=1, keepdims=True) + 1e-5)
    w0,b0,w1,b1 = fit_pair(x[train],y[train],mask[train].astype("float32"),a.epochs,a.seed)
    base = expit(x @ w0 + b0)
    masked = expit(x @ w1 + b1)
    # Held comparison uses the same explicit report-label rows for both arms.
    held_control = score(y[held],base[held],mask[held])
    held_masked = score(y[held],masked[held],mask[held])
    gold = official.reindex(ids)[TARGETS].to_numpy("float32")
    gold_rows = np.isfinite(gold).all(axis=1)
    if int(gold_rows.sum()) != 46 or np.any(train & gold_rows) or np.any(held & gold_rows):
        raise ValueError("Expert-label diagnostic is not segregated")
    gold_control = score(gold[gold_rows],base[gold_rows],np.ones((46,12),bool))
    gold_masked = score(gold[gold_rows],masked[gold_rows],np.ones((46,12),bool))
    held_ci = group_bootstrap_delta(y[held],base[held],masked[held],mask[held],
                                    m.loc[held,"report_group_sha256"].to_numpy())
    gold_ci = group_bootstrap_delta(gold[gold_rows],base[gold_rows],masked[gold_rows],
                                    np.ones((46,12),bool),
                                    m.loc[gold_rows,"report_group_sha256"].to_numpy())
    result = {
        "scope":"CPU frozen ConvNeXt head pilot; 869 report-derived held labels and 46 small, previously label-source-used expert rows; no independent V3 comparison",
        "epochs":a.epochs,"seed":a.seed,"train_rows":int(train.sum()),"held_rows":int(held.sum()),
        "gold_diagnostic_rows":int(gold_rows.sum()),"elapsed_seconds":round(time.perf_counter()-begun,2),
        "feature_sha256":EXPECTED_FEATURE_SHA,"manifest_sha256":sha256(a.manifest),
        "held_report_control":held_control,"held_report_masked":held_masked,
        "held_group_bootstrap_delta":held_ci,
        "gold_diagnostic_control":gold_control,"gold_diagnostic_masked":gold_masked,
        "gold_group_bootstrap_delta":gold_ci,
    }
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"held_report_control":held_control["macro_auc"],
                      "held_report_masked":held_masked["macro_auc"],
                      "gold_control":gold_control["macro_auc"],
                      "gold_masked":gold_masked["macro_auc"],
                      "held_delta_ci95":held_ci["ci95"],"gold_delta_ci95":gold_ci["ci95"],
                      "elapsed_seconds":result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
