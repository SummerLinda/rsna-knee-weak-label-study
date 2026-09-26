"""Paired CPU screen: Steven v2 versus credited Steven/lixin73 v4 labels.

The held report-derived verdicts are a different public labeler's output,
not independent expert MRI ground truth. Gold46 was used in source selection.
Neither endpoint supports claims about V3 or hidden-test improvement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from prepare_rsna_report_masks import TARGETS, UID
from rsna_cpu_mask_pilot import (EXPECTED_FEATURE_SHA, fit_pair, frozen_features,
                                 group_bootstrap_delta, score, sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--steven-v4", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4723)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or not 100 <= args.bootstrap <= 2000:
        raise ValueError("Pilot bounds: 1..20 epochs and 100..2000 bootstrap draws")
    begun = time.perf_counter()
    ids, x = frozen_features(args.features)
    manifest = pd.read_csv(args.manifest, dtype={UID: str}).set_index(UID)
    v4 = pd.read_csv(args.steven_v4, dtype={UID: str}).set_index(UID)
    official = pd.read_csv(args.train_csv, dtype={UID: str}).set_index(UID)
    if any(not f.index.is_unique for f in (manifest, v4, official)):
        raise ValueError("Duplicate UIDs")
    if set(manifest.index) != set(v4.index) or set(manifest.index) != set(official.index):
        raise ValueError("Study UID sets differ")
    m = manifest.reindex(ids)
    train, held = m.partition.eq("train").to_numpy(), m.partition.eq("heldout").to_numpy()
    if (int(train.sum()), int(held.sum())) != (3476, 869):
        raise ValueError("Locked partition/count changed")
    if set(m.loc[train,"report_group_sha256"]) & set(m.loc[held,"report_group_sha256"]):
        raise ValueError("Report group leakage")
    y2 = np.column_stack([m[f"{t}__soft_target"].to_numpy("float32") for t in TARGETS])
    y4 = v4.reindex(ids)[TARGETS].to_numpy("float32")
    for y in (y2, y4):
        if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1):
            raise ValueError("Invalid label range")
    verdict = np.column_stack([m[f"{t}__pilkwang_verdict"].to_numpy(str) for t in TARGETS])
    if not set(np.unique(verdict)).issubset({"YES", "NO", "UNK", "MISSING"}):
        raise ValueError("Unrecognized verdict")
    explicit = np.isin(verdict, ["YES", "NO"])
    truth = np.where(verdict == "YES", 1., 0.).astype("float32")
    gold = official.reindex(ids)[TARGETS].to_numpy("float32")
    gold_rows = np.isfinite(gold).all(axis=1)
    if int(gold_rows.sum()) != 46 or np.any(train & gold_rows) or np.any(held & gold_rows):
        raise ValueError("Gold diagnostic rows not segregated")
    x -= x.mean(axis=1, keepdims=True)
    x /= np.sqrt((x*x).mean(axis=1, keepdims=True) + 1e-5)
    weights2, biases2, weights4, biases4 = fit_pair(
        x[train], y2[train], np.ones_like(y2[train]), args.epochs, args.seed,
        candidate_targets=y4[train])
    pred2, pred4 = expit(x @ weights2 + biases2), expit(x @ weights4 + biases4)
    held2 = score(truth[held], pred2[held], explicit[held])
    held4 = score(truth[held], pred4[held], explicit[held])
    held_ci = group_bootstrap_delta(truth[held],pred2[held],pred4[held],explicit[held],
                                    m.loc[held,"report_group_sha256"].to_numpy(), args.bootstrap)
    mask_gold = np.ones((46, len(TARGETS)), dtype=bool)
    gold2 = score(gold[gold_rows],pred2[gold_rows],mask_gold)
    gold4 = score(gold[gold_rows],pred4[gold_rows],mask_gold)
    gold_ci = group_bootstrap_delta(gold[gold_rows],pred2[gold_rows],pred4[gold_rows],
                                    mask_gold,m.loc[gold_rows,"report_group_sha256"].to_numpy(),
                                    args.bootstrap)
    result = {
        "comparison":"same cached ConvNeXt features/head/batches; only Steven v2 vs Steven/lixin73 v4 soft training labels change",
        "caveat":"Pilkwang held verdicts are separate report-derived labels, not independent expert labels; Gold46 used in public label selection; no V3 comparison or hidden-test claim",
        "attribution":"Steven Lee Hans v2 vs Steven Lee Hans v4 blended with lixin73 report labels",
        "seed":args.seed,"epochs":args.epochs,"train_rows":int(train.sum()),
        "held_rows":int(held.sum()),"gold_diagnostic_rows":int(gold_rows.sum()),
        "source_sha256":{"features":EXPECTED_FEATURE_SHA,"manifest":sha256(args.manifest),
                         "v4":sha256(args.steven_v4),"official":sha256(args.train_csv)},
        "elapsed_seconds":round(time.perf_counter()-begun,2),
        "held_pilkwang_v2":held2,"held_pilkwang_v4":held4,
        "held_group_bootstrap_delta":held_ci,
        "gold46_v2":gold2,"gold46_v4":gold4,"gold_group_bootstrap_delta":gold_ci,
    }
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"seed":args.seed,"held_v2":held2["macro_auc"],
                      "held_v4":held4["macro_auc"],"held_ci":held_ci["ci95"],
                      "gold_v2":gold2["macro_auc"],"gold_v4":gold4["macro_auc"],
                      "gold_ci":gold_ci["ci95"],"elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__ == "__main__":
    main()
