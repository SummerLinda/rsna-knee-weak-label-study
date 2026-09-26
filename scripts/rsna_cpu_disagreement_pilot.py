"""Paired CPU pilot: lower weight on disagreements between report labelers.

The 869-study held labels are report-derived and the public labelers may
have consulted Gold58. This is a methods screen, not a leaderboard predictor.
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
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4723)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--ablation", choices=("half_all", "drop_acl_conflict"), default="half_all")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or not 100 <= args.bootstrap <= 2000:
        raise ValueError("Pilot runtime bound exceeded")
    begin = time.perf_counter()
    ids, x = frozen_features(args.features)
    m = pd.read_csv(args.manifest,dtype={UID:str}).set_index(UID)
    official = pd.read_csv(args.train_csv,dtype={UID:str}).set_index(UID)
    if not m.index.is_unique or not official.index.is_unique or set(m.index)!=set(official.index):
        raise ValueError("Manifest/official UID mismatch")
    m = m.reindex(ids)
    train, held = m.partition.eq("train").to_numpy(),m.partition.eq("heldout").to_numpy()
    gold=official.reindex(ids)[TARGETS].to_numpy("float32")
    gold_rows=np.isfinite(gold).all(axis=1)
    if (int(train.sum()),int(held.sum()),int(gold_rows.sum())) != (3476,869,46):
        raise ValueError("Frozen split sizes differ")
    if np.any((train | held) & gold_rows) or set(m.loc[train,"report_group_sha256"]) & set(m.loc[held,"report_group_sha256"]):
        raise ValueError("Split leakage")
    y = np.column_stack([m[f"{t}__soft_target"].to_numpy("float32") for t in TARGETS])
    explicit = np.column_stack([m[f"{t}__explicit_mask"].to_numpy(bool) for t in TARGETS])
    disagree = np.column_stack([m[f"{t}__readers_disagree"].to_numpy(bool) for t in TARGETS])
    if np.any(disagree & ~explicit) or not np.isfinite(y).all() or np.any((y<0)|(y>1)):
        raise ValueError("Invalid report labels or disagreement provenance")
    # Precommitted ablations. Both arms train on identical Steven v2 targets;
    # only per-study/target loss weights differ on observed conflict.
    if args.ablation == "half_all":
        weights = np.where(disagree[train],0.5,1.0).astype("float32")
    else:
        weights = np.ones_like(y[train], dtype="float32")
        weights[:, TARGETS.index("ACL")] = np.where(disagree[train, TARGETS.index("ACL")],0.,1.)
    verdict = np.column_stack([m[f"{t}__pilkwang_verdict"].to_numpy(str) for t in TARGETS])
    if not set(np.unique(verdict)).issubset({"YES","NO","UNK","MISSING"}):
        raise ValueError("Unknown verdict")
    pil_mask = np.isin(verdict,["YES","NO"])
    pil_y=np.where(verdict=="YES",1.,0.).astype("float32")
    x -= x.mean(axis=1,keepdims=True)
    x /= np.sqrt((x*x).mean(axis=1,keepdims=True)+1e-5)
    w0,b0,w1,b1=fit_pair(x[train],y[train],weights,args.epochs,args.seed)
    control,candidate=expit(x@w0+b0),expit(x@w1+b1)
    held_groups=m.loc[held,"report_group_sha256"].to_numpy()
    gold_groups=m.loc[gold_rows,"report_group_sha256"].to_numpy()
    endpoints={}
    for label,truth,mask,base,treatment,groups in (
        ("held_steven_explicit",y[held],explicit[held],control[held],candidate[held],held_groups),
        ("held_pilkwang_explicit",pil_y[held],pil_mask[held],control[held],candidate[held],held_groups),
        ("gold46_diagnostic",gold[gold_rows],np.ones((46,12),bool),control[gold_rows],candidate[gold_rows],gold_groups),
    ):
        endpoints[label]={"control":score(truth,base,mask),
                          "candidate":score(truth,treatment,mask),
                          "delta_bootstrap":group_bootstrap_delta(truth,base,treatment,mask,groups,args.bootstrap)}
    result={"scope":"CPU frozen image feature paired method pilot; not V3 or test-set evidence",
            "method":("Steven v2 targets, with 0.5 training loss factor for explicit Steven/Pilkwang contradictions" if args.ablation == "half_all" else "Steven v2 targets, dropping ACL loss for explicit Steven/Pilkwang ACL contradictions; all other target losses unchanged"),
            "ablation":args.ablation,
            "limitations":"Pilkwang held labels derive from the same algorithm used to identify training contradictions; Gold46 was available to public label developers; no independent expert holdout",
            "seed":args.seed,"epochs":args.epochs,"train_rows":int(train.sum()),"held_rows":int(held.sum()),
            "disagreement_weighted_target_entries":int((weights<1).sum()),
            "feature_sha256":EXPECTED_FEATURE_SHA,"manifest_sha256":sha256(args.manifest),
            "elapsed_seconds":round(time.perf_counter()-begin,2),"endpoints":endpoints}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"seed":args.seed,"disagreement_entries":result["disagreement_weighted_target_entries"],
                      "endpoints":{k:{"control":v["control"]["macro_auc"],
                                      "candidate":v["candidate"]["macro_auc"],
                                      "ci95":v["delta_bootstrap"]["ci95"]}
                                   for k,v in endpoints.items()},"elapsed":result["elapsed_seconds"]},indent=2))


if __name__ == "__main__":
    main()
