"""Audit RSNA report-derived labels and prepare an explicit-observation training mask.

The Steven v2 synovitis column contains effusion-proxy values for initially
unmentioned findings; only Steven v1 can identify the originally explicit rows.
No gold labels, MRI images, or Kaggle test data are read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def load(path: Path, expected: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={UID: str})
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError(f"{path}: null or duplicate study IDs")
    return frame.set_index(UID)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(steven_v1: Path, steven_v2: Path, pilkwang: Path,
          train: Path, split_seed: str) -> tuple[pd.DataFrame, dict]:
    old = load(steven_v1, TARGETS)
    new = load(steven_v2, TARGETS)
    pil = load(pilkwang, [f"{t}__verdict" for t in TARGETS])
    if set(old.index) != set(new.index):
        raise ValueError("Steven v1 and v2 study IDs differ")
    old = old.reindex(new.index)
    missing_pil = new.index.difference(pil.index).tolist()
    extra_pil = pil.index.difference(new.index).tolist()
    pil = pil.reindex(new.index)
    out = pd.DataFrame(index=new.index)
    audit = {
        "studies": len(new),
        "pilkwang_missing_ids": missing_pil,
        "pilkwang_extra_ids": extra_pil,
        "input_sha256": {str(p): sha256(p) for p in (steven_v1, steven_v2, pilkwang)},
        "targets": {},
    }
    for target in TARGETS:
        prev = pd.to_numeric(old[target], errors="raise")
        value = pd.to_numeric(new[target], errors="raise")
        if not (prev.between(0, 1).all() and value.between(0, 1).all()):
            raise ValueError(f"{target}: Steven score outside [0, 1]")
        if target != "Synovitis" and not prev.equals(value):
            raise ValueError(f"{target}: unexpected change from Steven v1 to v2")
        explicit = prev.ne(.5) if target == "Synovitis" else value.ne(.5)
        verdict = pil[f"{target}__verdict"]
        if not verdict.dropna().isin(["YES", "NO", "UNK"]).all():
            raise ValueError(f"{target}: unrecognized Pilkwang verdict")
        pil_observed = verdict.isin(["YES", "NO"])
        both = explicit & pil_observed
        disagree = both & (value.gt(.5) != verdict.eq("YES"))
        out[f"{target}__soft_target"] = value
        out[f"{target}__explicit_mask"] = explicit.astype("int8")
        out[f"{target}__pilkwang_verdict"] = verdict.fillna("MISSING")
        out[f"{target}__readers_disagree"] = disagree.astype("int8")
        audit["targets"][target] = {
            "steven_explicit": int(explicit.sum()),
            "steven_unknown": int((~explicit).sum()),
            "pilkwang_unknown": int(verdict.eq("UNK").sum()),
            "both_observed": int(both.sum()),
            "disagree_on_both_observed": int(disagree.sum()),
            "steven_v1_to_v2_changes": int(value.ne(prev).sum()),
            "synovitis_proxy_filled": int((prev.eq(.5) & value.ne(.5)).sum()),
        }
    meta = load(train, ["Report", *TARGETS]).reindex(new.index)
    if meta["Report"].isna().any() or len(meta) != len(new):
        raise ValueError("train.csv missing a report for a Steven study")
    if not set(new.index).issubset(set(load(train, ["Report", *TARGETS]).index)):
        raise ValueError("train.csv IDs do not include all Steven studies")
    label_present = meta[TARGETS].notna()
    if label_present.any(axis=1).ne(label_present.all(axis=1)).any():
        raise ValueError("partial gold rows need separate adjudication")
    report_norm = meta["Report"].str.lower().str.split().str.join(" ")
    group_id = report_norm.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    gold = label_present.any(axis=1)
    excluded = group_id.isin(set(group_id[gold]))
    eligible = (~excluded).groupby(group_id).sum()
    eligible = eligible[eligible.gt(0)]
    target_held = round(int((~excluded).sum()) * .2)
    order = sorted(eligible.index, key=lambda g: hashlib.sha256(
        (split_seed + "\0" + g).encode("utf-8")).hexdigest())
    selected: set[str] = set()
    held_count = 0
    for group in order:
        size = int(eligible[group])
        if abs(held_count + size - target_held) <= abs(held_count - target_held):
            selected.add(group)
            held_count += size
        if held_count == target_held:
            break
    partition = pd.Series("train", index=new.index)
    partition.loc[excluded] = "excluded_gold_group"
    partition.loc[group_id.isin(selected) & ~excluded] = "heldout"
    out["report_group_sha256"] = group_id
    out["partition"] = partition
    audit["split"] = {
        "seed": split_seed,
        "gold_rows": int(gold.sum()),
        "gold_group_excluded": int(excluded.sum()),
        "train_rows": int(partition.eq("train").sum()),
        "heldout_rows": int(partition.eq("heldout").sum()),
        "overlap_groups": len(set(group_id[partition.eq("train")]) & set(group_id[partition.eq("heldout")])),
        "heldout_targets": {},
    }
    for target in TARGETS:
        active = partition.eq("heldout") & out[f"{target}__explicit_mask"].eq(1)
        vals = out.loc[active, f"{target}__soft_target"]
        audit["split"]["heldout_targets"][target] = {
            "explicit": int(active.sum()),
            "positive_over_half": int(vals.gt(.5).sum()),
            "negative_below_half": int(vals.lt(.5).sum()),
        }
    if audit["split"]["overlap_groups"] or audit["split"]["gold_group_excluded"] < 58:
        raise AssertionError("split leakage")
    return out.reset_index(), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steven-v1", type=Path, required=True)
    parser.add_argument("--steven-v2", type=Path, required=True)
    parser.add_argument("--pilkwang", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split-seed", default="rsna-v4-mask-holdout-2026-09-25")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    frame, audit = build(args.steven_v1, args.steven_v2, args.pilkwang,
                         args.train, args.split_seed)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"csv": str(csv_path), "audit": str(json_path),
                      "studies": len(frame), "targets": len(TARGETS),
                      "synovitis_explicit": audit["targets"]["Synovitis"]["steven_explicit"],
                      "pilkwang_missing": len(audit["pilkwang_missing_ids"]),
                      "split": audit["split"]}, indent=2))


if __name__ == "__main__":
    main()
