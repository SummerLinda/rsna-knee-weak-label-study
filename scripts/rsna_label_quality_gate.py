"""Aggregate the RSNA report-label coverage gate without exporting patient rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_rsna_report_masks import TARGETS, UID


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    m = pd.read_csv(args.manifest, dtype={UID: str}).set_index(UID)
    source = pd.read_csv(args.train_csv, dtype={UID: str}).set_index(UID)
    if not m.index.is_unique or not source.index.is_unique or set(m.index) != set(source.index):
        raise ValueError("UID alignment mismatch")
    if m.partition.value_counts().to_dict() != {"train": 3476, "heldout": 869, "excluded_gold_group": 62}:
        raise ValueError("Unexpected split size")
    if set(m.loc[m.partition.eq("train"), "report_group_sha256"]) & set(m.loc[m.partition.eq("heldout"), "report_group_sha256"]):
        raise ValueError("Overlapping report groups")
    gold = source[TARGETS].notna().all(axis=1)
    if int(gold.sum()) != 58 or (m.loc[gold, "partition"] != "excluded_gold_group").any():
        raise ValueError("Expert split unexpected")
    summary = []
    for target in TARGETS:
        row: dict = {"target": target, "gold58_positive": int(source.loc[gold, target].sum())}
        for split in ("train", "heldout"):
            part = m[m.partition.eq(split)]
            explicit = part[f"{target}__explicit_mask"].astype(bool)
            disagree = part[f"{target}__readers_disagree"].astype(bool)
            verdict = part[f"{target}__pilkwang_verdict"].astype(str)
            unknown = verdict.isin(("UNK", "MISSING"))
            if int((disagree & ~explicit).sum()) or not verdict.isin(("YES", "NO", "UNK", "MISSING")).all():
                raise ValueError(f"Malformed {split}/{target} label provenance")
            row[split] = {
                "studies": len(part),
                "steven_explicit": int(explicit.sum()),
                "steven_unknown": int((~explicit).sum()),
                "pilkwang_yes": int(verdict.eq("YES").sum()),
                "pilkwang_no": int(verdict.eq("NO").sum()),
                "pilkwang_unknown": int(unknown.sum()),
                "explicit_contradiction": int(disagree.sum()),
            }
        summary.append(row)
    output = {
        "scope": "Aggregate quality audit of report-derived training labels only; no extra expert truth or leaderboard evidence",
        "input_sha256": {"manifest": digest(args.manifest), "train_csv": digest(args.train_csv)},
        "split_counts": m.partition.value_counts().to_dict(),
        "targets": summary,
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({r["target"]: {"unknown_train": r["train"]["pilkwang_unknown"], "contradictions_train": r["train"]["explicit_contradiction"]} for r in summary}, indent=2))


if __name__ == "__main__":
    main()
