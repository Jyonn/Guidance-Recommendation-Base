"""Prepare deterministic per-user chronological MovieLens-100K splits."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def preprocess(source, output):
    users = defaultdict(list)
    original = Counter()
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            user, item, score, timestamp = map(int, line.split())
            if not 1 <= score <= 5:
                raise ValueError(f"Invalid score on line {index + 1}")
            original[score] += 1
            users[user].append((timestamp, index, item, score))
    splits = {name: [] for name in ("train", "validation", "test")}
    before_filter = Counter()
    dropped = Counter()
    for user, events in sorted(users.items()):
        events.sort(key=lambda row: (row[0], row[2], row[1]))
        n = len(events)
        a = n * 8 // 10
        b = a + n // 10
        for name, part in zip(splits, (events[:a], events[a:b], events[b:])):
            before_filter[name] += len(part)
            dropped[name] += sum(score == 3 for _, _, _, score in part)
            splits[name].extend((user, item, score, timestamp, int(score > 3))
                                for timestamp, _, item, score in part if score != 3)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_rows": sum(original.values()),
        "dropped_score_3": original[3],
        "retained_rows": sum(map(len, splits.values())),
        "users": len(users),
        "items": len({r[1] for rows in splits.values() for r in rows}),
        "split_rule": "Per user on all interactions; timestamp then item_id order; train length floor(0.8*N), validation length floor(0.1*N), remainder test; then remove score=3 within each split",
        "splits": {},
    }
    for name, rows in splits.items():
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("user_id", "item_id", "score", "timestamp", "label"))
            writer.writerows(rows)
        summary["splits"][name] = {
            "rows_before_filter": before_filter[name],
            "dropped_score_3": dropped[name],
            "rows": len(rows), "positive": sum(r[4] for r in rows),
            "negative": sum(1 - r[4] for r in rows),
        }
    (ROOT / "stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/u.data")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    preprocess(args.input, args.output)
