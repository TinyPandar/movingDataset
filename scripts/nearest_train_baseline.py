from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a nearest-train coordinate baseline for dark-intersection datasets."
    )
    parser.add_argument("dataset_root", type=Path)
    return parser.parse_args()


def read_rows(path: Path, source: str) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "source": source,
                    "split": row["split"],
                    "rotation_deg": int(float(row["rotation_deg"])) % 360,
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                }
            )
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def nearest_distances(rows: list[dict[str, object]]) -> tuple[list[float], int]:
    train_by_rotation: dict[int, list[tuple[float, float]]] = defaultdict(list)
    val_rows: list[dict[str, object]] = []
    for row in rows:
        if row["split"] == "train":
            train_by_rotation[int(row["rotation_deg"])].append((float(row["center_x"]), float(row["center_y"])))
        elif row["split"] == "val":
            val_rows.append(row)

    distances: list[float] = []
    missing_rotation_count = 0
    for row in val_rows:
        train_points = train_by_rotation[int(row["rotation_deg"])]
        if not train_points:
            missing_rotation_count += 1
            continue
        x = float(row["center_x"])
        y = float(row["center_y"])
        best_sq = min((x - tx) * (x - tx) + (y - ty) * (y - ty) for tx, ty in train_points)
        distances.append(math.sqrt(best_sq))
    return distances, missing_rotation_count


def summarize(name: str, rows: list[dict[str, object]]) -> None:
    train_count = sum(1 for row in rows if row["split"] == "train")
    val_count = sum(1 for row in rows if row["split"] == "val")
    distances, missing_rotation_count = nearest_distances(rows)
    print(f"[{name}] rows={len(rows)} train={train_count} val={val_count}")
    if missing_rotation_count:
        print(f"  skipped val rows with no same-rotation train sample: {missing_rotation_count}")
    if not distances:
        print("  no validation rows")
        return
    print(
        "  nearest-train error px: "
        f"mean={sum(distances) / len(distances):.4f} "
        f"median={percentile(distances, 0.50):.4f} "
        f"p90={percentile(distances, 0.90):.4f} "
        f"p95={percentile(distances, 0.95):.4f} "
        f"max={max(distances):.4f}"
    )
    close_1 = sum(1 for value in distances if value <= 1.0)
    close_2 = sum(1 for value in distances if value <= 2.0)
    close_5 = sum(1 for value in distances if value <= 5.0)
    print(
        "  close val rows: "
        f"<=1px={close_1}/{len(distances)} ({close_1 / len(distances):.1%}) "
        f"<=2px={close_2}/{len(distances)} ({close_2 / len(distances):.1%}) "
        f"<=5px={close_5}/{len(distances)} ({close_5 / len(distances):.1%})"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    random_rows = read_rows(dataset_root / "random" / "annotations.csv", "random")
    straight_rows = read_rows(dataset_root / "straight" / "keyframes_manifest.csv", "straight")

    summarize("random", random_rows)
    summarize("straight", straight_rows)
    summarize("combined", random_rows + straight_rows)


if __name__ == "__main__":
    main()
