from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "artifacts" / "crypto_bench" / "results.csv"

METRICS = [
    "decrypt_ms",
    "jit_load_ms",
    "total_create_inner_ms",
    "total_create_wall_ms",
]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def summarize(results_path: Path) -> None:
    rows_by_algorithm: dict[str, list[dict[str, str]]] = defaultdict(list)
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_algorithm[row["algorithm"]].append(row)

    if not rows_by_algorithm:
        raise RuntimeError(f"no benchmark rows found in {results_path}")

    header = [
        "algorithm",
        "runs",
        "encrypted_mb",
        "plain_mb",
        "decrypt_avg_ms",
        "decrypt_min_ms",
        "jit_load_avg_ms",
        "total_create_avg_ms",
        "wall_avg_ms",
    ]
    print(",".join(header))
    for algorithm, rows in rows_by_algorithm.items():
        metric_values = {
            metric: [float(row[metric]) for row in rows]
            for metric in METRICS
        }
        encrypted_mb = float(rows[0]["encrypted_bytes"]) / (1024 * 1024)
        plain_mb = float(rows[0]["plaintext_bytes"]) / (1024 * 1024)
        print(
            ",".join(
                [
                    algorithm,
                    str(len(rows)),
                    fmt(encrypted_mb),
                    fmt(plain_mb),
                    fmt(mean(metric_values["decrypt_ms"])),
                    fmt(min(metric_values["decrypt_ms"])),
                    fmt(mean(metric_values["jit_load_ms"])),
                    fmt(mean(metric_values["total_create_inner_ms"])),
                    fmt(mean(metric_values["total_create_wall_ms"])),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize crypto model_create benchmark CSV output.")
    parser.add_argument("results", type=Path, nargs="?", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    summarize(args.results)


if __name__ == "__main__":
    main()
