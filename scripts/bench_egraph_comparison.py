#!/usr/bin/env python3
"""Generate run-1..12 math databases once, then measure a comparison build.

Build first: cargo build --release -p egraph-comparison --example benchmark
            cargo build --release -p egglog -p egraph-comparison --bins
Data/results stay below target/ by default. Comparisons use renamed/reordered
copies, never the same Database object. Parsing and CLI timing are separate.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def generate(output):
    source = (ROOT / "tests/math-microbenchmark.egg").read_text()
    assert source.count("(run 11)") == 1
    source = source.split("(run 11)")[0]
    corpus = []
    for n in range(1, 13):
        program = output / f"math-{n:02}.egg"
        program.write_text(source + f"(run {n})\n")
        start = time.perf_counter()
        with (output / f"math-{n:02}.log").open("w") as log:
            subprocess.run([ROOT / "target/release/egglog", "--to-comparison-json", program],
                           stdout=log, stderr=log, check=True, cwd=ROOT)
        elapsed = time.perf_counter() - start
        left = program.with_suffix(".comparison.json")
        database = json.loads(left.read_text())
        names = {name: f"renamed-{i:09}" for i, name in enumerate(reversed(database["classes"]))}
        database["classes"] = {names[key]: value for key, value in reversed(list(database["classes"].items()))}
        for row in database["rows"]:
            row["inputs"] = [names[key] for key in row["inputs"]]
            row["output"] = names[row["output"]]
        database["rows"].reverse()
        right = output / f"math-{n:02}.renamed.json"
        right.write_text(json.dumps(database, indent=2))
        record = {"run": n, "left": str(left), "right": str(right), "generate_seconds": elapsed,
                  "classes": len(database["classes"]), "rows": len(database["rows"])}
        corpus.append(record)
        print(json.dumps(record), flush=True)
    (output / "corpus.json").write_text(json.dumps(corpus, indent=2))


def measure(output, label, samples, min_ms, bin_dir, runs, canonical):
    corpus = json.loads((output / "corpus.json").read_text())
    results = []
    for case in corpus:
        if case["run"] not in runs:
            continue
        command = [bin_dir / "examples/benchmark", case["left"], case["right"],
                   "--samples", str(samples), "--min-sample-ms", str(min_ms)]
        if canonical:
            command.append("--canonical")
            if case["run"] == 12:
                command.append("--single-pass")
        data = subprocess.check_output(command, cwd=ROOT)
        row = {"run": case["run"], **json.loads(data)}
        cli = []
        for _ in range(0 if canonical else samples + 1):
            start = time.perf_counter()
            subprocess.run([bin_dir / "egraph-comparison", case["left"], case["right"]],
                           stdout=subprocess.DEVNULL, check=True, cwd=ROOT)
            cli.append((time.perf_counter() - start) * 1000)
        if cli:
            row["cli_median_ms"] = statistics.median(cli[1:])
        results.append(row)
        (output / f"{label}.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["generate", "measure"])
    parser.add_argument("--output", type=Path, default=ROOT / "target/comparison-performance")
    parser.add_argument("--canonical", action="store_true",
                        help="measure canonicalization; run 12 uses one pass; skip CLI timings")
    parser.add_argument("--label", default="measure")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--bin-dir", type=Path, default=ROOT / "target/release")
    parser.add_argument("--runs", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--min-sample-ms", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1 or args.min_sample_ms < 1:
        parser.error("samples and min-sample-ms must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        generate(output)
    else:
        measure(output, args.label, args.samples, args.min_sample_ms, args.bin_dir.resolve(), args.runs, args.canonical)


if __name__ == "__main__":
    main()
