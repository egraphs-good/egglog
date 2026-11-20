"""Build the Lean-emitted and pinned Rutgers kernels, validate all bit patterns, and time them."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from compare import check_load

PIN = "8533ec0db38b2f3ab55c1b296ddaec80d578e251"
FLAGS = ["-O3", "-fno-fast-math", "-ffp-contract=off"]
KINDS = ("positive", "all", "chain")


def capture(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def prepare(args, directory):
    upstream = args.upstream.resolve()
    sources = ["source/bfloat16/" + f + ".cpp" for f in ("log2", "log10")]
    headers = ["include/bfloat16.hpp", "include/bfloat16_math.hpp"]
    if capture(["git", "-C", upstream, "rev-parse", "HEAD"]) != PIN:
        raise SystemExit(f"Expected the Rutgers rlibm checkout at {PIN}")
    subprocess.run(["git", "-C", upstream, "diff", "--quiet", PIN, "--", *sources, *headers], check=True)
    # Run both Lean emitters. These emit the checked coefficients and the actual
    # extracted arithmetic schedules. Only these emitted routines are timed.
    with (directory / "rlibm_coefficients.h").open("w") as output:
        subprocess.run(["lake", "env", "lean", "--run", "RlibmTables.lean"],
                       cwd=LEAN, stdout=output, check=True)
    subprocess.run(["lake", "env", "lean", "--run", "RlibmEmit.lean", directory], cwd=LEAN, check=True)
    manifest = directory / "schedules.csv"
    with manifest.open() as source:
        schedules = list(csv.DictReader(source))
    entries = [(f, "upstream", "upstream_" + f) for f in ("log2", "log10")]
    for row in schedules:
        fn, name, symbol = [row[k] for k in ("function", "schedule", "symbol")]
        if (fn not in ("log2", "log10") or not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol)
                or any((f, n) == (fn, name) or s == symbol for f, n, s in entries)):
            raise ValueError(f"Invalid or duplicate schedule {row}")
        entries.append((fn, name, symbol))
    per_function = {fn: [n for f, n, _ in entries if f == fn] for fn in ("log2", "log10")}
    if (set(per_function["log2"]) != set(per_function["log10"])
            or "source" not in per_function["log2"] or len(per_function["log2"]) < 3):
        raise ValueError("Expected source and extracted schedules for both functions")
    registry = [f'extern "C" uint16_t {sym}(uint16_t);' for _, _, sym in entries]
    registry += ["static constexpr Entry entries[] = {",
                 *[f'    {{"{fn}", "{name}", &{sym}}},' for fn, name, sym in entries], "};"]
    (directory / "registry.inc").write_text("\n".join(registry) + "\n")

    mpfr_flags = shlex.split(capture(["pkg-config", "--cflags", "mpfr", "gmp"]))
    libs = shlex.split(capture(["pkg-config", "--libs", "mpfr", "gmp"]))
    inc = [f"-I{HERE}", f"-I{directory}"]
    objects = []
    generated = directory / "schedules.c"
    obj = directory / "schedules.o"
    subprocess.run([args.cc, "-std=c11", *FLAGS, *inc, "-c", generated, "-o", obj], check=True)
    objects.append(obj)
    for base in (2, 10):
        obj = directory / f"upstream{base}.o"
        subprocess.run([args.cxx, "-std=c++17", *FLAGS, f"-I{upstream}", f"-I{upstream / 'include'}",
                        *mpfr_flags, f"-DRLIBM_BASE={base}", "-c", HERE / "upstream.cpp", "-o", obj],
                       check=True)
        objects.append(obj)
    worker = directory / "driver"
    subprocess.run([args.cxx, "-std=c++17", *FLAGS, *inc, *mpfr_flags, HERE / "driver.cpp",
                    *objects, *libs, "-o", worker], check=True)
    cpu = platform.machine()
    if sys.platform == "darwin":
        probe = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if probe.returncode == 0:
            cpu = probe.stdout.strip()
        else:
            report = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
            chip = re.search(r"^\s*Chip:\s*(.+)$", report.stdout, re.MULTILINE)
            if chip:
                cpu = chip.group(1).strip()
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("model name", "Hardware")):
                cpu = line.partition(":")[2].strip()
                break
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "rutgers_rlibm_commit": PIN, "rutgers_sources": sources + headers,
        "mpfr_version": capture(["pkg-config", "--modversion", "mpfr"]),
        "c_compiler": capture([args.cc, "--version"]).splitlines()[0],
        "cxx_compiler": capture([args.cxx, "--version"]).splitlines()[0],
        "flags": FLAGS, "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
        "lean_version": capture(["lake", "env", "lean", "--version"], cwd=LEAN),
        "sweeps_per_sample": args.sweeps, "warmup_sweeps_per_process": 8,
        "native_lean_sweeps_per_sample": args.lean_sweeps, "word_sweeps_per_sample": args.word_sweeps,
        "comparison_groups": args.group,
        "balanced_blocks": args.blocks, "c_clock": "std::chrono::steady_clock",
        "lean_clock": "IO.monoNanosNow",
        "generated_sha256": {p.name: digest(p) for p in (generated, directory / "rlibm_coefficients.h", manifest)},
    }
    if "word" in args.group:
        word_sources = [LEAN / "Egglog/Examples/Rlibm/NativeWord.lean", LEAN / "RlibmWordBench.lean",
                        LEAN / ".lake/build/bin/rlibm-word-bench"]
        metadata["native_word_sha256"] = {str(p.relative_to(LEAN)): digest(p) for p in word_sources}
    (args.out / "machine.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return worker, per_function


def validate_lean(args, native, build, variants, comparison):
    with (args.out / "validation.csv").open() as source:
        expected = {row["function"]: row for row in csv.DictReader(source) if row["schedule"] == "upstream"}
    with (args.out / "validation.csv").open("a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=next(iter(expected.values())).keys(), lineterminator="\n")
        for fn, names in variants.items():
            for name in names:
                if name == "upstream":
                    continue
                # The worker checks each individual word before printing its hash.
                digest = int(capture([native, "validate", fn, name, build / f"oracle_{fn}.bin"]))
                report = expected[fn].copy()
                report.update(schedule=comparison + "_" + name, result_hash=f"{digest:016x}")
                if report["result_hash"] != report["oracle_hash"]:
                    raise ValueError(f"native Lean checksum mismatch: {fn}/{name}")
                writer.writerow(report)


def sample(worker, native, fn, name, kind, sweeps, env):
    comparison, _, variant = name.partition("_")
    if comparison not in native:
        output = capture([worker, fn, name, kind, str(sweeps)], env=env)
        elapsed, calls, checksum = map(int, output.split(","))
        return elapsed, calls, checksum, ""
    inputs = worker.parent / f"inputs_{kind}.bin"
    output = capture([native[comparison], "time", fn, variant, inputs, kind, str(sweeps)], env=env)
    return tuple(map(int, output.split(",")))


def measure(args, worker, native, variants):
    rows = []
    checksums = {}
    worker_env = dict(os.environ, LEAN_NUM_THREADS="1")
    fields = ("comparison", "function", "distribution", "block", "trial", "position", "schedule",
              "calls", "elapsed_ns", "checksum", "idle_pct", "optimizer_setup_ns")
    with (args.out / "samples.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for fn, c_variants in variants.items():
            def group(name):
                if name == "c":
                    return name, c_variants, args.sweeps
                names = ["upstream"] + [name + "_" + n for n in c_variants if n != "upstream"]
                sweeps = args.lean_sweeps if name == "lean" else args.word_sweeps
                return name, names, sweeps
            for kind in KINDS:
                for comparison, names, sweeps in map(group, args.group):
                    for block in range(args.blocks):
                        for offset in range(len(names)):
                            idle = check_load()
                            # Each complete Latin square puts every kernel in every
                            # position. Reverse the ordering in alternating blocks.
                            order = names if block % 2 == 0 else list(reversed(names))
                            order = order[offset:] + order[:offset]
                            for pos, name in enumerate(order, 1):
                                ns, calls, checksum, setup = sample(worker, native, fn, name, kind,
                                                                   sweeps, worker_env)
                                # The reference runs the same number of sweeps as
                                # its group, including uninterrupted dependent chains.
                                key = (fn, kind, sweeps)
                                if checksums.setdefault(key, (calls, checksum)) != (calls, checksum):
                                    raise ValueError(f"timed result mismatch: {comparison}/{fn}/{kind}/{name}")
                                row = dict(zip(fields, (comparison, fn, kind, block + 1, offset + 1, pos,
                                                        name, calls, ns, checksum, idle, setup)))
                                writer.writerow(row)
                                rows.append(row)
                            out.flush()
    fields = ("comparison", "function", "distribution", "schedule", "samples",
              "median_ns_per_call", "time_over_upstream")
    with (args.out / "summary.csv").open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(fields)
        for comparison in args.group:
            for fn in variants:
                for kind in KINDS:
                    samples = {}
                    for row in rows:
                        if (row["comparison"], row["function"], row["distribution"]) == (comparison, fn, kind):
                            samples.setdefault(row["schedule"], []).append(row["elapsed_ns"] / row["calls"])
                    baseline = median(samples["upstream"])
                    for name, values in samples.items():
                        cost = median(values)
                        writer.writerow((comparison, fn, kind, name, len(values), f"{cost:.4f}", f"{cost/baseline:.4f}"))
                        print(f"{comparison}/{fn}/{kind}/{name}: {cost:.3f} ns/call ({cost/baseline:.3f}x upstream)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="clean pinned Rutgers rlibm checkout")
    parser.add_argument("out", type=Path, help="fresh directory for executable, emitted code and CSVs")
    parser.add_argument("--verify-only", action="store_true", help="run the exhaustive checks without timing")
    parser.add_argument("--blocks", type=int, default=2, help="balanced rotations per distribution")
    parser.add_argument("--sweeps", type=int, default=32, help="timed full-domain sweeps per sample")
    parser.add_argument("--lean-sweeps", type=int, default=4, help="native Lean sweeps per sample")
    parser.add_argument("--word-sweeps", type=int, default=32, help="native machine-word sweeps per sample")
    parser.add_argument("--group", action="append", choices=["c", "lean", "word"],
                        help="measure this comparison against Rutgers (repeatable, defaults to all three)")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    args.group = list(dict.fromkeys(args.group or ["c", "lean", "word"]))
    if args.blocks < 1 or not all(1 <= s <= 65535 for s in (args.sweeps, args.lean_sweeps, args.word_sweeps)):
        parser.error("blocks must be positive and sweeps must be between 1 and 65535")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    directory = args.out / "build"
    directory.mkdir()
    targets = {"lean": "rlibm-bench", "word": "rlibm-word-bench"}
    native = {k: LEAN / ".lake/build/bin" / t for k, t in targets.items() if k in args.group}
    subprocess.run(["lake", "build", "Egglog.Examples.Rlibm.Emit", "Egglog.Examples.Rlibm.Cells",
                    "Egglog.Examples.Rlibm.Coefficients", *(p.name for p in native.values())], cwd=LEAN, check=True)
    worker, variants = prepare(args, directory)
    with (args.out / "validation.csv").open("w") as out:
        subprocess.run([worker, "validate", directory], stdout=out, check=True)
    for comparison, path in native.items():
        validate_lean(args, path, directory, variants, comparison)
    print("Exhaustive MPFR comparison passed.", flush=True)
    if not args.verify_only:
        measure(args, worker, native, variants)


if __name__ == "__main__":
    main()
