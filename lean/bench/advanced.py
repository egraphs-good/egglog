"""Single-threaded compiler workload timings. Run from the repository root."""
import argparse
import csv
from pathlib import Path
from random import Random
from statistics import median
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.compare import ROOT, check_load
from bench.large_graph import run as run_graphs
from parabit.compare import run_limited

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / ".lake/build/bin"
MODEL = ROOT / "tests/luminal-llama.egg"
SELECTS = {"select_simple": (1, 1), "if_dead_code": (1, 2), "if_dead_code_nested": (4, 6)}
GRAPH_CASES = {"llama-full": 2413, "llama-lowering": 2413, "llama-projections": 225,
               "2mm-regions": 2699}


def graphs(name, size):
    if name == "2mm-regions":
        from bench.eggcc_regions import eggcc_workload
        return eggcc_workload()
    if name == "llama-projections":
        from bench.luminal import certified_projection_workload
        return certified_projection_workload()
    from bench.luminal import luminal_full_workload, luminal_workload
    return (luminal_full_workload if name == "llama-full" else luminal_workload)()


def certified(args):
    rng = Random(53)
    egg_modes = ("egg-plain", "egg-track", "egg-explain")
    cases = ("llama-projections", "rust-egglog-projections", *egg_modes, *SELECTS)
    samples = {case: [] for case in cases}
    fields = "case trial status pass_ms wall_ms checked_stats host_idle_pct".split()
    with tempfile.TemporaryDirectory() as tmp, args.output.open("x", newline="") as out:
        projections = Path(tmp) / "projections.egg"
        subprocess.run([BIN / "luminal-certified", "--emit-projections", MODEL, projections],
                       stdout=subprocess.DEVNULL, check=True)
        from bench.luminal import projection_rows
        rows = Path(tmp) / "projections.tsv"
        rows.write_text(projection_rows(projections.read_text()))
        writer = csv.DictWriter(out, fields, lineterminator="\n")
        writer.writeheader()
        for trial in range(1 - args.warmup, args.runs + 1):
            for case in rng.sample(cases, len(cases)):
                idle_pct = check_load()
                if case == "llama-projections":
                    cmd = [BIN / "luminal-certified", MODEL]
                    expected = {"bindings": 2413, "outputs": 1, "fused": 225,
                                "live_before": 2413, "live_after": 2155}
                    clock = "compiler_ns"
                elif case == "rust-egglog-projections":
                    cmd = [ROOT / "target/release/examples/lean_comparison", "luminal", rows]
                    expected = {"layouts": 5, "roots": 225, "products": 225}
                    clock = "compiler_ns"
                elif case in egg_modes:
                    mode = case.removeprefix("egg-")
                    cmd = [HERE / "projection-egg/target/release/projection-egg", mode, projections]
                    expected = {"layouts": 5, "roots": 225, "products": 225,
                                "proofs": 225 if mode == "explain" else 0}
                    clock = "total_ns"
                else:
                    cmd = [BIN / "eggcc-certified", HERE / "eggcc" / (case + ".json")]
                    branches, selects = SELECTS[case]
                    expected = {"input_if": branches, "output_select": selects, "output_if": 0,
                                "output_bind": 0}
                    clock = "solver_ns"
                status, wall, raw, err = run_limited(cmd, args.seconds, max_rss=int(args.rss_gib * 1024**3))
                metrics = {}
                elapsed = ""
                if status == 0:
                    metrics = {k: int(v) for token in raw.decode().split()
                               for k, v in [token.split("=")]}
                    elapsed = metrics.pop(clock) / 1e6
                    status = "ok" if all(metrics.get(k) == v for k, v in expected.items()) else "mismatch"
                    if not (0 <= elapsed <= wall):
                        status = "clock-error"
                    if case == "egg-explain" and metrics["proof_steps"] < metrics["proofs"]:
                        status = "missing-explanation"
                writer.writerow(dict(case=case, trial=trial, status=status, pass_ms=elapsed, wall_ms=wall,
                                     checked_stats=" ".join(f"{k}={v}" for k, v in metrics.items()),
                                     host_idle_pct=idle_pct))
                out.flush()
                if status != "ok":
                    raise RuntimeError(f"{case}: {status} {err[-500:]!r}")
                if trial > 0:
                    samples[case].append(elapsed)
    for case in cases:
        print(f"{case}: {median(samples[case]):.3f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["graphs", "certified"])
    p.add_argument("output", type=Path)
    p.add_argument("--case", choices=GRAPH_CASES, action="append")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--rss-gib", type=float, default=8)
    args = p.parse_args()
    if args.warmup < 0 or min(args.runs, args.seconds, args.rss_gib) <= 0:
        p.error("expected positive budgets and a nonnegative warmup")
    if args.mode == "graphs":
        selected = args.case or GRAPH_CASES
        sys.exit(run_graphs(args, [(case, GRAPH_CASES[case]) for case in selected], graphs))
    elif args.case:
        p.error("--case is for graph workloads")
    else:
        certified(args)
