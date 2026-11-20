"""Export source-level equivalences and check them in one Lean process per input."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from parabit.compare import run_limited
from bench.compare import check_load

EXPORTER = PROJECT / ".lake/build/bin/parabit-certified"
UNPROVED = "the certified solver did not establish this equivalence"


def lean_environment():
    path = subprocess.check_output(["lake", "env", "printenv", "LEAN_PATH"],
                                   cwd=PROJECT, text=True).strip()
    return {**os.environ, "LEAN_PATH": path, "LEAN_NUM_THREADS": "1"}


def export(case, rounds, env, staged=False):
    options = ["--staged"] if staged else []
    return subprocess.run([EXPORTER, "--emit-lean", *options, case, str(rounds)], env=env,
                          cwd=PROJECT, capture_output=True, text=True)


def check(emitted, seconds, env):
    with tempfile.TemporaryFile() as source:
        source.write(emitted.encode())
        source.seek(0)
        status, elapsed, out, err = run_limited(["lean", "-j", "1", "--stdin"], seconds,
                                                cwd=PROJECT, env=env, stdin=source)
    if isinstance(status, str):
        return status, elapsed, ""
    return ("checked" if status == 0 else "kernel-error",
            elapsed, (out + err).decode(errors="replace").strip())


def smoke(env):
    # Check both the exporter and an importing module. The mathematical example
    # has independent input, intermediate, and output widths.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        file = root / "commutativity_add.bwlang"
        file.write_text(json.dumps({
            "preconditions": [],
            "lhs": "(bw r (+ (bw p a) (bw q b)))",
            "rhs": "(bw r (+ (bw q b) (bw p a)))",
        }))
        proof = export(file, 4, env)
        proof.check_returncode()
        module = root / "Certificate.lean"
        module.write_text(proof.stdout)
        compiler = ["lean", "-j", "1", "-R", root]
        subprocess.run(compiler + ["-o", module.with_suffix(".olean"), module],
                       cwd=PROJECT, env=env, check=True)
        source = ("import Certificate\n"
                  "example := BWLang.Certificates.Case_commutativity_add.equivalence\n")
        subprocess.run(compiler + ["--stdin"], input=source, text=True,
                       cwd=PROJECT, env={**env, "LEAN_PATH": str(root) + os.pathsep + env["LEAN_PATH"]},
                       check=True)
        # Ask the kernel to establish actual rejection. Elaborating an
        # intentionally failing `decide` would spend time printing a trace.
        altered = re.sub(r"^  rhs := .*", "  rhs := .literal 12", proof.stdout,
                         count=1, flags=re.MULTILINE)
        body = re.search(r"  \(\((problem\.(?:replay|prove)\?.*?)\)\.get"
                         r" \(by decide \+kernel\)\)\.down", altered, flags=re.DOTALL)
        if body is None:
            raise ValueError("missing reflection proof in exported theorem")
        rejected = (altered.split("theorem equivalence", 1)[0]
                    + "example : (" + body[1] + ").isNone = true := by decide +kernel\n"
                    + altered[altered.rindex("\nend "):])
        status, _, detail = check(rejected, 10, env)
        if status != "checked":
            raise RuntimeError(f"kernel rejection of modified RHS failed: {status} {detail}")
        file.write_text('{"preconditions":[],"lhs":"x","rhs":"y"}')
        failed = export(file, 4, env)
        if failed.returncode == 0 or failed.stdout or UNPROVED not in failed.stderr:
            raise RuntimeError("the exporter accepted two independent words")
        choice = "(sel c (bw 1 a) (bw 1 b))"
        file.write_text(json.dumps(dict(preconditions=[], lhs=f"(bw 1 {choice})", rhs=choice)))
        proof = export(file, 3, env, staged=True)
        proof.check_returncode()
        status, _, detail = check(proof.stdout, 15, env)
        if status != "checked" or "problem.proveStaged?" not in proof.stdout:
            raise RuntimeError(f"staged kernel export failed: {status} {detail}")


def corpus(args, env, exporter):
    base = args.artifact.resolve() / "benchmarks"
    files = sorted(base.glob("*/bwlang/*.bwlang"))
    if args.case:
        wanted = set(args.case)
        files = [p for p in files if p.relative_to(base).as_posix() in wanted]
        if len(files) != len(wanted):
            raise SystemExit("requested case missing from ARTIFACT/benchmarks")
    if not files:
        raise SystemExit("no BWLang cases found in ARTIFACT/benchmarks")
    counts = Counter()
    with args.output.open("x", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["case", "rounds", "replay_rounds", "hint", "status", "attempts", "emit_ms", "kernel_ms", "detail"])
        for case in files:
            if args.pause_file and args.pause_file.exists():
                print(f"paused before {case.name}", flush=True)
                while args.pause_file.exists():
                    time.sleep(0.25)
            if args.check_load:
                check_load()
            options = ["--staged"] if args.staged else []
            code, emit_ms, emitted, err = run_limited(
                [exporter, "--emit-lean", *options, case, str(args.rounds)], args.export_seconds,
                cwd=PROJECT, env=env)
            proof = emitted.decode(errors="replace")
            hint = ""
            attempts = 0
            replay = ""
            elapsed = 0
            if isinstance(code, str):
                status, detail = "export-" + code, ""
            elif code:
                detail = err.decode(errors="replace").strip()
                status = "unproved" if UNPROVED in detail else "export-error"
                if status == "unproved":
                    detail = ""
            else:
                marker = re.search(r"\(\(problem\.(?:proveStaged|prove|replay)\? (\d+)\b", proof)
                if marker is None:
                    raise ValueError(f"missing replay bound in exported theorem for {case}")
                replay = marker[1]
                kind = re.search(r"ParaBit\.ReplayHint\.(\w+)", proof)
                hint = kind[1] if kind else ("staged-search" if args.staged else "full-search")
                budgets = [args.seconds]
                if args.retry_seconds > args.seconds:
                    budgets.append(args.retry_seconds)
                for attempts, seconds in enumerate(budgets, 1):
                    status, duration, detail = check(proof, seconds, env)
                    elapsed += duration
                    if status != "timeout":
                        break
            counts[status] += 1
            writer.writerow([case.relative_to(base).as_posix(), args.rounds, replay, hint, status,
                             attempts, f"{emit_ms:.3f}", f"{elapsed:.3f}", detail.strip()[-1000:]])
            out.flush()
    print(dict(counts))
    if set(counts) - {"checked", "unproved"}:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("smoke")
    audit = modes.add_parser("corpus")
    audit.add_argument("artifact", type=Path)
    audit.add_argument("output", type=Path)
    audit.add_argument("--pause-file", type=Path,
                       help="pause between cases while this file exists")
    audit.add_argument("--rounds", type=int, default=6)
    audit.add_argument("--staged", action="store_true", help="export the staged solver's theorem")
    audit.add_argument("--case", action="append", help="check this named path relative to benchmarks")
    audit.add_argument("--check-load", action="store_true", help="sample host CPU idle before each case")
    audit.add_argument("--export-seconds", type=float, default=120)
    audit.add_argument("--seconds", type=float, default=5)
    audit.add_argument("--retry-seconds", type=float, default=15)
    args = parser.parse_args()
    env = lean_environment()
    if args.mode == "smoke":
        smoke(env)
    elif args.rounds < 0 or args.seconds <= 0 or args.retry_seconds < 0 or args.export_seconds <= 0:
        parser.error("the time budget must be positive and the round and retry limits nonnegative")
    else:
        # Another build may replace an olean while the audit is running.
        # Use one snapshot of both the exporting binary and the imported library.
        with tempfile.TemporaryDirectory(prefix="parabit-kernel-") as tmp:
            snapshot = Path(tmp)
            library = PROJECT / ".lake/build/lib/lean"
            local = snapshot / "lib"
            shutil.copytree(library, local)
            binary = shutil.copy2(EXPORTER, snapshot / EXPORTER.name)
            paths = env["LEAN_PATH"].split(os.pathsep)
            paths = [str(local) if Path(path).resolve() == library else path for path in paths]
            corpus(args, {**env, "LEAN_PATH": os.pathsep.join(paths)}, binary)


if __name__ == "__main__":
    main()
