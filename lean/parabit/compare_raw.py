#!/usr/bin/env python3
"""Run identical emitted ParaBit programs through the Lean and Rust egglog CLIs."""
import argparse
import csv
import math
from pathlib import Path
from random import Random
from statistics import fmean, median
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from bench.compare import check_load
from parabit.compare import run_limited

# These are the two artifact inputs with countermodels checked by Lean.
INVALID = {'Hydra/bwlang/opt_310.bwlang', 'Hydra/bwlang/opt_2703.bwlang'}


def sample(command, program, timeout):
    status, elapsed, out, err = run_limited([*command, program], timeout)
    if isinstance(status, str):
        return status, elapsed
    if status == 0:
        return 'solved', elapsed
    if status == 1 and b'check failed' in (out + err).lower():
        return 'check-failed', elapsed
    print(f'{command[0].name} {program.name}: {(out + err)[-500:]!r}', file=sys.stderr)
    return 'error', elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('artifact', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--timeout', type=float, default=30)
    p.add_argument('--manifest', type=Path,
                   help='read generated inputs relative to ARTIFACT using the case and expected columns')
    p.add_argument('--control-size', type=int, action='append',
                   help='with a scale manifest, limit generated negative controls to these sizes')
    args = p.parse_args()
    if min(args.trials, args.rounds, args.timeout) <= 0:
        p.error('trials, rounds, and timeout must be positive')
    if args.manifest:
        corpus = args.artifact
        with args.manifest.open() as manifest:
            entries = list(csv.DictReader(manifest))
        if args.control_size:
            entries = [r for r in entries if r['expected'] != 'inequivalent'
                       or int(r['size']) in args.control_size]
        cases = [Path(row['case']) for row in entries]
        if (not cases or len(cases) != len(set(cases))
                or any(row['expected'] not in {'solved', 'inequivalent'} for row in entries)
                or any(not (corpus / case).is_file() for case in cases)):
            p.error('expected distinct generated input files with known outcomes')
        invalid = {row['case'] for row in entries if row['expected'] == 'inequivalent'}
    else:
        if args.control_size:
            p.error('--control-size requires --manifest')
        corpus = args.artifact / 'benchmarks'
        cases = sorted(f.relative_to(corpus) for f in corpus.glob('*/bwlang/*.bwlang'))
        invalid = INVALID
        if len(cases) != 165 or not invalid.issubset({k.as_posix() for k in cases}):
            p.error('expected the pinned CAV artifact with 165 BWLang inputs')
    binaries = HERE.parent / '.lake/build/bin'
    commands = {'lean': [binaries / 'lean-egglog', '--raw'],
                'rust': [ROOT / 'target/release/egglog', '-j', '1']}
    rng = Random(42)
    times = {case: {'lean': [], 'rust': []} for case in cases}
    failures = set()
    with tempfile.TemporaryDirectory() as temp, args.output.open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['case', 'expected', 'trial', 'first', 'lean_status', 'rust_status',
                         'lean_wall_ms', 'rust_wall_ms'])
        programs = {}
        for idx, case in enumerate(cases):
            program = Path(temp) / f'{idx}.egg'
            with program.open('wb') as out:
                subprocess.run([binaries / 'parabit-relational', '--emit-egg', HERE / 'theory.egg',
                                corpus / case, str(args.rounds)], stdout=out, check=True)
            programs[case] = program
        for trial in range(1, args.trials + 1):
            rng.shuffle(cases)
            for idx, case in enumerate(cases):
                if idx % 32 == 0:
                    print(f'trial {trial}, {idx}/{len(cases)}', flush=True)
                    check_load()
                expected = 'check-failed' if case.as_posix() in invalid else 'solved'
                order = ('lean', 'rust') if (idx + trial) % 2 else ('rust', 'lean')
                results = {}
                for mode in order:
                    status, ms = sample(commands[mode], programs[case], args.timeout)
                    results[mode] = status, round(ms, 3)
                    if status != expected:
                        failures.add(case)
                    times[case][mode].append(ms)
                writer.writerow([case.as_posix(), expected, trial, order[0], results['lean'][0],
                                 results['rust'][0], results['lean'][1], results['rust'][1]])
                file.flush()
    matched = [t for c, t in times.items() if c not in failures and c.as_posix() not in invalid]
    if matched:
        lean = [median(t['lean']) for t in matched]
        rust = [median(t['rust']) for t in matched]
        ratio = math.exp(fmean(math.log(x/y) for x, y in zip(lean, rust)))
        print(f'{len(matched)} joint positive goals. Median wall ms {median(lean):.3f} Lean, '
              f'{median(rust):.3f} Rust. Paired geometric mean Lean/Rust {ratio:.3f}.')
    print(f'{len(cases) - len(failures)}/{len(cases)} inputs matched expectation in every trial.')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
