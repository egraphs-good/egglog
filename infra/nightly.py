#!/user/bin/env python3

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Determine directories
SCRIPT_DIR = Path(__file__).resolve().parent
POACH_ROOT = SCRIPT_DIR.parent
NIGHTLY_DIR = POACH_ROOT / "nightly"
POACH_BINARY = POACH_ROOT / "target" / "release" / "poach"

def main(benchmark_dir):
  print(benchmark_dir)

  (benchmark_results, failing_benchmarks) = run_benchmarks(benchmark_dir)

  data = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "failing_benchmarks": [str(b) for b in failing_benchmarks],
    "passing_benchmarks": benchmark_results
  }
  data_out_path = NIGHTLY_DIR / "output" / "data" / "data.json"
  data_out_path.parent.mkdir(parents=True, exist_ok=True)
  data_out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def run_command(cmd):
  started = time.perf_counter()
  cmd_result = subprocess.run(
    cmd,
    cwd=POACH_ROOT,
    capture_output=True,
    text=True # decode stderr/stdout as string instead of raw bytes
  )
  if cmd_result.returncode != 0:
    time_seconds = time.perf_counter() - started
    print(f"Command failed after {time_seconds:.2f}s: {' '.join(cmd)}", file=sys.stderr)
    return {
      "cmd": cmd,
      "status": "error"
    }

  report = json.loads(cmd_result.stderr)

  return {
    "cmd": " ".join(cmd),
    "status": "success",
    "report": summarize_report(report),
    "wall_time_s": time.perf_counter() - started
  }
  
def summarize_report(report):
  # aggregate timing steps by type
  rule_ms = 0
  extraction_ms = 0
  other_ms = 0
  total_ms = 0
  for time_step in report["timings"]:
    total_ms += time_step["total"]
    if "running_rules" in time_step["tags"]:
      rule_ms += time_step["total"]
    elif "extraction" in time_step["tags"]:
      extraction_ms += time_step["total"]
    else:
      other_ms += time_step["total"]

  # No sizes in vanilla egglog reports

  return {
    "rule_ms": rule_ms,
    "extraction_ms": extraction_ms,
    "other_ms": other_ms,
    "timing_steps": len(report["timings"])
  }

def run_benchmarks(benchmark_dir):
  report_dir = NIGHTLY_DIR / "reports"
  report_dir.mkdir(parents=True, exist_ok=True)

  # Find benchmarks
  # benchmark_dir is the root of the benchmark directory 
  benchmarks = list(Path(benchmark_dir).rglob("train/*.egg"))
  # For this treatment, we don't do anything at train time,
  # we just use the train benchmarks at serve time

  results = []
  failing_benchmarks = []
  for benchmark in benchmarks:
    relative_path = benchmark.relative_to(benchmark_dir)
    suite_name = str(relative_path.parent)
    benchmark_name = relative_path.name
    command = [
      str(POACH_BINARY),
      "serve",
      "--debug",
      "EMPTY.MODEL",
      "single",
      str(benchmark)
    ]
    
    result = run_command(command)
    result["benchmark_name"] = benchmark_name
    result["suite_name"] = suite_name
    if result["status"] == "success":
      print(f"Success: {benchmark_name}")
      results.append(result)
    else:
      failing_benchmarks.append(relative_path)

  return (results, failing_benchmarks)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise SystemExit(f"Usage: nightly.py <benchmark-dir>")
  
  main(sys.argv[1])