#!/usr/bin/env python3
"""Measure compressed Git growth between two refs."""

from __future__ import annotations

import os
import subprocess
import sys

DEFAULT_SOFT_LIMIT_BYTES = 512 * 1024
DEFAULT_HARD_LIMIT_BYTES = 2 * 1024 * 1024
LARGEST_BLOBS = 20


def usage() -> str:
    return f"""Usage:
  scripts/git-size-budget.py BASE_REF HEAD_REF

Measures compressed Git objects reachable from HEAD_REF but not BASE_REF.

Environment:
  SOFT_LIMIT_BYTES   Warning threshold. Default: {DEFAULT_SOFT_LIMIT_BYTES}.
  HARD_LIMIT_BYTES   Failure threshold. Default: {DEFAULT_HARD_LIMIT_BYTES}.
  GITHUB_STEP_SUMMARY, when set, receives the Markdown summary.
"""


def run_text(args: list[str], *, input_text: str | None = None) -> str:
    proc = subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def run_bytes(args: list[str], *, input_text: str) -> bytes:
    proc = subprocess.run(
        args,
        input=input_text.encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def rev_parse(ref: str) -> str:
    return run_text(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()


def pack_size(revs: list[str], *, thin: bool) -> int:
    args = [
        "git",
        "pack-objects",
        "--revs",
        "--stdout",
        "--window=50",
        "--depth=50",
        "--threads=1",
    ]
    if thin:
        args.insert(3, "--thin")

    pack = run_bytes(args, input_text="\n".join(revs) + "\n")
    return len(pack)


def parse_limit(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from None
    if parsed < 0:
        raise SystemExit(f"{name} must be non-negative, got {value!r}")
    return parsed


def fmt_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024

    if unit == "B":
        return f"{sign}{value} B"

    text = f"{size:.1f}".rstrip("0").rstrip(".")
    return f"{sign}{text} {unit}"


def github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotate(level: str, title: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        print(
            f"::{level} title={github_escape(title)}::{github_escape(message)}",
            file=sys.stderr,
        )
    else:
        print(f"{level}: {title}: {message}", file=sys.stderr)


def largest_new_blobs(base_sha: str, head_sha: str) -> list[tuple[int, str]]:
    objects = run_text(["git", "rev-list", "--objects", head_sha, f"^{base_sha}"])
    oids: list[str] = []
    paths: dict[str, str] = {}

    for line in objects.splitlines():
        oid, _, path = line.partition(" ")
        if not oid:
            continue
        oids.append(oid)
        paths.setdefault(oid, path)

    if not oids:
        return []

    batch_input = "\n".join(oids) + "\n"
    batch_output = run_text(
        ["git", "cat-file", "--batch-check=%(objectname)\t%(objecttype)\t%(objectsize)"],
        input_text=batch_input,
    )

    rows = []
    for line in batch_output.splitlines():
        oid, typ, size_text = line.split("\t")
        if typ == "blob":
            rows.append((int(size_text), paths.get(oid, "")))

    return sorted(rows, reverse=True)[:LARGEST_BLOBS]


def build_summary(
    *,
    base_sha: str,
    head_sha: str,
    delta_thin: int,
    delta_self_contained: int,
    soft_limit: int,
    hard_limit: int,
    blobs: list[tuple[int, str]],
) -> str:
    lines = [
        "## Git size budget",
        "",
        "| Metric | Size |",
        "|---|---:|",
        f"| Delta pack, thin/network-like | {fmt_bytes(delta_thin)} |",
        f"| Delta pack, self-contained | {fmt_bytes(delta_self_contained)} |",
        "",
        f"Base: `{base_sha}`",
        "",
        f"Head: `{head_sha}`",
        "",
        "Thresholds:",
        "",
        "| Level | Limit | Behavior |",
        "|---|---:|---|",
        f"| Soft | {fmt_bytes(soft_limit)} | pass with warning annotation |",
        f"| Hard | {fmt_bytes(hard_limit)} | fail check |",
        "",
        "### Largest newly reachable blobs",
        "",
        "```text",
    ]

    if blobs:
        for size, path in blobs:
            lines.append(f"{fmt_bytes(size):>12}  {path}")
    else:
        lines.append("(none)")

    lines.append("```")
    return "\n".join(lines) + "\n"


def write_summary(summary: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary)
    else:
        print(summary, end="")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] in {"-h", "--help"}:
        print(usage(), end="")
        return 0 if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"} else 2

    base_sha = rev_parse(sys.argv[1])
    head_sha = rev_parse(sys.argv[2])
    revs = [head_sha, f"^{base_sha}"]

    soft_limit = parse_limit("SOFT_LIMIT_BYTES", DEFAULT_SOFT_LIMIT_BYTES)
    hard_limit = parse_limit("HARD_LIMIT_BYTES", DEFAULT_HARD_LIMIT_BYTES)
    delta_thin = pack_size(revs, thin=True)
    delta_self_contained = pack_size(revs, thin=False)
    blobs = largest_new_blobs(base_sha, head_sha)

    write_summary(
        build_summary(
            base_sha=base_sha,
            head_sha=head_sha,
            delta_thin=delta_thin,
            delta_self_contained=delta_self_contained,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            blobs=blobs,
        )
    )

    message = f"Git delta is {fmt_bytes(delta_thin)}."
    if delta_thin > hard_limit:
        annotate("error", "Git size budget exceeded", message)
        return 1
    if delta_thin > soft_limit:
        annotate("warning", "Git size budget warning", message)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
