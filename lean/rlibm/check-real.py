"""Check the real-log interpretation using an existing built mathlib Lake project."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="mathlib checkout or Lake project with mathlib in .lake/packages")
    args = parser.parse_args()
    project = args.project.resolve()
    packages = project / ".lake/packages"
    lib = Path(".lake/build/lib/lean")
    paths = [project / lib, *(p / lib for p in sorted(packages.glob("*")))]
    if not any((p / "Mathlib/Analysis/SpecialFunctions/Log/Base.olean").is_file() for p in paths):
        parser.error("mathlib's built Mathlib.Analysis.SpecialFunctions.Log.Base is required")
    version = (project / "lean-toolchain").read_text().strip()
    lean = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as work:
        dest = Path(work) / "Egglog/Examples/Rlibm"
        dest.mkdir(parents=True)
        env = {**os.environ, "LEAN_PATH": os.pathsep.join(map(str, [work, *paths]))}
        cmd = ["elan", "run", version, "lean", f"--root={lean}"]
        for module in ("Types", "Coefficients", "Polynomial", "Spec", "KernelPages", "Certificate"):
            subprocess.run([*cmd, "-o", str(dest / (module + ".olean")),
                            str(lean / "Egglog/Examples/Rlibm" / (module + ".lean"))],
                           env=env, check=True)
        subprocess.run([*cmd, str(lean / "rlibm/mathlib/RealLog.lean")], env=env, check=True)


if __name__ == "__main__":
    main()
