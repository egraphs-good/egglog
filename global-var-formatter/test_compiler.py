#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def test_file(filepath: str) -> bool:
    """Test a file with cargo run. Returns True if no warnings and exit code is 0."""
    result = subprocess.run(
        ['cargo', 'run', filepath],
        capture_output=True,
        text=True
    )
    expected_exit_code = 0
    if 'fail' in filepath:
        expected_exit_code = 1
    
    return result.returncode == expected_exit_code and 'WARN' not in result.stderr


def test_directory(directory: str, suffix: str) -> None:
    """Test all files with given suffix in directory recursively."""
    root = Path(directory)
    failed_files = []
    total_files = 0
    
    for filepath in root.rglob(f'*{suffix}'):
        total_files += 1
        print(f"Testing: {filepath}")
        
        if not test_file(str(filepath)):
            failed_files.append(str(filepath))
    
    print(f"\n{'='*60}")
    print(f"Tested {total_files} file(s)")
    
    if failed_files:
        print(f"\n{len(failed_files)} file(s) failed:")
        for filepath in failed_files:
            print(f"  - {filepath}")
    else:
        print("\nAll files passed!")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory> <suffix>")
        print(f"Example: {sys.argv[0]} ./src .egg")
        sys.exit(1)
    
    test_directory(sys.argv[1], sys.argv[2])
