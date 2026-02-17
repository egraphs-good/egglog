#!/usr/bin/env python3
import sys
from pathlib import Path
from global_var_formatter import format_file


def format_directory(directory: str, suffix: str) -> None:
    """Format all files with given suffix in directory recursively."""
    root = Path(directory)
    files_with_conflicts = []
    total_files = 0
    
    for filepath in root.rglob(f'*{suffix}'):
        total_files += 1
        print(f"Processing: {filepath}")
        
        conflicts = format_file(str(filepath))
        if conflicts:
            files_with_conflicts.append((str(filepath), conflicts))
    
    print(f"\n{'='*60}")
    print(f"Processed {total_files} file(s)")
    
    if files_with_conflicts:
        print(f"\n{len(files_with_conflicts)} file(s) need manual review:")
        for filepath, conflicts in files_with_conflicts:
            print(f"\n  {filepath}")
            for var in conflicts:
                print(f"    - {var}")
    else:
        print("\nAll files formatted successfully!")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory> <suffix>")
        print(f"Example: {sys.argv[0]} ./src .egg")
        sys.exit(1)
    
    format_directory(sys.argv[1], sys.argv[2])