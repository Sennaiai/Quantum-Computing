"""Validate tracked notebook schemas without importing or executing their code."""

from pathlib import Path
import subprocess
import nbformat

ROOT = Path(__file__).resolve().parents[2]


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.ipynb"], cwd=ROOT
    ).decode("utf-8").split("\0")
    names = [name for name in names if name]
    if not names:
        raise SystemExit("No tracked notebooks found")
    for name in names:
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        print("PASS", name)
    print(f"Validated {len(names)} notebook structures; no code executed.")


if __name__ == "__main__":
    main()
