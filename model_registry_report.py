"""Generate a simple model registry report for M19."""

from __future__ import annotations

from pathlib import Path
import hashlib

REGISTRY_DIR = Path("models")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("# Model registry report\n")

    if not REGISTRY_DIR.exists():
        print(f"Registry folder not found: `{REGISTRY_DIR}`")
        return

    files = sorted([p for p in REGISTRY_DIR.rglob("*") if p.is_file()])

    if not files:
        print(f"No files found in `{REGISTRY_DIR}`")
        return

    print(f"Registry path: `{REGISTRY_DIR}`")
    print(f"Number of files: **{len(files)}**\n")

    for p in files:
        size = p.stat().st_size
        print(f"## `{p.as_posix()}`")
        print(f"- Size: {size} bytes")
        print(f"- SHA256: `{sha256(p)}`\n")


if __name__ == "__main__":
    main()
