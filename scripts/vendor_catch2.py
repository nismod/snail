#!/usr/bin/env python3
"""Regenerate the vendored Catch2 header.

Downloads the header-only merged file from a pinned Catch2 v2 revision and
keeps its upstream license beside it.

Run::

    python scripts/vendor_catch2.py
"""

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

COMMIT = "9712bc8fe569b10b033b4d753c21658397fa17de"
BASE_URL = f"https://raw.githubusercontent.com/catchorg/Catch2/{COMMIT}"
VENDOR_DIR = (
    Path(__file__).resolve().parent.parent / "extension" / "extern" / "Catch2"
)
FILES = {
    "catch.hpp": (
        f"{BASE_URL}/single_include/catch2/catch.hpp",
        "fc0241722d4af9ad3f7dd02e831510dd5fb91326fa7f8b23c0f50ea25919d7688a40777d65ec65b7826231393636cc6ff15f73a6879c8df9cdc9493e2850237e",
    ),
    "LICENSE.txt": (
        f"{BASE_URL}/LICENSE.txt",
        "d6078467835dba8932314c1c1e945569a64b065474d7aced27c9a7acc391d52e9f234138ed9f1aa9cd576f25f12f557e0b733c14891d42c16ecdc4a7bd4d60b8",
    ),
}


def fetch(into: Path) -> None:
    for name, (url, expected) in FILES.items():
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        digest = hashlib.sha512(content).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"checksum mismatch for {name}\n"
                f"  expected {expected}\n"
                f"  got      {digest}"
            )
        (into / name).write_bytes(content)
        print(f"checksum ok: {name}")


def main() -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        fetched = Path(work_dir)
        fetch(fetched)

        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(fetched / name, VENDOR_DIR / name)

    print(f"\nvendored Catch2 {COMMIT[:12]} into {VENDOR_DIR}")
    for name in FILES:
        print(f"  {name}")


if __name__ == "__main__":
    main()
