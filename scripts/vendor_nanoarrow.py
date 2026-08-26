#!/usr/bin/env python3
"""Regenerate the vendored nanoarrow amalgamation.

nanoarrow is not distributed as a system library and has no pre-built
amalgamation to download - the two-file (plus C++ header) bundle is generated
from a source release by a script that ships with it. This wraps that, so the
vendored copy in ``extension/vendor/nanoarrow`` can be reproduced or moved to a
new release without anyone having to remember the incantation.

Run from anywhere::

    python scripts/vendor_nanoarrow.py

To move to a new release, bump ``VERSION`` and ``SHA512`` together - the
checksum is the one published alongside the release tarball, and is verified
before anything is generated.

The symbol namespace matters and is not cosmetic: every public nanoarrow
symbol is macro-prefixed with it, so that this extension can be loaded into the
same interpreter as pyarrow, or any other wheel that vendors nanoarrow, without
the two colliding at link time.
"""

import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.9.0"
SHA512 = "2fbdfe3274da9dcba5e3215ba0a7ff66da9f65395d1800841f0dc9a6bbc00b8cc224f900bcb946c91969b3c6e79d132ad5077c9a537f861502c4763dbffb33b8"
SYMBOL_NAMESPACE = "Snail"

RELEASE = f"apache-arrow-nanoarrow-{VERSION}"
URL = (
    "https://github.com/apache/arrow-nanoarrow/releases/download/"
    f"{RELEASE}/{RELEASE}.tar.gz"
)
VENDOR_DIR = (
    Path(__file__).resolve().parent.parent / "extension" / "vendor" / "nanoarrow"
)
# nanoarrow.hpp and nanoarrow.c both #include "nanoarrow.h" as a sibling, so
# all three live in one directory and that directory is the include path
GENERATED = ("nanoarrow.h", "nanoarrow.hpp", "nanoarrow.c")
LICENCE_FILES = ("LICENSE.txt", "NOTICE.txt")


def fetch(into: Path) -> Path:
    tarball = into / f"{RELEASE}.tar.gz"
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        tarball.write_bytes(response.read())

    digest = hashlib.sha512(tarball.read_bytes()).hexdigest()
    if digest != SHA512:
        raise SystemExit(
            f"checksum mismatch for {tarball.name}\n"
            f"  expected {SHA512}\n"
            f"  got      {digest}"
        )
    print("checksum ok")
    return tarball


def bundle(source: Path, into: Path) -> Path:
    out = into / "bundle"
    subprocess.run(
        [
            sys.executable,
            str(source / "ci" / "scripts" / "bundle.py"),
            "--symbol-namespace",
            SYMBOL_NAMESPACE,
            # flat headers: we want "nanoarrow.h", not "nanoarrow/nanoarrow.h"
            "--header-namespace",
            "",
            "--include-output-dir",
            str(out / "include"),
            "--source-output-dir",
            str(out / "src"),
        ],
        check=True,
    )
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        tarball = fetch(work)
        with tarfile.open(tarball) as archive:
            archive.extractall(work, filter="data")
        source = work / RELEASE

        out = bundle(source, work)
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        for name in GENERATED:
            found = next(out.rglob(name))
            shutil.copyfile(found, VENDOR_DIR / name)
        for name in LICENCE_FILES:
            shutil.copyfile(source / name, VENDOR_DIR / name)

    print(f"\nvendored nanoarrow {VERSION} into {VENDOR_DIR}")
    for name in (*GENERATED, *LICENCE_FILES):
        print(f"  {name}")


if __name__ == "__main__":
    main()
