#!/usr/bin/env python3
"""Regenerate the vendored geoarrow-c amalgamation.

geoarrow-c decodes WKB and walks any GeoArrow encoding through a visitor,
which is what lets the extension read a GeoParquet file written with
geopandas' defaults and split a layer holding more than one geometry type.
Like nanoarrow it has no amalgamation to download - the two-file bundle is
generated from a source tree by its own CMake - so this wraps that, and the
vendored copy in ``extension/vendor/geoarrow`` can be reproduced or moved to a
new revision without anyone having to remember the incantation.

Run from anywhere::

    python scripts/vendor_geoarrow.py

Unlike nanoarrow, geoarrow-c has never made a release: the last C tag is
``v0.1.2`` (February 2024) and ``main`` has been ``0.2.0-SNAPSHOT`` since. So
this pins a **commit**, and verifies it by asking git to check out that exact
SHA - which is a content hash, so it is a stronger guarantee than the checksum
of a generated tarball. To move to a newer revision, bump ``COMMIT`` and
``COMMIT_DATE`` together.

Three build options matter and none of them is cosmetic:

``GEOARROW_NAMESPACE``
    Macro-prefixes every exported symbol, so this extension can be loaded into
    the same interpreter as pyarrow or any other wheel that vendors geoarrow-c.
    It must match the nanoarrow namespace, because geoarrow-c's own CMake ties
    the two together - see its ``NANOARROW_NAMESPACE=${GEOARROW_NAMESPACE}``.

``GEOARROW_USE_RYU=OFF``, ``GEOARROW_USE_FAST_FLOAT=OFF``
    Both are third-party libraries used only to make WKT number parsing and
    printing fast. We read and write WKB, never WKT, so leaving them on would
    vendor two more dependencies to speed up a code path nothing calls. With
    them off the bundle needs nothing beyond nanoarrow and libc.

The bundle deliberately excludes nanoarrow, so the generated ``geoarrow.c``
compiles against the copy already vendored at ``extension/vendor/nanoarrow``
and there is exactly one nanoarrow in the build.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

# Pinned revision of https://github.com/geoarrow/geoarrow-c
COMMIT = "725a560132b1f41519ec7370c33433af7908c1ff"
COMMIT_DATE = "2026-07-23"
SYMBOL_NAMESPACE = "Snail"

URL = "https://github.com/geoarrow/geoarrow-c.git"
VENDOR_DIR = (
    Path(__file__).resolve().parent.parent / "extension" / "vendor" / "geoarrow"
)
# geoarrow.c includes "geoarrow/geoarrow.h" and "nanoarrow/nanoarrow.h", so
# extension/vendor is the include path and each library sits in its own
# directory below it
GENERATED = ("geoarrow.h", "geoarrow.c")
LICENCE_FILES = {"LICENSE": "LICENSE.txt"}


def fetch(into: Path) -> Path:
    source = into / "geoarrow-c"
    print(f"cloning {URL} at {COMMIT[:12]} ({COMMIT_DATE})")
    subprocess.run(["git", "clone", "--quiet", URL, str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--quiet", COMMIT], check=True
    )

    at = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if at != COMMIT:
        raise SystemExit(f"checked out {at}, expected {COMMIT}")
    print("commit ok")
    return source


def bundle(source: Path, into: Path) -> Path:
    out = into / "bundle"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(into / "build"),
            "-DGEOARROW_BUNDLE=ON",
            f"-DGEOARROW_NAMESPACE={SYMBOL_NAMESPACE}",
            # no WKT, so no ryu and no fast_float - see the module docstring
            "-DGEOARROW_USE_RYU=OFF",
            "-DGEOARROW_USE_FAST_FLOAT=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--install", str(into / "build"), "--prefix", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        source = fetch(work)
        out = bundle(source, work)

        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        for name in GENERATED:
            shutil.copyfile(next(out.rglob(name)), VENDOR_DIR / name)
        for name, as_name in LICENCE_FILES.items():
            shutil.copyfile(source / name, VENDOR_DIR / as_name)

    print(f"\nvendored geoarrow-c {COMMIT[:12]} into {VENDOR_DIR}")
    for name in (*GENERATED, *LICENCE_FILES.values()):
        print(f"  {name}")


if __name__ == "__main__":
    main()
