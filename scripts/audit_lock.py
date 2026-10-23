"""Audit every Poetry lock entry, including other platforms and Python versions."""

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lockfile", nargs="?", type=Path, default=Path("poetry.lock"))
    args = parser.parse_args()
    with args.lockfile.open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    if not packages:
        parser.error("The lockfile contains no packages to audit")

    # pip-audit rejects multiple versions of a name in one requirements file.
    # Separate those versions into batches without discarding any lock entries.
    batches = []
    for package in packages:
        if package.get("source", {}).get("type", "pypi") != "pypi":
            parser.error(f"Cannot audit non-PyPI source for {package['name']}")
        name = package["name"].lower().replace("_", "-").replace(".", "-")
        version = package["version"]
        for batch in batches:
            if name not in batch or batch[name] == version:
                batch[name] = version
                break
        else:
            batches.append({name: version})

    failed = False
    with TemporaryDirectory(prefix="drfetools-audit-") as directory:
        for index, batch in enumerate(batches):
            requirements = Path(directory) / f"requirements-{index}.txt"
            requirements.write_text(
                "".join(f"{name}=={version}\n" for name, version in sorted(batch.items())),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-m", "pip_audit", "--strict", "--no-deps",
                 "--disable-pip", "--requirement", str(requirements)],
                check=False,
            )
            failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
