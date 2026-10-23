"""Ensure the security gate covers all environments and propagates failures."""

from pathlib import Path
import subprocess

import pytest

from scripts import audit_lock


@pytest.mark.parametrize("exit_codes, expected", [([0, 0], 0), ([1, 0], 1), ([0, 2], 1)])
def test_audit_covers_all_versions_and_platforms(tmp_path, monkeypatch, exit_codes, expected):
    lockfile = tmp_path / "poetry.lock"
    lockfile.write_text('''
[[package]]
name = "example-package"
version = "1.0.0"
groups = ["main"]
markers = "python_version < '3.12'"
[[package]]
name = "example-package"
version = "2.0.0"
groups = ["main"]
markers = "python_version >= '3.12'"
[[package]]
name = "windows-only"
version = "3.0.0"
groups = ["docs"]
markers = "sys_platform == 'win32'"
''', encoding="utf-8")
    audited = set()
    calls = []

    def run_audit(command, check):
        requirements = Path(command[-1]).read_text().splitlines()
        assert len({line.split("==")[0] for line in requirements}) == len(requirements)
        assert {"--strict", "--no-deps", "--disable-pip"} <= set(command)
        audited.update(requirements)
        calls.append(command)
        return subprocess.CompletedProcess(command, exit_codes[len(calls) - 1])

    monkeypatch.setattr(audit_lock.subprocess, "run", run_audit)
    monkeypatch.setattr("sys.argv", ["audit_lock.py", str(lockfile)])
    assert audit_lock.main() == expected
    assert len(calls) == 2
    assert audited == {"example-package==1.0.0", "example-package==2.0.0", "windows-only==3.0.0"}


@pytest.mark.parametrize("contents", [
    "package = []",
    '[[package]]\nname = "custom"\nversion = "1.0"\n[package.source]\ntype = "git"',
])
def test_audit_rejects_empty_or_unsupported_inventory(tmp_path, monkeypatch, contents):
    lockfile = tmp_path / "poetry.lock"
    lockfile.write_text(contents, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["audit_lock.py", str(lockfile)])
    with pytest.raises(SystemExit) as error:
        audit_lock.main()
    assert error.value.code == 2
