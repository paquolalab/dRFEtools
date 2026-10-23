# Changelog

Versions follow the git tags (`vX.Y.Z`). Pushing a tag publishes that version to PyPI and creates a GitHub release whose notes are the matching section below.

## 0.4.1 - 2026-10-23

Security release: no API changes.

### Security

* Raised the minimum versions of dependencies with published vulnerabilities: `pillow>=13.0.0` at runtime, and `certifi`, `idna`, `pygments`, `requests` and `urllib3` in the docs group. `poetry.lock` and `docs/requirements.txt` are regenerated to match.

### Added

* `Security and compatibility` workflow: audits every locked dependency (`scripts/audit_lock.py`) and the docs requirements with `pip-audit`, checks that the lock file and `docs/requirements.txt` agree, and runs the tests, the CLI, the docs build and a clean-wheel install on Python 3.11 and 3.14. Runs on pull requests, on pushes to `main`, weekly, and before every release.
* Dependabot security updates for Python dependencies and GitHub Actions.
* `SECURITY.md` with the supported versions and how to report a vulnerability.
* Tests for the audit script (`tests/test_security_audit.py`) and end-to-end `rf_rfe`/`dev_rfe` runs with plot output (`tests/test_workflows.py`).
* Release workflow: a `v*` tag runs the security and compatibility checks, builds with poetry, publishes to PyPI through trusted publishing and creates the GitHub release.

## 0.4.0 - 2025-11-30

See the [v0.4.0 release](https://github.com/LieberInstitute/dRFEtools/releases/tag/v0.4.0).
