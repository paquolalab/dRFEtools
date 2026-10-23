# Security maintenance

Report suspected vulnerabilities privately through
[GitHub security advisories](https://github.com/LieberInstitute/dRFEtools/security/advisories/new).
Avoid including sensitive data in public issues.

The development baseline requires Python 3.11 or later. Poetry defines the
dependencies; `docs/requirements.txt` is generated from its lockfile and includes
both runtime and documentation dependencies.

## Updating security patches

Use the same tooling as CI:

```bash
python -m pip install poetry==2.2.1 poetry-plugin-export==1.9.0 pip-audit==2.10.1
```

Raise affected dependency minimums in `pyproject.toml`, mirroring runtime security
minimums in `setup.py`. Update only the affected packages, for example:

```bash
poetry update pillow urllib3 requests pygments idna certifi --lock
poetry export --with docs --without-hashes --format requirements.txt --output docs/requirements.txt
poetry check --lock
python scripts/audit_lock.py
python -m pip_audit --strict --no-deps --disable-pip --requirement docs/requirements.txt
```

The lock audit covers every locked version, including packages marked for other
platforms or Python versions. Audit errors and known vulnerabilities fail CI;
the dependency database must be reachable. An audit does not establish whether
an application exercises a vulnerable code path.

Verify compatibility in a clean environment on Python 3.11 and 3.14:

```bash
python -m pip install -r docs/requirements.txt pytest==8.4.2 build==1.3.0
python -m pip install --no-deps .
python -m pip check
MPLBACKEND=Agg python -m pytest -q
python -m dRFEtools.cli --help
python -m sphinx -b html docs docs/_build/html
python -m build --wheel
```

CI also installs the wheel in a fresh environment with the audited requirements
as constraints, checks its dependencies, and exercises its installed CLI.

## Automated checks

Security and compatibility checks run on pull requests, pushes to `main`, every
Monday at 12:23 UTC, and manual dispatch. They reject documentation requirements
that differ from a fresh Poetry export.

Dependabot is configured for security PRs only, including security fixes that
require a major version upgrade. Its security updates must also be enabled in
the repository's GitHub settings; the YAML configuration alone does not enable
them. PRs require review and are not automatically merged. When reviewing a
Dependabot PR, synchronize the dependency minimums, lockfile, and generated
documentation requirements using the commands above before merging.

After a fix reaches `main`, check that GitHub resolves the associated Dependabot
alerts. Existing published distributions require a separate release to carry
the updated dependency minimums to package-index users.
