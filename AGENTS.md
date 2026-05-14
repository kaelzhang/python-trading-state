# Repo Adoption — OST.ai Standards

This repository follows the OST.ai standards at
`ost/standards/AGENTS.md`. Local notes only — defer everything else
to the canonical standards.

## Scope (FS-8)
- Handwritten code: `trading_state/`, `test/`
- Generated / exempt: `build/`, `dist/`, `*.egg-info/`, `no_track/`
- File-size ceiling: 1000 physical lines (`platform/FILE_SIZE_STANDARD.md`)
- No oversize files at adoption time.

## Review Gate (FS-8)
GitHub Actions (`.github/workflows/python.yml`) runs `make test-ci`
on every push. Reviewers reject:
- new handwritten file > 1000 lines
- existing 1001–1500 line file growing
- `make test-ci` failing

## Test Entrypoints (TC-7)
- Unit + coverage (local): `make test`
- Unit + coverage (CI): `make test-ci`
- Lint: `make lint` (ruff + mypy)
- Test helpers: `test/fixtures/`

Integration / race / stress standards do not apply — this is a pure
library with no PostgreSQL, Redis, or runtime workload.
