#!/usr/bin/env bash
# Every gate CI runs, in CI's order, so a local pass cannot diverge from a CI pass.
set -euo pipefail
PY=.venv/bin/python
# CI runs the console script, which keeps the CWD off sys.path, unlike `-m pytest`.
PYTEST=.venv/bin/pytest
SHIPPED="src tools"
# Fakes mirror third-party signatures, so their unused parameters are the contract, not dead code.
UNUSED_BY_DESIGN="pytestmark,pytest_*,config,chunksize,a,k,exc,padding,truncation,max_length,return_tensors,normalize_embeddings"
$PY -m ruff check .
$PY -m ruff format --check .
$PY -m mypy --strict $SHIPPED
$PY -m deptry $SHIPPED
$PY -m vulture $SHIPPED tests .vulture-whitelist.py --min-confidence 60 \
  --ignore-decorators "@pytest.fixture" --ignore-names "$UNUSED_BY_DESIGN"
$PYTEST -q -m "not integration"
