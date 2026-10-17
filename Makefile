.PHONY: setup lint typecheck test test-all

setup:
	uv python install 3.14.4
	uv sync --locked --group dev --extra vlm

lint:
	.venv/bin/ruff check eldercare tests scripts
	.venv/bin/ruff format --check eldercare/config.py eldercare/models.py eldercare/provenance.py eldercare/logging_utils.py scripts/fetch_models.py tests/test_config.py tests/test_models.py tests/test_model_assets.py tests/test_provenance.py tests/test_logging.py tests/test_goldens.py

typecheck:
	.venv/bin/mypy
	.venv/bin/mypy --strict --follow-imports=silent eldercare/config.py eldercare/models.py eldercare/provenance.py

test:
	.venv/bin/python -m pytest -q

test-all:
	.venv/bin/python scripts/fetch_models.py --verify-only
	.venv/bin/python -m pytest -q
