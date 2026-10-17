"""Optional integration check for locally supplied, pinned model files."""

from __future__ import annotations

from pathlib import Path

import pytest

from eldercare.models import load_manifest, model_directory, verify_model


@pytest.mark.needs_models
def test_local_model_assets_match_manifest() -> None:
    manifest = load_manifest()
    for name, entry in manifest["models"].items():
        directory = model_directory(name)
        missing = [filename for filename in entry["files"] if not (directory / filename).is_file()]
        if missing:
            pytest.skip(
                f"Local {name} model missing ({', '.join(missing)}); "
                "run scripts/fetch_models.py with verified local sources"
            )
        verified = verify_model(name)
        assert verified
        assert all(Path(path).is_file() for path in verified)
