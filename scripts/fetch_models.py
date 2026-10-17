"""Copy only locally supplied, hash-pinned OpenVINO model assets."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eldercare.models import load_manifest, model_directory, sha256_file, verify_model  # noqa: E402


def import_model(name: str, source: Path) -> None:
    entry = load_manifest()["models"][name]
    if not source.is_dir():
        raise ValueError(f"{name} source directory does not exist: {source}")
    destination = model_directory(name)
    planned: list[tuple[Path, Path]] = []
    for filename, expected in entry["files"].items():
        candidate = source / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing {name} source file: {candidate}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(
                f"{name} source hash mismatch for {candidate}: expected {expected}, got {actual}"
            )
        target = destination / filename
        if target.exists() and sha256_file(target) != expected:
            raise ValueError(f"Refusing to replace corrupted destination: {target}")
        planned.append((candidate, target))

    destination.mkdir(parents=True, exist_ok=True)
    for source_file, target in planned:
        if target.exists():
            continue
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".model-", delete=False) as temp:
            temporary = Path(temp.name)
        try:
            shutil.copyfile(source_file, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    for path, digest in verify_model(name).items():
        print(f"verified {name}: {path} sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-source", type=Path)
    parser.add_argument("--pose-source", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        for name in ("detector", "pose"):
            for path, digest in verify_model(name).items():
                print(f"verified {name}: {path} sha256={digest}")
        return
    if args.detector_source is None or args.pose_source is None:
        parser.error(
            "--detector-source and --pose-source are required unless --verify-only is used"
        )
    import_model("detector", args.detector_source)
    import_model("pose", args.pose_source)


if __name__ == "__main__":
    main()
