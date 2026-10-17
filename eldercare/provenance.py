"""Run manifests, verified artifacts, and guarded output-folder archiving."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .models import PROJECT_ROOT, sha256_file

ARTIFACT_NAMES = frozenset(
    {
        "config.json",
        "events.json",
        "evaluation.json",
        "overlay.mp4",
        "run.json",
        "run.log",
        "summary.json",
        "ticks.jsonl",
        "ticks.meta.json",
        "timeline.json",
        "timeline.txt",
        "vlm_reviews.json",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    actual = path.resolve(strict=True)
    return {
        "path": str(actual),
        "size_bytes": actual.stat().st_size,
        "sha256": sha256_file(actual),
    }


def model_file_hashes(xml_path: str) -> dict[str, str]:
    """Hash the resolved IR pair, including a companion BIN when present."""
    xml = Path(xml_path).resolve(strict=True)
    files = [xml]
    companion = xml.with_suffix(".bin")
    if companion.is_file():
        files.append(companion)
    return {str(path): sha256_file(path) for path in files}


def input_identity(video: str | None, ticks: str | None) -> dict[str, Any]:
    """Use original-video SHA when known, and retain the exact tick-file SHA."""
    if video:
        result: dict[str, Any] = {"kind": "video", "video": file_identity(Path(video))}
        result["source_sha256"] = result["video"]["sha256"]
        if ticks:
            result["kind"] = "ticks_with_video"
            result["ticks"] = file_identity(Path(ticks))
        return result
    if not ticks:
        raise ValueError("A video or ticks file is required")
    ticks_path = Path(ticks)
    result = {"kind": "ticks", "ticks": file_identity(ticks_path)}
    meta = ticks_path.with_suffix(".meta.json")
    if meta.is_file():
        with meta.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        source = metadata.get("source_video", {})
        if isinstance(source, dict) and isinstance(source.get("sha256"), str):
            result["source_video_from_ticks_meta"] = source
            result["source_sha256"] = source["sha256"]
    if "source_sha256" not in result:
        result["source_sha256"] = result["ticks"]["sha256"]
    return result


def _forbidden_output_path(path: Path) -> str | None:
    root = PROJECT_ROOT.resolve()
    home = Path.home().resolve()
    if path == root:
        return "the repository root"
    if path == home:
        return "the home directory"
    if path in root.parents:
        return "a parent of the repository"
    if path == Path(path.anchor):
        return "a filesystem root"
    if path.is_symlink():
        return "a symbolic link"
    return None


def _checked_artifacts(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    artifacts = []
    for item in path.iterdir():
        if item.is_symlink() or not item.is_file() or item.name not in ARTIFACT_NAMES:
            raise ValueError(
                f"Refusing to archive {path}: unknown or non-file entry {item.name!r}. "
                "Move it manually or choose a different --out directory."
            )
        artifacts.append(item)
    return artifacts


def prepare_output(
    path_string: str, source_sha256: str, settings_hash: str, overwrite: bool
) -> tuple[Path, Path | None]:
    """Refuse unsafe targets; archive only approved generated artifacts."""
    raw_path = Path(path_string).expanduser().absolute()
    if raw_path.is_symlink():
        raise ValueError(f"Refusing symlink output directory: {raw_path}")
    path = raw_path.resolve()
    forbidden = _forbidden_output_path(path)
    if forbidden:
        raise ValueError(f"Refusing --out {path}: it is {forbidden}")
    if not path.exists():
        path.mkdir(parents=True)
        return path, None

    artifacts = _checked_artifacts(path)
    if not artifacts:
        return path, None
    run_file = path / "run.json"
    if not overwrite:
        if not run_file.is_file():
            raise ValueError(
                f"Existing legacy output {path} has no run.json; pass --overwrite "
                "to archive its known artifacts first."
            )
        try:
            with run_file.open(encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid run.json in {path}; pass --overwrite") from exc
        old_source = previous.get("input", {}).get("source_sha256")
        old_config = previous.get("config_sha256")
        if old_source != source_sha256 or old_config != settings_hash:
            raise ValueError(
                f"Output provenance mismatch in {path}: source/config differs. "
                "Use a new --out, or pass --overwrite to archive old artifacts."
            )
        return path, None

    size = sum(item.stat().st_size for item in artifacts)
    archive_root = path.parent / "_archive"
    if archive_root.is_symlink():
        raise ValueError(f"Refusing symlink archive directory: {archive_root}")
    archive_root.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = archive_root / f"{path.name}-{stamp}"
    if archive.exists():
        raise ValueError(f"Archive destination already exists: {archive}")
    path.rename(archive)
    path.mkdir()
    print(f"Archived prior output to {archive} ({size} bytes)")
    return path, archive


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _host_info() -> dict[str, Any]:
    cpu = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        with cpuinfo.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    try:
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except OSError, ValueError:
        ram = None
    return {"cpu_model": cpu, "ram_bytes": ram, "platform": platform.platform()}


def redacted_command(argv: Sequence[str]) -> list[str]:
    output: list[str] = []
    redact_next = False
    for part in argv:
        if redact_next:
            output.append("<redacted>")
            redact_next = False
            continue
        lower = part.lower()
        sensitive = any(word in lower for word in ("api-key", "password", "token", "secret"))
        if sensitive and "=" in part:
            output.append(part.split("=", 1)[0] + "=<redacted>")
        else:
            output.append(part)
            redact_next = sensitive and "=" not in part
    return output


def manifest_template(
    argv: Sequence[str], source: dict[str, Any], settings: dict[str, Any], requested_device: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(),
        "status": "running",
        "git": _git_state(),
        "package_version": _version("eldercare-monitor") or "0.1.0",
        "command": redacted_command(argv),
        "python_version": platform.python_version(),
        "openvino_version": _version("openvino"),
        "opencv_version": _version("opencv-python"),
        "input": source,
        "config_sha256": config_hash(settings),
        "config": settings,
        "models": {
            "detector": {
                "requested_device": requested_device,
                "execution_devices": None,
                "files": {},
            },
            "pose": {"requested_device": requested_device, "execution_devices": None, "files": {}},
        },
        "warnings": [],
        "host": _host_info(),
        "artifacts": {},
    }


def write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    target = folder / "run.json"
    temp = folder / "run.json.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temp.replace(target)


def artifact_hashes(folder: Path) -> dict[str, str]:
    return {
        item.name: sha256_file(item)
        for item in folder.iterdir()
        if item.is_file() and item.name in ARTIFACT_NAMES and item.name != "run.json"
    }


def verify_result(folder: Path) -> list[str]:
    failures: list[str] = []
    run_file = folder / "run.json"
    if not run_file.is_file():
        return [f"Missing run.json: {run_file}"]
    try:
        with run_file.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read run.json: {exc}"]

    if manifest.get("config_sha256") != config_hash(manifest.get("config", {})):
        failures.append("config hash mismatch")
    source = manifest.get("input", {})
    for key in ("video", "ticks"):
        item = source.get(key)
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            failures.append(f"source {key} file missing: {path}")
        elif sha256_file(path) != item.get("sha256"):
            failures.append(f"source {key} hash mismatch: {path}")
    for model_name, item in manifest.get("models", {}).items():
        for filename, expected in item.get("files", {}).items():
            path = Path(filename)
            if not path.is_file():
                failures.append(f"{model_name} model file missing: {path}")
            elif sha256_file(path) != expected:
                failures.append(f"{model_name} model hash mismatch: {path}")
    expected_artifacts = manifest.get("artifacts", {})
    if not isinstance(expected_artifacts, dict):
        failures.append("artifact inventory is invalid")
    else:
        current = artifact_hashes(folder)
        for name, expected in expected_artifacts.items():
            if name not in current:
                failures.append(f"artifact missing: {name}")
            elif current[name] != expected:
                failures.append(f"artifact hash mismatch: {name}")
        for name in current.keys() - expected_artifacts.keys():
            failures.append(f"unrecorded artifact: {name}")
    try:
        _checked_artifacts(folder)
    except ValueError as exc:
        failures.append(str(exc))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="verify a result folder against run.json")
    verify.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    failures = verify_result(args.result_dir)
    if failures:
        for failure in failures:
            print(f"MISMATCH: {failure}")
        raise SystemExit(1)
    print(f"OK: {args.result_dir}")


if __name__ == "__main__":
    main()
