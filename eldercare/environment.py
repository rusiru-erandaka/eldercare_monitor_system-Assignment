"""Load local environment variables used by optional integrations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Union[str, Path]) -> Path:
  """Load KEY=VALUE entries without replacing exported environment variables."""
  env_path = Path(path)
  if not env_path.is_file():
    return env_path

  with env_path.open(encoding="utf-8") as stream:
    for line_number, raw_line in enumerate(stream, 1):
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      if line.startswith("export "):
        line = line[7:].strip()

      key, separator, value = line.partition("=")
      key = key.strip()
      if not separator or not key or not key.replace("_", "a").isalnum():
        raise ValueError(
            f"Invalid environment entry at {env_path}:{line_number}")

      value = value.strip()
      if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
      os.environ.setdefault(key, value)
  return env_path


def load_project_env() -> Path:
  """Load the project's private .env file when it exists."""
  return load_env_file(PROJECT_ROOT / ".env")
