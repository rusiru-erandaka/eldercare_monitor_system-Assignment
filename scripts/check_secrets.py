"""Reject obvious credentials in staged text; not a substitute for a full scanner."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{32,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def main() -> int:
    found = False
    for name in sys.argv[1:]:
        path = Path(name)
        if not path.is_file():
            continue
        # Never inspect local credentials, even if accidentally passed explicitly.
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if any(pattern.search(line) for pattern in PATTERNS):
                print(f"Potential credential in {path}:{number}")
                found = True
    return int(found)


if __name__ == "__main__":
    raise SystemExit(main())
