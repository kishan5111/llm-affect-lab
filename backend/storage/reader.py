"""Read helpers for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

