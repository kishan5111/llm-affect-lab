"""JSONL storage helpers."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

try:
    import fcntl
except ImportError:  # pragma: no cover - this project currently targets Unix-like runners.
    fcntl = None


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[resolved] = lock
        return lock


@contextmanager
def _locked_file(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _thread_lock_for(path)
    with thread_lock:
        with path.open(mode, encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield f
                f.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: Path, record: BaseModel | dict) -> None:
    if isinstance(record, BaseModel):
        line = record.model_dump_json()
    else:
        line = json.dumps(record, ensure_ascii=False)
    with _locked_file(path, "a") as f:
        f.write(line + "\n")


def write_jsonl(path: Path, records: Iterable[BaseModel | dict]) -> None:
    with _locked_file(path, "w") as f:
        for record in records:
            if isinstance(record, BaseModel):
                f.write(record.model_dump_json() + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: BaseModel | dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
