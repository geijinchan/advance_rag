"""Tiny atomic-JSON persistence helper.

Used by the session/feedback stores so conversation history and answer
ratings survive service restarts (previously: in-memory only, reset on every
restart — documented risk #2 from the prior round).

Writes are atomic (tmp file + rename) so a crash mid-write can never corrupt
the store. ``write_json`` never raises: persistence is a best-effort
convenience, not a correctness gate.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag-agent.persist")


def load_json(path: str | os.PathLike) -> Any | None:
    try:
        p = Path(path)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("persist: failed to load %s (%s) — starting empty", path, exc)
        return None


def save_json(path: str | os.PathLike, payload: Any) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return True
    except OSError as exc:
        logger.warning("persist: failed to save %s (%s)", path, exc)
        return False
