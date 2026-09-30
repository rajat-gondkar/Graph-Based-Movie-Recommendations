"""Helpers for the JSON caches (TMDB responses, LLM themes, LLM explanations)."""
import json
import os
import tempfile
from pathlib import Path


def load_json(path: Path) -> dict:
    """Read a JSON cache file, or return an empty dict if it doesn't exist yet."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path) -> None:
    """Write JSON atomically (D13): write a temp file in the same folder, then rename it over the target."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
