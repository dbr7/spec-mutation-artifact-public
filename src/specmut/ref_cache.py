"""Cache for reference programs and their test results."""

import json
import logging
import threading
from pathlib import Path

from specmut.config import Config

logger = logging.getLogger(__name__)

_CACHE_FILE = "ref_programs.json"

# Module-level lock that serialises read-modify-write of the
# ref_programs.json file. The cache is a single JSON file shared
# across all (task_id, model) entries, so concurrent save_ref
# calls from parallel pipeline workers can race and lose updates
# without this lock. get_ref is also locked, both for consistency
# and to avoid the rare case where a reader sees a partially
# written file.
_ref_lock = threading.Lock()


def _cache_path(config: Config) -> Path:
    path = config.cache_dir / _CACHE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cache(config: Config) -> dict:
    path = _cache_path(config)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_cache(config: Config, cache: dict):
    path = _cache_path(config)
    path.write_text(json.dumps(cache, indent=2))


def _key(task_id: str, model: str) -> str:
    return f"{task_id}::{model}"


def get_ref(
    config: Config,
    task_id: str,
    model: str,
    *,
    include_failed: bool = True,
) -> dict | None:
    """Get cached reference program. Returns dict with 'code' and 'passes' or None."""
    with _ref_lock:
        cache = _load_cache(config)
        entry = cache.get(_key(task_id, model))
    if entry:
        if not include_failed and not entry.get("passes", False):
            logger.info("  Cached reference failed previously; regenerating")
            return None
        logger.info(f"  Reference program loaded from cache")
    return entry


def save_ref(config: Config, task_id: str, model: str, code: str, passes: bool):
    """Save reference program and its pass/fail status to cache."""
    with _ref_lock:
        cache = _load_cache(config)
        cache[_key(task_id, model)] = {
            "task_id": task_id,
            "model": model,
            "code": code,
            "passes": passes,
        }
        _save_cache(config, cache)
