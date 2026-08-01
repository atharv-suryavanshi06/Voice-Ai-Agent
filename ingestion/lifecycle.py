"""Durable, atomic ingestion status journal."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict


VALID_INGESTION_STATES = {
    "pending", "extracted", "validated", "indexed", "failed", "active"
}


class IngestionJournal:
    def __init__(self, path: str) -> None:
        self.path = path

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"ingestions": {}}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"ingestions": {}}

    def update(self, ingestion_id: str, state: str, **details: Any) -> None:
        if state not in VALID_INGESTION_STATES:
            raise ValueError(f"Unsupported ingestion state: {state}")
        data = self._load()
        ingestions = data.setdefault("ingestions", {})
        record = dict(ingestions.get(ingestion_id) or {})
        record.update(details)
        record["state"] = state
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        ingestions[ingestion_id] = record
        self._atomic_write(data)

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix="ingestion-", suffix=".json.tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
