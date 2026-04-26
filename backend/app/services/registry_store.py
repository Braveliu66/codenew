from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.algorithms.registry import AlgorithmRegistry, AlgorithmRegistryEntry
from backend.app.core.config import get_settings
from backend.app.db import models
from backend.app.services.serializers import algorithm_to_dict


def seed_algorithm_registry(db: Session, registry_path: Path | None = None) -> None:
    path = registry_path or get_settings().algorithm_registry_path
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data.get("algorithms", []):
        name = str(item["name"])
        existing = db.scalar(select(models.AlgorithmRegistryRecord).where(models.AlgorithmRegistryRecord.name == name))
        if existing:
            continue
        db.add(
            models.AlgorithmRegistryRecord(
                name=name,
                repo_url=item.get("repo_url"),
                license=item.get("license"),
                commit_hash=item.get("commit_hash"),
                weight_source=item.get("weight_source"),
                local_path=item.get("local_path"),
                enabled=bool(item.get("enabled", False)),
                notes=item.get("notes"),
                commands=item.get("commands") or {},
                weight_paths=item.get("weight_paths") or [],
                source_type=str(item.get("source_type") or "git"),
            )
        )
    db.commit()


def list_algorithm_records(db: Session) -> list[models.AlgorithmRegistryRecord]:
    return list(db.scalars(select(models.AlgorithmRegistryRecord).order_by(models.AlgorithmRegistryRecord.name)))


def registry_to_response(db: Session) -> dict[str, Any]:
    records = list_algorithm_records(db)
    return {"algorithms": [algorithm_to_dict(record) for record in records]}


def load_registry_from_db(db: Session) -> AlgorithmRegistry:
    records = list_algorithm_records(db)
    entries = [
        AlgorithmRegistryEntry.from_mapping(
            {
                "name": record.name,
                "repo_url": record.repo_url,
                "license": record.license,
                "commit_hash": record.commit_hash,
                "weight_source": record.weight_source,
                "local_path": record.local_path,
                "enabled": record.enabled,
                "notes": record.notes,
                "commands": record.commands or {},
                "weight_paths": record.weight_paths or [],
                "source_type": record.source_type,
            }
        )
        for record in records
    ]
    return AlgorithmRegistry(entries)
