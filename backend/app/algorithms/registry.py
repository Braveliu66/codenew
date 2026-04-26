from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmRegistryEntry:
    name: str
    repo_url: str | None = None
    license: str | None = None
    commit_hash: str | None = None
    weight_source: str | None = None
    local_path: Path | None = None
    enabled: bool = False
    notes: str | None = None
    weight_paths: tuple[Path, ...] = ()
    commands: dict[str, list[str]] = field(default_factory=dict)
    source_type: str = "git"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AlgorithmRegistryEntry":
        local_path = data.get("local_path")
        weight_paths = data.get("weight_paths") or []
        commands = data.get("commands") or {}

        normalized_commands: dict[str, list[str]] = {}
        for key, command in commands.items():
            if isinstance(command, str):
                normalized_commands[key] = [command]
            else:
                normalized_commands[key] = [str(part) for part in command]

        return cls(
            name=str(data["name"]),
            repo_url=data.get("repo_url"),
            license=data.get("license"),
            commit_hash=data.get("commit_hash"),
            weight_source=data.get("weight_source"),
            local_path=Path(str(local_path)) if local_path else None,
            enabled=bool(data.get("enabled", False)),
            notes=data.get("notes"),
            weight_paths=tuple(Path(str(path)) for path in weight_paths),
            commands=normalized_commands,
            source_type=str(data.get("source_type") or "git"),
        )

    def command(self, key: str) -> list[str] | None:
        command = self.commands.get(key)
        return list(command) if command else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo_url": self.repo_url,
            "license": self.license,
            "commit_hash": self.commit_hash,
            "weight_source": self.weight_source,
            "local_path": str(self.local_path) if self.local_path else None,
            "enabled": self.enabled,
            "notes": self.notes,
            "weight_paths": [str(path) for path in self.weight_paths],
            "commands": self.commands,
            "source_type": self.source_type,
        }


class AlgorithmRegistry:
    def __init__(self, entries: list[AlgorithmRegistryEntry] | None = None) -> None:
        self._entries = {entry.name: entry for entry in entries or []}

    @classmethod
    def from_json_file(cls, path: str | Path) -> "AlgorithmRegistry":
        registry_path = Path(path)
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        entries = [
            AlgorithmRegistryEntry.from_mapping(item)
            for item in data.get("algorithms", [])
        ]
        return cls(entries)

    def get(self, name: str) -> AlgorithmRegistryEntry | None:
        return self._entries.get(name)

    def list_entries(self) -> list[AlgorithmRegistryEntry]:
        return list(self._entries.values())

    def to_dict(self) -> dict[str, Any]:
        return {"algorithms": [entry.to_dict() for entry in self.list_entries()]}
