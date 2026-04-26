from __future__ import annotations

import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.security import hash_password
from backend.app.db import models
from backend.app.services.registry_store import seed_algorithm_registry


def seed_database(db: Session) -> None:
    seed_algorithm_registry(db)
    username = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
    password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")
    email = os.environ.get("DEFAULT_ADMIN_EMAIL")
    existing = db.scalar(select(models.User).where(models.User.username == username))
    if existing:
        return
    db.add(
        models.User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role="admin",
        )
    )
    db.commit()
