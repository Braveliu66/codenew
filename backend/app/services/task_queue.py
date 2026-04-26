from __future__ import annotations

from backend.app.core.config import Settings, get_settings


class TaskQueueError(RuntimeError):
    pass


class PreviewTaskQueue:
    queue_name = "preview_tasks"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    def enqueue_preview(self, task_id: str) -> None:
        self._redis().rpush(self.queue_name, task_id)

    def pop_preview(self, timeout_seconds: int = 5) -> str | None:
        result = self._redis().blpop(self.queue_name, timeout=timeout_seconds)
        if not result:
            return None
        _, task_id = result
        return task_id.decode("utf-8") if isinstance(task_id, bytes) else str(task_id)

    def _redis(self):
        if self._client is not None:
            return self._client
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise TaskQueueError("Redis client is not installed; install backend/requirements.txt") from exc
        self._client = redis.Redis.from_url(self.settings.redis_url)
        try:
            self._client.ping()
        except Exception as exc:
            raise TaskQueueError(f"Redis queue is unavailable: {exc}") from exc
        return self._client
