from __future__ import annotations

from backend.app.core.config import Settings, get_settings


class TaskQueueError(RuntimeError):
    pass


class PreviewTaskQueue:
    queue_name = "preview_tasks"
    image_queue_name = "preview_image_tasks"
    video_queue_name = "preview_video_tasks"
    camera_queue_name = "preview_camera_tasks"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    def enqueue_preview(self, task_id: str, input_type: str = "images") -> None:
        self._redis().rpush(self._queue_for_input_type(input_type), task_id)

    def enqueue_image_preview(self, task_id: str) -> None:
        self.enqueue_preview(task_id, "images")

    def enqueue_video_preview(self, task_id: str) -> None:
        self.enqueue_preview(task_id, "video")

    def enqueue_camera_preview(self, task_id: str) -> None:
        self.enqueue_preview(task_id, "camera")

    def pop_preview(self, timeout_seconds: int = 5, input_type: str = "images") -> str | None:
        result = self._redis().blpop(self._queue_for_input_type(input_type), timeout=timeout_seconds)
        if not result:
            return None
        _, task_id = result
        return task_id.decode("utf-8") if isinstance(task_id, bytes) else str(task_id)

    def _queue_for_input_type(self, input_type: str) -> str:
        if input_type == "video":
            return self.video_queue_name
        if input_type == "camera":
            return self.camera_queue_name
        if input_type == "images":
            return self.image_queue_name
        return self.queue_name

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
