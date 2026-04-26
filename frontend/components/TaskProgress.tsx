import type { Task } from "@/lib/types";

export function TaskProgress({ task }: { task?: Task | null }) {
  if (!task) return <div className="empty-state">暂无任务</div>;
  return (
    <div className={`task-progress ${task.status}`}>
      <div className="row between">
        <strong>{task.current_stage || task.type}</strong>
        <span className={`status-pill ${task.status}`}>{task.status}</span>
      </div>
      <div className="progress-track" aria-label="task progress">
        <span style={{ width: `${Math.max(0, Math.min(100, task.progress || 0))}%` }} />
      </div>
      <div className="muted small">{task.error_message || `${task.progress || 0}%`}</div>
    </div>
  );
}
