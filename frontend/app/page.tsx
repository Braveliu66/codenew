"use client";

import Link from "next/link";
import { Camera, FolderKanban, UploadCloud } from "lucide-react";
import { useEffect, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import type { Task } from "@/lib/types";
import { TaskProgress } from "@/components/TaskProgress";

export default function HomePage() {
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [resources, setResources] = useState<{ gpu?: Record<string, unknown> }>({});
  const [tasks, setTasks] = useState<Task[]>([]);

  useEffect(() => {
    void api.projectSummary().then(setSummary).catch(() => undefined);
    void api.resources().then(setResources).catch(() => setResources({ gpu: { available: false, message: "仅管理员可查看完整资源状态" } }));
    void api.adminTasks()
      .then((taskData) => setTasks(taskData.tasks.filter((task) => task.status === "running" || task.status === "queued")))
      .catch(() => setTasks([]));
  }, []);

  const activeTask = tasks[0];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">3D Gaussian Splatting</p>
          <h1>重建工作台</h1>
        </div>
        <div className="actions">
          <Link className="button" href="/upload"><UploadCloud size={18} />上传素材</Link>
          <Link className="ghost-button" href="/camera"><Camera size={18} />实时视频</Link>
        </div>
      </header>

      <section className="grid four">
        <div className="panel stat"><span className="muted small">项目数</span><strong>{summary.project_count ?? 0}</strong></div>
        <div className="panel stat"><span className="muted small">训练中</span><strong>{summary.training_count ?? 0}</strong></div>
        <div className="panel stat"><span className="muted small">已完成</span><strong>{summary.completed_count ?? 0}</strong></div>
        <div className="panel stat"><span className="muted small">总占用</span><strong>{formatBytes(summary.total_size_bytes)}</strong></div>
      </section>

      <section className="grid two">
        <div className="panel stack">
          <h2>系统资源</h2>
          <div className="data-row"><span>CPU</span><strong>可用</strong></div>
          <div className="data-row"><span>GPU</span><strong>{resources.gpu?.available ? "可用" : "不可用"}</strong></div>
          <p className="muted small">{String(resources.gpu?.message ?? "Docker/WSL CUDA 环境会显示 GPU 状态。")}</p>
        </div>
        <div className="panel stack">
          <div className="row between">
            <h2>当前任务</h2>
            <Link href="/projects" className="ghost-button"><FolderKanban size={16} />项目</Link>
          </div>
          <TaskProgress task={activeTask} />
        </div>
      </section>
    </div>
  );
}
