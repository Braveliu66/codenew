"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import type { Project, ViewerConfig } from "@/lib/types";
import { SplatViewer } from "@/components/SplatViewer";
import { TaskProgress } from "@/components/TaskProgress";

export default function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [viewer, setViewer] = useState<ViewerConfig | null>(null);

  useEffect(() => {
    if (!params.id) return;
    void Promise.all([api.project(params.id), api.viewerConfig(params.id)])
      .then(([projectData, viewerData]) => {
        setProject(projectData);
        setViewer(viewerData);
      })
      .catch(() => undefined);
  }, [params.id]);

  const latestTask = project?.tasks?.[0];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Project Detail</p>
          <h1>{project?.name ?? "加载项目"}</h1>
        </div>
        <div className="actions">
          <button className="ghost-button" disabled type="button">精细重建未接入</button>
          <button className="ghost-button" disabled type="button">导出 Mesh 未接入</button>
        </div>
      </header>

      <SplatViewer modelUrl={viewer?.status === "ready" ? viewer.model_url : null} />
      {viewer?.status === "unavailable" ? <div className="error-box">{viewer.message}</div> : null}
      {project?.error_message ? <div className="error-box">{project.error_message}</div> : null}

      <section className="grid three">
        <div className="panel stat"><span className="muted small">状态</span><strong>{project?.status ?? "-"}</strong></div>
        <div className="panel stat"><span className="muted small">输入</span><strong>{project?.input_type ?? "-"}</strong></div>
        <div className="panel stat"><span className="muted small">占用</span><strong>{formatBytes(project?.total_size_bytes)}</strong></div>
      </section>

      <section className="grid two">
        <div className="panel stack">
          <h2>最新任务</h2>
          <TaskProgress task={latestTask} />
        </div>
        <div className="panel stack">
          <h2>素材</h2>
          {(project?.media ?? []).map((item) => (
            <div className="file-row" key={item.id}>
              <span>{item.file_name}</span>
              <span className="muted small">{formatBytes(item.file_size)}</span>
            </div>
          ))}
          {!project?.media?.length ? <div className="empty-state">暂无素材</div> : null}
        </div>
      </section>
    </div>
  );
}
