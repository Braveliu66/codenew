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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!params.id) return;
    void Promise.all([api.project(params.id), api.viewerConfig(params.id)])
      .then(([projectData, viewerData]) => {
        setProject(projectData);
        setViewer(viewerData);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load project"));
  }, [params.id]);

  const latestTask = project?.tasks?.[0];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Project Detail</p>
          <h1>{project?.name ?? "Loading project"}</h1>
        </div>
        <div className="actions">
          <button className="ghost-button" disabled type="button">Fine reconstruction unavailable</button>
          <button className="ghost-button" disabled type="button">Mesh export unavailable</button>
        </div>
      </header>

      <SplatViewer modelUrl={viewer?.status === "ready" ? viewer.model_url : null} />
      {viewer?.status === "unavailable" ? <div className="error-box">{viewer.message}</div> : null}
      {project?.error_message ? <div className="error-box">{project.error_message}</div> : null}
      {error ? <div className="error-box">{error}</div> : null}

      <section className="grid three">
        <div className="panel stat"><span className="muted small">Status</span><strong>{project?.status ?? "-"}</strong></div>
        <div className="panel stat"><span className="muted small">Input</span><strong>{project?.input_type ?? "-"}</strong></div>
        <div className="panel stat"><span className="muted small">Storage</span><strong>{formatBytes(project?.total_size_bytes)}</strong></div>
      </section>

      <section className="grid two">
        <div className="panel stack">
          <h2>Latest Task</h2>
          <TaskProgress task={latestTask} />
        </div>
        <div className="panel stack">
          <h2>Media</h2>
          {(project?.media ?? []).map((item) => (
            <div className="file-row" key={item.id}>
              <span>{item.file_name}</span>
              <span className="muted small">{formatBytes(item.file_size)}</span>
            </div>
          ))}
          {!project?.media?.length ? <div className="empty-state">No media</div> : null}
        </div>
      </section>
    </div>
  );
}
