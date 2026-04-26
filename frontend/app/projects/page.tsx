"use client";

import Link from "next/link";
import { Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import type { Project } from "@/lib/types";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");

  useEffect(() => {
    void api.projects().then((data) => setProjects(data.projects)).catch(() => undefined);
  }, []);

  const visible = useMemo(() => {
    return projects.filter((project) => {
      const matchesQuery = project.name.toLowerCase().includes(query.toLowerCase());
      const matchesStatus = status === "all" || project.status === status;
      return matchesQuery && matchesStatus;
    });
  }, [projects, query, status]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Projects</p>
          <h1>项目管理</h1>
        </div>
        <Link className="button" href="/upload">新建项目</Link>
      </header>
      <section className="panel row between">
        <div className="field" style={{ flex: 1 }}>
          <label>搜索</label>
          <div className="row">
            <Search size={18} className="muted" />
            <input className="input" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
        </div>
        <div className="field">
          <label>状态</label>
          <select className="select" value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="all">全部</option>
            <option value="PREVIEW_RUNNING">训练中</option>
            <option value="PREVIEW_READY">预览完成</option>
            <option value="FAILED">失败</option>
          </select>
        </div>
      </section>
      <section className="grid three">
        {visible.map((project) => (
          <Link className="panel project-card" href={`/projects/${project.id}`} key={project.id}>
            <div className="preview-tile">{project.status === "PREVIEW_READY" ? "SPZ" : project.input_type}</div>
            <div className="row between">
              <h3>{project.name}</h3>
              <span className={`status-pill ${project.status}`}>{project.status}</span>
            </div>
            <p className="muted small">{project.tags.join(" / ") || "无标签"}</p>
            <div className="row between small muted">
              <span>{formatBytes(project.total_size_bytes)}</span>
              <span>{new Date(project.updated_at).toLocaleString()}</span>
            </div>
          </Link>
        ))}
        {visible.length === 0 ? <div className="empty-state">暂无匹配项目</div> : null}
      </section>
    </div>
  );
}
