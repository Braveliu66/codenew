"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AlgorithmEntry, Task, User } from "@/lib/types";

export default function AdminPage() {
  const [user, setUser] = useState<User | null>(null);
  const [algorithms, setAlgorithms] = useState<AlgorithmEntry[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [resources, setResources] = useState<Record<string, unknown>>({});
  const [workers, setWorkers] = useState<unknown[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.me().then(setUser).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!user) return;
    if (user.role !== "admin") {
      setError("当前账号没有管理员权限。");
      return;
    }
    void Promise.all([api.adminAlgorithms(), api.adminTasks(), api.resources(), api.workers()])
      .then(([algorithmData, taskData, resourceData, workerData]) => {
        setAlgorithms(algorithmData.algorithms);
        setTasks(taskData.tasks);
        setResources(resourceData as unknown as Record<string, unknown>);
        setWorkers(workerData.workers);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "管理接口读取失败"));
  }, [user]);

  if (error) {
    return (
      <div className="page">
        <header className="page-header">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>管理面板</h1>
          </div>
        </header>
        <div className="error-box">{error}</div>
      </div>
    );
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>管理面板</h1>
        </div>
      </header>
      <section className="grid three">
        <div className="panel stat"><span className="muted small">任务数</span><strong>{tasks.length}</strong></div>
        <div className="panel stat"><span className="muted small">Worker</span><strong>{workers.length}</strong></div>
        <div className="panel stat"><span className="muted small">启用算法</span><strong>{algorithms.filter((item) => item.enabled).length}</strong></div>
      </section>
      <section className="grid two">
        <div className="panel stack">
          <h2>资源</h2>
          <pre className="empty-state">{JSON.stringify(resources, null, 2)}</pre>
        </div>
        <div className="panel stack">
          <h2>任务</h2>
          {tasks.slice(0, 8).map((task) => (
            <div className="data-row" key={task.id}>
              <span>{task.type}</span>
              <span className={`status-pill ${task.status}`}>{task.status}</span>
            </div>
          ))}
          {tasks.length === 0 ? <div className="empty-state">暂无任务</div> : null}
        </div>
      </section>
      <section className="panel stack">
        <h2>算法登记</h2>
        <div className="table">
          <div className="table-row header"><span>名称</span><span>许可证</span><span>Commit</span><span>状态</span></div>
          {algorithms.map((item) => (
            <div className="table-row" key={item.name}>
              <span>{item.name}</span>
              <span>{item.license ?? "-"}</span>
              <span className="small muted">{item.commit_hash ?? "未登记"}</span>
              <span className={`status-pill ${item.enabled ? "ready" : "failed"}`}>{item.enabled ? "enabled" : "disabled"}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
