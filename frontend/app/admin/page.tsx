"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AlgorithmEntry, RuntimePreflight, Task, User } from "@/lib/types";

export default function AdminPage() {
  const [user, setUser] = useState<User | null>(null);
  const [algorithms, setAlgorithms] = useState<AlgorithmEntry[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [resources, setResources] = useState<Record<string, unknown>>({});
  const [workers, setWorkers] = useState<unknown[]>([]);
  const [preflight, setPreflight] = useState<RuntimePreflight | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.me().then(setUser).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!user) return;
    if (user.role !== "admin") {
      setError("Admin role required.");
      return;
    }
    void Promise.all([api.adminAlgorithms(), api.adminTasks(), api.resources(), api.workers(), api.runtimePreflight()])
      .then(([algorithmData, taskData, resourceData, workerData, preflightData]) => {
        setAlgorithms(algorithmData.algorithms);
        setTasks(taskData.tasks);
        setResources(resourceData as unknown as Record<string, unknown>);
        setWorkers(workerData.workers);
        setPreflight(preflightData);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to read admin APIs"));
  }, [user]);

  if (error) {
    return (
      <div className="page">
        <header className="page-header">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>Admin Panel</h1>
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
          <h1>Admin Panel</h1>
        </div>
      </header>
      <section className="grid three">
        <div className="panel stat"><span className="muted small">Tasks</span><strong>{tasks.length}</strong></div>
        <div className="panel stat"><span className="muted small">Workers</span><strong>{workers.length}</strong></div>
        <div className="panel stat"><span className="muted small">Enabled Algorithms</span><strong>{algorithms.filter((item) => item.enabled).length}</strong></div>
      </section>

      <section className="panel stack">
        <h2>Runtime Preflight</h2>
        <div className="grid three">
          <div className="stat"><span className="muted small">Python</span><strong>{String(preflight?.python.version ?? "-")}</strong></div>
          <div className="stat"><span className="muted small">CUDA</span><strong>{String(preflight?.torch.cuda_available ?? false)}</strong></div>
          <div className="stat"><span className="muted small">GPU</span><strong>{String(preflight?.gpu.available ?? false)}</strong></div>
        </div>
        {preflight?.errors.length ? <div className="error-box">{preflight.errors.join("\n")}</div> : null}
        <div className="table">
          <div className="table-row header"><span>Algorithm</span><span>License</span><span>Commit</span><span>Status</span></div>
          {(preflight?.algorithms ?? []).map((item) => (
            <div className="table-row" key={item.name}>
              <span>{item.name}</span>
              <span>{item.license ?? "-"}</span>
              <span className="small muted">{item.commit_hash ?? "-"}</span>
              <span className={`status-pill ${item.ready ? "ready" : "failed"}`}>{item.ready ? "ready" : item.enabled ? "check" : "disabled"}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="grid two">
        <div className="panel stack">
          <h2>Resources</h2>
          <pre className="empty-state">{JSON.stringify(resources, null, 2)}</pre>
        </div>
        <div className="panel stack">
          <h2>Recent Tasks</h2>
          {tasks.slice(0, 8).map((task) => (
            <div className="data-row" key={task.id}>
              <span>{task.type}</span>
              <span className={`status-pill ${task.status}`}>{task.status}</span>
            </div>
          ))}
          {tasks.length === 0 ? <div className="empty-state">No tasks</div> : null}
        </div>
      </section>
    </div>
  );
}
