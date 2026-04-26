"use client";

import Link from "next/link";
import { FileUp, Play, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import type { MediaAsset, Project, Task } from "@/lib/types";
import { TaskProgress } from "@/components/TaskProgress";

export default function UploadPage() {
  const [name, setName] = useState("新建重建项目");
  const [inputType, setInputType] = useState<Project["input_type"]>("images");
  const [tags, setTags] = useState("preview, research");
  const [project, setProject] = useState<Project | null>(null);
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [task, setTask] = useState<Task | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalBytes = useMemo(() => media.reduce((sum, item) => sum + item.file_size, 0), [media]);

  useEffect(() => {
    if (!task || task.status === "failed" || task.status === "succeeded" || task.status === "canceled") return;
    const timer = window.setInterval(() => {
      void api.task(task.id).then(setTask).catch(() => undefined);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [task]);

  async function ensureProject() {
    if (project) return project;
    const created = await api.createProject({
      name,
      input_type: inputType,
      tags: tags.split(",").map((item) => item.trim()).filter(Boolean)
    });
    setProject(created);
    return created;
  }

  async function onFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    try {
      const active = await ensureProject();
      const uploaded: MediaAsset[] = [];
      for (const file of Array.from(files)) {
        uploaded.push(await api.uploadMedia(active.id, file));
      }
      setMedia((items) => [...items, ...uploaded]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setBusy(false);
    }
  }

  async function startPreview() {
    if (!project) return;
    setBusy(true);
    setError(null);
    try {
      setTask(await api.startPreview(project.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "预览任务创建失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Upload</p>
          <h1>上传素材并发起真实极速预览</h1>
        </div>
        {project ? <Link className="ghost-button" href={`/projects/${project.id}`}>项目详情</Link> : null}
      </header>

      <section className="grid two">
        <div className="panel stack">
          <div className="field">
            <label>项目名称</label>
            <input className="input" value={name} onChange={(event) => setName(event.target.value)} disabled={Boolean(project)} />
          </div>
          <div className="field">
            <label>输入类型</label>
            <select className="select" value={inputType} onChange={(event) => setInputType(event.target.value as Project["input_type"])} disabled={Boolean(project)}>
              <option value="images">图片</option>
              <option value="video">视频</option>
            </select>
          </div>
          <div className="field">
            <label>标签</label>
            <input className="input" value={tags} onChange={(event) => setTags(event.target.value)} disabled={Boolean(project)} />
          </div>
          <label className="dropzone">
            <FileUp size={22} />
            <strong>选择真实素材文件</strong>
            <span className="muted small">图片或视频会写入后端对象存储，失败任务不会生成假 preview.spz。</span>
            <input hidden type="file" multiple={inputType === "images"} accept={inputType === "images" ? "image/*" : "video/*"} onChange={(event) => void onFiles(event.target.files)} />
          </label>
          <button className="button" type="button" onClick={() => void startPreview()} disabled={!project || media.length === 0 || busy}>
            {busy ? <RefreshCw size={17} /> : <Play size={17} />}极速预览
          </button>
          {error ? <div className="error-box">{error}</div> : null}
        </div>

        <div className="panel stack">
          <h2>素材统计</h2>
          <div className="grid three">
            <div className="stat"><span className="muted small">文件数</span><strong>{media.length}</strong></div>
            <div className="stat"><span className="muted small">图片</span><strong>{media.filter((item) => item.kind === "image").length}</strong></div>
            <div className="stat"><span className="muted small">大小</span><strong>{formatBytes(totalBytes)}</strong></div>
          </div>
          <div className="file-list">
            {media.map((item) => (
              <div className="file-row" key={item.id}>
                <span>{item.file_name}</span>
                <span className="muted small">{formatBytes(item.file_size)}</span>
              </div>
            ))}
            {media.length === 0 ? <div className="empty-state">等待上传素材</div> : null}
          </div>
          <TaskProgress task={task} />
        </div>
      </section>
    </div>
  );
}
