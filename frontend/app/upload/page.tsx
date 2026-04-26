"use client";

import Link from "next/link";
import { FileUp, Play, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import type { MediaAsset, Project, Task } from "@/lib/types";
import { TaskProgress } from "@/components/TaskProgress";

const MIN_INPUT_FRAMES = 8;
const MAX_INPUT_FRAMES = 800;

export default function UploadPage() {
  const [name, setName] = useState("New reconstruction");
  const [inputType, setInputType] = useState<Project["input_type"]>("images");
  const [tags, setTags] = useState("preview, research");
  const [project, setProject] = useState<Project | null>(null);
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [task, setTask] = useState<Task | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalBytes = useMemo(() => media.reduce((sum, item) => sum + item.file_size, 0), [media]);
  const imageCount = media.filter((item) => item.kind === "image").length;
  const canStartPreview = project && media.length > 0 && (inputType === "video" || imageCount >= MIN_INPUT_FRAMES);

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
      setError(err instanceof Error ? err.message : "Upload failed");
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
      setError(err instanceof Error ? err.message : "Preview task creation failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Upload</p>
          <h1>Upload media and start a real preview</h1>
        </div>
        {project ? <Link className="ghost-button" href={`/projects/${project.id}`}>Project detail</Link> : null}
      </header>

      <section className="grid two">
        <div className="panel stack">
          <div className="field">
            <label>Project name</label>
            <input className="input" value={name} onChange={(event) => setName(event.target.value)} disabled={Boolean(project)} />
          </div>
          <div className="field">
            <label>Input type</label>
            <select className="select" value={inputType} onChange={(event) => setInputType(event.target.value as Project["input_type"])} disabled={Boolean(project)}>
              <option value="images">Images</option>
              <option value="video">Video</option>
            </select>
          </div>
          <div className="field">
            <label>Tags</label>
            <input className="input" value={tags} onChange={(event) => setTags(event.target.value)} disabled={Boolean(project)} />
          </div>
          <label className="dropzone">
            <FileUp size={22} />
            <strong>Select real media files</strong>
            <span className="muted small">
              Preview uses at least {MIN_INPUT_FRAMES} input frames and samples at most {MAX_INPUT_FRAMES}. Failed tasks never create fake preview.spz artifacts.
            </span>
            <input hidden type="file" multiple={inputType === "images"} accept={inputType === "images" ? "image/*" : "video/*"} onChange={(event) => void onFiles(event.target.files)} />
          </label>
          <button className="button" type="button" onClick={() => void startPreview()} disabled={!canStartPreview || busy}>
            {busy ? <RefreshCw size={17} /> : <Play size={17} />} Start preview
          </button>
          {inputType === "images" && imageCount > 0 && imageCount < MIN_INPUT_FRAMES ? (
            <div className="error-box">Upload at least {MIN_INPUT_FRAMES} images before starting preview.</div>
          ) : null}
          {error ? <div className="error-box">{error}</div> : null}
        </div>

        <div className="panel stack">
          <h2>Media stats</h2>
          <div className="grid three">
            <div className="stat"><span className="muted small">Files</span><strong>{media.length}</strong></div>
            <div className="stat"><span className="muted small">Images</span><strong>{imageCount}</strong></div>
            <div className="stat"><span className="muted small">Size</span><strong>{formatBytes(totalBytes)}</strong></div>
          </div>
          <div className="file-list">
            {media.map((item) => (
              <div className="file-row" key={item.id}>
                <span>{item.file_name}</span>
                <span className="muted small">{formatBytes(item.file_size)}</span>
              </div>
            ))}
            {media.length === 0 ? <div className="empty-state">Waiting for uploads</div> : null}
          </div>
          <TaskProgress task={task} />
        </div>
      </section>
    </div>
  );
}
