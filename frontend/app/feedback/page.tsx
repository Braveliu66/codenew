"use client";

import { Send } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";

export default function FeedbackPage() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setMessage(null);
    setError(null);
    try {
      await api.feedback({ title, content });
      setTitle("");
      setContent("");
      setMessage("反馈已记录。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交失败");
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Feedback</p>
          <h1>问题反馈</h1>
        </div>
      </header>
      <section className="panel stack">
        <div className="field"><label>标题</label><input className="input" value={title} onChange={(event) => setTitle(event.target.value)} /></div>
        <div className="field"><label>描述</label><textarea className="textarea" value={content} onChange={(event) => setContent(event.target.value)} /></div>
        <button className="button" type="button" onClick={() => void submit()} disabled={!title || !content}><Send size={17} />提交</button>
        {message ? <div className="empty-state">{message}</div> : null}
        {error ? <div className="error-box">{error}</div> : null}
      </section>
    </div>
  );
}
