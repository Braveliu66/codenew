"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AlgorithmEntry } from "@/lib/types";

export default function AboutPage() {
  const [algorithms, setAlgorithms] = useState<AlgorithmEntry[]>([]);

  useEffect(() => {
    void api.algorithms().then((data) => setAlgorithms(data.algorithms)).catch(() => undefined);
  }, []);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Compliance</p>
          <h1>算法与许可证</h1>
        </div>
      </header>
      <section className="panel stack">
        <p className="muted">
          本系统仅面向毕业设计、实验室内部验证和非商业研究。算法必须登记仓库、许可证、commit hash、权重来源和启用状态；未完成配置时后端会返回明确失败，不生成假模型或假产物。
        </p>
        <div className="table">
          <div className="table-row header"><span>算法</span><span>许可证</span><span>仓库</span><span>状态</span></div>
          {algorithms.map((item) => (
            <div className="table-row" key={item.name}>
              <span>{item.name}</span>
              <span>{item.license ?? "-"}</span>
              <span className="small muted">{item.repo_url ?? "-"}</span>
              <span className={`status-pill ${item.enabled ? "ready" : "failed"}`}>{item.enabled ? "enabled" : "disabled"}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
