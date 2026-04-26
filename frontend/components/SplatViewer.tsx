"use client";

import { Maximize2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { artifactUrl } from "@/lib/api";

export function SplatViewer({ modelUrl }: { modelUrl?: string | null }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [message, setMessage] = useState("真实 preview_spz 产物生成后会在这里加载。");

  useEffect(() => {
    if (!modelUrl || !hostRef.current) {
      setState("idle");
      setMessage("真实 preview_spz 产物生成后会在这里加载。");
      return;
    }
    const resolvedModelUrl = modelUrl;
    let cancelled = false;
    let cleanup: (() => void) | undefined;
    async function mountViewer() {
      try {
        setState("loading");
        setMessage("正在初始化 Spark Viewer");
        const THREE = await import("three");
        const Spark = (await import("@sparkjsdev/spark")) as Record<string, unknown>;
        const SplatMesh = Spark.SplatMesh as undefined | (new (options: { url: string }) => any);
        if (!SplatMesh) {
          throw new Error("当前 @sparkjsdev/spark 包未暴露 SplatMesh");
        }
        const host = hostRef.current;
        if (!host || cancelled) return;
        host.innerHTML = "";
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(55, host.clientWidth / Math.max(host.clientHeight, 1), 0.01, 1000);
        camera.position.set(0, 0, 3);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(host.clientWidth, host.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        host.appendChild(renderer.domElement);
        const splat = new SplatMesh({ url: artifactUrl(resolvedModelUrl) });
        scene.add(splat);
        let frame = 0;
        const render = () => {
          frame = requestAnimationFrame(render);
          splat.rotation.y += 0.002;
          renderer.render(scene, camera);
        };
        render();
        cleanup = () => {
          cancelAnimationFrame(frame);
          renderer.dispose();
          host.innerHTML = "";
        };
        setState("ready");
        setMessage("Spark Viewer 已加载真实 SPZ。");
      } catch (error) {
        setState("error");
        setMessage(error instanceof Error ? error.message : "Spark Viewer 加载失败");
      }
    }
    void mountViewer();
    return () => {
      cancelled = true;
      cleanup?.();
    };
  }, [modelUrl]);

  return (
    <section className="viewer-shell">
      <div ref={hostRef} className="viewer-canvas" />
      <div className={`viewer-overlay ${state}`}>
        <span>{message}</span>
        <button className="icon-button" type="button" onClick={() => hostRef.current?.requestFullscreen?.()} aria-label="全屏">
          <Maximize2 size={17} />
        </button>
      </div>
    </section>
  );
}
