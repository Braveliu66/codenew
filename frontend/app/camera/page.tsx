import Link from "next/link";
import { Camera, UploadCloud, VideoOff } from "lucide-react";

export default function CameraPage() {
  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Realtime</p>
          <h1>实时视频重建</h1>
        </div>
        <div className="actions">
          <Link className="button" href="/upload"><UploadCloud size={18} />改用离线上传</Link>
        </div>
      </header>

      <section className="split-workspace">
        <div className="panel fill">
          <div className="panel-head">
            <h2>摄像头通道</h2>
            <span className="status-pill failed">未接入</span>
          </div>
          <div className="panel-body scrollable" style={{ padding: 0 }}>
            <div className="preview-stage">
              <div className="preview-placeholder">
                <span className="preview-icon"><VideoOff size={30} /></span>
                <h2>Camera preview 尚未实现</h2>
                <p className="muted">后端当前会返回 camera preview is not implemented，因此这里不显示模拟视频流或假点云。</p>
              </div>
            </div>
          </div>
          <div className="sticky-actions">
            <button className="ghost-button" disabled type="button"><Camera size={17} />连接设备</button>
            <Link className="button secondary" href="/upload"><UploadCloud size={17} />上传图片/视频</Link>
          </div>
        </div>

        <div className="panel padded stack">
          <h2>可用工作流</h2>
          <div className="notice-box">当前版本支持图片序列和视频上传后的真实极速预览。实时相机入口保留为未来接入点。</div>
          <div className="data-row"><span>实时输入</span><strong>不可用</strong></div>
          <div className="data-row"><span>离线图片</span><strong>可用</strong></div>
          <div className="data-row"><span>离线视频</span><strong>可用</strong></div>
        </div>
      </section>
    </div>
  );
}
