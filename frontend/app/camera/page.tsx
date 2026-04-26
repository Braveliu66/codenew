import { Camera } from "lucide-react";

export default function CameraPage() {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Realtime</p>
          <h1>实时视频重建</h1>
        </div>
      </header>
      <section className="grid two">
        <div className="panel stack">
          <div className="preview-tile"><Camera size={38} /></div>
          <button className="button" disabled type="button">Stream3R 未接入</button>
        </div>
        <div className="panel stack">
          <h2>实时粗重建</h2>
          <div className="empty-state">实时摄像头管线未接入，本批次只接图片/视频极速预览。</div>
        </div>
      </section>
    </div>
  );
}
