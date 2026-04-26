"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, Boxes, Camera, FolderKanban, Home, Info, LogIn, LogOut, MessageSquare, UploadCloud } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, clearToken, getToken, isPublicPath } from "@/lib/api";
import type { User } from "@/lib/types";

const nav = [
  { href: "/", label: "首页", icon: Home, adminOnly: false },
  { href: "/upload", label: "上传", icon: UploadCloud, adminOnly: false },
  { href: "/projects", label: "项目", icon: FolderKanban, adminOnly: false },
  { href: "/camera", label: "实时视频", icon: Camera, adminOnly: false },
  { href: "/admin", label: "管理", icon: Activity, adminOnly: true },
  { href: "/feedback", label: "反馈", icon: MessageSquare, adminOnly: false },
  { href: "/about", label: "关于", icon: Info, adminOnly: false }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (!getToken()) {
      setUser(null);
      if (!isPublicPath(pathname)) router.replace("/login");
      return;
    }
    void api.me()
      .then(setUser)
      .catch(() => {
        setUser(null);
        if (!isPublicPath(pathname)) router.replace("/login");
      });
  }, [pathname, router]);

  const visibleNav = useMemo(() => nav.filter((item) => !item.adminOnly || user?.role === "admin"), [user?.role]);

  function logout() {
    clearToken();
    setUser(null);
    router.replace("/login");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" href="/">
          <span className="brand-mark"><Boxes size={20} /></span>
          <span>3DGS Platform</span>
        </Link>
        <nav className="nav-list">
          {visibleNav.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link className={`nav-link ${active ? "active" : ""}`} href={item.href} key={item.href}>
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          {user ? (
            <>
              <div className="user-card">
                <strong>{user.username}</strong>
                <span>{user.role}</span>
              </div>
              <button className="ghost-button full" type="button" onClick={logout}><LogOut size={16} />退出</button>
            </>
          ) : (
            <Link className="ghost-button full" href="/login"><LogIn size={16} />登录</Link>
          )}
        </div>
      </aside>
      <main className="main-panel">{children}</main>
    </div>
  );
}
