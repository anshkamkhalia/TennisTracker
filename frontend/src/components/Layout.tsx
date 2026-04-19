import { useEffect, useState } from "react";
import { Outlet, NavLink, useNavigate, useLocation } from "react-router-dom";
import { useUser, useClerk } from "@clerk/clerk-react";
import { Video, User, LayoutDashboard, LogOut, History, ChevronDown, ChevronRight } from "lucide-react";
import { supabase } from "../lib/supabase";

export default function Layout() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const navigate = useNavigate();
  const location = useLocation();

  const [sessionsOpen, setSessionsOpen] = useState(true);
  const [sessions, setSessions] = useState<any[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);

  const currentSessionId = (location.state as any)?.id;

  const initials = user?.firstName && user?.lastName
    ? `${user.firstName[0]}${user.lastName[0]}`
    : user?.emailAddresses?.[0]?.emailAddress?.[0]?.toUpperCase() ?? "?";

  // Load sessions for the current user
  useEffect(() => {
    if (!user?.id) return;
    supabase
      .from("sessions")
      .select("id, title, created_at, video_type")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .then(({ data, error }) => {
        if (error) console.error(error);
        else setSessions(data || []);
        setLoadingSessions(false);
      });
  }, [user?.id, location.pathname]); // refresh when route changes (e.g. after a new upload)

  async function handleLogout() {
    await signOut();
    navigate("/");
  }

  function openSession(id: string) {
    navigate("/result", { state: { id } });
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Sidebar */}
      <aside style={{
        width: 240, background: "#0f1f4a", display: "flex", flexDirection: "column",
        position: "fixed", top: 0, left: 0, bottom: 0, zIndex: 50,
      }}>
        {/* Logo */}
        <div style={{ padding: "28px 24px 20px", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 26 }}>🎾</span>
            <span style={{ fontSize: 17, fontWeight: 800, color: "#fff" }}>TennisTracker</span>
          </div>
        </div>

        {/* Nav — scrollable because sessions list can grow */}
        <nav style={{ padding: "16px 12px", flex: 1, overflowY: "auto" }}>
          <p style={{ fontSize: 11, fontWeight: 700, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: 1.5, padding: "4px 12px", marginBottom: 6 }}>
            Main
          </p>
          <SideNavItem to="/home" icon={<LayoutDashboard size={18} />} label="Dashboard" />
          <SideNavItem to="/record" icon={<Video size={18} />} label="Record / Upload" />

          {/* Sessions expandable section */}
          <button
            onClick={() => setSessionsOpen(o => !o)}
            style={{
              display: "flex", alignItems: "center", gap: 10, width: "100%",
              padding: "10px 12px", borderRadius: 10, marginBottom: 2,
              color: "rgba(255,255,255,0.55)", background: "transparent",
              fontSize: 14, fontWeight: 400, textAlign: "left",
              border: "none", borderLeft: "3px solid transparent",
              cursor: "pointer", transition: "all 0.15s",
            }}
          >
            <History size={18} />
            <span style={{ flex: 1 }}>Sessions</span>
            {sessionsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>

          {sessionsOpen && (
            <div style={{ marginLeft: 8, marginTop: 4, marginBottom: 8, paddingLeft: 10, borderLeft: "1px solid rgba(255,255,255,0.1)" }}>
              {loadingSessions ? (
                <p style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", padding: "6px 10px" }}>Loading...</p>
              ) : sessions.length === 0 ? (
                <p style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", padding: "6px 10px", fontStyle: "italic" }}>
                  No sessions yet
                </p>
              ) : (
                sessions.map(s => {
                  const isActive = currentSessionId === s.id;
                  return (
                    <button
                      key={s.id}
                      onClick={() => openSession(s.id)}
                      title={s.title || "Untitled session"}
                      style={{
                        display: "block", width: "100%", textAlign: "left",
                        padding: "7px 10px", marginBottom: 2, borderRadius: 8,
                        background: isActive ? "rgba(96,165,250,0.15)" : "transparent",
                        color: isActive ? "#fff" : "rgba(255,255,255,0.55)",
                        border: "none", cursor: "pointer", fontSize: 12.5,
                        fontWeight: isActive ? 600 : 400,
                        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                        transition: "all 0.15s",
                      }}
                      onMouseEnter={e => {
                        if (!isActive) e.currentTarget.style.background = "rgba(255,255,255,0.06)";
                      }}
                      onMouseLeave={e => {
                        if (!isActive) e.currentTarget.style.background = "transparent";
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{
                          width: 5, height: 5, borderRadius: "50%",
                          background: isActive ? "#60a5fa" : "rgba(255,255,255,0.3)",
                          flexShrink: 0,
                        }} />
                        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                          {s.title || new Date(s.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                        </span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          )}

          <SideNavItem to="/profile" icon={<User size={18} />} label="Profile" />
        </nav>

        {/* User area */}
        <div style={{ padding: "16px", borderTop: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            {user?.imageUrl ? (
              <img src={user.imageUrl} alt="avatar" style={{ width: 36, height: 36, borderRadius: "50%", objectFit: "cover" }} />
            ) : (
              <div style={{
                width: 36, height: 36, borderRadius: "50%", background: "#1A3263",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "#fff", fontWeight: 700, fontSize: 13, border: "2px solid rgba(255,255,255,0.2)",
              }}>
                {initials}
              </div>
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ color: "#fff", fontWeight: 600, fontSize: 13, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {user?.fullName || user?.emailAddresses?.[0]?.emailAddress}
              </p>
              <p style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>Free plan</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            style={{
              display: "flex", alignItems: "center", gap: 8, width: "100%",
              background: "rgba(255,255,255,0.06)", border: "none", borderRadius: 8,
              padding: "8px 12px", color: "rgba(255,255,255,0.6)", fontSize: 13,
              cursor: "pointer", transition: "all 0.2s",
            }}
          >
            <LogOut size={15} /> Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main style={{
        marginLeft: 240, flex: 1, minHeight: "100vh",
        background: "#f4f6fa", overflowY: "auto",
      }}>
        <Outlet />
      </main>
    </div>
  );
}

function SideNavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      style={({ isActive }) => ({
        display: "flex", alignItems: "center", gap: 10,
        padding: "10px 12px", borderRadius: 10, marginBottom: 2,
        color: isActive ? "#fff" : "rgba(255,255,255,0.55)",
        background: isActive ? "rgba(255,255,255,0.12)" : "transparent",
        fontSize: 14, fontWeight: isActive ? 600 : 400,
        transition: "all 0.15s", textDecoration: "none",
        borderLeft: isActive ? "3px solid #60a5fa" : "3px solid transparent",
      })}
    >
      {icon}
      {label}
    </NavLink>
  );
}