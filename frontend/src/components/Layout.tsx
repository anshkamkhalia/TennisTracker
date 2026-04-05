import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useUser, useClerk } from "@clerk/clerk-react";
import { Home, Video, User, LayoutDashboard, LogOut, Settings } from "lucide-react";

export default function Layout() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const navigate = useNavigate();

  const initials = user?.firstName && user?.lastName
    ? `${user.firstName[0]}${user.lastName[0]}`
    : user?.emailAddresses?.[0]?.emailAddress?.[0]?.toUpperCase() ?? "?";

  async function handleLogout() {
    await signOut();
    navigate("/");
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

        {/* Nav */}
        <nav style={{ padding: "16px 12px", flex: 1 }}>
          <p style={{ fontSize: 11, fontWeight: 700, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: 1.5, padding: "4px 12px", marginBottom: 6 }}>
            Main
          </p>
          <SideNavItem to="/home" icon={<LayoutDashboard size={18} />} label="Dashboard" />
          <SideNavItem to="/record" icon={<Video size={18} />} label="Record / Upload" />
          <SideNavItem to="/profile" icon={<User size={18} />} label="Profile" />
        </nav>

        {/* User area */}
        <div style={{
          padding: "16px", borderTop: "1px solid rgba(255,255,255,0.08)",
        }}>
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
