import { useSignIn } from "@clerk/clerk-react";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function Login() {
  const { signIn, isLoaded, setActive } = useSignIn();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    if (!isLoaded) return;
    setLoading(true);
    setError("");
    try {
      const result = await signIn.create({ identifier: email, password });
      if (result.status === "complete") {
        await setActive({ session: result.createdSessionId });
        navigate("/home");
      } else {
        setError(`Unexpected status: ${result.status}`);
      }
    } catch (err: any) {
      setError(err.errors?.[0]?.message || "Login failed. Check your credentials.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ minHeight: "100vh", display: "grid", gridTemplateColumns: "1fr 1fr" }}>
      <div style={{
        background: "linear-gradient(135deg, #0f1f4a 0%, #1A3263 100%)",
        display: "flex", flexDirection: "column",
        justifyContent: "center", alignItems: "center",
        padding: 48, textAlign: "center",
      }}>
        <span style={{ fontSize: 64, marginBottom: 24 }}>🎾</span>
        <h2 style={{ fontSize: 36, fontWeight: 800, color: "#fff", marginBottom: 16, lineHeight: 1.2 }}>
          Welcome back,<br />champion.
        </h2>
        <p style={{ color: "rgba(255,255,255,0.6)", fontSize: 16, maxWidth: 320, lineHeight: 1.7 }}>
          Your stats, sessions, and AI insights are waiting for you.
        </p>
        <div style={{ marginTop: 48, display: "flex", flexDirection: "column", gap: 14, width: "100%", maxWidth: 280 }}>
          {["Video analysis on every session", "Real-time AI feedback", "Track progress over time"].map(f => (
            <div key={f} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#60a5fa", flexShrink: 0 }} />
              <span style={{ color: "rgba(255,255,255,0.7)", fontSize: 14, textAlign: "left" }}>{f}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{
        display: "flex", flexDirection: "column", justifyContent: "center",
        alignItems: "center", padding: 48, background: "#fff",
      }}>
        <div style={{ width: "100%", maxWidth: 400 }}>
          <Link to="/" style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 40, color: "#6b7280", fontSize: 14 }}>
            ← Back to home
          </Link>
          <h1 style={{ fontSize: 30, fontWeight: 800, color: "#111827", marginBottom: 8 }}>Sign in</h1>
          <p style={{ color: "#6b7280", marginBottom: 32 }}>
            Don't have an account?{" "}
            <Link to="/signup" style={{ color: "#1A3263", fontWeight: 600 }}>Create one free</Link>
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <label style={labelStyle}>Email address</label>
              <input type="email" placeholder="you@example.com" value={email}
                onChange={e => setEmail(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>Password</label>
              <input type="password" placeholder="••••••••" value={password}
                onChange={e => setPassword(e.target.value)} style={inputStyle} />
            </div>
            {error && <div style={errorStyle}>{error}</div>}
            <button type="button" onClick={handleSubmit} disabled={loading} style={{
              ...btnStyle, opacity: loading ? 0.7 : 1, cursor: loading ? "not-allowed" : "pointer",
            }}>
              {loading ? "Signing in..." : "Sign in →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block", fontSize: 13, fontWeight: 600, color: "#374151", marginBottom: 6,
};
const inputStyle: React.CSSProperties = {
  width: "100%", padding: "12px 14px", border: "1px solid #e5e7eb",
  borderRadius: 10, fontSize: 15, outline: "none", background: "#f9fafb", color: "#111827",
};
const btnStyle: React.CSSProperties = {
  background: "#1A3263", color: "#fff", border: "none",
  borderRadius: 10, padding: "14px 0", fontSize: 16, fontWeight: 700, marginTop: 8,
};
const errorStyle: React.CSSProperties = {
  background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8,
  padding: "10px 14px", color: "#dc2626", fontSize: 14,
};