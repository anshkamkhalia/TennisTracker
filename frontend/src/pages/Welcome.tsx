import { useNavigate, Link } from "react-router-dom";
import { Video, BarChart2, Zap, Shield, ArrowRight, CheckCircle, Star } from "lucide-react";

export default function Welcome() {
  const navigate = useNavigate();

  return (
    <div style={{ background: "#fff", minHeight: "100vh" }}>
      {/* Navbar */}
      <nav style={{
        position: "sticky", top: 0, zIndex: 100,
        background: "rgba(255,255,255,0.95)", backdropFilter: "blur(10px)",
        borderBottom: "1px solid #e5e7eb",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 48px", height: 68,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 28 }}>🎾</span>
          <span style={{ fontSize: 20, fontWeight: 700, color: "#1A3263" }}>TennisTracker</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Link to="/login" style={{
            padding: "8px 20px", border: "1px solid #e5e7eb", borderRadius: 8,
            color: "#374151", fontWeight: 500, fontSize: 14,
          }}>
            Login
          </Link>
          <button
            onClick={() => navigate("/signup")}
            style={{
              padding: "8px 20px", background: "#1A3263", border: "none",
              borderRadius: 8, color: "#fff", fontWeight: 600, fontSize: 14,
            }}
          >
            Get Started Free
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section style={{
        background: "linear-gradient(135deg, #0f1f4a 0%, #1A3263 60%, #2a4a8a 100%)",
        padding: "100px 48px 120px",
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", inset: 0, opacity: 0.05,
          backgroundImage: "radial-gradient(circle at 20% 50%, white 1px, transparent 1px), radial-gradient(circle at 80% 50%, white 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }} />

        <div style={{ position: "relative", maxWidth: 800, margin: "0 auto" }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            background: "rgba(255,255,255,0.1)", borderRadius: 999,
            padding: "6px 16px", marginBottom: 28, border: "1px solid rgba(255,255,255,0.2)",
          }}>
            <Zap size={14} color="#fbbf24" fill="#fbbf24" />
            <span style={{ color: "#fbbf24", fontSize: 13, fontWeight: 600 }}>
              AI-Powered Tennis Analysis
            </span>
          </div>

          <h1 style={{
            fontSize: "clamp(40px, 6vw, 72px)", fontWeight: 800, color: "#fff",
            lineHeight: 1.1, marginBottom: 24, letterSpacing: "-1px",
          }}>
            Elevate Your Game<br />
            <span style={{ color: "#60a5fa" }}>With AI Analysis</span>
          </h1>

          <p style={{
            fontSize: 20, color: "rgba(255,255,255,0.75)", maxWidth: 560,
            margin: "0 auto 40px", lineHeight: 1.6,
          }}>
            Record your sessions, get instant AI feedback on your technique,
            track your progress and outperform your competition.
          </p>

          <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
            <button
              onClick={() => navigate("/signup")}
              style={{
                padding: "16px 36px", background: "#fff", border: "none",
                borderRadius: 12, color: "#1A3263", fontWeight: 700, fontSize: 16,
                display: "flex", alignItems: "center", gap: 8,
              }}
            >
              Start for Free <ArrowRight size={18} />
            </button>
            <button
              onClick={() => navigate("/login")}
              style={{
                padding: "16px 36px", background: "transparent",
                border: "2px solid rgba(255,255,255,0.4)",
                borderRadius: 12, color: "#fff", fontWeight: 600, fontSize: 16,
              }}
            >
              Sign In
            </button>
          </div>

          <p style={{ marginTop: 20, color: "rgba(255,255,255,0.45)", fontSize: 13 }}>
            No credit card required · Free forever plan
          </p>
        </div>
      </section>

      {/* Stats bar */}
      <section style={{
        background: "#1A3263", padding: "32px 48px",
      }}>
        <div style={{
          maxWidth: 1100, margin: "0 auto",
          display: "grid", gridTemplateColumns: "repeat(3, 1fr)",
          gap: 24, textAlign: "center",
        }}>
          {[
            { value: "50K+", label: "Shots Analyzed" },
            { value: "98%", label: "Accuracy Rate" },
            { value: "12K+", label: "Active Players" },
          ].map(s => (
            <div key={s.label}>
              <p style={{ fontSize: 36, fontWeight: 800, color: "#fff" }}>{s.value}</p>
              <p style={{ color: "rgba(255,255,255,0.6)", fontSize: 14, marginTop: 4 }}>{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section style={{ padding: "100px 48px", background: "#fff" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <p style={{ color: "#4f8ef7", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: 2, textAlign: "center", marginBottom: 12 }}>
            Features
          </p>
          <h2 style={{ fontSize: 42, fontWeight: 800, textAlign: "center", marginBottom: 16, color: "#111827" }}>
            Everything you need to improve
          </h2>
          <p style={{ color: "#6b7280", textAlign: "center", maxWidth: 540, margin: "0 auto 64px", fontSize: 18 }}>
            From recording to analysis to tracking — TennisTracker has your game covered.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 32 }}>
            {[
              {
                icon: <Video size={28} color="#1A3263" />,
                bg: "#e8f0ff",
                title: "Video Analysis",
                desc: "Record your match or practice session directly in the browser or upload an existing video. Our AI breaks down every shot.",
              },
              {
                icon: <BarChart2 size={28} color="#059669" />,
                bg: "#d1fae5",
                title: "Performance Stats",
                desc: "Track winners, accuracy, serve speed, and dozens of other metrics over time to see exactly where you're improving.",
              },
              {
                icon: <Zap size={28} color="#d97706" />,
                bg: "#fef3c7",
                title: "Instant Feedback",
                desc: "Get real-time coaching tips powered by AI after every session. Know what to fix before your next match.",
              },
              {
                icon: <Shield size={28} color="#7c3aed" />,
                bg: "#ede9fe",
                title: "Secure & Private",
                desc: "Your videos and data are encrypted and stored securely. Only you have access to your performance history.",
              },
            ].map(f => (
              <div key={f.title} style={{
                padding: 32, borderRadius: 20, border: "1px solid #f3f4f6",
                transition: "box-shadow 0.2s",
              }}>
                <div style={{
                  width: 60, height: 60, borderRadius: 16, background: f.bg,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  marginBottom: 20,
                }}>
                  {f.icon}
                </div>
                <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 10, color: "#111827" }}>{f.title}</h3>
                <p style={{ color: "#6b7280", lineHeight: 1.7, fontSize: 15 }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section style={{ padding: "100px 48px", background: "#f9fafb" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <p style={{ color: "#4f8ef7", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: 2, textAlign: "center", marginBottom: 12 }}>
            How It Works
          </p>
          <h2 style={{ fontSize: 42, fontWeight: 800, textAlign: "center", marginBottom: 64, color: "#111827" }}>
            Three simple steps
          </h2>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 40 }}>
            {[
              { step: "01", title: "Record or Upload", desc: "Use your webcam to record a session, or upload a video from your device." },
              { step: "02", title: "AI Analyzes", desc: "Our AI model processes your video and identifies technique patterns, shot types, and areas of improvement." },
              { step: "03", title: "Review & Improve", desc: "Get detailed feedback, track your stats over time, and watch your game improve session after session." },
            ].map(s => (
              <div key={s.step} style={{ textAlign: "center" }}>
                <div style={{
                  width: 64, height: 64, borderRadius: "50%", background: "#1A3263",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  margin: "0 auto 20px", fontSize: 18, fontWeight: 800, color: "#60a5fa",
                }}>
                  {s.step}
                </div>
                <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 10, color: "#111827" }}>{s.title}</h3>
                <p style={{ color: "#6b7280", lineHeight: 1.7, fontSize: 15 }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section style={{ padding: "100px 48px", background: "#fff" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <h2 style={{ fontSize: 42, fontWeight: 800, textAlign: "center", marginBottom: 64, color: "#111827" }}>
            Loved by players worldwide
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 28 }}>
            {[
              {
                name: "Sarah M.", role: "Club Player",
                text: "TennisTracker helped me identify that my backhand was consistently going wide. Within 3 weeks of targeted practice, I fixed it completely.",
                rating: 5,
              },
              {
                name: "James R.", role: "Tennis Coach",
                text: "I recommend TennisTracker to all my students. The AI analysis catches things even I sometimes miss, and the stats are incredibly detailed.",
                rating: 5,
              },
              {
                name: "Priya K.", role: "Tournament Player",
                text: "My serve speed went from 85 to 102mph in two months. The feedback on my toss position was game-changing.",
                rating: 5,
              },
            ].map(t => (
              <div key={t.name} style={{
                padding: 28, borderRadius: 20, border: "1px solid #e5e7eb",
                background: "#f9fafb",
              }}>
                <div style={{ display: "flex", marginBottom: 16 }}>
                  {Array.from({ length: t.rating }).map((_, i) => (
                    <Star key={i} size={16} color="#f59e0b" fill="#f59e0b" />
                  ))}
                </div>
                <p style={{ color: "#374151", lineHeight: 1.7, marginBottom: 20, fontSize: 15 }}>"{t.text}"</p>
                <div>
                  <p style={{ fontWeight: 700, color: "#111827" }}>{t.name}</p>
                  <p style={{ fontSize: 13, color: "#6b7280" }}>{t.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{
        padding: "100px 48px",
        background: "linear-gradient(135deg, #1A3263, #2a4a8a)",
        textAlign: "center",
      }}>
        <div style={{ maxWidth: 700, margin: "0 auto" }}>
          <h2 style={{ fontSize: 48, fontWeight: 800, color: "#fff", marginBottom: 16 }}>
            Ready to level up?
          </h2>
          <p style={{ color: "rgba(255,255,255,0.7)", fontSize: 18, marginBottom: 40 }}>
            Join thousands of players using AI to get better, faster.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap", marginBottom: 24 }}>
            {["No credit card", "Free to start", "Cancel anytime"].map(f => (
              <div key={f} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <CheckCircle size={16} color="#60a5fa" />
                <span style={{ color: "rgba(255,255,255,0.8)", fontSize: 14 }}>{f}</span>
              </div>
            ))}
          </div>
          <button
            onClick={() => navigate("/signup")}
            style={{
              padding: "18px 48px", background: "#fff", border: "none",
              borderRadius: 12, color: "#1A3263", fontWeight: 800, fontSize: 18,
              display: "inline-flex", alignItems: "center", gap: 8,
            }}
          >
            Get Started Free <ArrowRight size={20} />
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        padding: "40px 48px",
        background: "#0f1f4a",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexWrap: "wrap", gap: 16,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 22 }}>🎾</span>
          <span style={{ fontWeight: 700, color: "#fff", fontSize: 16 }}>TennisTracker</span>
        </div>
        <p style={{ color: "rgba(255,255,255,0.4)", fontSize: 13 }}>
          © 2025 TennisTracker. All rights reserved.
        </p>
        <div style={{ display: "flex", gap: 24 }}>
          {["Privacy", "Terms", "Contact"].map(l => (
            <a key={l} href="#" style={{ color: "rgba(255,255,255,0.5)", fontSize: 13, transition: "color 0.2s" }}>{l}</a>
          ))}
        </div>
      </footer>
    </div>
  );
}
