import { useSignUp } from "@clerk/clerk-react";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function Signup() {
  const { signUp, isLoaded, setActive } = useSignUp();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSignUp() {
    console.log("clicked, isLoaded:", isLoaded);
    if (!isLoaded) return;
    setLoading(true);
    setError("");
    try {
      const result = await signUp.create({ emailAddress: email, password });
      console.log("result:", result.status);
      if (result.status === "complete") {
        await setActive({ session: result.createdSessionId });
        navigate("/home");
      } else {
        console.log("not complete:", result.status);
        setError(`Unexpected status: ${result.status}`);
      }
    } catch (err: any) {
      console.log("full error:", JSON.stringify(err.errors, null, 2));
      setError(err.errors?.[0]?.message || "Sign up failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
      {/* Left */}
      <div
        className="flex flex-col justify-center items-center p-12 text-center"
        style={{ background: "linear-gradient(135deg, #0f1f4a 0%, #1A3263 100%)" }}
      >
        <span className="text-6xl mb-6">🎾</span>
        <h2 className="text-4xl font-extrabold text-white mb-4 leading-tight">
          Your AI coach<br />awaits.
        </h2>
        <p className="text-white/60 text-base max-w-xs leading-relaxed">
          Create your free account and start getting smarter feedback on your tennis game today.
        </p>
        <div className="mt-10 bg-white/10 rounded-2xl px-7 py-6 w-full max-w-xs">
          <p className="text-white/50 text-xs uppercase tracking-widest font-bold mb-3">
            Free plan includes
          </p>
          {["5 video analyses per month", "Full performance dashboard", "Session history & trends"].map(f => (
            <div key={f} className="flex items-center gap-3 mt-3">
              <span className="text-emerald-400 font-bold">✓</span>
              <span className="text-white/80 text-sm">{f}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right */}
      <div className="flex flex-col justify-center items-center p-12 bg-white">
        <div className="w-full max-w-sm">
          <Link to="/" className="flex items-center gap-2 text-gray-500 text-sm mb-10 no-underline hover:text-gray-700 transition">
            ← Back to home
          </Link>
          <h1 className="text-3xl font-extrabold text-gray-900 mb-2">Create account</h1>
          <p className="text-gray-500 mb-8">
            Already have one?{" "}
            <Link to="/login" className="text-[#1A3263] font-semibold no-underline hover:underline">
              Sign in
            </Link>
          </p>

          <div className="flex flex-col gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1.5">Email address</label>
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm bg-gray-50 text-gray-900 outline-none focus:border-[#1A3263] transition"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1.5">Password</label>
              <input
                type="password"
                placeholder="Min. 8 characters"
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm bg-gray-50 text-gray-900 outline-none focus:border-[#1A3263] transition"
              />
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-600">
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={handleSignUp}
              disabled={loading}
              className="w-full bg-[#1A3263] text-white rounded-xl py-3.5 text-base font-bold mt-1 hover:bg-[#15295a] transition cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? "Creating account..." : "Create account →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}