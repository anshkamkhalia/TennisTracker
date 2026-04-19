import { useNavigate } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";
import { Video, ArrowRight, MoreHorizontal } from "lucide-react";
import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";

type Session = {
  id: string;
  video_url: string;
  created_at: string;
  title: string | null;
};

export default function Home() {
  const { user } = useUser();
  const navigate = useNavigate();
  const firstName = user?.firstName || user?.emailAddresses?.[0]?.emailAddress?.split("@")[0] || "Player";
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  const [sessions, setSessions] = useState<Session[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);

  useEffect(() => {
    if (!user) return;
    async function fetchSessions() {
      const { data, error } = await supabase
        .from("sessions")
        .select("id, video_url, created_at, title")
        .eq("user_id", user!.id)
        .order("created_at", { ascending: false })
        .limit(5);
      if (!error && data) setSessions(data);
      setLoadingSessions(false);
    }
    fetchSessions();
  }, [user]);

  return (
    <div className="px-10 py-8 max-w-3xl mx-auto">
      <div className="flex justify-between items-start mb-8">
        <div>
          <p className="text-gray-500 text-sm mb-1">{greeting} 👋</p>
          <h1 className="text-3xl font-extrabold text-gray-900">{firstName}'s Dashboard</h1>
        </div>
        <button
          onClick={() => navigate("/record")}
          className="flex items-center gap-2 bg-[#1A3263] text-white rounded-xl px-5 py-3 text-sm font-semibold hover:bg-[#15295a] transition cursor-pointer"
        >
          <Video size={16} /> New Recording
        </button>
      </div>

      <div className="flex flex-col gap-6">
        <div>
          <h2 className="text-lg font-bold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-4">
            <ActionCard bg="linear-gradient(135deg, #E8E3FF, #d5ccff)" icon="🎥" title="Analyze Video" subtitle="Upload & get AI feedback" onClick={() => navigate("/record")} />
            <ActionCard bg="linear-gradient(135deg, #E0F2FE, #bae6fd)" icon="📂" title="Session History" subtitle={`${sessions.length} sessions recorded`} onClick={() => navigate("/history")} />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
          <div className="px-6 py-5 border-b border-gray-100 flex justify-between items-center">
            <h2 className="text-base font-bold text-gray-900">Recent Sessions</h2>
            <button
              onClick={() => navigate("/history")}
              className="text-[#1A3263] text-sm font-semibold flex items-center gap-1 bg-transparent border-none cursor-pointer"
            >
              View all <ArrowRight size={14} />
            </button>
          </div>

          {loadingSessions ? (
            <div className="px-6 py-8 text-center text-gray-400 text-sm">Loading sessions...</div>
          ) : sessions.length === 0 ? (
            <div className="px-6 py-8 text-center">
              <p className="text-gray-400 text-sm">No sessions yet.</p>
              <button
                onClick={() => navigate("/record")}
                className="mt-3 text-[#1A3263] text-sm font-semibold hover:underline bg-transparent border-none cursor-pointer"
              >
                Upload your first video →
              </button>
            </div>
          ) : (
            sessions.map((session, i) => (
              <div
                key={session.id}
                onClick={() => navigate("/result", { state: { id: session.id } })}
                className="flex items-center gap-4 px-6 py-3.5 border-b last:border-b-0 border-gray-50 hover:bg-gray-50 cursor-pointer transition"
              >
                <div className="w-11 h-11 rounded-xl bg-gray-100 flex items-center justify-center text-xl shrink-0">
                  🎾
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-gray-900 text-sm">
                    {session.title || `Session ${sessions.length - i}`}
                  </p>
                  <p className="text-gray-400 text-xs mt-0.5">
                    {new Date(session.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                  </p>
                </div>
                <MoreHorizontal size={16} className="text-gray-300" />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function ActionCard({ bg, icon, title, subtitle, onClick }: {
  bg: string; icon: string; title: string; subtitle: string; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex flex-col items-start gap-3 p-5 rounded-2xl border-none cursor-pointer text-left hover:-translate-y-0.5 hover:shadow-lg transition-all"
      style={{ background: bg }}
    >
      <span className="text-4xl">{icon}</span>
      <div>
        <p className="font-bold text-[#1A3263] text-sm">{title}</p>
        <p className="text-xs mt-0.5" style={{ color: "rgba(26,50,99,0.6)" }}>{subtitle}</p>
      </div>
    </button>
  );
}