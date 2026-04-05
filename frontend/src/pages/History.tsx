import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";
import { ArrowLeft, Play, Trash2 } from "lucide-react";
import { supabase } from "../lib/supabase";

type Session = {
  id: string;
  video_url: string;
  created_at: string;
};

export default function History() {
  const { user } = useUser();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;
    async function fetchSessions() {
      const { data, error } = await supabase
        .from("sessions")
        .select("*")
        .eq("user_id", user!.id)
        .order("created_at", { ascending: false });
      if (!error && data) setSessions(data);
      setLoading(false);
    }
    fetchSessions();
  }, [user]);

  async function deleteSession(id: string) {
    await supabase.from("sessions").delete().eq("id", id);
    setSessions(prev => prev.filter(s => s.id !== id));
  }

  return (
    <div className="px-10 py-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <button
          onClick={() => navigate("/home")}
          className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 transition cursor-pointer"
        >
          <ArrowLeft size={15} /> Back
        </button>
        <div>
          <h1 className="text-2xl font-extrabold text-gray-900">Session History</h1>
          <p className="text-sm text-gray-500 mt-0.5">{sessions.length} sessions recorded</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
        {loading ? (
          <div className="px-6 py-12 text-center text-gray-400 text-sm">Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <p className="text-4xl mb-3">🎾</p>
            <p className="text-gray-500 font-medium mb-1">No sessions yet</p>
            <p className="text-gray-400 text-sm mb-4">Upload a video to get started</p>
            <button
              onClick={() => navigate("/record")}
              className="bg-[#1A3263] text-white px-5 py-2.5 rounded-xl text-sm font-semibold hover:bg-[#15295a] transition cursor-pointer border-none"
            >
              Upload Video
            </button>
          </div>
        ) : (
          sessions.map((session, i) => (
            <div
              key={session.id}
              className="flex items-center gap-4 px-6 py-4 border-b last:border-b-0 border-gray-50 hover:bg-gray-50 transition"
            >
              {/* Thumbnail placeholder */}
              <div className="w-20 h-14 rounded-xl bg-gray-100 flex items-center justify-center shrink-0 overflow-hidden">
                <video src={session.video_url} className="w-full h-full object-cover" />
              </div>

              <div className="flex-1">
                <p className="font-semibold text-gray-900 text-sm">Session {sessions.length - i}</p>
                <p className="text-gray-400 text-xs mt-0.5">
                  {new Date(session.created_at).toLocaleDateString("en-US", {
                    weekday: "short", month: "short", day: "numeric", year: "numeric",
                  })}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => navigate("/result", { state: { url: session.video_url } })}
                  className="flex items-center gap-1.5 bg-[#1A3263] text-white px-3.5 py-2 rounded-lg text-xs font-semibold hover:bg-[#15295a] transition cursor-pointer border-none"
                >
                  <Play size={12} fill="white" /> Watch
                </button>
                <button
                  onClick={() => deleteSession(session.id)}
                  className="flex items-center justify-center w-8 h-8 rounded-lg bg-red-50 hover:bg-red-100 transition cursor-pointer border-none"
                >
                  <Trash2 size={14} className="text-red-500" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}