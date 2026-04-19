import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";
import { ArrowLeft, Play, Trash2, Pencil, Check, X } from "lucide-react";
import { supabase } from "../lib/supabase";

type Session = {
  id: string;
  video_url: string;
  created_at: string;
  title: string | null;
  description: string | null;
};

export default function History() {
  const { user } = useUser();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDesc, setEditDesc] = useState("");

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

  function startEdit(session: Session) {
    setEditing(session.id);
    setEditTitle(session.title || "");
    setEditDesc(session.description || "");
  }

  async function saveEdit(id: string) {
    const { error } = await supabase
      .from("sessions")
      .update({ title: editTitle || null, description: editDesc || null })
      .eq("id", id);
    if (!error) {
      setSessions(prev =>
        prev.map(s => s.id === id ? { ...s, title: editTitle || null, description: editDesc || null } : s)
      );
    }
    setEditing(null);
  }

  return (
    <div className="px-10 py-8 max-w-4xl mx-auto">
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
              className="flex items-start gap-4 px-6 py-4 border-b last:border-b-0 border-gray-50 hover:bg-gray-50 transition"
            >
              <div className="w-20 h-14 rounded-xl bg-gray-100 flex items-center justify-center shrink-0 overflow-hidden">
                <video src={session.video_url} className="w-full h-full object-cover" />
              </div>

              <div className="flex-1 min-w-0">
                {editing === session.id ? (
                  <div className="flex flex-col gap-2">
                    <input
                      value={editTitle}
                      onChange={e => setEditTitle(e.target.value)}
                      placeholder="Session title"
                      className="text-sm font-semibold border border-gray-200 rounded-lg px-3 py-1.5 w-full focus:outline-none focus:border-[#1A3263]"
                    />
                    <textarea
                      value={editDesc}
                      onChange={e => setEditDesc(e.target.value)}
                      placeholder="Add a description (optional)"
                      rows={2}
                      className="text-xs border border-gray-200 rounded-lg px-3 py-1.5 w-full resize-none focus:outline-none focus:border-[#1A3263]"
                    />
                  </div>
                ) : (
                  <>
                    <p className="font-semibold text-gray-900 text-sm">
                      {session.title || `Session ${sessions.length - i}`}
                    </p>
                    <p className="text-gray-400 text-xs mt-0.5">
                      {new Date(session.created_at).toLocaleDateString("en-US", {
                        weekday: "short", month: "short", day: "numeric", year: "numeric",
                      })}
                    </p>
                    {session.description && (
                      <p className="text-gray-500 text-xs mt-1 truncate">{session.description}</p>
                    )}
                  </>
                )}
              </div>

              <div className="flex items-center gap-2 shrink-0">
                {editing === session.id ? (
                  <>
                    <button
                      onClick={() => saveEdit(session.id)}
                      className="flex items-center justify-center w-8 h-8 rounded-lg bg-green-50 hover:bg-green-100 transition cursor-pointer border-none"
                    >
                      <Check size={14} className="text-green-600" />
                    </button>
                    <button
                      onClick={() => setEditing(null)}
                      className="flex items-center justify-center w-8 h-8 rounded-lg bg-gray-100 hover:bg-gray-200 transition cursor-pointer border-none"
                    >
                      <X size={14} className="text-gray-500" />
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => startEdit(session)}
                      className="flex items-center justify-center w-8 h-8 rounded-lg bg-gray-50 hover:bg-gray-100 transition cursor-pointer border-none"
                    >
                      <Pencil size={13} className="text-gray-400" />
                    </button>
                    <button
                      onClick={() => navigate("/result", { state: { id: session.id } })}
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
                  </>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}