import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FileVideo, Upload } from "lucide-react";
import { useUser } from "@clerk/clerk-react";
import { supabase } from "../lib/supabase";

const API_URL = import.meta.env.VITE_API_URL || "";

async function processVideo(file: File): Promise<{ url: string }> {
  const formData = new FormData();
  formData.append("video", file);
  const res = await fetch(`${API_URL}/process-video`, { method: "POST", body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "Upload failed");
  }
  return res.json();
}

export default function Record() {
  const navigate = useNavigate();
  const { user } = useUser();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);

  async function sendToServer(file: File) {
  setLoading(true);
  setError("");

  const now = new Date();
  try {
    const res = await processVideo(file);

    console.log(res);

    let sessionId: string | null = null;

    if (res.video_type === "court") {
      const { data, error } = await supabase.from("sessions").insert({
        user_id: user?.id,
        title: "Session from " + now.toLocaleDateString('en-US'),
        video_url: res.url,
        video_type: res.video_type,
        right_wrist_v: res.right_wrist_v,
        left_wrist_v: res.left_wrist_v,
        right_wrist_avg: res.right_wrist_avg,
        left_wrist_avg: res.left_wrist_avg,
        total_shots: res.total_shots,
        n_shots_by_poi: res.n_shots_by_POI,
        forehand_percent: res.forehand_percent,
        backhand_percent: res.backhand_percent,
        slice_volley_percent: res.slice_volley_percent,
        serve_overhead_percent: res.serve_overhead_percent,
      }).select("id").single();
      if (error) throw error;
      sessionId = data?.id;
    } else {
      const { data, error } = await supabase.from("sessions").insert({
        user_id: user?.id,
        video_url: res.url,
        video_type: res.video_type,
        heatmap: res.heatmap,
        ball_speeds: res.ball_speeds,
      }).select("id").single();
      if (error) throw error;
      sessionId = data?.id;
    }

    navigate("/result", { state: { id: sessionId } });
  } catch (err: any) {
    setError(err.message || "Upload failed");
  } finally {
    setLoading(false);
  }
}


  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    await sendToServer(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("video/")) sendToServer(file);
  }

  return (
    <div className="min-h-screen bg-gray-50 px-6 py-10 max-w-3xl mx-auto">
      <div className="flex items-center gap-4 mb-10">
        <button
          onClick={() => navigate("/home")}
          className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 transition"
        >
          <ArrowLeft size={15} /> Back
        </button>
        <div>
          <h1 className="text-2xl font-extrabold text-gray-900">Analyze Video</h1>
          <p className="text-sm text-gray-500 mt-0.5">Upload a tennis video for AI analysis</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-gray-200 p-8 mb-6 shadow-sm">
        <h3 className="text-base font-bold text-gray-900 mb-1">Upload Video</h3>
        <p className="text-sm text-gray-500 mb-6 leading-relaxed">
          Already have a recording? Upload it directly for instant AI analysis.
        </p>

        <div
          onClick={() => !loading && fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`
            flex flex-col items-center justify-center gap-4 w-full rounded-xl border-2 border-dashed
            py-14 cursor-pointer transition-all
            ${dragOver ? "border-blue-400 bg-blue-50" : "border-gray-200 bg-gray-50 hover:border-gray-300 hover:bg-gray-100"}
            ${loading ? "opacity-60 cursor-not-allowed pointer-events-none" : ""}
          `}
        >
          {loading ? (
            <>
              <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
              <div className="text-center">
                <p className="text-sm font-semibold text-gray-700">Processing your video...</p>
                <p className="text-xs text-gray-400 mt-1">This may take a while</p>
              </div>
            </>
          ) : (
            <>
              <div className="w-14 h-14 rounded-full bg-[#1A3263]/10 flex items-center justify-center">
                <FileVideo size={26} className="text-[#1A3263]" />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-gray-700">
                  Drop your video here, or <span className="text-[#1A3263] underline underline-offset-2">browse</span>
                </p>
                <p className="text-xs text-gray-400 mt-1">Supports MP4, MOV, WebM · Max 500MB</p>
              </div>
            </>
          )}
        </div>

        <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={handleFileChange} />

        {!loading && (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="mt-4 flex items-center justify-center gap-2 w-full py-3 bg-[#1A3263] text-white text-sm font-semibold rounded-xl hover:bg-[#15295a] transition"
          >
            <Upload size={16} /> Choose video file
          </button>
        )}

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}
      </div>

      <div className="rounded-2xl p-6" style={{ background: "linear-gradient(135deg, #1A3263, #2a4a8a)" }}>
        <p className="text-xs font-bold uppercase tracking-widest text-white/40 mb-4">Recording Tips</p>
        {[
          "Ensure good lighting — avoid harsh shadows",
          "Record full rallies, not just isolated shots",
          "At least 30 seconds gives better AI analysis",
          "Having less balls on the court is preferred—ball tracking may get confused",
          "60 FPS for best results",
        ].map((tip, i) => (
          <div key={i} className="flex gap-3 mb-3 last:mb-0">
            <span className="text-blue-400 font-bold text-sm shrink-0">{i + 1}.</span>
            <span className="text-white/75 text-sm leading-relaxed">{tip}</span>
          </div>
        ))}
      </div>
    </div>
  );
}