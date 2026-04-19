import { useEffect, useState, useRef, useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Video, Home, Check, X, Play, Pause, Volume2, VolumeX, Maximize, RotateCcw, Download } from "lucide-react";
import { supabase } from "../lib/supabase";
import Chart from "react-apexcharts";


export default function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const sessionId = (location.state as any)?.id;


  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editDesc, setEditDesc] = useState("");


  const [leftPct, setLeftPct] = useState(55);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);


  const onMouseDown = useCallback(() => { dragging.current = true; }, []);


  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setLeftPct(Math.min(75, Math.max(30, pct)));
  }, []);


  const onMouseUp = useCallback(() => { dragging.current = false; }, []);


  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);


  useEffect(() => {
    if (!sessionId) return;
    supabase
      .from("sessions")
      .select("*")
      .eq("id", sessionId)
      .single()
      .then(({ data, error }) => {
        if (error) console.error(error);
        else {
          setSession(data);
          setEditTitle(data.title || "");
          setEditDesc(data.description || "");
        }
        setLoading(false);
      });
  }, [sessionId]);


  async function saveEdit() {
    const { error } = await supabase
      .from("sessions")
      .update({ title: editTitle || null, description: editDesc || null })
      .eq("id", sessionId);
    if (!error) setSession((prev: any) => ({ ...prev, title: editTitle || null, description: editDesc || null }));
    setEditing(false);
  }


  if (!sessionId) {
    return (
      <div className="px-10 py-20 text-center">
        <p className="text-6xl mb-4">🎾</p>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">No session found</h2>
        <p className="text-gray-500 mb-6">Go back and upload a video first.</p>
        <button
          onClick={() => navigate("/record")}
          className="bg-[#1A3263] text-white border-none rounded-xl px-6 py-3 font-semibold cursor-pointer hover:bg-[#15295a] transition"
        >
          Go to Record
        </button>
      </div>
    );
  }


  if (loading) {
    return (
      <div className="px-10 py-20 text-center">
        <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
        <p className="text-gray-500 text-sm">Loading session...</p>
      </div>
    );
  }


  const tabs = ["Visualization", "Statistics", "Settings"];
  const isCourt = session.video_type === "court";


  // Shared chart styling so every graph matches the rest of the UI
  const brandColor = "#1A3263";
  const palette = ["#1A3263", "#3b82f6", "#60a5fa", "#93c5fd"];


  const baseChartOptions: any = {
    chart: {
      toolbar: { show: false },
      fontFamily: "inherit",
      animations: { enabled: true, speed: 400 },
    },
    grid: {
      borderColor: "#f3f4f6",
      strokeDashArray: 4,
      padding: { left: 10, right: 10, top: 0, bottom: 0 },
    },
    dataLabels: { enabled: false },
    tooltip: {
      theme: "light",
      style: { fontSize: "11px", fontFamily: "inherit" },
      y: { formatter: (v: number) => (v != null ? v.toFixed(2) : "") },
    },
    xaxis: {
      labels: { style: { fontSize: "10px", colors: "#9ca3af" } },
      axisTicks: { show: false },
      axisBorder: { show: false },
    },
    yaxis: {
      labels: {
        style: { fontSize: "10px", colors: "#9ca3af" },
        formatter: (v: number) => (v != null ? v.toFixed(2) : ""),
      },
    },
    legend: {
      fontSize: "11px",
      fontFamily: "inherit",
      labels: { colors: "#6b7280" },
      markers: { width: 8, height: 8, radius: 4 },
      itemMargin: { horizontal: 8 },
    },
  };


  const statsItems = [
    { label: "Total shots", value: session.total_shots },
    { label: "Shots by POI", value: session.n_shots_by_poi },
    { label: "Forehand", value: session.forehand_percent != null ? `${session.forehand_percent.toFixed(2)}%` : null },
    { label: "Backhand", value: session.backhand_percent != null ? `${session.backhand_percent.toFixed(2)}%` : null },
    { label: "Slice / volley", value: session.slice_volley_percent != null ? `${session.slice_volley_percent.toFixed(2)}%` : null },
    { label: "Serve / overhead", value: session.serve_overhead_percent != null ? `${session.serve_overhead_percent.toFixed(2)}%` : null },
    { label: "Right wrist avg", value: session.right_wrist_avg != null ? `${session.right_wrist_avg.toFixed(2)} mph` : null },
    { label: "Left wrist avg", value: session.left_wrist_avg != null ? `${session.left_wrist_avg.toFixed(2)} mph` : null },
  ].filter(item => item.value != null);


  return (
    <div className="px-7 py-8 mx-auto">
      <div className="flex items-start gap-4 mb-8">
        <button
          onClick={() => navigate("/record")}
          className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 transition cursor-pointer shrink-0"
        >
          <ArrowLeft size={15} /> Back to Record
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-extrabold text-gray-900">
            {session.title || "Analysis Results"}
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {session.description ||
              new Date(session.created_at).toLocaleDateString("en-US", {
                weekday: "short", month: "short", day: "numeric", year: "numeric",
              })}
          </p>
        </div>
      </div>


      {/* Draggable split layout */}
      <div ref={containerRef} className="flex px-10 gap-0 select-none" style={{ height: "calc(100vh - 180px)" }}>
        {/* Left panel */}
        <div style={{ width: `${leftPct}%` }} className="flex flex-col pr-4 overflow-hidden">
          <VideoPlayer src={session.video_url} />
          <div className="mt-3 flex justify-end">
            <a
              href={session.video_url}
              download={`${(session.title || "session").replace(/[^a-z0-9]+/gi, "_")}.mp4`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 hover:border-[#1A3263] hover:text-[#1A3263] transition cursor-pointer no-underline"
            >
              <Download size={15} /> Download video
            </a>
          </div>
        </div>


        {/* Divider */}
        <div
          onMouseDown={onMouseDown}
          className="w-2 flex items-center justify-center cursor-col-resize shrink-0 group"
        >
          <div className="w-0.5 h-full bg-gray-200 group-hover:bg-[#1A3263] transition-colors rounded-full" />
        </div>


        {/* Right panel */}
        <div style={{ width: `${100 - leftPct}%` }} className="flex flex-col pl-4 overflow-y-auto">
          <div className="flex gap-2 mb-6 border-b border-gray-200">
            {tabs.map((tab, i) => (
              <button
                key={i}
                onClick={() => setActiveTab(i)}
                className={`px-4 py-2.5 text-sm font-semibold border-none cursor-pointer transition rounded-none border-b-2 -mb-px ${
                  activeTab === i
                    ? "border-[#1A3263] text-[#1A3263] bg-transparent"
                    : "border-transparent text-gray-400 bg-transparent hover:text-gray-600"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>


          {/* ================= VISUALIZATION ================= */}
          {activeTab === 0 && (
            <div className="grid grid-cols-2 gap-3">
              {isCourt ? (
                <>
                  {/* Wrist velocity — full width smooth area chart */}
                  {(session.right_wrist_v?.length > 0 || session.left_wrist_v?.length > 0) && (
                    <div className="col-span-2 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                      <p className="text-xs text-gray-400 mb-2">Wrist velocity</p>
                      <Chart
                        type="area"
                        height={200}
                        series={[
                          { name: "Right", data: session.right_wrist_v || [] },
                          { name: "Left", data: session.left_wrist_v || [] },
                        ]}
                        options={{
                          ...baseChartOptions,
                          colors: [palette[0], palette[2]],
                          stroke: { curve: "smooth", width: 2 },
                          fill: {
                            type: "gradient",
                            gradient: {
                              shadeIntensity: 1,
                              opacityFrom: 0.25,
                              opacityTo: 0.02,
                              stops: [0, 100],
                            },
                          },
                          xaxis: { ...baseChartOptions.xaxis, labels: { show: false } },
                        }}
                      />
                    </div>
                  )}


                  {/* Shot distribution donut */}
                  <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                    <p className="text-xs text-gray-400 mb-2">Shot distribution</p>
                    <Chart
                      type="donut"
                      height={220}
                      series={[
                        session.forehand_percent || 0,
                        session.backhand_percent || 0,
                        session.slice_volley_percent || 0,
                        session.serve_overhead_percent || 0,
                      ]}
                      options={{
                        ...baseChartOptions,
                        colors: palette,
                        labels: ["Forehand", "Backhand", "Slice", "Serve"],
                        legend: { ...baseChartOptions.legend, position: "bottom" },
                        stroke: { width: 0 },
                        plotOptions: {
                          pie: {
                            donut: {
                              size: "65%",
                              labels: {
                                show: true,
                                name: { fontSize: "11px", color: "#9ca3af" },
                                value: {
                                  fontSize: "18px",
                                  fontWeight: 700,
                                  color: "#111827",
                                  formatter: (v: string) => `${Number(v).toFixed(2)}%`,
                                },
                                total: {
                                  show: true,
                                  label: "Total",
                                  fontSize: "11px",
                                  color: "#9ca3af",
                                  formatter: () => "100.00%",
                                },
                              },
                            },
                          },
                        },
                      }}
                    />
                  </div>


                  {/* Shot breakdown bar */}
                  <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                    <p className="text-xs text-gray-400 mb-2">Shot breakdown</p>
                    <Chart
                      type="bar"
                      height={220}
                      series={[{
                        name: "Percent",
                        data: [
                          session.forehand_percent || 0,
                          session.backhand_percent || 0,
                          session.slice_volley_percent || 0,
                          session.serve_overhead_percent || 0,
                        ],
                      }]}
                      options={{
                        ...baseChartOptions,
                        colors: [brandColor],
                        plotOptions: {
                          bar: { borderRadius: 6, columnWidth: "55%" },
                        },
                        xaxis: {
                          ...baseChartOptions.xaxis,
                          categories: ["FH", "BH", "Slice", "Serve"],
                        },
                        yaxis: {
                          ...baseChartOptions.yaxis,
                          labels: {
                            style: { fontSize: "10px", colors: "#9ca3af" },
                            formatter: (v: number) => `${v.toFixed(2)}%`,
                          },
                        },
                      }}
                    />
                  </div>


                  {/* Avg wrist speed — full width */}
                  {(session.left_wrist_avg != null || session.right_wrist_avg != null) && (
                    <div className="col-span-2 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                      <p className="text-xs text-gray-400 mb-2">Avg wrist speed (mph)</p>
                      <Chart
                        type="bar"
                        height={160}
                        series={[{
                          name: "mph",
                          data: [
                            session.left_wrist_avg || 0,
                            session.right_wrist_avg || 0,
                          ],
                        }]}
                        options={{
                          ...baseChartOptions,
                          colors: [palette[1]],
                          plotOptions: {
                            bar: { borderRadius: 6, columnWidth: "40%" },
                          },
                          xaxis: {
                            ...baseChartOptions.xaxis,
                            categories: ["Left wrist", "Right wrist"],
                          },
                        }}
                      />
                    </div>
                  )}
                </>
              ) : (
                <>
                  {/* Ball speed — top view */}
                  {session.ball_speeds?.length > 0 && (
                    <div className="col-span-2 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                      <p className="text-xs text-gray-400 mb-2">Ball speed per shot (mph)</p>
                      <Chart
                        type="area"
                        height={220}
                        series={[{ name: "mph", data: session.ball_speeds }]}
                        options={{
                          ...baseChartOptions,
                          colors: [brandColor],
                          stroke: { curve: "smooth", width: 2 },
                          fill: {
                            type: "gradient",
                            gradient: {
                              shadeIntensity: 1,
                              opacityFrom: 0.3,
                              opacityTo: 0.02,
                              stops: [0, 100],
                            },
                          },
                          markers: {
                            size: 4,
                            colors: ["#fff"],
                            strokeColors: brandColor,
                            strokeWidth: 2,
                            hover: { size: 6 },
                          },
                          xaxis: {
                            ...baseChartOptions.xaxis,
                            categories: session.ball_speeds.map((_: any, i: number) => `${i + 1}`),
                            title: { text: "Shot #", style: { fontSize: "10px", color: "#9ca3af", fontWeight: 500 } },
                          },
                        }}
                      />
                    </div>
                  )}


                  {session.heatmap && (
                    <div className="col-span-2 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                      <p className="text-xs text-gray-400 mb-2">Court heatmap</p>
                      <img
                        src={session.heatmap}
                        className="w-full h-[220px] object-contain rounded-lg"
                      />
                    </div>
                  )}


                  {!session.ball_speeds?.length && !session.heatmap && (
                    <div className="col-span-2 py-12 text-center text-gray-400 text-sm border border-dashed border-gray-200 rounded-2xl">
                      No visualization data available
                    </div>
                  )}
                </>
              )}
            </div>
          )}


          {/* ================= STATISTICS ================= */}
          {activeTab === 1 && (
            <div>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {statsItems.map(({ label, value }) => (
                  <div key={label} className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                    <p className="text-xs text-gray-400 mb-1">{label}</p>
                    <p className="text-lg font-bold text-gray-900">{value}</p>
                  </div>
                ))}
              </div>


              {session.ball_speeds?.length > 0 && (
                <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                  <p className="text-xs text-gray-400 mb-2">Ball speeds</p>
                  <div className="flex flex-wrap gap-2">
                    {session.ball_speeds.map((s: number, i: number) => (
                      <span key={i} className="bg-white border border-gray-200 rounded-lg px-3 py-1 text-sm font-semibold text-gray-700">
                        Shot {i + 1}: {s.toFixed(2)} mph
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}


          {/* ================= SETTINGS ================= */}
          {activeTab === 2 && (
            <div className="flex flex-col gap-4">
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Session title</label>
                <input
                  value={editTitle}
                  onChange={e => { setEditTitle(e.target.value); setEditing(true); }}
                  placeholder="Add a title..."
                  className="text-sm font-semibold border border-gray-200 rounded-lg px-3 py-2 w-full focus:outline-none focus:border-[#1A3263]"
                />
              </div>


              <div>
                <label className="text-xs text-gray-400 mb-1 block">Date</label>
                <div className="text-sm border border-gray-200 rounded-lg px-3 py-2 bg-gray-50 text-gray-500">
                  {new Date(session.created_at).toLocaleDateString("en-US", {
                    weekday: "short", month: "short", day: "numeric", year: "numeric",
                  })}
                </div>
              </div>


              <div>
                <label className="text-xs text-gray-400 mb-1 block">Description</label>
                <textarea
                  value={editDesc}
                  onChange={e => { setEditDesc(e.target.value); setEditing(true); }}
                  placeholder="Add a description (optional)..."
                  rows={3}
                  className="text-sm border border-gray-200 rounded-lg px-3 py-2 w-full resize-none focus:outline-none focus:border-[#1A3263] text-gray-500"
                />
              </div>


              {editing && (
                <div className="flex gap-2">
                  <button
                    onClick={saveEdit}
                    className="flex items-center gap-2 bg-[#1A3263] text-white border-none rounded-lg px-4 py-2 text-sm font-semibold cursor-pointer hover:bg-[#15295a] transition"
                  >
                    <Check size={14} /> Save
                  </button>
                  <button
                    onClick={() => { setEditing(false); setEditTitle(session.title || ""); setEditDesc(session.description || ""); }}
                    className="flex items-center gap-2 bg-gray-100 text-gray-600 border-none rounded-lg px-4 py-2 text-sm font-semibold cursor-pointer hover:bg-gray-200 transition"
                  >
                    <X size={14} /> Cancel
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


/* ============================================================
   Custom video player — matches the app's navy/gray aesthetic
   ============================================================ */
function VideoPlayer({ src }: { src: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showControls, setShowControls] = useState(true);
  const [scrubbing, setScrubbing] = useState(false);

  const hideTimer = useRef<number | null>(null);

  // Auto-hide controls while playing
  const kickHideTimer = useCallback(() => {
    setShowControls(true);
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    if (playing && !scrubbing) {
      hideTimer.current = window.setTimeout(() => setShowControls(false), 2200);
    }
  }, [playing, scrubbing]);

  useEffect(() => { kickHideTimer(); }, [kickHideTimer]);

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play(); else v.pause();
  }

  function toggleMute() {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  }

  function onVolumeChange(e: React.ChangeEvent<HTMLInputElement>) {
    const v = videoRef.current;
    if (!v) return;
    const val = parseFloat(e.target.value);
    v.volume = val;
    v.muted = val === 0;
    setVolume(val);
    setMuted(val === 0);
  }

  function seekFromEvent(e: React.MouseEvent | MouseEvent) {
    const v = videoRef.current;
    const bar = progressRef.current;
    if (!v || !bar || !duration) return;
    const rect = bar.getBoundingClientRect();
    const pct = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    v.currentTime = pct * duration;
    setCurrent(pct * duration);
  }

  function onScrubStart(e: React.MouseEvent) {
    setScrubbing(true);
    seekFromEvent(e);
  }

  useEffect(() => {
    if (!scrubbing) return;
    const move = (e: MouseEvent) => seekFromEvent(e);
    const up = () => setScrubbing(false);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [scrubbing]);

  function restart() {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = 0;
    v.play();
  }

  function toggleFullscreen() {
    const el = wrapRef.current;
    if (!el) return;
    if (document.fullscreenElement) document.exitFullscreen();
    else el.requestFullscreen();
  }

  // Keyboard shortcuts (space, arrows, M, F)
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!wrapRef.current?.contains(document.activeElement) && document.activeElement !== document.body) return;
      const v = videoRef.current;
      if (!v) return;
      if (e.key === " " || e.key === "k") { e.preventDefault(); togglePlay(); }
      else if (e.key === "ArrowRight") { v.currentTime = Math.min(duration, v.currentTime + 5); }
      else if (e.key === "ArrowLeft") { v.currentTime = Math.max(0, v.currentTime - 5); }
      else if (e.key === "m") { toggleMute(); }
      else if (e.key === "f") { toggleFullscreen(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [duration]);

  function fmt(t: number) {
    if (!isFinite(t)) return "0:00";
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  const pct = duration ? (current / duration) * 100 : 0;

  return (
    <div
      ref={wrapRef}
      onMouseMove={kickHideTimer}
      onMouseLeave={() => playing && !scrubbing && setShowControls(false)}
      className="relative bg-black rounded-2xl overflow-hidden aspect-video group"
    >
      <video
        ref={videoRef}
        src={src}
        autoPlay
        onClick={togglePlay}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={e => setCurrent((e.target as HTMLVideoElement).currentTime)}
        onLoadedMetadata={e => setDuration((e.target as HTMLVideoElement).duration)}
        onVolumeChange={e => {
          const v = e.target as HTMLVideoElement;
          setMuted(v.muted);
          setVolume(v.volume);
        }}
        className="w-full h-full object-contain cursor-pointer"
      />

      {/* Big center play button when paused */}
      {!playing && (
        <button
          onClick={togglePlay}
          className="absolute inset-0 flex items-center justify-center bg-black/20 border-none cursor-pointer"
          aria-label="Play"
        >
          <div className="w-16 h-16 rounded-full bg-white/95 flex items-center justify-center shadow-lg hover:scale-110 transition">
            <Play size={26} className="text-[#1A3263] ml-1" fill="#1A3263" />
          </div>
        </button>
      )}

      {/* Controls bar */}
      <div
        className={`absolute left-0 right-0 bottom-0 px-4 pb-3 pt-8 bg-gradient-to-t from-black/80 via-black/40 to-transparent transition-opacity duration-200 ${
          showControls || !playing ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        {/* Progress bar */}
        <div
          ref={progressRef}
          onMouseDown={onScrubStart}
          className="relative h-1.5 bg-white/25 rounded-full cursor-pointer mb-3 group/bar hover:h-2 transition-all"
        >
          <div
            className="absolute left-0 top-0 bottom-0 bg-[#60a5fa] rounded-full"
            style={{ width: `${pct}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 bg-white rounded-full shadow opacity-0 group-hover/bar:opacity-100 transition-opacity"
            style={{ left: `${pct}%` }}
          />
        </div>

        {/* Buttons row */}
        <div className="flex items-center gap-3 text-white">
          <button
            onClick={togglePlay}
            className="bg-transparent border-none text-white cursor-pointer p-1 hover:text-[#60a5fa] transition"
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
          </button>

          <button
            onClick={restart}
            className="bg-transparent border-none text-white cursor-pointer p-1 hover:text-[#60a5fa] transition"
            aria-label="Restart"
          >
            <RotateCcw size={16} />
          </button>

          {/* Volume */}
          <div className="flex items-center gap-2 group/vol">
            <button
              onClick={toggleMute}
              className="bg-transparent border-none text-white cursor-pointer p-1 hover:text-[#60a5fa] transition"
              aria-label={muted ? "Unmute" : "Mute"}
            >
              {muted || volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={muted ? 0 : volume}
              onChange={onVolumeChange}
              className="w-0 group-hover/vol:w-20 transition-all duration-200 accent-[#60a5fa] cursor-pointer"
              style={{ height: 3 }}
            />
          </div>

          {/* Time */}
          <div className="text-xs font-medium text-white/80 tabular-nums">
            {fmt(current)} <span className="text-white/40">/</span> {fmt(duration)}
          </div>

          <div className="flex-1" />

          <button
            onClick={toggleFullscreen}
            className="bg-transparent border-none text-white cursor-pointer p-1 hover:text-[#60a5fa] transition"
            aria-label="Fullscreen"
          >
            <Maximize size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}