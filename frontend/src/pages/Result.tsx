import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Check, Download, Maximize, Pause, Play, RotateCcw, Sparkles, Trash2, Volume2, VolumeX, X } from "lucide-react";
import { supabase } from "../lib/supabase";
import Chart from "react-apexcharts";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

type AnalysisPayload = {
  url?: string;
  video_type?: string;
  n_shots_by_POI?: number;
  total_shots?: number;
  forehand_percent?: number;
  backhand_percent?: number;
  serve_overhead_percent?: number;
  right_wrist_avg?: number;
  right_wrist_v?: number[];
  heatmap?: string;
  ball_speeds?: number[] | null;
  pose_landmarks_3d?: number[][][];
};

export default function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const locationState = (location.state as any) || {};
  const sessionId = locationState.id;
  const analysis = (locationState.analysis || locationState.result || {}) as AnalysisPayload;

  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editDesc, setEditDesc] = useState("");

  const mergedSession = { ...session, ...analysis };
  const poseFrames = mergedSession.pose_landmarks_3d || [];
  const hasPoseData = Array.isArray(poseFrames) && poseFrames.length > 0;

  useEffect(() => {
    if (!sessionId) {
      setLoading(false);
      return;
    }

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

  async function deleteSession() {
    if (!sessionId) return;
    const ok = window.confirm("Delete this session? This cannot be undone.");
    if (!ok) return;

    const { error } = await supabase.from("sessions").delete().eq("id", sessionId);
    if (!error) {
      navigate("/history");
    }
  }

  if (!sessionId && !hasPoseData && !analysis.url) {
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

  if (loading && !analysis.url) {
    return (
      <div className="px-10 py-20 text-center">
        <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
        <p className="text-gray-500 text-sm">Loading session...</p>
      </div>
    );
  }

  const isCourt = mergedSession.video_type === "court";
  const chartColors = ["#1A3263", "#3b82f6", "#14b8a6", "#f59e0b"];

  const statsItems = [
    { label: "Total shots", value: mergedSession.total_shots ?? mergedSession.n_shots_by_poi },
    { label: "Shots by POI", value: mergedSession.n_shots_by_POI ?? mergedSession.n_shots_by_poi },
    { label: "Forehand", value: mergedSession.forehand_percent != null ? `${Number(mergedSession.forehand_percent).toFixed(2)}%` : null },
    { label: "Backhand", value: mergedSession.backhand_percent != null ? `${Number(mergedSession.backhand_percent).toFixed(2)}%` : null },
    { label: "Serve / overhead", value: mergedSession.serve_overhead_percent != null ? `${Number(mergedSession.serve_overhead_percent).toFixed(2)}%` : null },
    { label: "Right wrist avg", value: mergedSession.right_wrist_avg != null ? `${Number(mergedSession.right_wrist_avg).toFixed(2)} mph` : null },
  ].filter(item => item.value != null);

  const baseChartOptions: any = {
    chart: {
      toolbar: { show: false },
      fontFamily: "inherit",
      animations: { enabled: true, speed: 400 },
    },
    grid: {
      borderColor: "#eef2f7",
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

  const tabs = ["Overview", "Pose demo", "Settings"];

  return (
    <div className="px-6 py-8 mx-auto max-w-7xl">
      <div className="flex flex-col md:flex-row md:items-start gap-4 mb-8">
        <button
          onClick={() => navigate("/record")}
          className="flex items-center gap-2 bg-white border border-gray-200 rounded-xl px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 transition cursor-pointer shrink-0"
        >
          <ArrowLeft size={15} /> Back to Record
        </button>
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="inline-flex items-center rounded-full bg-[#1A3263]/10 px-3 py-1 text-xs font-bold text-[#1A3263]">
              {mergedSession.video_type || "session"}
            </span>
            <span className="inline-flex items-center rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700">
              <Sparkles size={12} className="mr-1" /> Analysis ready
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl font-extrabold text-gray-900">
            {mergedSession.title || "Analysis Results"}
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            {mergedSession.description ||
              new Date(mergedSession.created_at || Date.now()).toLocaleDateString("en-US", {
                weekday: "short",
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
          </p>
        </div>
        <button
          onClick={deleteSession}
          className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-xl px-4 py-2 text-sm font-semibold text-red-600 hover:bg-red-100 transition cursor-pointer shrink-0"
        >
          <Trash2 size={15} /> Delete session
        </button>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-6">
          <VideoPlayer src={mergedSession.video_url || mergedSession.url || analysis.url || ""} />

          <div className="bg-white rounded-[1.75rem] border border-gray-200 p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Session summary</p>
                <h2 className="text-xl font-extrabold text-gray-900">Key metrics</h2>
              </div>
              <button
                onClick={() => navigate("/record")}
                className="text-sm text-[#1A3263] font-semibold bg-transparent border-none"
              >
                Upload another
              </button>
            </div>

            <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-3">
              {statsItems.map(({ label, value }) => (
                <MetricCard key={label} label={label} value={String(value)} />
              ))}
            </div>
          </div>

          {isCourt ? (
            <div className="grid gap-6 lg:grid-cols-2">
              {(mergedSession.right_wrist_v?.length > 0) && (
                <div className="bg-white rounded-[1.75rem] border border-gray-200 p-5 shadow-sm lg:col-span-2">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Wrist speed</p>
                      <h3 className="text-lg font-extrabold text-gray-900">Right wrist velocity</h3>
                    </div>
                  </div>
                  <Chart
                    type="area"
                    height={260}
                    series={[{ name: "Right wrist", data: mergedSession.right_wrist_v || [] }]}
                    options={{
                      ...baseChartOptions,
                      colors: [chartColors[0]],
                      stroke: { curve: "smooth", width: 2 },
                      fill: {
                        type: "gradient",
                        gradient: {
                          shadeIntensity: 1,
                          opacityFrom: 0.28,
                          opacityTo: 0.02,
                          stops: [0, 100],
                        },
                      },
                      xaxis: {
                        ...baseChartOptions.xaxis,
                        labels: { show: false },
                      },
                    }}
                  />
                </div>
              )}

              <div className="bg-white rounded-[1.75rem] border border-gray-200 p-5 shadow-sm">
                <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Shot mix</p>
                <Chart
                  type="donut"
                  height={260}
                  series={[
                    Number(mergedSession.forehand_percent || 0),
                    Number(mergedSession.backhand_percent || 0),
                    Number(mergedSession.serve_overhead_percent || 0),
                  ]}
                  options={{
                    ...baseChartOptions,
                    colors: chartColors,
                    labels: ["Forehand", "Backhand", "Serve / overhead"],
                    legend: { ...baseChartOptions.legend, position: "bottom" },
                    stroke: { width: 0 },
                    plotOptions: {
                      pie: {
                        donut: {
                          size: "68%",
                          labels: {
                            show: true,
                            name: { fontSize: "11px", color: "#9ca3af" },
                            value: {
                              fontSize: "18px",
                              fontWeight: 700,
                              color: "#111827",
                              formatter: (v: string) => `${Number(v).toFixed(1)}%`,
                            },
                            total: {
                              show: true,
                              label: "Total",
                              fontSize: "11px",
                              color: "#9ca3af",
                              formatter: () => "100%",
                            },
                          },
                        },
                      },
                    },
                  }}
                />
              </div>

              <div className="bg-white rounded-[1.75rem] border border-gray-200 p-5 shadow-sm">
                <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Shot breakdown</p>
                <Chart
                  type="bar"
                  height={260}
                  series={[
                    {
                      name: "Percent",
                      data: [
                        Number(mergedSession.forehand_percent || 0),
                        Number(mergedSession.backhand_percent || 0),
                        Number(mergedSession.serve_overhead_percent || 0),
                      ],
                    },
                  ]}
                  options={{
                    ...baseChartOptions,
                    colors: [chartColors[1]],
                    plotOptions: {
                      bar: { borderRadius: 8, columnWidth: "50%" },
                    },
                    xaxis: {
                      ...baseChartOptions.xaxis,
                      categories: ["Forehand", "Backhand", "Serve"],
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
            </div>
          ) : (
            <div className="grid gap-6 lg:grid-cols-2">
              {mergedSession.ball_speeds?.length > 0 ? (
                <div className="bg-white rounded-[1.75rem] border border-gray-200 p-5 shadow-sm lg:col-span-2">
                  <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Ball speed</p>
                  <Chart
                    type="area"
                    height={260}
                    series={[{ name: "mph", data: mergedSession.ball_speeds }]}
                    options={{
                      ...baseChartOptions,
                      colors: [chartColors[0]],
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
                        strokeColors: chartColors[0],
                        strokeWidth: 2,
                        hover: { size: 6 },
                      },
                      xaxis: {
                        ...baseChartOptions.xaxis,
                        categories: mergedSession.ball_speeds.map((_: any, i: number) => `${i + 1}`),
                        title: { text: "Shot #", style: { fontSize: "10px", color: "#9ca3af", fontWeight: 500 } },
                      },
                    }}
                  />
                </div>
              ) : null}

              {mergedSession.heatmap && (
                <div className="bg-white rounded-[1.75rem] border border-gray-200 p-5 shadow-sm lg:col-span-2">
                  <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Court heatmap</p>
                  <img
                    src={mergedSession.heatmap}
                    className="w-full h-[260px] object-contain rounded-2xl bg-gray-50"
                  />
                </div>
              )}
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div className="bg-white rounded-[1.75rem] border border-gray-200 overflow-hidden shadow-sm">
            <div className="px-5 pt-5 pb-4 border-b border-gray-100 flex gap-2">
              {tabs.map((tab, i) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(i)}
                  className={`px-4 py-2.5 text-sm font-semibold border-none cursor-pointer transition rounded-xl ${
                    activeTab === i
                      ? "bg-[#1A3263] text-white"
                      : "bg-gray-50 text-gray-500 hover:bg-gray-100"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            <div className="p-5">
              {activeTab === 0 && (
                <div className="space-y-4">
                  <div className="rounded-2xl bg-gray-50 border border-gray-200 p-4">
                    <p className="text-xs text-gray-400 mb-2">Video location</p>
                    <a
                      href={mergedSession.video_url}
                      download={`${(mergedSession.title || "session").replace(/[^a-z0-9]+/gi, "_")}.mp4`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-semibold text-[#1A3263] break-all"
                    >
                      {mergedSession.video_url}
                    </a>
                  </div>
                  {mergedSession.right_wrist_v?.length > 0 && (
                    <div className="rounded-2xl bg-gray-50 border border-gray-200 p-4">
                      <p className="text-xs text-gray-400 mb-2">Right wrist samples</p>
                      <div className="flex flex-wrap gap-2">
                        {mergedSession.right_wrist_v.slice(0, 12).map((s: number, i: number) => (
                          <span key={i} className="bg-white border border-gray-200 rounded-lg px-3 py-1 text-sm font-semibold text-gray-700">
                            {s.toFixed(2)} mph
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 1 && (
                <PoseReplay frames={poseFrames} />
              )}

              {activeTab === 2 && (
                <div className="space-y-4">
                  <div>
                    <label className="text-xs text-gray-400 mb-1 block">Session title</label>
                    <input
                      value={editTitle}
                      onChange={e => { setEditTitle(e.target.value); setEditing(true); }}
                      placeholder="Add a title..."
                      className="text-sm font-semibold border border-gray-200 rounded-xl px-3 py-2 w-full focus:outline-none focus:border-[#1A3263]"
                    />
                  </div>

                  <div>
                    <label className="text-xs text-gray-400 mb-1 block">Date</label>
                    <div className="text-sm border border-gray-200 rounded-xl px-3 py-2 bg-gray-50 text-gray-500">
                      {new Date(mergedSession.created_at || Date.now()).toLocaleDateString("en-US", {
                        weekday: "short",
                        month: "short",
                        day: "numeric",
                        year: "numeric",
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
                      className="text-sm border border-gray-200 rounded-xl px-3 py-2 w-full resize-none focus:outline-none focus:border-[#1A3263] text-gray-500"
                    />
                  </div>

                  {editing && (
                    <div className="flex gap-2">
                      <button
                        onClick={saveEdit}
                        className="flex items-center gap-2 bg-[#1A3263] text-white border-none rounded-xl px-4 py-2 text-sm font-semibold cursor-pointer hover:bg-[#15295a] transition"
                      >
                        <Check size={14} /> Save
                      </button>
                      <button
                        onClick={() => { setEditing(false); setEditTitle(session?.title || ""); setEditDesc(session?.description || ""); }}
                        className="flex items-center gap-2 bg-gray-100 text-gray-600 border-none rounded-xl px-4 py-2 text-sm font-semibold cursor-pointer hover:bg-gray-200 transition"
                      >
                        <X size={14} /> Cancel
                      </button>
                    </div>
                  )}

                  <button
                    onClick={deleteSession}
                    className="flex items-center justify-center gap-2 w-full bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm font-semibold text-red-600 hover:bg-red-100 transition cursor-pointer"
                  >
                    <Trash2 size={14} /> Delete this session
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-gray-50 border border-gray-200 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-gray-400 font-bold">{label}</p>
      <p className="text-lg md:text-xl font-extrabold text-gray-900 mt-2">{value}</p>
    </div>
  );
}

function PoseReplay({ frames }: { frames: number[][][] }) {
  const [playing, setPlaying] = useState(false);
  const [targetPlayhead, setTargetPlayhead] = useState(0);
  const [displayPlayhead, setDisplayPlayhead] = useState(0);
  const [zoom, setZoom] = useState(1.25);
  const [baseAngle, setBaseAngle] = useState([0.15, 0.7, 0] as [number, number, number]);
  const frameCount = frames?.length || 0;
  const poseFrames = useMemo(() => normalizePoseFrames(frames), [frames]);
  const frameIndex = Math.floor(displayPlayhead) % Math.max(frameCount, 1);
  const blend = displayPlayhead - Math.floor(displayPlayhead);
  const currentFrame = useMemo(() => {
    if (!poseFrames.length) return undefined;
    const a = poseFrames[frameIndex];
    const b = poseFrames[(frameIndex + 1) % poseFrames.length];
    return interpolateFrames(a, b, blend);
  }, [blend, frameIndex, poseFrames]);
  const frameDurationMs = 1000 / 14;
  const playbackRef = useRef({ playhead: 0, lastTs: 0 });

  useEffect(() => {
    if (!playing || !frameCount) {
      playbackRef.current.lastTs = 0;
      return;
    }

    let raf = 0;
    const tick = (ts: number) => {
      const state = playbackRef.current;
      if (!state.lastTs) state.lastTs = ts;
      const delta = ts - state.lastTs;
      state.lastTs = ts;

      const nextPlayhead = (state.playhead + delta / frameDurationMs) % frameCount;
      state.playhead = nextPlayhead;
      setTargetPlayhead(nextPlayhead);
      raf = window.requestAnimationFrame(tick);
    };

    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [frameCount, frameDurationMs, playing]);

  useEffect(() => {
    if (targetPlayhead >= frameCount && frameCount > 0) setTargetPlayhead(0);
  }, [frameCount, targetPlayhead]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setDisplayPlayhead(prev => {
        const delta = targetPlayhead - prev;
        if (Math.abs(delta) < 0.0005) return targetPlayhead;
        const next = prev + delta * 0.2;
        return Math.abs(targetPlayhead - next) < 0.0005 ? targetPlayhead : next;
      });
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [targetPlayhead]);

  const onScrub = useCallback((value: number) => {
    if (!frameCount) return;
    const nextPlayhead = Math.min(frameCount - 1, Math.max(0, value));
    playbackRef.current.playhead = nextPlayhead;
    playbackRef.current.lastTs = 0;
    setTargetPlayhead(nextPlayhead);
    if (!playing) setDisplayPlayhead(nextPlayhead);
  }, [frameCount]);

  if (!frameCount) {
    return (
      <div className="rounded-2xl border border-dashed border-gray-200 bg-gray-50 p-8 text-center text-sm text-gray-400">
        Pose frames were not included for this session.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-[1.75rem] overflow-hidden bg-gradient-to-br from-slate-950 via-[#102448] to-[#1A3263] border border-slate-800 shadow-lg">
        <div className="flex flex-col gap-3 px-5 py-4 border-b border-white/10 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-white/40 font-bold mb-1">Interactive pose demo</p>
            <h3 className="text-lg font-extrabold text-white">3D human mesh playback</h3>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-white/55">
            <span>Drag to rotate</span>
            <span>•</span>
            <span>Use the slider to scrub</span>
            <span>•</span>
            <span>Frame {Math.floor(displayPlayhead) + 1} / {frameCount}</span>
          </div>
        </div>

        <div className="p-4">
          <div className="w-full h-[400px] rounded-[1.25rem] overflow-hidden bg-black/35">
            <Canvas
              camera={{ position: [0, 0.8, 5.5], fov: 40, near: 0.1, far: 100 }}
              dpr={[1, 2]}
            >
              <color attach="background" args={["#030712"]} />
              <fog attach="fog" args={["#030712", 8, 16]} />
              <ambientLight intensity={1.6} />
              <directionalLight position={[4, 6, 6]} intensity={2.2} color="#ffffff" />
              <directionalLight position={[-4, 2, 3]} intensity={1.3} color="#60a5fa" />
              <group rotation={baseAngle} scale={zoom}>
                <PoseHuman frame={currentFrame} />
              </group>
              <OrbitControls
                makeDefault
                enablePan={false}
                enableZoom={false}
                target={[0, 0.25, 0]}
                onChange={e => {
                  const ctl = e?.target as any;
                  const x = ctl?.getPolarAngle?.() ?? 0;
                  const y = ctl?.getAzimuthalAngle?.() ?? 0;
                  setBaseAngle([0.15 + (x - Math.PI / 2) * 0.15, 0.7 + y * 0.2, 0]);
                }}
              />
              <gridHelper args={[12, 24, "#1f3b74", "#14213d"]} position={[0, -2.3, 0]} />
            </Canvas>
          </div>
        </div>

        <div className="px-4 pb-4 space-y-4">
            <input
              type="range"
              min={0}
              max={Math.max(0, frameCount - 1)}
              step={0.001}
              value={targetPlayhead}
              onChange={e => onScrub(Number(e.target.value))}
              className="w-full accent-sky-400"
            />

          <div className="flex flex-col gap-3 text-white md:flex-row md:items-center">
            <button
              onClick={() => setPlaying(v => !v)}
              className="bg-white/10 border border-white/10 text-white rounded-xl px-4 py-2.5 text-sm font-semibold flex items-center gap-2 hover:bg-white/15 transition"
            >
              {playing ? <Pause size={16} /> : <Play size={16} fill="currentColor" />} {playing ? "Pause" : "Play"}
            </button>
            <button
              onClick={() => {
                setTargetPlayhead(0);
                setDisplayPlayhead(0);
                setPlaying(false);
                playbackRef.current.playhead = 0;
                playbackRef.current.lastTs = 0;
              }}
              className="bg-white/10 border border-white/10 text-white rounded-xl px-4 py-2.5 text-sm font-semibold flex items-center gap-2 hover:bg-white/15 transition"
            >
              <RotateCcw size={16} /> Restart
            </button>
            <div className="flex items-center gap-2 md:ml-auto">
              <span className="text-xs text-white/55 uppercase tracking-[0.18em] font-bold">Zoom</span>
              <input
                type="range"
                min={0.9}
                max={1.8}
                step={0.01}
                value={zoom}
                onChange={e => setZoom(Number(e.target.value))}
                className="w-28 accent-sky-400"
              />
            </div>
            <div className="text-xs text-white/45 md:ml-2">
              Smooth playback with frame interpolation
            </div>
          </div>
        </div>
      </div>
      <p className="text-xs text-gray-400 leading-relaxed">
        This preview keeps the pose anchored in the middle of the scene, flips it upright, and renders a human-like body mesh so the motion is easier to read from any angle.
      </p>
    </div>
  );
}

type PosePoint = { x: number; y: number; z: number };

const POSE_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [15, 17], [17, 19],
  [12, 14], [14, 16], [16, 18], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32],
  [11, 24], [12, 23],
  [23, 26], [24, 25],
];

function PoseHuman({ frame }: { frame?: PosePoint[] }) {
  if (!frame || frame.length < 33) return null;

  return (
    <group position={[0, 0, 0]}>
      <HumanBody frame={frame} />
      <UpperArm frame={frame} side="left" />
      <UpperArm frame={frame} side="right" />
      <LowerArm frame={frame} side="left" />
      <LowerArm frame={frame} side="right" />
      <UpperLeg frame={frame} side="left" />
      <UpperLeg frame={frame} side="right" />
      <LowerLeg frame={frame} side="left" />
      <LowerLeg frame={frame} side="right" />
      <Hand frame={frame} side="left" />
      <Hand frame={frame} side="right" />
      <Foot frame={frame} side="left" />
      <Foot frame={frame} side="right" />
    </group>
  );
}

function HumanBody({ frame }: { frame: PosePoint[] }) {
  const hips = avg(frame[23], frame[24]) || { x: 0, y: -0.1, z: 0 };
  const shoulders = avg(frame[11], frame[12]) || { x: 0, y: 0.55, z: 0 };
  const chest = avg(shoulders, hips) || { x: 0, y: 0.2, z: 0 };
  const pelvis = hips;
  const torsoHeight = Math.max(0.7, Math.abs(shoulders.y - hips.y) * 1.15);
  const shoulderWidth = Math.max(0.3, Math.abs(frame[11]?.x - frame[12]?.x) * 1.12);
  const hipWidth = Math.max(0.25, Math.abs(frame[23]?.x - frame[24]?.x) * 1.08);
  const bodyLean = clamp((shoulders.x - hips.x) * 0.5, -0.12, 0.12);

  return (
    <group rotation={[0, bodyLean, 0]}>
      <mesh position={[chest.x, chest.y + torsoHeight * 0.14, chest.z]} scale={[shoulderWidth * 1.08, torsoHeight * 0.6, 0.4]}>
        <capsuleGeometry args={[0.44, 0.96, 10, 18]} />
        <meshStandardMaterial color="#60a5fa" wireframe transparent opacity={0.65} />
      </mesh>
      <mesh position={[chest.x, chest.y - torsoHeight * 0.12, chest.z]} scale={[hipWidth * 1.15, torsoHeight * 0.38, 0.46]}>
        <capsuleGeometry args={[0.4, 0.72, 10, 18]} />
        <meshStandardMaterial color="#93c5fd" wireframe transparent opacity={0.55} />
      </mesh>
      <mesh position={[pelvis.x, pelvis.y - 0.12, pelvis.z]} scale={[hipWidth * 1.05, 0.4, 0.46]}>
        <sphereGeometry args={[1, 28, 28]} />
        <meshStandardMaterial color="#38bdf8" wireframe transparent opacity={0.52} />
      </mesh>
      <mesh position={[shoulders.x, shoulders.y + 0.1, shoulders.z]} scale={[shoulderWidth * 1.05, 0.12, 0.24]}>
        <capsuleGeometry args={[0.11, 0.18, 8, 14]} />
        <meshStandardMaterial color="#e0f2fe" wireframe transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

function UpperArm({ frame, side }: { frame: PosePoint[]; side: "left" | "right" }) {
  const shoulderIndex = side === "left" ? 11 : 12;
  const elbowIndex = side === "left" ? 13 : 14;
  const shoulder = frame[shoulderIndex];
  const elbow = frame[elbowIndex];
  if (!shoulder || !elbow) return null;

  return (
    <LimbSegment
      a={shoulder}
      b={elbow}
      color={side === "left" ? "#f1c7a5" : "#f0b996"}
      radius={0.085}
      capRadius={0.105}
    />
  );
}

function LowerArm({ frame, side }: { frame: PosePoint[]; side: "left" | "right" }) {
  const elbowIndex = side === "left" ? 13 : 14;
  const wristIndex = side === "left" ? 15 : 16;
  const elbow = frame[elbowIndex];
  const wrist = frame[wristIndex];
  if (!elbow || !wrist) return null;

  return (
    <LimbSegment
      a={elbow}
      b={wrist}
      color={side === "left" ? "#f1c7a5" : "#e7b18b"}
      radius={0.075}
      capRadius={0.09}
    />
  );
}

function UpperLeg({ frame, side }: { frame: PosePoint[]; side: "left" | "right" }) {
  const hipIndex = side === "left" ? 23 : 24;
  const kneeIndex = side === "left" ? 25 : 26;
  const hip = frame[hipIndex];
  const knee = frame[kneeIndex];
  if (!hip || !knee) return null;

  return (
    <LimbSegment
      a={hip}
      b={knee}
      color={side === "left" ? "#23344e" : "#1f2d43"}
      radius={0.11}
      capRadius={0.13}
    />
  );
}

function LowerLeg({ frame, side }: { frame: PosePoint[]; side: "left" | "right" }) {
  const kneeIndex = side === "left" ? 25 : 26;
  const ankleIndex = side === "left" ? 27 : 28;
  const knee = frame[kneeIndex];
  const ankle = frame[ankleIndex];
  if (!knee || !ankle) return null;

  return (
    <LimbSegment
      a={knee}
      b={ankle}
      color={side === "left" ? "#23344e" : "#1f2d43"}
      radius={0.095}
      capRadius={0.11}
    />
  );
}

function Hand({ frame, side }: { frame: PosePoint[]; side: "left" | "right" }) {
  const wristIndex = side === "left" ? 15 : 16;
  const wrist = frame[wristIndex];
  if (!wrist) return null;
  return (
    <mesh position={[wrist.x, wrist.y, wrist.z]}>
      <sphereGeometry args={[0.08, 22, 22]} />
      <meshStandardMaterial color={side === "left" ? "#67e8f9" : "#f59e0b"} wireframe transparent opacity={0.85} />
    </mesh>
  );
}

function Foot({ frame, side }: { frame: PosePoint[]; side: "left" | "right" }) {
  const ankleIndex = side === "left" ? 27 : 28;
  const footIndex = side === "left" ? 31 : 32;
  const ankle = frame[ankleIndex];
  const foot = frame[footIndex];
  if (!ankle || !foot) return null;
  const start = new THREE.Vector3(ankle.x, ankle.y, ankle.z);
  const end = new THREE.Vector3(foot.x, foot.y, foot.z);
  const dir = new THREE.Vector3().subVectors(end, start);
  const length = Math.max(dir.length(), 1e-4);
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize()
  );

  return (
    <mesh position={[midpoint.x, midpoint.y, midpoint.z]} quaternion={quat}>
      <capsuleGeometry args={[0.06, Math.max(length - 0.06, 0.05), 8, 14]} />
      <meshStandardMaterial color="#e2e8f0" wireframe transparent opacity={0.9} />
    </mesh>
  );
}

function LimbSegment({ a, b, color, radius, capRadius }: { a: PosePoint; b: PosePoint; color: string; radius: number; capRadius: number }) {
  const start = new THREE.Vector3(a.x, a.y, a.z);
  const end = new THREE.Vector3(b.x, b.y, b.z);
  const dir = new THREE.Vector3().subVectors(end, start);
  const length = Math.max(dir.length(), 1e-4);
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize()
  );

  return (
    <mesh position={[midpoint.x, midpoint.y, midpoint.z]} quaternion={quat}>
      <capsuleGeometry args={[radius, Math.max(length - radius * 1.2, 0.08), 10, 20]} />
      <meshStandardMaterial color={color} wireframe transparent opacity={0.8} />
    </mesh>
  );
}

function normalizePoseFrames(frames: number[][][] = []) {
  const rawScaleSamples = frames
    .map(frame => torsoSize(frame))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0.0001);

  const globalScale = median(rawScaleSamples) || 1;

  return frames.map(frame => {
    const root = avg(frame[23], frame[24]) || avg(frame[11], frame[12]) || { x: 0, y: 0, z: 0 };

    return frame.map((point = [0, 0, 0]) => ({
      x: ((point[0] || 0) - root.x) / globalScale,
      y: (-(point[1] || 0) + root.y) / globalScale,
      z: ((point[2] || 0) - root.z) / globalScale,
    }));
  });
}

function interpolateFrames(a?: PosePoint[], b?: PosePoint[], t = 0) {
  if (!a && !b) return undefined;
  if (!a) return b;
  if (!b) return a;
  return a.map((point, index) => {
    const next = b[index] || point;
    return {
      x: lerp(point.x, next.x, t),
      y: lerp(point.y, next.y, t),
      z: lerp(point.z, next.z, t),
    };
  });
}

function torsoSize(frame: number[][]) {
  const shoulders = avg(frame[11], frame[12]);
  const hips = avg(frame[23], frame[24]);
  if (!shoulders || !hips) return null;
  return distance(shoulders, hips);
}

function avg(a?: PosePoint | number[], b?: PosePoint | number[]): PosePoint | null {
  if (!a || !b) return null;
  const aPoint = Array.isArray(a)
    ? { x: a[0] || 0, y: a[1] || 0, z: a[2] || 0 }
    : a;
  const bPoint = Array.isArray(b)
    ? { x: b[0] || 0, y: b[1] || 0, z: b[2] || 0 }
    : b;
  return {
    x: (aPoint.x + bPoint.x) / 2,
    y: (aPoint.y + bPoint.y) / 2,
    z: (aPoint.z + bPoint.z) / 2,
  };
}

function distance(a: PosePoint, b: PosePoint) {
  return Math.sqrt(
    (a.x - b.x) ** 2 +
    (a.y - b.y) ** 2 +
    (a.z - b.z) ** 2
  );
}

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

/* ============================================================
   Custom video player
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
      className="relative bg-black rounded-[1.75rem] overflow-hidden aspect-video group shadow-lg"
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
        className="w-full h-full object-contain cursor-pointer bg-black"
      />

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

      <div
        className={`absolute left-0 right-0 bottom-0 px-4 pb-3 pt-8 bg-gradient-to-t from-black/80 via-black/40 to-transparent transition-opacity duration-200 ${
          showControls || !playing ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
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
