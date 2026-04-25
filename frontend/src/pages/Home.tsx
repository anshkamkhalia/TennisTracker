import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";
import { ArrowRight, BarChart3, Play, Sparkles, Video } from "lucide-react";
import Chart from "react-apexcharts";
import { supabase } from "../lib/supabase";

type Session = {
  id: string;
  video_url: string;
  created_at: string;
  title: string | null;
  description: string | null;
  video_type: string | null;
  total_shots: number | null;
  n_shots_by_poi: number | null;
  forehand_percent: number | null;
  backhand_percent: number | null;
  serve_overhead_percent: number | null;
  right_wrist_avg: number | null;
  ball_speeds: number[] | null;
};

const brandColor = "#1A3263";
const accentColor = "#3b82f6";
const tealColor = "#14b8a6";
const goldColor = "#f59e0b";
const chartColors = [brandColor, accentColor, tealColor, goldColor];

export default function Home() {
  const { user } = useUser();
  const navigate = useNavigate();
  const firstName = user?.firstName || user?.emailAddresses?.[0]?.emailAddress?.split("@")[0] || "Player";
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  const [sessions, setSessions] = useState<Session[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);

  useEffect(() => {
    const userId = user?.id;
    if (!userId) return;

    async function fetchSessions() {
      const { data, error } = await supabase
        .from("sessions")
        .select("id, video_url, created_at, title, description, video_type, total_shots, n_shots_by_poi, forehand_percent, backhand_percent, serve_overhead_percent, right_wrist_avg, ball_speeds")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(8);

      if (!error && data) setSessions(data as Session[]);
      setLoadingSessions(false);
    }

    fetchSessions();
  }, [user?.id]);

  const latestSession = sessions[0] || null;
  const recentCourtSessions = sessions.filter(session => session.video_type === "court");

  const globalStats = useMemo(() => {
    const courtCount = recentCourtSessions.length;
    const totalShots = recentCourtSessions.reduce((sum, session) => sum + (session.total_shots || session.n_shots_by_poi || 0), 0);
    const totalPoiShots = recentCourtSessions.reduce((sum, session) => sum + (session.n_shots_by_poi || 0), 0);
    const wristValues = recentCourtSessions
      .map(session => session.right_wrist_avg)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const avgRightWrist = wristValues.length
      ? wristValues.reduce((sum, value) => sum + value, 0) / wristValues.length
      : null;

    const fh = recentCourtSessions.reduce((sum, session) => sum + (session.forehand_percent || 0), 0);
    const bh = recentCourtSessions.reduce((sum, session) => sum + (session.backhand_percent || 0), 0);
    const so = recentCourtSessions.reduce((sum, session) => sum + (session.serve_overhead_percent || 0), 0);
    const divisor = courtCount || 1;

    return {
      totalSessions: sessions.length,
      courtCount,
      totalShots,
      totalPoiShots,
      avgRightWrist,
      avgForehand: fh / divisor,
      avgBackhand: bh / divisor,
      avgServe: so / divisor,
    };
  }, [recentCourtSessions, sessions.length]);

  const dashboardOptions: any = {
    chart: {
      toolbar: { show: false },
      fontFamily: "inherit",
      animations: { enabled: true, speed: 350 },
    },
    grid: {
      borderColor: "#eef2f7",
      strokeDashArray: 4,
      padding: { left: 6, right: 8, top: 0, bottom: 0 },
    },
    dataLabels: { enabled: false },
    tooltip: {
      theme: "light",
      style: { fontSize: "12px", fontFamily: "inherit" },
    },
    legend: {
      fontSize: "12px",
      fontFamily: "inherit",
      labels: { colors: "#6b7280" },
      markers: { width: 8, height: 8, radius: 4 },
      itemMargin: { horizontal: 8 },
    },
    xaxis: {
      labels: { style: { fontSize: "11px", colors: "#9ca3af" } },
      axisTicks: { show: false },
      axisBorder: { show: false },
    },
    yaxis: {
      labels: { style: { fontSize: "11px", colors: "#9ca3af" } },
    },
  };

  return (
    <div className="px-6 py-8 max-w-6xl mx-auto">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6 mb-8">
        <div className="max-w-2xl">
          <p className="text-sm text-gray-500 mb-2">{greeting} 👋</p>
          <h1 className="text-4xl font-extrabold text-gray-900 leading-tight">
            {firstName}'s Dashboard
          </h1>
          <p className="text-gray-500 mt-3 max-w-xl">
            Your latest sessions, global training stats, and the most recent match video are all in one place.
          </p>
        </div>

        <button
          onClick={() => navigate("/record")}
          className="inline-flex items-center justify-center gap-2 bg-[#1A3263] text-white rounded-2xl px-5 py-3.5 text-sm font-semibold hover:bg-[#15295a] transition cursor-pointer shadow-lg shadow-[#1A3263]/15"
        >
          <Video size={16} /> New Recording
        </button>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr] mb-6">
        <div className="bg-gradient-to-br from-[#102448] via-[#1A3263] to-[#274982] text-white rounded-[2rem] p-6 md:p-7 shadow-xl shadow-[#1A3263]/10 overflow-hidden relative">
          <div className="absolute -right-10 -top-10 w-36 h-36 rounded-full bg-white/10 blur-2xl" />
          <div className="absolute -left-6 bottom-0 w-28 h-28 rounded-full bg-sky-400/20 blur-2xl" />
          <div className="relative flex items-center justify-between gap-4 mb-5">
            <div>
              <p className="text-white/60 text-xs uppercase tracking-[0.24em] font-bold mb-2">Featured video</p>
              <h2 className="text-2xl md:text-3xl font-extrabold">
                {latestSession?.title || "Most recent session"}
              </h2>
              <p className="text-white/70 text-sm mt-2">
                {latestSession
                  ? new Date(latestSession.created_at).toLocaleDateString("en-US", {
                      weekday: "short",
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })
                  : "Upload a video to populate this panel"}
              </p>
            </div>
            <div className="hidden sm:flex items-center gap-2 rounded-full bg-white/10 px-3 py-2 text-xs font-semibold text-white/80">
              <Sparkles size={14} /> Latest session
            </div>
          </div>

          {latestSession ? (
            <div className="grid lg:grid-cols-[1.3fr_0.7fr] gap-5 relative">
              <div className="rounded-[1.5rem] overflow-hidden bg-black/20 border border-white/10 shadow-2xl">
                <video
                  src={latestSession.video_url}
                  controls
                  playsInline
                  className="w-full aspect-video object-contain bg-black"
                />
              </div>

              <div className="flex flex-col justify-between gap-4">
                <div className="grid grid-cols-2 gap-3">
                  <MiniMetric label="Sessions" value={String(globalStats.totalSessions)} />
                  <MiniMetric label="Court runs" value={String(globalStats.courtCount)} />
                  <MiniMetric label="Shots" value={String(globalStats.totalShots)} />
                  <MiniMetric label="POI" value={String(globalStats.totalPoiShots)} />
                </div>
                <button
                  onClick={() => navigate("/result", { state: { id: latestSession.id } })}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-white text-[#1A3263] px-4 py-3 font-semibold hover:bg-slate-50 transition"
                >
                  Watch analysis <Play size={15} fill="currentColor" />
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-[1.5rem] border border-white/10 bg-white/5 p-8 text-white/70 text-sm">
              No sessions yet. Record your first video to see the dashboard come alive.
            </div>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
          <StatCard label="Total sessions" value={globalStats.totalSessions} helper="All uploads in your library" />
          <StatCard label="Court sessions" value={globalStats.courtCount} helper="Videos with shot and wrist analysis" />
          <StatCard label="Average right wrist" value={globalStats.avgRightWrist != null ? `${globalStats.avgRightWrist.toFixed(2)} mph` : "—"} helper="Across court sessions" />
          <StatCard label="Total shots" value={globalStats.totalShots} helper="Estimated shot count from analysis" />
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-2 mb-6">
        <div className="bg-white rounded-[2rem] border border-gray-200 p-6 shadow-sm">
          <div className="flex items-center justify-between mb-5">
            <div>
              <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Global stats</p>
              <h3 className="text-xl font-extrabold text-gray-900">Shot mix</h3>
            </div>
            <BarChart3 size={18} className="text-[#1A3263]" />
          </div>
          {recentCourtSessions.length > 0 ? (
            <Chart
              type="donut"
              height={320}
              series={[
                globalStats.avgForehand,
                globalStats.avgBackhand,
                Math.max(0, 100 - globalStats.avgForehand - globalStats.avgBackhand - globalStats.avgServe),
                globalStats.avgServe,
              ]}
              options={{
                ...dashboardOptions,
                colors: chartColors,
                labels: ["Forehand", "Backhand", "Slice / volley", "Serve / overhead"],
                legend: { ...dashboardOptions.legend, position: "bottom" },
                stroke: { width: 0 },
                plotOptions: {
                  pie: {
                    donut: {
                      size: "68%",
                      labels: {
                        show: true,
                        name: { fontSize: "12px", color: "#9ca3af" },
                        value: {
                          fontSize: "24px",
                          fontWeight: 700,
                          color: "#111827",
                          formatter: (v: string) => `${Number(v).toFixed(1)}%`,
                        },
                        total: {
                          show: true,
                          label: "Average",
                          fontSize: "12px",
                          color: "#9ca3af",
                          formatter: () => "100%",
                        },
                      },
                    },
                  },
                },
              }}
            />
          ) : (
            <EmptyState text="Shot mix charts appear after you process a court-view session." />
          )}
        </div>

        <div className="bg-white rounded-[2rem] border border-gray-200 p-6 shadow-sm">
          <div className="flex items-center justify-between mb-5">
            <div>
              <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Training trend</p>
              <h3 className="text-xl font-extrabold text-gray-900">Right wrist speed</h3>
            </div>
            <BarChart3 size={18} className="text-[#1A3263]" />
          </div>
          {recentCourtSessions.some(session => session.right_wrist_avg != null) ? (
            <Chart
              type="bar"
              height={320}
              series={[
                {
                  name: "mph",
                  data: recentCourtSessions
                    .slice(0, 6)
                    .reverse()
                    .map(session => session.right_wrist_avg || 0),
                },
              ]}
              options={{
                ...dashboardOptions,
                colors: [brandColor],
                plotOptions: {
                  bar: { borderRadius: 10, columnWidth: "48%" },
                },
                xaxis: {
                  ...dashboardOptions.xaxis,
                  categories: recentCourtSessions.slice(0, 6).reverse().map((session, index) => `Session ${index + 1}`),
                },
                yaxis: {
                  ...dashboardOptions.yaxis,
                  labels: {
                    style: { fontSize: "11px", colors: "#9ca3af" },
                    formatter: (value: number) => `${value.toFixed(1)}`,
                  },
                },
                tooltip: {
                  theme: "light",
                  y: { formatter: (value: number) => `${value.toFixed(2)} mph` },
                },
              }}
            />
          ) : (
            <EmptyState text="Right-wrist speed data appears once a court session finishes processing." />
          )}
        </div>
      </div>

      <div className="bg-white rounded-[2rem] border border-gray-200 overflow-hidden shadow-sm">
        <div className="px-6 md:px-7 py-5 border-b border-gray-100 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-gray-400 font-bold mb-2">Recent activity</p>
            <h2 className="text-xl font-extrabold text-gray-900">Latest sessions</h2>
          </div>
          <button
            onClick={() => navigate("/history")}
            className="text-[#1A3263] text-sm font-semibold flex items-center gap-1 bg-transparent border-none cursor-pointer"
          >
            View all <ArrowRight size={14} />
          </button>
        </div>

        {loadingSessions ? (
          <div className="px-6 py-10 text-center text-gray-400 text-sm">Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <p className="text-gray-400 text-sm">No sessions yet.</p>
            <button
              onClick={() => navigate("/record")}
              className="mt-3 text-[#1A3263] text-sm font-semibold hover:underline bg-transparent border-none cursor-pointer"
            >
              Upload your first video →
            </button>
          </div>
        ) : (
          <div className="divide-y divide-gray-50">
            {sessions.map((session, i) => (
              <div
                key={session.id}
                onClick={() => navigate("/result", { state: { id: session.id } })}
                className="grid md:grid-cols-[180px_1fr_auto] gap-4 px-6 md:px-7 py-4 hover:bg-gray-50 cursor-pointer transition"
              >
                <div className="rounded-2xl overflow-hidden bg-gray-100 aspect-video md:h-[108px]">
                  <video src={session.video_url} className="w-full h-full object-cover" />
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="inline-flex items-center rounded-full bg-[#1A3263]/10 px-2.5 py-1 text-[11px] font-bold text-[#1A3263]">
                      {session.video_type || "session"}
                    </span>
                    <p className="text-xs text-gray-400">
                      {new Date(session.created_at).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </p>
                  </div>
                  <p className="font-semibold text-gray-900 text-base truncate">
                    {session.title || `Session ${sessions.length - i}`}
                  </p>
                  <p className="text-sm text-gray-500 mt-1 line-clamp-2">
                    {session.description || "Tap to inspect the latest analysis and charts."}
                  </p>
                </div>
                <div className="flex items-center justify-end">
                  <ArrowRight size={18} className="text-gray-300" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, helper }: { label: string; value: number | string; helper: string }) {
  return (
    <div className="bg-white rounded-[1.5rem] border border-gray-200 p-5 shadow-sm">
      <p className="text-xs uppercase tracking-[0.2em] text-gray-400 font-bold">{label}</p>
      <p className="text-3xl font-extrabold text-gray-900 mt-3">{value}</p>
      <p className="text-sm text-gray-500 mt-2 leading-relaxed">{helper}</p>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-white/10 border border-white/10 p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-white/55 font-bold">{label}</p>
      <p className="text-xl font-extrabold mt-1">{value}</p>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="h-[320px] rounded-[1.5rem] border border-dashed border-gray-200 bg-gray-50 flex items-center justify-center text-center px-8">
      <p className="text-sm text-gray-400 max-w-sm">{text}</p>
    </div>
  );
}
