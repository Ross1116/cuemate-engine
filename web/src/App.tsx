import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import QRCode from "qrcode";
import {
  AlertTriangle,
  BarChart3,
  Check,
  ChevronDown,
  CircleHelp,
  CircleDot,
  Eye,
  FileAudio,
  FolderPlus,
  FolderOpen,
  Library,
  ListMusic,
  PlayCircle,
  Radar,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  TerminalSquare,
} from "lucide-react";
import { Bar, BarChart, LabelList, ResponsiveContainer, XAxis, YAxis } from "recharts";
import {
  api,
  FeedbackSummary,
  LaneItem,
  PickPathRequest,
  Playlist,
  PlaylistAnalysisStatus,
  RecommendationResponse,
  SetupStatus,
  ToolCommandRequest,
  ToolCommandResult,
  ToolRunStatus,
  Track,
} from "./api";

const targets = ["maintain", "build", "reset", "jump", "contrast"];

const targetDescriptions: Record<string, string> = {
  maintain: "Target mode: preserve the current energy and keep the handoff steady.",
  build: "Target mode: nudge momentum upward without making a hard jump.",
  reset: "Target mode: create room, reduce pressure, or reframe the set.",
  jump: "Target mode: make a bigger energy or character move.",
  contrast: "Target mode: surface useful alternatives that intentionally differ from the current track.",
};

const laneDescriptions: Record<string, string> = {
  maintain: "Result lane: closest continuity picks for staying near the current feel.",
  build: "Result lane: candidates that add lift or pressure from here.",
  reset: "Result lane: options that open space or cool the room before the next move.",
  jump: "Result lane: bigger changes in energy, character, or direction.",
  contrast: "Result lane: higher-contrast alternatives that may still be playable.",
};

const weightDescriptions: Record<string, string> = {
  bass_transition: "Low-end handoff quality between the current track and candidate.",
  harmonic: "Camelot/key compatibility, adjusted by key confidence.",
  history_fit: "How well the pick avoids repetition and stagnant recent sequencing.",
  rhythmic_continuity: "Groove continuity across the handoff. Limited when rhythm features are unavailable.",
  target_energy: "How closely the pick matches the selected target mode's energy direction.",
  tempo: "BPM proximity, including ratio-aware tempo matches.",
  transition_support: "Intro/outro window support for a clean mix. Stubbed when window features are unavailable.",
  vocal_transition: "Vocal overlap or contrast risk during the handoff. Stubbed when vocal features are unavailable.",
};

const metricDescriptions: Record<string, string> = {
  "Fit score": "0-100 fit score for this candidate in its result lane. Higher means stronger scorer fit.",
  Move: "The transition strategy the scorer thinks this pick supports.",
  Risk: "Estimated transition difficulty. Low is safer; high needs more care.",
  Confidence: "How confident the scorer is in the suggested move label.",
  Events: "Recorded recommendation outcomes for this playlist.",
  "Top 1": "Share of recorded choices where the selected song was the top-ranked recommendation.",
  "Top 3": "Share of recorded choices where the selected song was in the top three recommendations.",
  Pairs: "Pairwise comparisons collected from skipped-over recommendations.",
};

function pct(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}

function titleFor(track?: Pick<Track, "title" | "artist" | "track_id"> | null) {
  if (!track) return "No track selected";
  const title = track.title || track.track_id;
  return track.artist ? `${track.artist} - ${title}` : title;
}

function helpFor(map: Record<string, string>, key: string, fallback: string) {
  return map[key.toLowerCase()] ?? fallback;
}

function TooltipAnchor({
  text,
  children,
  className,
  focusable = true,
}: {
  text: string;
  children: React.ReactNode;
  className?: string;
  focusable?: boolean;
}) {
  const tooltipId = useId();
  const anchorRef = useRef<HTMLSpanElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number; placement: "top" | "bottom" } | null>(null);

  const closeTooltip = useCallback(() => setPosition(null), []);

  const updatePosition = useCallback(() => {
    const anchor = anchorRef.current;
    if (!anchor) return;

    const rect = anchor.getBoundingClientRect();
    const placement = rect.top < 96 ? "bottom" : "top";
    const maxTooltipWidth = Math.min(280, window.innerWidth - 24);
    const unclampedLeft = rect.left + rect.width / 2;
    const left = Math.min(window.innerWidth - 12 - maxTooltipWidth / 2, Math.max(12 + maxTooltipWidth / 2, unclampedLeft));
    const top = placement === "bottom" ? rect.bottom + 10 : rect.top - 10;
    window.dispatchEvent(new CustomEvent("cuemate-tooltip-open", { detail: { id: tooltipId } }));
    setPosition({ left, top, placement });
  }, [tooltipId]);

  useEffect(() => {
    const handleOtherTooltip = (event: Event) => {
      const nextTooltipId = (event as CustomEvent<{ id?: string }>).detail?.id;
      if (nextTooltipId !== tooltipId) closeTooltip();
    };
    window.addEventListener("cuemate-tooltip-open", handleOtherTooltip);
    return () => window.removeEventListener("cuemate-tooltip-open", handleOtherTooltip);
  }, [closeTooltip, tooltipId]);

  useEffect(() => {
    if (!position) return undefined;
    const handleMove = () => updatePosition();
    const handlePointerMove = (event: PointerEvent) => {
      const anchor = anchorRef.current;
      if (anchor && event.target instanceof Node && !anchor.contains(event.target)) closeTooltip();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeTooltip();
    };
    window.addEventListener("resize", handleMove);
    window.addEventListener("scroll", handleMove, true);
    window.addEventListener("blur", closeTooltip);
    window.addEventListener("keydown", handleKeyDown, true);
    document.addEventListener("pointermove", handlePointerMove, true);
    document.addEventListener("pointerdown", closeTooltip, true);
    return () => {
      window.removeEventListener("resize", handleMove);
      window.removeEventListener("scroll", handleMove, true);
      window.removeEventListener("blur", closeTooltip);
      window.removeEventListener("keydown", handleKeyDown, true);
      document.removeEventListener("pointermove", handlePointerMove, true);
      document.removeEventListener("pointerdown", closeTooltip, true);
    };
  }, [closeTooltip, position, updatePosition]);

  return (
    <span
      ref={anchorRef}
      className={className ? `tooltip-anchor ${className}` : "tooltip-anchor"}
      tabIndex={focusable ? 0 : undefined}
      aria-label={focusable ? text : undefined}
      onPointerEnter={updatePosition}
      onPointerLeave={closeTooltip}
      onPointerCancel={closeTooltip}
      onFocus={focusable ? updatePosition : undefined}
      onBlur={focusable ? closeTooltip : undefined}
    >
      {children}
      {position
        ? createPortal(
            <span id={tooltipId} className={`floating-tooltip ${position.placement}`} role="tooltip" style={{ left: position.left, top: position.top }}>
              {text}
            </span>,
            document.body,
          )
        : null}
    </span>
  );
}

function InfoHint({ text, focusable = true }: { text: string; focusable?: boolean }) {
  return (
    <TooltipAnchor text={text} className="info-hint" focusable={focusable}>
      <CircleHelp size={13} />
    </TooltipAnchor>
  );
}

function ScoreBadge({ value }: { value: number }) {
  const text = `Fit score ${value} out of 100. Higher means a stronger scorer fit for this result lane.`;
  return (
    <TooltipAnchor text={text} className="score-tip" focusable={false}>
      <span className="score">
        <small>Fit</small>
        <strong>{value}</strong>
        <em>/100</em>
      </span>
    </TooltipAnchor>
  );
}

function pillClass(value?: string | null) {
  const key = (value ?? "").toLowerCase();
  if (key.includes("peak") || key.includes("high") || key.includes("risk")) return "pill hot";
  if (key.includes("drive") || key.includes("build")) return "pill lime";
  if (key.includes("ready") || key.includes("available")) return "pill cyan";
  if (key.includes("stale") || key.includes("unanalysed") || key.includes("outdated")) return "pill amber";
  return "pill";
}

function operationStartedMessage(action: string) {
  switch (action) {
    case "list_dj_playlists":
      return "Reading DJ playlists...";
    case "import_playlist":
    case "import_dj_playlist":
      return "Import started...";
    case "analyze_playlist":
      return "Analysis started...";
    case "run_analysis_worker":
      return "Worker running...";
    case "download_essentia_models":
      return "Downloading models...";
    case "prewarm_model_services":
      return "Prewarming model services...";
    default:
      return "Working...";
  }
}

function parseDJPlaylistNames(output?: string) {
  if (!output) return [];
  const seen = new Set<string>();
  const ansiEscapePattern = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, "g");
  const names = output
    .split(/\r?\n/)
    .map((line) => line.replace(ansiEscapePattern, "").trim())
    .map((line) => {
      const bulletMatch = line.match(/^[-*]\s+(.+?)\s*$/);
      if (bulletMatch) return bulletMatch[1].trim();
      const numberedMatch = line.match(/^\d+[.)]\s+(.+?)\s*$/);
      if (numberedMatch) return numberedMatch[1].trim();
      return "";
    })
    .filter(Boolean)
    .filter((name) => {
      if (seen.has(name)) return false;
      seen.add(name);
      return true;
    });
  return names;
}

function isListDJPlaylistsResult(result: ToolCommandResult | null) {
  return result?.command?.some((part) => part === "list-dj-playlists") ?? false;
}

function boundedPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

const ANALYSIS_WORKER_BATCH_LIMIT = 15;

export function App() {
  const queryClient = useQueryClient();
  const [selectedPlaylistId, setSelectedPlaylistId] = useState<string>(() => localStorage.getItem("cuemate.playlist") ?? "");
  const [currentTrackId, setCurrentTrackId] = useState<string>(() => localStorage.getItem("cuemate.current") ?? "");
  const [history, setHistory] = useState<string[]>(() => JSON.parse(localStorage.getItem("cuemate.history") ?? "[]") as string[]);
  const [target, setTarget] = useState("maintain");
  const [trackQuery, setTrackQuery] = useState("");
  const [selectedCandidate, setSelectedCandidate] = useState<LaneItem | null>(null);
  const [mobileTab, setMobileTab] = useState<"recommend" | "library" | "feedback" | "admin">("recommend");
  const [workMode, setWorkMode] = useState<"live" | "full">(() => (localStorage.getItem("cuemate.mode") === "full" ? "full" : "live"));
  const [lastToolResult, setLastToolResult] = useState<ToolCommandResult | null>(null);
  const [remotePairUrl, setRemotePairUrl] = useState("");
  const [remotePairExpiresAt, setRemotePairExpiresAt] = useState("");
  const [remotePairQr, setRemotePairQr] = useState("");
  const [operationMessage, setOperationMessage] = useState("");
  const [activeRunId, setActiveRunId] = useState("");
  const consumedPairTokenRef = useRef("");
  const chainedWorkerRunRef = useRef("");

  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000 });
  const readiness = useQuery({ queryKey: ["ready"], queryFn: api.readiness, refetchInterval: 15_000 });
  const metadata = useQuery({ queryKey: ["metadata"], queryFn: api.metadata, refetchInterval: 30_000 });
  const setupStatus = useQuery({ queryKey: ["setupStatus"], queryFn: api.setupStatus, refetchInterval: 15_000 });
  const playlists = useQuery({ queryKey: ["playlists"], queryFn: api.playlists });
  const tracks = useQuery({
    queryKey: ["playlistTracks", selectedPlaylistId, trackQuery],
    queryFn: () => api.playlistTracks(selectedPlaylistId, trackQuery),
    enabled: !!selectedPlaylistId,
  });
  const jobs = useQuery({
    queryKey: ["jobs", selectedPlaylistId],
    queryFn: () => api.jobs(selectedPlaylistId || undefined),
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? [];
      return items.some((job) => job.status === "pending" || job.status === "running") ? 3_000 : 20_000;
    },
  });
  const analysisStatus = useQuery({
    queryKey: ["analysisStatus", selectedPlaylistId],
    queryFn: () => api.playlistAnalysisStatus(selectedPlaylistId),
    enabled: !!selectedPlaylistId,
    refetchInterval: (query) => {
      const status = query.state.data;
      return status && (status.jobs.pending > 0 || status.jobs.running > 0) ? 3_000 : 15_000;
    },
  });
  const activeToolRun = useQuery({
    queryKey: ["toolRun", activeRunId],
    queryFn: () => api.toolRun(activeRunId),
    enabled: !!activeRunId,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 2_000 : false),
  });
  const feedback = useQuery({
    queryKey: ["feedback", selectedPlaylistId],
    queryFn: () => api.feedback(selectedPlaylistId),
    enabled: !!selectedPlaylistId,
  });
  const remoteStatus = useQuery({ queryKey: ["remoteStatus"], queryFn: api.remoteStatus, refetchInterval: 30_000 });
  const recommendations = useQuery({
    queryKey: ["recommendations", selectedPlaylistId, currentTrackId, target, history],
    queryFn: () =>
      api.recommendations({
        playlist_id: selectedPlaylistId,
        current_track_id: currentTrackId,
        target,
        history_track_ids: history.slice(-8),
        max_per_lane: 5,
      }),
    enabled: !!selectedPlaylistId && !!currentTrackId,
  });

  const playMutation = useMutation({
    mutationFn: (trackId: string) => {
      const eventId = recommendations.data?.meta.recommendation_event_id;
      if (!eventId) throw new Error("No recommendation event is available to record.");
      return api.played({ recommendation_event_id: eventId, chosen_track_id: trackId });
    },
    onSuccess: (_result, trackId) => {
      const nextHistory = currentTrackId ? [...history.slice(-7), currentTrackId] : history;
      setHistory(nextHistory);
      setCurrentTrackId(trackId);
      localStorage.setItem("cuemate.current", trackId);
      localStorage.setItem("cuemate.history", JSON.stringify(nextHistory));
      setSelectedCandidate(null);
      void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      void queryClient.invalidateQueries({ queryKey: ["feedback"] });
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const refreshMutation = useMutation({
    mutationFn: async (body?: { force?: boolean; analysis_mode?: "fast_pass" | "staged" | "full" }) => {
      const refresh = await api.refreshPlaylistAnalysis(selectedPlaylistId, body);
      const worker = await api.toolCommand({ action: "run_analysis_worker", limit: ANALYSIS_WORKER_BATCH_LIMIT });
      return { refresh, worker };
    },
    onMutate: (body) => setOperationMessage(body?.force ? "Force reanalysis queued. Starting worker..." : "Smart refresh queued. Starting worker..."),
    onSuccess: (result) => {
      setLastToolResult(result.worker);
      if (result.worker.run_id) setActiveRunId(result.worker.run_id);
      const queued = result.refresh.queued_count;
      setOperationMessage(
        queued
          ? `${queued} analysis job${queued === 1 ? "" : "s"} queued. Processing ${ANALYSIS_WORKER_BATCH_LIMIT} at a time.`
          : "No new jobs were queued. Worker checked for any pending analysis.",
      );
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["analysisStatus"] });
      void queryClient.invalidateQueries({ queryKey: ["playlistTracks"] });
      void queryClient.invalidateQueries({ queryKey: ["playlists"] });
    },
    onError: (error) => setOperationMessage(`Failed: ${error instanceof Error ? error.message : "analysis refresh failed"}`),
  });
  const removePlaylistMutation = useMutation({
    mutationFn: (playlistId: string) => api.removePlaylist(playlistId),
    onMutate: () => setOperationMessage("Removing playlist from CueMate..."),
    onSuccess: (result) => {
      setOperationMessage("Playlist removed from CueMate. Your music files were not touched.");
      queryClient.setQueryData<{ items: Playlist[] }>(["playlists"], (current) =>
        current ? { items: current.items.filter((item) => item.playlist_id !== result.playlist_id) } : current,
      );
      if (selectedPlaylistId === result.playlist_id) {
        setSelectedPlaylistId("");
        setCurrentTrackId("");
        setHistory([]);
        localStorage.removeItem("cuemate.playlist");
        localStorage.removeItem("cuemate.current");
        localStorage.removeItem("cuemate.history");
      }
      void queryClient.invalidateQueries({ queryKey: ["playlists"] });
      void queryClient.invalidateQueries({ queryKey: ["playlistTracks"] });
      void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      void queryClient.invalidateQueries({ queryKey: ["feedback"] });
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["analysisStatus"] });
    },
    onError: (error) => setOperationMessage(`Remove failed: ${error instanceof Error ? error.message : "unknown error"}`),
  });

  const remotePairMutation = useMutation({
    mutationFn: () => api.remotePairingToken(),
    onSuccess: async (result) => {
      setRemotePairUrl(result.pair_url);
      setRemotePairExpiresAt(result.expires_at);
      setRemotePairQr(await QRCode.toDataURL(result.pair_url, { margin: 1, width: 220 }));
    },
  });
  const remoteConsumePairMutation = useMutation({
    mutationFn: (token: string) => api.remotePair(token),
    onSuccess: () => {
      const url = new URL(window.location.href);
      url.searchParams.delete("pair_token");
      window.history.replaceState({}, "", url.toString());
      void queryClient.invalidateQueries({ queryKey: ["remoteStatus"] });
      void queryClient.invalidateQueries({ queryKey: ["health"] });
      void queryClient.invalidateQueries({ queryKey: ["playlists"] });
    },
  });

  const toolMutation = useMutation({
    mutationFn: (payload: ToolCommandRequest) => api.toolCommand(payload),
    onMutate: (payload) => setOperationMessage(operationStartedMessage(payload.action)),
    onSuccess: (result) => {
      setLastToolResult(result);
      if (result.run_id) setActiveRunId(result.run_id);
      setOperationMessage(result.mode === "background" ? "Started. Progress will update below." : "Completed.");
      void queryClient.invalidateQueries({ queryKey: ["playlists"] });
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      void queryClient.invalidateQueries({ queryKey: ["analysisStatus"] });
    },
    onError: (error) => setOperationMessage(`Failed: ${error instanceof Error ? error.message : "tool command failed"}`),
  });

  useEffect(() => {
    if (!activeToolRun.data || activeToolRun.data.status === "running") return;
    void queryClient.invalidateQueries({ queryKey: ["playlists"] });
    void queryClient.invalidateQueries({ queryKey: ["playlistTracks"] });
    void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    void queryClient.invalidateQueries({ queryKey: ["analysisStatus"] });
    void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    if (
      activeToolRun.data.status === "completed" &&
      selectedPlaylistId &&
      activeToolRun.data.run_id !== chainedWorkerRunRef.current &&
      activeToolRun.data.command?.some((part) => part === "run-analysis-worker")
    ) {
      chainedWorkerRunRef.current = activeToolRun.data.run_id;
      void queryClient
        .fetchQuery({
          queryKey: ["analysisStatus", selectedPlaylistId],
          queryFn: () => api.playlistAnalysisStatus(selectedPlaylistId),
        })
        .then((status) => {
          const pending = status.jobs.pending;
          const running = status.jobs.running;
          if (pending > 0 && running === 0) {
            setOperationMessage(`${pending} analysis job${pending === 1 ? "" : "s"} remaining. Starting next ${ANALYSIS_WORKER_BATCH_LIMIT}.`);
            toolMutation.mutate({ action: "run_analysis_worker", limit: ANALYSIS_WORKER_BATCH_LIMIT });
          }
        })
        .catch(() => undefined);
    }
  }, [activeToolRun.data, queryClient, selectedPlaylistId, toolMutation]);

  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get("pair_token");
    const showcase = setupStatus.data?.mode === "showcase" || setupStatus.data?.read_only === true;
    if (!showcase && token && consumedPairTokenRef.current !== token) {
      consumedPairTokenRef.current = token;
      remoteConsumePairMutation.mutate(token);
    }
  }, [remoteConsumePairMutation, setupStatus.data?.mode, setupStatus.data?.read_only]);

  const selectedPlaylist = playlists.data?.items.find((item) => item.playlist_id === selectedPlaylistId) ?? playlists.data?.items[0];
  const currentTrack = tracks.data?.items.find((item) => item.track_id === currentTrackId);
  const isShowcase = setupStatus.data?.mode === "showcase" || setupStatus.data?.read_only === true;

  const firstReadyTrack = tracks.data?.items.find((item) => item.analysis_state === "ready");

  const advanceCurrentTrack = useCallback(
    (trackId: string) => {
      const nextHistory = currentTrackId ? [...history.slice(-7), currentTrackId] : history;
      setHistory(nextHistory);
      setCurrentTrackId(trackId);
      localStorage.setItem("cuemate.current", trackId);
      localStorage.setItem("cuemate.history", JSON.stringify(nextHistory));
      setSelectedCandidate(null);
      void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
    [currentTrackId, history, queryClient],
  );

  const confirmCandidate = useCallback(
    (trackId: string) => {
      if (isShowcase) {
        advanceCurrentTrack(trackId);
        return;
      }
      playMutation.mutate(trackId);
    },
    [advanceCurrentTrack, isShowcase, playMutation],
  );

  useEffect(() => {
    if (!selectedPlaylistId && selectedPlaylist) {
      setSelectedPlaylistId(selectedPlaylist.playlist_id);
      localStorage.setItem("cuemate.playlist", selectedPlaylist.playlist_id);
    }
  }, [selectedPlaylist, selectedPlaylistId]);

  useEffect(() => {
    if (!currentTrackId && firstReadyTrack) {
      setCurrentTrackId(firstReadyTrack.track_id);
      localStorage.setItem("cuemate.current", firstReadyTrack.track_id);
    }
  }, [currentTrackId, firstReadyTrack]);

  useEffect(() => {
    if (isShowcase && mobileTab === "admin") {
      setMobileTab("recommend");
    }
  }, [isShowcase, mobileTab]);

  const shellClass = `app-shell mode-${workMode} tab-${mobileTab}`;

  return (
    <div className={shellClass}>
      <header className="topbar">
        <div>
          <p className="eyebrow">CueMate Engine</p>
          <h1>{isShowcase ? "Showcase Control" : "Performance Control"}</h1>
        </div>
        <div className="status-row">
          <div className="mode-switch" aria-label="Workspace mode">
            {(["live", "full"] as const).map((mode) => (
              <button
                key={mode}
                className={workMode === mode ? "active" : ""}
                onClick={() => {
                  setWorkMode(mode);
                  localStorage.setItem("cuemate.mode", mode);
                }}
              >
                {mode}
              </button>
            ))}
          </div>
          <StatusDot label="API" ok={health.data?.status === "ok"} loading={health.isLoading} />
          <StatusDot label="Scorer" ok={readiness.data?.status === "ready"} loading={readiness.isLoading} />
          <span className={metadata.data?.breaker_open ? "pill hot" : "pill cyan"}>
            {metadata.data?.breaker_open ? "breaker open" : "breaker calm"}
          </span>
        </div>
      </header>
      <div className="global-status-stack">
        {remoteConsumePairMutation.isPending || remoteConsumePairMutation.isSuccess || remoteConsumePairMutation.error ? (
          <div className="remote-pair-banner">
            {remoteConsumePairMutation.isPending ? "Pairing this phone..." : null}
            {remoteConsumePairMutation.isSuccess ? "Phone paired. CueMate is ready here." : null}
            {remoteConsumePairMutation.error ? `Pairing failed: ${remoteConsumePairMutation.error.message}` : null}
          </div>
        ) : null}
        <SetupStatusBanner status={setupStatus.data} />
        {!isShowcase ? <OperationBanner message={operationMessage} run={activeToolRun.data} status={analysisStatus.data} /> : null}
      </div>

      <aside className="library-pane panel mobile-library">
        <PaneTitle icon={<Library />} title="Library" action={`${playlists.data?.items.length ?? 0} playlists`} />
        {playlists.isLoading ? (
          <SkeletonRows count={4} />
        ) : (playlists.data?.items.length ?? 0) === 0 ? (
          <div className="empty-library">
            <strong>No playlists yet</strong>
            <span>{isShowcase ? "This showcase snapshot does not include any playlists yet." : "Import local files or a DJ library from Full Mode to get started."}</span>
            {!isShowcase ? (
              <button
                className="wide-action secondary"
                onClick={() => {
                  setWorkMode("full");
                  setMobileTab("admin");
                  localStorage.setItem("cuemate.mode", "full");
                }}
              >
                Open import tools
              </button>
            ) : null}
          </div>
        ) : (
          <PlaylistList
            playlists={playlists.data?.items ?? []}
            selectedId={selectedPlaylistId}
            onSelect={(id) => {
              setSelectedPlaylistId(id);
              localStorage.setItem("cuemate.playlist", id);
              setCurrentTrackId("");
              setHistory([]);
              localStorage.removeItem("cuemate.current");
              localStorage.removeItem("cuemate.history");
            }}
          />
        )}
        <div className="searchbox">
          <Search size={16} />
          <input value={trackQuery} onChange={(event) => setTrackQuery(event.target.value)} placeholder="Search tracks" />
        </div>
        {tracks.isLoading ? (
          <SkeletonRows count={6} />
        ) : (
          <TrackList
            tracks={tracks.data?.items ?? []}
            currentTrackId={currentTrackId}
            onSelect={(track) => {
              setCurrentTrackId(track.track_id);
              localStorage.setItem("cuemate.current", track.track_id);
            }}
          />
        )}
      </aside>

      <main className="recommend-pane mobile-recommend">
        <section className="hero-panel">
          <div>
            <p className="eyebrow">{selectedPlaylist?.name ?? "No playlist"}</p>
            <h2>{titleFor(currentTrack)}</h2>
            <div className="meta-line">
              <span>{currentTrack?.bpm ? `${currentTrack.bpm.toFixed(1)} BPM` : "BPM pending"}</span>
              <span>{currentTrack?.key ?? "Key pending"}</span>
              <span>{currentTrack?.intensity_band ?? "No band"}</span>
            </div>
          </div>
          <div className="confidence-ring">
            <span>{pct(recommendations.data?.recommendation_confidence)}</span>
            <small>confidence</small>
          </div>
        </section>

        <section className="target-strip">
          <span className="strip-label">
            Target mode
            <InfoHint text="Target mode is your intent for the next transition. The lanes below are grouped recommendation results." focusable={false} />
          </span>
          {targets.map((item) => (
            <button
              key={item}
              className={item === target ? "seg active" : "seg"}
              aria-label={`${item}: ${targetDescriptions[item]}`}
              onClick={() => setTarget(item)}
            >
              <span>{item}</span>
              <InfoHint text={targetDescriptions[item]} focusable={false} />
            </button>
          ))}
        </section>

        <RecommendationBoard
          response={recommendations.data}
          loading={recommendations.isLoading}
          error={recommendations.error}
          selected={selectedCandidate}
          onSelect={setSelectedCandidate}
        />
      </main>

      <aside className="detail-pane panel mobile-feedback">
        <CurrentTrackInfo track={currentTrack} playlistId={selectedPlaylist?.playlist_id} />
        {workMode === "live" ? (
          <>
            <LiveCandidatePanel
              candidate={selectedCandidate}
              response={recommendations.data}
              onConfirm={confirmCandidate}
              confirming={!isShowcase && playMutation.isPending}
              showcaseMode={isShowcase}
            />
            <CandidateSignalPanel candidate={selectedCandidate} playlistId={selectedPlaylist?.playlist_id} />
          </>
        ) : (
          <>
            <CandidateDetail
              candidate={selectedCandidate}
              response={recommendations.data}
              onConfirm={confirmCandidate}
              confirming={!isShowcase && playMutation.isPending}
              showcaseMode={isShowcase}
            />
            <FullCandidateAnalysisPanel candidate={selectedCandidate} playlistId={selectedPlaylist?.playlist_id} />
            <FeedbackPanel feedback={feedback.data} />
          </>
        )}
      </aside>

      {workMode === "full" && !isShowcase ? (
        <aside className="admin-pane panel mobile-admin">
          <FullToolsPanel
            playlist={selectedPlaylist}
            candidate={selectedCandidate}
            playlistId={selectedPlaylist?.playlist_id}
            jobs={jobs.data?.items ?? []}
            analysisStatus={analysisStatus.data}
            analysisStatusLoading={analysisStatus.isLoading}
            onSmartRefresh={() => refreshMutation.mutate({ analysis_mode: "staged" })}
            onForceRefresh={() => refreshMutation.mutate({ analysis_mode: "staged", force: true })}
            queueBusy={refreshMutation.isPending}
            queueResult={refreshMutation.data?.refresh.queued_count}
            onRemovePlaylist={(id) => removePlaylistMutation.mutate(id)}
            removeBusy={removePlaylistMutation.isPending}
            onRunWorker={() => toolMutation.mutate({ action: "run_analysis_worker", limit: ANALYSIS_WORKER_BATCH_LIMIT })}
            onTool={(payload) => toolMutation.mutate(payload)}
            toolBusy={toolMutation.isPending}
            toolResult={lastToolResult}
            toolError={toolMutation.error}
            toolRun={activeToolRun.data}
            setupStatus={setupStatus.data}
            remoteStatus={remoteStatus.data}
            remoteStatusLoading={remoteStatus.isLoading}
            remotePairUrl={remotePairUrl}
            remotePairQr={remotePairQr}
            remotePairExpiresAt={remotePairExpiresAt}
            onGenerateRemotePair={() => remotePairMutation.mutate()}
            remotePairBusy={remotePairMutation.isPending}
            remotePairError={remotePairMutation.error}
          />
        </aside>
      ) : null}

      <nav className="mobile-nav">
        <button className={mobileTab === "recommend" ? "active" : ""} onClick={() => setMobileTab("recommend")}>
          <Radar size={18} /> Recommend
        </button>
        <button className={mobileTab === "library" ? "active" : ""} onClick={() => setMobileTab("library")}>
          <ListMusic size={18} /> Library
        </button>
        <button className={mobileTab === "feedback" ? "active" : ""} onClick={() => setMobileTab("feedback")}>
          <BarChart3 size={18} /> Feedback
        </button>
        <button className={mobileTab === "admin" ? "active" : ""} onClick={() => setMobileTab("admin")} disabled={workMode !== "full" || isShowcase}>
          <Settings size={18} /> Full
        </button>
      </nav>
    </div>
  );
}

function StatusDot({ label, ok, loading }: { label: string; ok: boolean; loading: boolean }) {
  return (
    <span className={ok ? "status ok" : "status bad"}>
      <CircleDot size={14} />
      {loading ? `${label}...` : label}
    </span>
  );
}

function SetupStatusBanner({ status }: { status?: SetupStatus }) {
  if (!status?.available) return null;
  if (status.mode === "showcase" || status.read_only) {
    return (
      <div className="setup-banner">
        <Eye size={16} />
        <span>Showcase mode is read-only. Browse the curated library and try live recommendations without importing files.</span>
      </div>
    );
  }
  const isBlocked = status.status === "blocked" || status.status === "failed";
  const modelPending = status.core_ready && (!status.docker_ready || !status.model_ready);
  if (!isBlocked && !modelPending) return null;
  const message = isBlocked
    ? status.message || "CueMate setup needs attention before all features are available."
    : "CueMate is open. Docker/model setup is still pending, so full analysis may be unavailable until setup resumes.";
  return (
    <div className={isBlocked ? "setup-banner blocked" : "setup-banner"}>
      <AlertTriangle size={16} />
      <span>{message}</span>
      {status.log_dir ? <small>Logs: {status.log_dir}</small> : null}
    </div>
  );
}

function OperationBanner({ message, run, status }: { message: string; run?: ToolRunStatus; status?: PlaylistAnalysisStatus }) {
  const activeJobs = (status?.jobs.pending ?? 0) + (status?.jobs.running ?? 0);
  const total = status?.total_tracks ?? 0;
  const ready = status?.ready_tracks ?? 0;
  const percent = total ? Math.round((ready / total) * 100) : 0;
  const runMessage =
    run?.status === "running"
      ? "Analysis worker is running in the background."
      : run?.status === "completed"
        ? "Analysis worker finished."
        : run?.status === "failed"
          ? `Background tool failed${run.error ? `: ${run.error}` : "."}`
          : "";
  if (!message && !runMessage && !activeJobs) return null;
  return (
    <div className={run?.status === "failed" ? "operation-banner danger" : activeJobs ? "operation-banner active" : "operation-banner"}>
      <span>{runMessage || message || "Analysis work is in progress."}</span>
      {activeJobs ? (
        <div className="operation-progress" aria-label={`Analysis progress ${ready} of ${total} ready`}>
          <div className="operation-progress-head">
            <strong>{percent}% current</strong>
            <small>
              {ready}/{total} ready · {status?.jobs.running ?? 0} running · {status?.jobs.pending ?? 0} queued
            </small>
          </div>
          <div className="progress-track animated">
            <span style={{ width: `${boundedPercent(percent)}%` }} />
          </div>
        </div>
      ) : null}
      {run?.log_path ? <small>Log: {run.log_path}</small> : null}
    </div>
  );
}

function PaneTitle({ icon, title, action }: { icon: React.ReactNode; title: string; action?: string }) {
  return (
    <div className="pane-title">
      <span>
        {icon}
        {title}
      </span>
      {action ? <small>{action}</small> : null}
    </div>
  );
}

function PlaylistList({ playlists, selectedId, onSelect }: { playlists: Playlist[]; selectedId: string; onSelect: (id: string) => void }) {
  return (
    <div className="playlist-list">
      {playlists.map((playlist) => (
        <button key={playlist.playlist_id} className={playlist.playlist_id === selectedId ? "playlist active" : "playlist"} onClick={() => onSelect(playlist.playlist_id)}>
          <span>{playlist.name}</span>
          <small>
            {playlist.track_count_analyzed}/{playlist.track_count} ready
          </small>
        </button>
      ))}
    </div>
  );
}

function SkeletonRows({ count }: { count: number }) {
  return (
    <div className="skeleton-list" aria-label="Loading">
      {Array.from({ length: count }).map((_, index) => (
        <span key={index} />
      ))}
    </div>
  );
}

function TrackList({ tracks, currentTrackId, onSelect }: { tracks: Track[]; currentTrackId: string; onSelect: (track: Track) => void }) {
  return (
    <div className="track-list">
      {tracks.map((track) => (
        <button key={track.track_id} className={track.track_id === currentTrackId ? "track active" : "track"} onClick={() => onSelect(track)}>
          <span className="track-index">{track.position || "-"}</span>
          <span className="track-main">
            <strong>{track.title || track.track_id}</strong>
            <small>{track.artist || "Unknown artist"}</small>
          </span>
          <span className={pillClass(track.analysis_state)}>{track.analysis_state}</span>
        </button>
      ))}
    </div>
  );
}

function RecommendationBoard({
  response,
  loading,
  error,
  selected,
  onSelect,
}: {
  response?: RecommendationResponse;
  loading: boolean;
  error: Error | null;
  selected: LaneItem | null;
  onSelect: (item: LaneItem) => void;
}) {
  if (loading) return <div className="empty-state">Scoring the room...</div>;
  if (error) return <div className="empty-state danger">{error.message}</div>;
  if (!response) return <div className="empty-state">Choose a ready track to open the lanes.</div>;
  if (response.recommendations_status !== "available") {
    return (
      <div className="empty-state danger">
        <AlertTriangle />
        <strong>{response.recommendations_status}</strong>
        <span>{response.meta.status_note ?? response.meta.fallback_note ?? "Recommendations are unavailable."}</span>
      </div>
    );
  }

  return (
    <section className="lanes">
      {response.lane_order.map((lane) => {
        const group = response.lanes[lane];
        return (
          <article key={lane} className="lane">
            <div className="lane-head">
              <div>
                <small>Result lane</small>
                <h3>
                  {lane}
                  <InfoHint text={helpFor(laneDescriptions, lane, "Result lane for this recommendation group.")} />
                </h3>
              </div>
              <span>{group.items.length}</span>
            </div>
            {group.items.length === 0 ? <p className="empty-lane">{group.empty_reason}</p> : null}
            {group.items.map((item) => (
              <div
                key={`${lane}-${item.track_id}`}
                className={selected?.track_id === item.track_id ? "candidate active" : "candidate"}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(item)}
                onKeyDown={(event) => {
                  if (event.currentTarget !== event.target) return;
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(item);
                  }
                }}
              >
                <ScoreBadge value={Math.round(item.score * 100)} />
                <span className="candidate-main">
                  <TooltipAnchor text={`Track title: ${item.title || item.track_id}`} className="candidate-title-tip" focusable={false}>
                    <strong>{item.title || item.track_id}</strong>
                  </TooltipAnchor>
                  {item.artist ? (
                    <TooltipAnchor text={`Artist: ${item.artist}`} className="candidate-title-tip" focusable={false}>
                      <span className="candidate-artist">{item.artist}</span>
                    </TooltipAnchor>
                  ) : null}
                  <TooltipAnchor text="Tempo/key summary for the transition from the current base track." className="candidate-title-tip" focusable={false}>
                    <span className="candidate-stats">{item.tempo_key.tempo_text || item.move}</span>
                  </TooltipAnchor>
                  <div className="candidate-abs-meta">
                    <span>{candidateBpmText(item)}</span>
                    <span>{candidateKeyText(item)}</span>
                  </div>
                  <TooltipAnchor text={metricDescriptions.Risk} className="candidate-title-tip" focusable={false}>
                    <span className={pillClass(item.risk)}>{item.risk}</span>
                  </TooltipAnchor>
                </span>
              </div>
            ))}
          </article>
        );
      })}
    </section>
  );
}

function CandidateDetail({
  candidate,
  response,
  onConfirm,
  confirming,
  showcaseMode = false,
}: {
  candidate: LaneItem | null;
  response?: RecommendationResponse;
  onConfirm: (trackId: string) => void;
  confirming: boolean;
  showcaseMode?: boolean;
}) {
  return (
    <section className="detail-block">
      <PaneTitle icon={<Sparkles />} title="Candidate" action={showcaseMode ? "showcase" : response?.meta.recommendation_event_id ? "event armed" : "preview"} />
      {!candidate ? (
        <p className="muted">Select a recommendation to inspect why it works, what to watch, and how to hand it off.</p>
      ) : (
        <>
          <h2>{candidate.artist ? `${candidate.artist} - ${candidate.title}` : candidate.title}</h2>
          <div className="candidate-meta-strip">
            <span>{candidateBpmText(candidate)}</span>
            <span>{candidateKeyText(candidate)}</span>
            <span>{stringValueFromUnknown(candidate.candidate_features.intensity_band) || "band unknown"}</span>
          </div>
          <div className="metric-grid">
            <Metric label="Fit score" value={`${Math.round(candidate.score * 100)}/100`} hint={metricDescriptions["Fit score"]} />
            <Metric label="Move" value={candidate.move} hint={metricDescriptions.Move} />
            <Metric label="Risk" value={candidate.risk} hint={metricDescriptions.Risk} />
            <Metric label="Confidence" value={pct(candidate.move_confidence)} hint={metricDescriptions.Confidence} />
          </div>
          <button className="wide-action confirm-next" disabled={confirming || (!showcaseMode && !response?.meta.recommendation_event_id)} onClick={() => onConfirm(candidate.track_id)}>
            <Check size={16} />
            {confirming ? "Setting next song..." : showcaseMode ? "Use as current track" : "Set as next song and make current"}
          </button>
          <p className="action-note">
            {showcaseMode
              ? "Advances the demo session in this browser only; the public showcase database stays unchanged."
              : "Records this recommendation as played, adds the previous base to history, and rescans from this track."}
          </p>
          <NoteList title="Reasons" notes={candidate.reasons} />
          <NoteList title="Watchouts" notes={candidate.watchouts} />
          <NoteList title="Handoff" notes={candidate.explanation.handoff?.notes ?? []} />
        </>
      )}
    </section>
  );
}

function CurrentTrackInfo({ track, playlistId }: { track?: Track; playlistId?: string }) {
  const [expanded, setExpanded] = useState(false);
  const detailId = useId();
  const features = useQuery({
    queryKey: ["track-features", playlistId, track?.track_id],
    queryFn: () => api.trackFeatures(playlistId ?? "", track?.track_id ?? ""),
    enabled: expanded && Boolean(playlistId && track?.track_id),
  });

  useEffect(() => {
    setExpanded(false);
  }, [track?.track_id]);

  const insight = useMemo(() => {
    if (!track) return null;
    const detail = features.data;
    const basic = detail?.basic ?? {};
    const absolute = detail?.absolute ?? {};
    const semantic = detail?.semantic ?? {};
    const relative = detail?.relative ?? {};
    const bpm = firstNumber(basic.bpm, track.bpm);
    const key = stringValueFromUnknown(basic.key) || track.key || "key pending";
    const energy = firstNumber(relative.energy_rel, absolute.energy_abs, semantic.energy_essentia_fused);
    const bass = firstNumber(relative.bass_rel, absolute.bass_abs);
    const drums = firstNumber(relative.drums_rel, absolute.drums_abs);
    const groove = firstNumber(relative.groove_rel, absolute.groove_abs);
    const vocals = firstNumber(relative.vocals_rel, absolute.vocals_abs);
    const danceability = maybeNumber(semantic.danceability_abs);
    const drive = firstNumber(semantic.arousal_abs, absolute.energy_sustained);
    const moodTone = maybeNumber(semantic.valence_abs);
    const mood = dominantMood(semantic);

    return {
      bpm,
      key,
      intensity: stringValueFromUnknown(relative.intensity_band) || track.intensity_band || "band pending",
      glance: [
        {
          label: "Tempo",
          value: bpm == null ? "pending" : `${bpm.toFixed(1)} BPM`,
          detail: "absolute BPM for the current base track",
          hint: "The actual analyzed tempo of the track currently driving recommendations.",
        },
        {
          label: "Key",
          value: key,
          detail: "ocelot/camelot key used for harmonic decisions",
          hint: "The analyzed musical key for harmonic compatibility checks.",
        },
        {
          label: "Energy",
          value: percentOrMissing(energy),
          detail: energyLevelCopy(energy),
          hint: "Playlist-relative energy where available, with analyzer energy as fallback.",
        },
        {
          label: "Mood",
          value: mood.label,
          detail: mood.detail,
          hint: "The dominant semantic mood detected during full analysis.",
        },
      ],
      groups: [
        {
          title: "Current Track Body",
          note: "The sound profile this recommendation round is anchored to.",
          metrics: [
            percentBarMetric("Energy", energy, "How much room pressure the current track carries."),
            percentBarMetric("Bass", bass, "Low-end weight in the current base track."),
            percentBarMetric("Drums", drums, "Percussive strength in the current base track."),
            percentBarMetric("Groove", groove, "Groove or rhythmic drive in the current base track."),
            percentBarMetric("Vocals", vocals, "How much vocal content the current base track carries."),
          ],
        },
        {
          title: "Movement + Mood",
          note: "Semantic values that explain the current floor feel.",
          metrics: [
            percentBarMetric("Danceability", danceability, "How strongly the model hears dance-floor movement."),
            percentBarMetric("Drive", drive, "Activation and intensity separate from BPM."),
            percentBarMetric("Mood tone", moodTone, "Semantic valence: emotional tone from darker to more positive. This is not bass or timbre brightness."),
            percentBarMetric("Party", semantic.mood_party_abs, "Party or club-forward character."),
            percentBarMetric("Relaxed", semantic.mood_relaxed_abs, "Laid-back or smoother character."),
          ],
        },
      ],
    };
  }, [features.data, track]);

  return (
    <section className={expanded ? "detail-block current-track-card expanded" : "detail-block current-track-card"}>
      <PaneTitle icon={<Radar />} title="Current Track" action={expanded ? "open" : "base"} />
      {!track ? (
        <p className="muted">Choose a ready track to set the base.</p>
      ) : (
        <>
          <button className="current-track-toggle" type="button" aria-expanded={expanded} aria-controls={detailId} onClick={() => setExpanded((value) => !value)}>
            <span>
              <strong>{track.artist ? `${track.artist} - ${track.title}` : track.title || track.track_id}</strong>
              <small>{expanded ? "Hide base-track data" : "Show tempo, energy, mood, and analysis"}</small>
            </span>
            <ChevronDown className="disclosure-icon" size={18} />
          </button>
          {expanded ? (
            <div className="current-track-details" id={detailId}>
              <div className="candidate-meta-strip">
                <span>{insight?.bpm == null ? "BPM pending" : `${insight.bpm.toFixed(1)} BPM`}</span>
                <span>{insight?.key ?? "key pending"}</span>
                <span>{insight?.intensity ?? "band pending"}</span>
                <span>{track.analysis_state}</span>
              </div>
              {track.role_hints.length ? (
                <div className="mini-tags">
                  {track.role_hints.slice(0, 3).map((hint) => (
                    <span key={hint}>{labelize(hint)}</span>
                  ))}
                </div>
              ) : null}
              {features.isFetching ? <p className="action-note">Loading current track analysis...</p> : null}
              {features.error ? <p className="action-note">Current track analysis could not be loaded; showing playlist row values only.</p> : null}
              {insight ? (
                <>
                  <div className="current-glance-grid">
                    {insight.glance.map((item) => (
                      <InsightTile key={item.label} {...item} />
                    ))}
                  </div>
                  <div className="analysis-groups current-analysis-groups">
                    {insight.groups.map((group) => (
                      <AnalysisBarGroup key={group.title} title={group.title} note={group.note} metrics={group.metrics} />
                    ))}
                  </div>
                </>
              ) : null}
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function LiveCandidatePanel({
  candidate,
  response,
  onConfirm,
  confirming,
  showcaseMode = false,
}: {
  candidate: LaneItem | null;
  response?: RecommendationResponse;
  onConfirm: (trackId: string) => void;
  confirming: boolean;
  showcaseMode?: boolean;
}) {
  return (
    <section className="detail-block live-candidate-card">
      <PaneTitle icon={<Sparkles />} title="Selected Next" action={showcaseMode ? "showcase" : response?.meta.recommendation_event_id ? "ready" : "preview"} />
      {!candidate ? (
        <p className="muted">Select a recommendation to inspect the next-song read.</p>
      ) : (
        <>
          <h2>{candidate.artist ? `${candidate.artist} - ${candidate.title}` : candidate.title}</h2>
          <div className="candidate-meta-strip">
            <span>{candidateBpmText(candidate)}</span>
            <span>{candidateKeyText(candidate)}</span>
            <span>{candidate.move}</span>
          </div>
          <div className="live-score-strip">
            <span>
              <small>Fit</small>
              <strong>{Math.round(candidate.score * 100)}</strong>
            </span>
            <span>
              <small>Risk</small>
              <strong>{candidate.risk}</strong>
            </span>
            <span>
              <small>Confidence</small>
              <strong>{pct(candidate.move_confidence)}</strong>
            </span>
          </div>
          <button className="wide-action confirm-next" disabled={confirming || (!showcaseMode && !response?.meta.recommendation_event_id)} onClick={() => onConfirm(candidate.track_id)}>
            <Check size={16} />
            {confirming ? "Setting next song..." : showcaseMode ? "Use as current track" : "Set as next song and make current"}
          </button>
        </>
      )}
    </section>
  );
}

function CandidateSignalPanel({ candidate, playlistId }: { candidate: LaneItem | null; playlistId?: string }) {
  const features = useQuery({
    queryKey: ["track-features", playlistId, candidate?.track_id],
    queryFn: () => api.trackFeatures(playlistId ?? "", candidate?.track_id ?? ""),
    enabled: Boolean(playlistId && candidate?.track_id),
  });

  const insight = useMemo(() => {
    if (!candidate) return null;
    const detail = features.data;
    const candidateRel = candidate.candidate_features;
    const relative = detail?.relative ?? {};
    const semantic = detail?.semantic ?? {};
    const transition = candidate.transition_features ?? {};

    const energyRel = firstNumber(relative.energy_rel, candidateRel.energy_rel);
    const bassRel = firstNumber(relative.bass_rel, candidateRel.bass_rel);
    const deltaEnergy = maybeNumber(transition.delta_energy_rel);
    const deltaBass = maybeNumber(transition.delta_bass_rel);
    const danceability = maybeNumber(semantic.danceability_abs);
    const arousal = maybeNumber(semantic.arousal_abs);
    const moodTone = maybeNumber(semantic.valence_abs);
    const mood = dominantMood(semantic);
    const bpmDistance = maybeNumber(transition.effective_bpm_distance);
    const keyFit = stringValueFromUnknown(transition.key_compat_label) || candidate.tempo_key.key_text || "unknown";
    const vocalPhrase = vocalHandoffPhrase(maybeNumber(transition.current_vocals_rel), maybeNumber(transition.candidate_vocals_rel));
    const tempoPhrase = candidate.tempo_key.tempo_text || (bpmDistance == null ? "unknown" : `${bpmDistance.toFixed(1)} BPM away`);
    const energyPhrase = deltaEnergy == null ? "unknown" : signedPercent(deltaEnergy);

    const drivers = Object.entries(candidate.component_scores ?? {})
      .map(([key, score]) => ({
        key,
        label: labelize(key),
        score,
        weight: candidate.weights_used?.[key],
        confidence: candidate.component_confidences?.[key],
        strength: score * (candidate.weights_used?.[key] ?? 1),
      }))
      .sort((a, b) => b.strength - a.strength || componentSort(a.key) - componentSort(b.key))
      .slice(0, 3);

    return {
      summary: buildDJReadSummary({ energyChange: deltaEnergy, risk: candidate.risk, tempo: tempoPhrase, keyFit, mood }),
      hero: [
        {
          label: "Energy move",
          value: energyPhrase,
          detail: energyMoveCopy(deltaEnergy, candidate.move),
          hint: "How this pick changes room pressure from the current base track.",
        },
        {
          label: "Mix safety",
          value: `${riskToSafety(candidate.risk_score)}%`,
          detail: `${candidate.risk} risk / ${keyFit}`,
          hint: "A quick read of transition difficulty using risk, harmonic fit, and scorer confidence.",
        },
        {
          label: "Crowd feel",
          value: mood.label,
          detail: mood.detail,
          hint: "The musical character this candidate brings into the next moment.",
        },
      ],
      mix: [
        {
          label: "Tempo",
          value: tempoPhrase,
          detail: tempoDecisionCopy(bpmDistance),
          hint: "How much BPM adjustment or ratio-aware mixing the handoff needs.",
        },
        {
          label: "Harmony",
          value: keyFit,
          detail: keyDecisionCopy(maybeNumber(transition.key_distance)),
          hint: "Whether the key relationship is likely to sound smooth or tense.",
        },
        {
          label: "Low end",
          value: deltaBass == null ? "unknown" : signedPercent(deltaBass),
          detail: bassDecisionCopy(deltaBass, bassRel),
          hint: "Whether the next track adds, removes, or preserves bass pressure.",
        },
        {
          label: "Vocals",
          value: vocalPhrase.value,
          detail: vocalPhrase.detail,
          hint: "Whether vocal content might clash across the transition.",
        },
      ],
      room: [
        {
          label: "Candidate energy",
          value: percentOrMissing(energyRel),
          detail: `${stringValueFromUnknown(relative.intensity_band) || stringValueFromUnknown(candidateRel.intensity_band) || "band unknown"} / ${energyLevelCopy(energyRel)}`,
          hint: "Where this track sits in this playlist's energy range.",
        },
        {
          label: "Danceability",
          value: percentOrMissing(danceability),
          detail: danceability == null ? "run full semantic analysis to unlock" : danceabilityCopy(danceability),
          hint: "How strongly the semantic model hears dance-floor movement.",
        },
        {
          label: "Drive",
          value: percentOrMissing(arousal),
          detail: arousal == null ? "semantic value missing" : arousalCopy(arousal),
          hint: "How activated or intense the track feels, separate from BPM.",
        },
        {
          label: "Mood tone",
          value: percentOrMissing(moodTone),
          detail: moodTone == null ? "semantic value missing" : valenceCopy(moodTone),
          hint: "Semantic valence: emotional tone from darker to more positive. This is separate from bassline darkness or low-end weight.",
        },
      ],
      drivers,
      loading: features.isFetching,
    };
  }, [candidate, features.data, features.isFetching]);

  return (
    <section className="detail-block">
      <PaneTitle icon={<BarChart3 />} title="DJ Read" action={candidate ? "selected track" : "waiting"} />
      {!candidate ? (
        <p className="muted">Select a recommendation to see candidate-specific values here.</p>
      ) : (
        <>
          <p className="context-note">
            {insight?.loading ? "Loading full track analysis..." : insight?.summary}
          </p>
          {features.error ? <p className="action-note">Full analysis details could not be loaded; showing scorer-returned fields only.</p> : null}
          {insight ? (
            <>
              <div className="dj-read-hero">
                {insight.hero.map((item) => (
                  <InsightTile key={item.label} {...item} featured />
                ))}
              </div>
              <InsightSection title="Can I mix it cleanly?" note="The handoff checks that matter in the booth." items={insight.mix} />
              <InsightSection title="What happens to the room?" note="Energy and feel, not just raw score." items={insight.room} />
              <ScoreDriverList drivers={insight.drivers} />
            </>
          ) : null}
        </>
      )}
    </section>
  );
}

type InsightItem = {
  label: string;
  value: string;
  detail: string;
  hint: string;
};

function InsightSection({ title, note, items }: { title: string; note: string; items: InsightItem[] }) {
  return (
    <section className="insight-section">
      <div className="insight-section-title">
        <h3>{title}</h3>
        <p>{note}</p>
      </div>
      <div className="insight-grid">
        {items.map((item) => (
          <InsightTile key={item.label} {...item} />
        ))}
      </div>
    </section>
  );
}

function InsightTile({ label, value, detail, hint, featured = false }: InsightItem & { featured?: boolean }) {
  return (
    <div className={featured ? "insight-tile featured" : "insight-tile"}>
      <span>
        {label}
        <InfoHint text={hint} />
      </span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

type ScoreDriver = {
  key: string;
  label: string;
  score: number;
  weight?: number;
  confidence?: number;
  strength: number;
};

function ScoreDriverList({ drivers }: { drivers: ScoreDriver[] }) {
  if (drivers.length === 0) return null;
  return (
    <section className="insight-section">
      <div className="insight-section-title">
        <h3>Why it scored well</h3>
        <p>The top scorer components behind this recommendation.</p>
      </div>
      <div className="driver-list">
        {drivers.map((driver) => (
          <div key={driver.key} className="driver-row">
            <span>
              {driver.label}
              <InfoHint text={helpFor(weightDescriptions, driver.key, "Scoring component used by the recommendation engine.")} />
            </span>
            <strong>{Math.round(clamp01(driver.score) * 100)}%</strong>
            <div className="driver-bar">
              <i style={{ width: `${Math.round(clamp01(driver.score) * 100)}%` }} />
            </div>
            <small>
              weight {driver.weight == null ? "n/a" : `${Math.round(driver.weight * 100)}%`} / confidence {driver.confidence == null ? "n/a" : `${Math.round(driver.confidence * 100)}%`}
            </small>
          </div>
        ))}
      </div>
    </section>
  );
}

function FullCandidateAnalysisPanel({ candidate, playlistId }: { candidate: LaneItem | null; playlistId?: string }) {
  const features = useQuery({
    queryKey: ["track-features", playlistId, candidate?.track_id],
    queryFn: () => api.trackFeatures(playlistId ?? "", candidate?.track_id ?? ""),
    enabled: Boolean(playlistId && candidate?.track_id),
  });

  const groups = useMemo(() => {
    if (!candidate) return [];
    const detail = features.data;
    const candidateRel = candidate.candidate_features;
    const relative = detail?.relative ?? {};
    const absolute = detail?.absolute ?? {};
    const semantic = detail?.semantic ?? {};
    const transition = candidate.transition_features ?? {};
    const components = Object.entries(candidate.component_scores ?? {}).sort(([a], [b]) => componentSort(a) - componentSort(b));

    return [
      {
        title: "Relative Playlist Profile",
        note: "Where this track sits compared with the selected playlist.",
        metrics: [
          percentBarMetric("Energy", firstNumber(relative.energy_rel, candidateRel.energy_rel), "Playlist-relative energy."),
          percentBarMetric("Bass", firstNumber(relative.bass_rel, candidateRel.bass_rel), "Playlist-relative low-end weight."),
          percentBarMetric("Drums", firstNumber(relative.drums_rel, candidateRel.drums_rel), "Playlist-relative drum/percussive strength."),
          percentBarMetric("Groove", firstNumber(relative.groove_rel, candidateRel.groove_rel), "Playlist-relative groove estimate."),
          percentBarMetric("Vocals", firstNumber(relative.vocals_rel, candidateRel.vocals_rel), "Playlist-relative vocal content; missing means unknown."),
        ],
      },
      {
        title: "Mood And Crowd Feel",
        note: "Semantic model readings from full analysis.",
        metrics: [
          percentBarMetric("Danceability", semantic.danceability_abs, "Dance-floor movement estimate."),
          percentBarMetric("Drive", semantic.arousal_abs, "Activation/intensity separate from BPM."),
          percentBarMetric("Mood tone", semantic.valence_abs, "Semantic valence: emotional tone from darker to more positive. This is not bass or timbre brightness."),
          percentBarMetric("Party", semantic.mood_party_abs, "Party/club mood strength."),
          percentBarMetric("Aggressive", semantic.mood_aggressive_abs, "Forceful or aggressive character."),
          percentBarMetric("Relaxed", semantic.mood_relaxed_abs, "Relaxed or laid-back character."),
          percentBarMetric("Semantic energy", semantic.energy_essentia_fused, "Fused semantic energy reading."),
        ],
      },
      {
        title: "Audio Body",
        note: "Absolute analyzer readings from the file.",
        metrics: [
          percentBarMetric("Energy", absolute.energy_abs, "Canonical absolute energy."),
          percentBarMetric("Sustained", absolute.energy_sustained, "Sustained energy over the track."),
          percentBarMetric("Peak", absolute.energy_peak, "Peak energy moments."),
          percentBarMetric("Bass", absolute.bass_abs, "Absolute low-end energy."),
          percentBarMetric("Drums", absolute.drums_abs, "Absolute drum/percussive content."),
          percentBarMetric("Groove", absolute.groove_abs, "Absolute groove estimate."),
          percentBarMetric("Vocals", absolute.vocals_abs, "Absolute vocal presence; missing means unknown."),
        ],
      },
      {
        title: "Transition From Current",
        note: "How this candidate changes the handoff.",
        metrics: [
          signedBarMetric("Energy change", transition.delta_energy_rel, "Positive builds pressure; negative creates space."),
          signedBarMetric("Bass change", transition.delta_bass_rel, "Positive adds low end; negative lightens the handoff."),
          percentBarMetric("Current vocals", transition.current_vocals_rel, "Vocal content in the current base track."),
          percentBarMetric("Candidate vocals", transition.candidate_vocals_rel, "Vocal content in the selected candidate."),
          percentBarMetric("Current low end", transition.current_outro_low_end, "Low-end content near the current outro."),
          percentBarMetric("Candidate low end", transition.candidate_intro_low_end, "Low-end content near the candidate intro."),
        ],
      },
      {
        title: "Scorer Components",
        note: "Every scoring component returned by the engine.",
        metrics: components.map(([key, score]) => {
          const weight = candidate.weights_used?.[key];
          const confidence = candidate.component_confidences?.[key];
          return percentBarMetric(
            labelize(key),
            score,
            helpFor(weightDescriptions, key, "Scoring component used by the recommendation engine."),
            `weight ${weight == null ? "n/a" : `${Math.round(weight * 100)}%`} / confidence ${confidence == null ? "n/a" : `${Math.round(confidence * 100)}%`}`,
          );
        }),
      },
    ];
  }, [candidate, features.data]);

  return (
    <section className="detail-block full-analysis-panel">
      <PaneTitle icon={<BarChart3 />} title="Full Analysis" action={candidate ? (features.isFetching ? "loading" : "bars") : "waiting"} />
      {!candidate ? (
        <p className="muted">Select a recommendation to view its full analysis profile.</p>
      ) : (
        <>
          <div className="candidate-meta-strip">
            <span>{candidateBpmText(candidate)}</span>
            <span>{candidateKeyText(candidate)}</span>
            <span>{stringValueFromUnknown(candidate.candidate_features.intensity_band) || "band unknown"}</span>
          </div>
          {features.error ? <p className="action-note">Full track analysis could not be loaded; showing scorer-returned fields only.</p> : null}
          <div className="analysis-groups">
            {groups.map((group) => (
              <AnalysisBarGroup key={group.title} title={group.title} note={group.note} metrics={group.metrics} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

type BarMetric = {
  label: string;
  value: number | null;
  display: string;
  hint: string;
  detail?: string;
  signed?: boolean;
};

function AnalysisBarGroup({ title, note, metrics }: { title: string; note: string; metrics: BarMetric[] }) {
  return (
    <section className="analysis-group">
      <div className="analysis-title">
        <h3>{title}</h3>
        <p>{note}</p>
      </div>
      <div className="analysis-bars">
        {metrics.map((metric) => (
          <div key={metric.label} className={metric.value == null ? "analysis-bar-row missing" : "analysis-bar-row"}>
            <div className="analysis-bar-label">
              <span>
                {metric.label}
                <InfoHint text={metric.hint} />
              </span>
              <strong>{metric.display}</strong>
            </div>
            <div className={metric.signed ? "analysis-meter signed" : "analysis-meter"}>
              <i style={{ width: `${barWidth(metric)}%` }} />
            </div>
            {metric.detail ? <small>{metric.detail}</small> : null}
          </div>
        ))}
      </div>
    </section>
  );
}

function firstNumber(...values: unknown[]) {
  for (const value of values) {
    const parsed = maybeNumber(value);
    if (parsed != null) return parsed;
  }
  return null;
}

function maybeNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function candidateBpmText(candidate: LaneItem) {
  const bpm = maybeNumber(candidate.candidate_features.bpm);
  return bpm == null ? "BPM pending" : `${bpm.toFixed(1)} BPM`;
}

function candidateKeyText(candidate: LaneItem) {
  return stringValueFromUnknown(candidate.candidate_features.key) || candidate.tempo_key.key_text || "key pending";
}

function percentBarMetric(label: string, rawValue: unknown, hint: string, detail?: string): BarMetric {
  const value = maybeNumber(rawValue);
  return {
    label,
    value,
    display: value == null ? "missing" : `${Math.round(clamp01(value) * 100)}%`,
    hint,
    detail,
  };
}

function signedBarMetric(label: string, rawValue: unknown, hint: string): BarMetric {
  const value = maybeNumber(rawValue);
  return {
    label,
    value,
    display: value == null ? "missing" : signedPercent(value),
    hint,
    signed: true,
  };
}

function barWidth(metric: BarMetric) {
  if (metric.value == null) return 0;
  if (metric.signed) return clampPercent(Math.round(Math.abs(metric.value) * 100));
  return clampPercent(Math.round(clamp01(metric.value) * 100));
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function componentSort(key: string) {
  const order = ["target_energy", "bass_transition", "harmonic", "tempo", "rhythmic_continuity", "vocal_transition", "history_fit", "transition_support"];
  const index = order.indexOf(key);
  return index === -1 ? 999 : index;
}

function labelize(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function clampPercent(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function riskToSafety(riskScore: number) {
  const riskPercent = riskScore <= 1 ? riskScore * 100 : riskScore;
  return clampPercent(100 - Math.round(riskPercent));
}

function stringValueFromUnknown(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function signedPercent(value: number) {
  return `${value >= 0 ? "+" : ""}${Math.round(value * 100)}%`;
}

function percentOrMissing(value: number | null) {
  return value == null ? "missing" : `${Math.round(clamp01(value) * 100)}%`;
}

function dominantMood(semantic: Record<string, number | string | null>) {
  const moods = [
    { label: "party", value: maybeNumber(semantic.mood_party_abs), detail: "club-forward and lively" },
    { label: "relaxed", value: maybeNumber(semantic.mood_relaxed_abs), detail: "smoother and more laid-back" },
    { label: "aggressive", value: maybeNumber(semantic.mood_aggressive_abs), detail: "harder and more forceful" },
  ].filter((item) => item.value != null) as { label: string; value: number; detail: string }[];
  if (moods.length === 0) return { label: "unknown", detail: "semantic mood not analyzed yet" };
  moods.sort((a, b) => b.value - a.value);
  return { label: `${moods[0].label} ${Math.round(moods[0].value * 100)}%`, detail: moods[0].detail };
}

function buildDJReadSummary({ energyChange, risk, tempo, keyFit, mood }: { energyChange: number | null; risk: string; tempo: string; keyFit: string; mood: { label: string } }) {
  const energy = energyChange == null ? "Energy impact is unknown" : energyMoveCopy(energyChange, "");
  return `${energy}. ${risk} risk, ${tempo}, ${keyFit}. Crowd feel: ${mood.label}.`;
}

function energyMoveCopy(deltaEnergy: number | null, move: string) {
  if (deltaEnergy == null) return move ? `${move} move; energy delta unavailable` : "energy delta unavailable";
  if (deltaEnergy > 0.08) return "lifts the room";
  if (deltaEnergy < -0.08) return "drops pressure for reset";
  return "keeps energy steady";
}

function tempoDecisionCopy(distance: number | null) {
  if (distance == null) return "tempo relationship unavailable";
  if (distance <= 2) return "tight tempo match";
  if (distance <= 6) return "workable with pitch or phrasing";
  return "wide tempo move; plan the transition";
}

function keyDecisionCopy(distance: number | null) {
  if (distance == null) return "key relationship unavailable";
  if (distance <= 1) return "harmonically comfortable";
  if (distance <= 3) return "usable with attention";
  return "harmonic contrast; mix carefully";
}

function bassDecisionCopy(deltaBass: number | null, bassRel: number | null) {
  if (deltaBass == null) return bassRel == null ? "low-end data unavailable" : `candidate bass ${percentOrMissing(bassRel)}`;
  if (deltaBass > 0.08) return "adds low-end pressure";
  if (deltaBass < -0.08) return "lightens the low end";
  return "keeps bass pressure stable";
}

function vocalHandoffPhrase(current: number | null, candidate: number | null) {
  if (current == null || candidate == null) return { value: "unknown", detail: "vocal analysis missing; use ears" };
  if (current > 0.65 && candidate > 0.65) return { value: "busy", detail: "both tracks are vocal-heavy" };
  if (current < 0.25 && candidate < 0.25) return { value: "clear", detail: "low vocal overlap" };
  if (candidate > current + 0.25) return { value: "incoming vocal", detail: "candidate brings vocals forward" };
  if (candidate < current - 0.25) return { value: "opens space", detail: "candidate reduces vocal density" };
  return { value: "balanced", detail: "similar vocal density" };
}

function energyLevelCopy(value: number | null) {
  if (value == null) return "relative energy missing";
  if (value >= 0.78) return "peak pressure";
  if (value >= 0.58) return "driving";
  if (value >= 0.35) return "groove zone";
  return "lower-pressure";
}

function danceabilityCopy(value: number) {
  if (value >= 0.75) return "strong dance-floor pull";
  if (value >= 0.5) return "moderate movement";
  return "less dance-driven";
}

function arousalCopy(value: number) {
  if (value >= 0.7) return "high activation";
  if (value >= 0.45) return "medium drive";
  return "calmer";
}

function valenceCopy(value: number) {
  if (value >= 0.7) return "more positive mood";
  if (value >= 0.55) return "near-neutral mood";
  if (value >= 0.4) return "slightly darker mood";
  return "darker mood";
}

function FeedbackPanel({ feedback }: { feedback?: FeedbackSummary }) {
  const chartData = useMemo(
    () =>
      Object.entries(feedback?.weights.effective ?? {}).map(([name, value]) => ({
        key: name,
        name: name.replace(/_/g, " "),
        shortName: name.replace("_transition", "").replace("target_", "").replace(/_/g, " "),
        value: Math.round(value * 100),
      })),
    [feedback],
  );
  return (
    <section className="detail-block">
      <PaneTitle icon={<BarChart3 />} title="Playlist Tuning" action={feedback?.weights.source ? `${feedback.weights.source} weights` : "static weights"} />
      <p className="context-note">
        Playlist-wide scoring weights for {feedback?.playlist_name ?? "the selected playlist"}. These change after feedback tuning or when you choose another playlist, not when you click a result card.
      </p>
      <div className="metric-grid">
        <Metric label="Events" value={(feedback?.metrics.total_events ?? 0).toString()} hint={metricDescriptions.Events} />
        <Metric label="Top 1" value={pct(feedback?.metrics.chosen_top1_rate)} hint={metricDescriptions["Top 1"]} />
        <Metric label="Top 3" value={pct(feedback?.metrics.chosen_top3_rate)} hint={metricDescriptions["Top 3"]} />
        <Metric label="Pairs" value={(feedback?.metrics.pairwise_comparison_count ?? 0).toString()} hint={metricDescriptions.Pairs} />
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={chartData}>
            <defs>
              <linearGradient id="playlistWeightBar" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#b7bf77" />
                <stop offset="55%" stopColor="#8f9f66" />
                <stop offset="100%" stopColor="#657744" />
              </linearGradient>
            </defs>
            <XAxis dataKey="shortName" tick={{ fill: "#8f9f66", fontSize: 10 }} axisLine={{ stroke: "rgba(143, 159, 102, 0.26)" }} tickLine={false} />
            <YAxis hide domain={[0, 100]} />
            <Bar dataKey="value" fill="url(#playlistWeightBar)" stroke="rgba(231, 236, 235, 0.12)" strokeWidth={1} radius={[4, 4, 0, 0]}>
              <LabelList dataKey="value" position="top" formatter={(value: number) => `${value}%`} fill="#e7eceb" fontSize={11} fontWeight={800} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="weight-list">
        {chartData.map((item) => (
          <div key={item.key} className="weight-row">
            <span>
              {item.name}
              <InfoHint text={helpFor(weightDescriptions, item.key, "Scoring weight used by the recommendation engine.")} />
            </span>
            <strong>{item.value}%</strong>
          </div>
        ))}
      </div>
      <NoteList title="Tuning notes" notes={feedback?.tuning.notes ?? []} />
    </section>
  );
}

function FullToolsPanel({
  playlist,
  candidate,
  playlistId,
  jobs,
  analysisStatus,
  analysisStatusLoading,
  onSmartRefresh,
  onForceRefresh,
  queueBusy,
  queueResult,
  onRemovePlaylist,
  removeBusy,
  onRunWorker,
  onTool,
  toolBusy,
  toolResult,
  toolError,
  toolRun,
  setupStatus,
  remoteStatus,
  remoteStatusLoading,
  remotePairUrl,
  remotePairQr,
  remotePairExpiresAt,
  onGenerateRemotePair,
  remotePairBusy,
  remotePairError,
}: {
  playlist?: Playlist;
  candidate: LaneItem | null;
  playlistId?: string;
  jobs: { id: number; status: string; track_id: string | null; created_at: string; error_message: string | null }[];
  analysisStatus?: PlaylistAnalysisStatus;
  analysisStatusLoading: boolean;
  onSmartRefresh: () => void;
  onForceRefresh: () => void;
  queueBusy: boolean;
  queueResult?: number;
  onRemovePlaylist: (playlistId: string) => void;
  removeBusy: boolean;
  onRunWorker: () => void;
  onTool: (payload: ToolCommandRequest) => void;
  toolBusy: boolean;
  toolResult: ToolCommandResult | null;
  toolError: Error | null;
  toolRun?: ToolRunStatus;
  setupStatus?: SetupStatus;
  remoteStatus?: { enabled: boolean; remote_url: string | null; paired: boolean; request_local: boolean };
  remoteStatusLoading: boolean;
  remotePairUrl: string;
  remotePairQr: string;
  remotePairExpiresAt: string;
  onGenerateRemotePair: () => void;
  remotePairBusy: boolean;
  remotePairError: Error | null;
}) {
  const [localName, setLocalName] = useState("");
  const [localPaths, setLocalPaths] = useState("");
  const [djSource, setDjSource] = useState<"rekordbox" | "traktor" | "serato">("rekordbox");
  const [djLibrary, setDjLibrary] = useState("");
  const [djPlaylist, setDjPlaylist] = useState("");
  const [djPlaylistOptions, setDjPlaylistOptions] = useState<string[]>([]);
  const [djPlaylistLoad, setDjPlaylistLoad] = useState<{ status: "idle" | "loading" | "loaded" | "error"; error?: string }>({ status: "idle" });
  const [djName, setDjName] = useState("");
  const djPlaylistLoadRequestRef = useRef(0);
  const pickPathMutation = useMutation({ mutationFn: api.pickPath });

  const localPathList = localPaths
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
  const pickerBusy = pickPathMutation.isPending;
  const pickerError = pickPathMutation.error instanceof Error ? pickPathMutation.error.message : null;
  const localImportDisabledReason = toolBusy
    ? "Import is already running."
    : !localPathList.length
      ? "Choose files/folders or paste at least one path."
      : !localName.trim()
        ? "Add a CueMate playlist name."
        : "";

  const applyDJPlaylistOptions = useCallback((names: string[]) => {
    setDjPlaylistOptions(names);
    if (names.length > 0) {
      setDjPlaylist((current) => (names.includes(current) ? current : names[0]));
    }
  }, []);

  const loadDJPlaylistOptions = useCallback(
    async (source: "rekordbox" | "traktor" | "serato", libraryPath: string) => {
      const library = libraryPath.trim();
      const requestId = ++djPlaylistLoadRequestRef.current;
      if (!library) {
        setDjPlaylistOptions([]);
        setDjPlaylistLoad({ status: "idle" });
        return;
      }

      setDjPlaylistOptions([]);
      setDjPlaylist("");
      setDjPlaylistLoad({ status: "loading" });
      try {
        const result = await api.toolCommand({ action: "list_dj_playlists", source, library });
        if (requestId !== djPlaylistLoadRequestRef.current) return;
        const names = parseDJPlaylistNames(result.output);
        applyDJPlaylistOptions(names);
        setDjPlaylistLoad({ status: "loaded" });
      } catch (error) {
        if (requestId !== djPlaylistLoadRequestRef.current) return;
        setDjPlaylistOptions([]);
        setDjPlaylistLoad({ status: "error", error: error instanceof Error ? error.message : "Could not load playlists." });
      }
    },
    [applyDJPlaylistOptions],
  );

  useEffect(() => {
    if (!isListDJPlaylistsResult(toolResult)) return;
    const names = parseDJPlaylistNames(toolResult?.output);
    applyDJPlaylistOptions(names);
    setDjPlaylistLoad({ status: "loaded" });
  }, [applyDJPlaylistOptions, toolResult]);

  useEffect(() => {
    const library = djLibrary.trim();
    if (!library) {
      setDjPlaylistOptions([]);
      setDjPlaylist("");
      setDjPlaylistLoad({ status: "idle" });
      return;
    }

    const timeout = window.setTimeout(() => {
      void loadDJPlaylistOptions(djSource, library);
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [djLibrary, djSource, loadDJPlaylistOptions]);

  const appendLocalPaths = (paths: string[]) => {
    const cleanPaths = paths.map((item) => item.trim()).filter(Boolean);
    if (!cleanPaths.length) return;
    if (!localName.trim()) {
      const first = cleanPaths[0].split(/[\\/]/).filter(Boolean).pop();
      if (first) setLocalName(first.replace(/\.[^.]+$/, ""));
    }
    setLocalPaths((current) => {
      const existing = current
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean);
      const next = Array.from(new Set([...existing, ...cleanPaths]));
      return next.join("\n");
    });
  };

  const pickPaths = async (kind: PickPathRequest["kind"]) => {
    try {
      const result = await pickPathMutation.mutateAsync({ kind });
      return result.paths;
    } catch {
      return [];
    }
  };

  const chooseLocalFolder = async () => {
    appendLocalPaths(await pickPaths("folder"));
  };

  const chooseAudioFiles = async () => {
    appendLocalPaths(await pickPaths("audio_files"));
  };

  const chooseDJLibrary = async () => {
    const kind: PickPathRequest["kind"] = djSource === "serato" ? "folder" : "dj_library_file";
    const paths = await pickPaths(kind);
    if (paths[0]) {
      setDjLibrary(paths[0]);
      setDjPlaylist("");
      setDjPlaylistOptions([]);
    }
  };

  return (
    <div className="full-tools">
      <PaneTitle icon={<TerminalSquare />} title="Full Mode" action="setup" />
      <p className="mode-note">Full Mode starts with the selected song analysis. Playlist setup and imports sit below it when you need maintenance tasks.</p>

      <FullCandidateAnalysisPanel candidate={candidate} playlistId={playlistId} />

      <ToolSection icon={<PlayCircle />} title="Playlist Health" description="Refresh stale analysis, run queued work, or remove this playlist from CueMate.">
        <PlaylistSetupPanel
          playlist={playlist}
          status={analysisStatus}
          loading={analysisStatusLoading}
          jobs={jobs}
          queueBusy={queueBusy}
          queueResult={queueResult}
          workerBusy={toolBusy}
          onSmartRefresh={onSmartRefresh}
          onForceRefresh={onForceRefresh}
          onRunWorker={onRunWorker}
          onRemovePlaylist={onRemovePlaylist}
          removeBusy={removeBusy}
        />
      </ToolSection>

      <ToolSection icon={<FolderPlus />} title="Import local files" description="Build a CueMate playlist from folders or individual audio files.">
        <div className="field-group">
          <label className="field-label">CueMate playlist name</label>
          <input value={localName} onChange={(event) => setLocalName(event.target.value)} placeholder="New playlist name" />
        </div>
        <div className="split-actions">
          <button className="wide-action secondary" disabled={pickerBusy} onClick={chooseLocalFolder}>
            <FolderOpen size={16} /> Choose folder
          </button>
          <button className="wide-action secondary" disabled={pickerBusy} onClick={chooseAudioFiles}>
            <FileAudio size={16} /> Choose audio files
          </button>
        </div>
        <div className="field-group">
          <label className="field-label">Selected files or folders</label>
          <textarea value={localPaths} onChange={(event) => setLocalPaths(event.target.value)} placeholder={"One audio file or folder path per line"} rows={3} />
          <p className="selection-summary">{localPathList.length ? `${localPathList.length} source path${localPathList.length === 1 ? "" : "s"} ready` : "Choose files/folders or paste paths manually."}</p>
        </div>
        <button
          className="wide-action"
          disabled={Boolean(localImportDisabledReason)}
          onClick={() => onTool({ action: "import_playlist", name: localName, paths: localPathList })}
        >
          <FolderPlus size={16} /> Import local playlist
        </button>
        {localImportDisabledReason ? <p className="selection-summary">{localImportDisabledReason}</p> : null}
      </ToolSection>

      <ToolSection icon={<ListMusic />} title="Import DJ library" description="Pull an existing Rekordbox, Traktor, or Serato playlist into CueMate.">
        <div className="field-group">
          <label className="field-label">DJ source</label>
          <select
            value={djSource}
            onChange={(event) => {
              setDjSource(event.target.value as "rekordbox" | "traktor" | "serato");
              setDjPlaylist("");
              setDjPlaylistOptions([]);
            }}
          >
            <option value="rekordbox">Rekordbox XML</option>
            <option value="traktor">Traktor NML</option>
            <option value="serato">Serato crate</option>
          </select>
        </div>
        <div className="field-group">
          <label className="field-label">{djSource === "serato" ? "Serato crate folder" : "Library export file"}</label>
          <div className="pick-row">
            <input
              value={djLibrary}
              onChange={(event) => {
                setDjLibrary(event.target.value);
                setDjPlaylist("");
                setDjPlaylistOptions([]);
              }}
              placeholder={djSource === "serato" ? "Choose a Serato crate folder" : "Choose a Rekordbox XML or Traktor NML export"}
            />
            <button className="path-pick" disabled={pickerBusy} onClick={chooseDJLibrary}>
              <FolderOpen size={16} /> Browse
            </button>
          </div>
        </div>
        <div className="field-group">
          <label className="field-label">Source playlist or crate</label>
          <select
            value={djPlaylistOptions.includes(djPlaylist) ? djPlaylist : ""}
            disabled={djPlaylistLoad.status === "loading" || djPlaylistOptions.length === 0}
            onChange={(event) => setDjPlaylist(event.target.value)}
          >
            <option value="">
              {djPlaylistLoad.status === "loading"
                ? "Loading playlists..."
                : djPlaylistOptions.length > 0
                  ? "Select a source playlist or crate"
                  : "Choose a library file to load playlists"}
            </option>
            {djPlaylistOptions.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          {djPlaylistOptions.length > 0 ? (
            <p className="selection-summary">{djPlaylistOptions.length} playlist{djPlaylistOptions.length === 1 ? "" : "s"} found. Choose one, then import.</p>
          ) : djPlaylistLoad.status === "loading" ? (
            <p className="selection-summary">Reading playlists from the selected library...</p>
          ) : djPlaylistLoad.status === "error" ? (
            <>
              <input value={djPlaylist} onChange={(event) => setDjPlaylist(event.target.value)} placeholder="Paste a source name manually" />
              <p className="selection-summary danger">Could not load playlists: {djPlaylistLoad.error}</p>
            </>
          ) : (
            <>
              <input value={djPlaylist} onChange={(event) => setDjPlaylist(event.target.value)} placeholder="Or paste a source name manually" />
              <p className="selection-summary">Choose a library file to load playlists automatically.</p>
            </>
          )}
        </div>
        <div className="field-group">
          <label className="field-label">CueMate playlist name</label>
          <input value={djName} onChange={(event) => setDjName(event.target.value)} placeholder={djPlaylist || "Optional name"} />
        </div>
        <div className="split-actions">
          <button className="wide-action secondary" disabled={toolBusy || djPlaylistLoad.status === "loading" || !djLibrary.trim()} onClick={() => void loadDJPlaylistOptions(djSource, djLibrary)}>
            {djPlaylistLoad.status === "loading" ? "Loading..." : "Refresh list"}
          </button>
          <button
            className="wide-action"
            disabled={toolBusy || !djLibrary.trim() || !djPlaylist.trim()}
            onClick={() => onTool({ action: "import_dj_playlist", source: djSource, library: djLibrary, playlist: djPlaylist, name: djName || undefined })}
          >
            Import
          </button>
        </div>
      </ToolSection>

      <ToolSection icon={<Radar />} title="Connect phone" description="Generate a short-lived QR link for the mobile web app.">
        <div className="remote-card">
          <div>
            <p className={remoteStatus?.enabled ? "pill cyan" : "pill amber"}>
              {remoteStatusLoading ? "checking" : remoteStatus?.enabled ? "remote ready" : "remote setup needed"}
            </p>
            <p className="muted">
              {remoteStatus?.remote_url ??
                "Mobile access requires Tailscale on this PC and your phone. Local CueMate works without it; launch again after signing in to enable QR pairing."}
            </p>
            {setupStatus?.available && !setupStatus.mobile_ready ? (
              <p className="action-note">Tailscale was skipped or is not ready. Install and sign in to Tailscale on both devices to use phone access.</p>
            ) : null}
          </div>
          <button className="wide-action secondary" disabled={!remoteStatus?.enabled || remotePairBusy} onClick={onGenerateRemotePair}>
            {remotePairBusy ? "Generating..." : "Generate QR"}
          </button>
        </div>
        {remotePairQr ? (
          <div className="qr-wrap">
            <img src={remotePairQr} alt="CueMate mobile pairing QR code" />
            <div>
              <p className="field-label">Pairing link</p>
              <p className="muted breakable">{remotePairUrl}</p>
              <p className="action-note">Expires {remotePairExpiresAt ? new Date(remotePairExpiresAt).toLocaleTimeString() : "soon"}.</p>
            </div>
          </div>
        ) : null}
        {remotePairError ? <p className="action-note">{remotePairError.message}</p> : null}
      </ToolSection>

      {pickerError ? <div className="tool-result danger">File picker failed: {pickerError}</div> : null}

      <ToolResultBox result={toolResult} error={toolError} busy={toolBusy} run={toolRun} />
    </div>
  );
}

function PlaylistSetupPanel({
  playlist,
  status,
  loading,
  jobs,
  queueBusy,
  queueResult,
  workerBusy,
  onSmartRefresh,
  onForceRefresh,
  onRunWorker,
  onRemovePlaylist,
  removeBusy,
}: {
  playlist?: Playlist;
  status?: PlaylistAnalysisStatus;
  loading: boolean;
  jobs: { id: number; status: string; track_id: string | null; created_at: string; error_message: string | null }[];
  queueBusy: boolean;
  queueResult?: number;
  workerBusy: boolean;
  onSmartRefresh: () => void;
  onForceRefresh: () => void;
  onRunWorker: () => void;
  onRemovePlaylist: (playlistId: string) => void;
  removeBusy: boolean;
}) {
  const total = status?.total_tracks ?? playlist?.track_count ?? 0;
  const ready = status?.ready_tracks ?? playlist?.track_count_analyzed ?? 0;
  const outdated = status?.outdated_tracks ?? 0;
  const queued = status?.jobs.pending ?? 0;
  const running = status?.jobs.running ?? 0;
  const failed = status?.jobs.failed ?? 0;
  const missing = Math.max(0, total - ready - outdated);
  const activeJobs = queued + running;
  const percent = status?.percent_complete ?? (total ? Math.round((ready / total) * 100) : 0);
  const stateLabel = setupStateLabel(status, playlist);
  const latestFailed = jobs.find((job) => job.status === "failed");
  const confirmRemove = () => {
    if (!playlist) return;
    if (window.confirm(`Remove "${playlist.name}" from CueMate? This will not delete any music files.`)) {
      onRemovePlaylist(playlist.playlist_id);
    }
  };
  return (
    <div className="playlist-setup-card">
      {!playlist ? (
        <div className="empty-state compact">
          <strong>No playlist selected</strong>
          <span>Import a playlist or choose one from the library to start analysis.</span>
        </div>
      ) : (
        <>
          <div className="setup-head">
            <div>
              <p className="eyebrow">Selected playlist</p>
              <h3>{playlist.name}</h3>
            </div>
            <span className={pillClass(stateLabel)}>{loading ? "checking" : stateLabel}</span>
          </div>
          <div className="progress-row">
            <div className="progress-track">
              <span style={{ width: `${boundedPercent(percent)}%` }} />
            </div>
            <strong>{ready}/{total} current</strong>
          </div>
          <div className={activeJobs ? "analysis-progress active" : "analysis-progress"}>
            <div className="analysis-progress-bar" aria-label="Playlist analysis breakdown">
              <span className="ready" style={{ width: `${total ? boundedPercent((ready / total) * 100) : 0}%` }} />
              <span className="outdated" style={{ width: `${total ? boundedPercent((outdated / total) * 100) : 0}%` }} />
              <span className="missing" style={{ width: `${total ? boundedPercent((missing / total) * 100) : 0}%` }} />
            </div>
            <div className="analysis-legend">
              <span><i className="ready" /> Current {ready}</span>
              {outdated ? <span><i className="outdated" /> Outdated {outdated}</span> : null}
              {missing ? <span><i className="missing" /> Missing {missing}</span> : null}
              {running ? <span><i className="running" /> Running {running}</span> : null}
              {queued ? <span><i className="queued" /> Queued {queued}</span> : null}
            </div>
          </div>
          <div className="metric-grid">
            <Metric label="Queued" value={queued.toString()} />
            <Metric label="Running" value={running.toString()} />
            <Metric label="Failed" value={failed.toString()} />
            {outdated > 0 ? <Metric label="Outdated" value={outdated.toString()} /> : null}
          </div>
          <p className="setup-guidance">{setupGuidance(status, total, ready)}</p>
          {latestFailed?.error_message || status?.latest_error ? <p className="action-note danger">Latest failure: {latestFailed?.error_message ?? status?.latest_error}</p> : null}
          <div className="split-actions">
            <button className="wide-action" disabled={queueBusy} onClick={onSmartRefresh}>
              <RefreshCw size={16} /> {queueBusy ? "Starting analysis..." : "Refresh and analyse playlist"}
            </button>
            <button className="wide-action secondary" disabled={queueBusy} onClick={onForceRefresh}>
              Force reanalyse and run worker
            </button>
          </div>
          <button className="wide-action secondary" disabled={workerBusy} onClick={onRunWorker}>
            <PlayCircle size={16} /> {workerBusy ? "Worker running..." : "Run analysis worker"}
          </button>
          {queueResult != null ? <p className="muted">{queueResult} analysis jobs queued.</p> : null}
          <button className="wide-action danger" disabled={removeBusy} onClick={confirmRemove}>
            {removeBusy ? "Removing..." : "Remove playlist from CueMate"}
          </button>
        </>
      )}
    </div>
  );
}

function setupStateLabel(status?: PlaylistAnalysisStatus, playlist?: Playlist) {
  if (status?.jobs.failed) return "Failed";
  if (status?.jobs.running) return "Running";
  if (status?.jobs.pending) return "Queued";
  if (status?.is_stale || playlist?.is_stale) return "Out of date";
  if (status?.outdated_tracks) return "Outdated";
  const total = status?.total_tracks ?? playlist?.track_count ?? 0;
  const ready = status?.ready_tracks ?? playlist?.track_count_analyzed ?? 0;
  if (total > 0 && ready >= total) return "Ready";
  return "Needs analysis";
}

function setupGuidance(status: PlaylistAnalysisStatus | undefined, total: number, ready: number) {
  if (!total) return "Import tracks first. CueMate will analyse them before recommendations become useful.";
  if (status?.jobs.failed) return "Some analysis jobs failed. Check the tool output below, then try Smart refresh again.";
  if (status?.jobs.running) return status.outdated_tracks ? "Reanalysis is running. Existing rows are old and recommendations unlock as current jobs finish." : "Analysis is running. This page will update as tracks become ready.";
  if (status?.jobs.pending) return status.outdated_tracks ? "Reanalysis is queued. Run the analysis worker if nothing is moving." : "Analysis is queued. Run the analysis worker if nothing is moving.";
  if (status?.is_stale) return `Out of date: ${status.stale_reason || "playlist analysis needs refresh"}. Smart refresh will queue only what changed.`;
  if (status?.outdated_tracks) return `${status.outdated_tracks} track${status.outdated_tracks === 1 ? "" : "s"} need reanalysis for the current scoring engine.`;
  if (ready < total) return "Some tracks still need analysis. Smart refresh queues only missing or outdated work.";
  return "Ready. Recommendations can use this playlist now.";
}

function ToolSection({ icon, title, description, children }: { icon: React.ReactNode; title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="tool-section">
      <div className="tool-section-title">
        {icon}
        <div>
          <span>{title}</span>
          {description ? <p>{description}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function ToolResultBox({ result, error, busy, run }: { result: ToolCommandResult | null; error: Error | null; busy: boolean; run?: ToolRunStatus }) {
  if (busy) return <div className="tool-result">Running tool...</div>;
  if (error) return <div className="tool-result danger">{error.message}</div>;
  if (!result && !run) return null;
  return (
    <div className="tool-result">
      <strong>{run ? `Tool ${run.status}` : result?.mode === "background" ? "Started" : "Completed"}</strong>
      {result?.command ? <small>{result.command.join(" ")}</small> : null}
      {result?.pid ? <span>PID {result.pid}</span> : null}
      {result?.log_path ? <span>Log: {result.log_path}</span> : null}
      {result?.output ? <pre>{result.output}</pre> : null}
      {run?.output_tail ? <pre>{run.output_tail}</pre> : null}
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="metric">
      <small>
        {label}
        {hint ? <InfoHint text={hint} /> : null}
      </small>
      <strong>{value}</strong>
    </div>
  );
}

function NoteList({ title, notes }: { title: string; notes: string[] }) {
  if (!notes.length) return null;
  return (
    <div className="notes">
      <p>{title}</p>
      {notes.map((note) => (
        <span key={note}>{note}</span>
      ))}
    </div>
  );
}
