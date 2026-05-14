import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  BarChart3,
  Check,
  CircleHelp,
  CircleDot,
  Library,
  ListMusic,
  Radar,
  RefreshCw,
  Search,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Waves,
} from "lucide-react";
import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, FeedbackSummary, LaneItem, Playlist, RecommendationResponse, Track } from "./api";

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
  const anchorRef = useRef<HTMLSpanElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number; placement: "top" | "bottom" } | null>(null);

  const updatePosition = useCallback(() => {
    const anchor = anchorRef.current;
    if (!anchor) return;

    const rect = anchor.getBoundingClientRect();
    const placement = rect.top < 96 ? "bottom" : "top";
    const maxTooltipWidth = Math.min(280, window.innerWidth - 24);
    const unclampedLeft = rect.left + rect.width / 2;
    const left = Math.min(window.innerWidth - 12 - maxTooltipWidth / 2, Math.max(12 + maxTooltipWidth / 2, unclampedLeft));
    const top = placement === "bottom" ? rect.bottom + 10 : rect.top - 10;
    setPosition({ left, top, placement });
  }, []);

  useEffect(() => {
    if (!position) return undefined;
    const handleMove = () => updatePosition();
    window.addEventListener("resize", handleMove);
    window.addEventListener("scroll", handleMove, true);
    return () => {
      window.removeEventListener("resize", handleMove);
      window.removeEventListener("scroll", handleMove, true);
    };
  }, [position, updatePosition]);

  return (
    <span
      ref={anchorRef}
      className={className ? `tooltip-anchor ${className}` : "tooltip-anchor"}
      tabIndex={focusable ? 0 : undefined}
      aria-label={focusable ? text : undefined}
      onMouseEnter={updatePosition}
      onMouseLeave={() => setPosition(null)}
      onFocus={focusable ? updatePosition : undefined}
      onBlur={focusable ? () => setPosition(null) : undefined}
    >
      {children}
      {position
        ? createPortal(
            <span className={`floating-tooltip ${position.placement}`} role="tooltip" style={{ left: position.left, top: position.top }}>
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
  if (key.includes("stale") || key.includes("unanalysed")) return "pill amber";
  return "pill";
}

export function App() {
  const queryClient = useQueryClient();
  const [selectedPlaylistId, setSelectedPlaylistId] = useState<string>(() => localStorage.getItem("cuemate.playlist") ?? "");
  const [currentTrackId, setCurrentTrackId] = useState<string>(() => localStorage.getItem("cuemate.current") ?? "");
  const [history, setHistory] = useState<string[]>(() => JSON.parse(localStorage.getItem("cuemate.history") ?? "[]") as string[]);
  const [target, setTarget] = useState("maintain");
  const [trackQuery, setTrackQuery] = useState("");
  const [selectedCandidate, setSelectedCandidate] = useState<LaneItem | null>(null);
  const [mobileTab, setMobileTab] = useState<"recommend" | "library" | "feedback" | "admin">("recommend");

  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000 });
  const readiness = useQuery({ queryKey: ["ready"], queryFn: api.readiness, refetchInterval: 15_000 });
  const metadata = useQuery({ queryKey: ["metadata"], queryFn: api.metadata, refetchInterval: 30_000 });
  const playlists = useQuery({ queryKey: ["playlists"], queryFn: api.playlists });
  const tracks = useQuery({
    queryKey: ["playlistTracks", selectedPlaylistId, trackQuery],
    queryFn: () => api.playlistTracks(selectedPlaylistId, trackQuery),
    enabled: !!selectedPlaylistId,
  });
  const jobs = useQuery({
    queryKey: ["jobs", selectedPlaylistId],
    queryFn: () => api.jobs(selectedPlaylistId || undefined),
    refetchInterval: 20_000,
  });
  const feedback = useQuery({
    queryKey: ["feedback", selectedPlaylistId],
    queryFn: () => api.feedback(selectedPlaylistId),
    enabled: !!selectedPlaylistId,
  });
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

  const enqueueMutation = useMutation({
    mutationFn: () => api.enqueueAnalysis(selectedPlaylistId, true),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const snapshotMutation = useMutation({ mutationFn: () => api.snapshot(selectedPlaylistId) });

  const correctionMutation = useMutation({
    mutationFn: (payload: { field: "bpm" | "key"; new_value: number | string }) =>
      api.correction({ track_id: currentTrackId, ...payload }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const selectedPlaylist = playlists.data?.items.find((item) => item.playlist_id === selectedPlaylistId) ?? playlists.data?.items[0];
  const currentTrack = tracks.data?.items.find((item) => item.track_id === currentTrackId);

  const firstReadyTrack = tracks.data?.items.find((item) => item.analysis_state === "ready");

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

  const shellClass = `app-shell tab-${mobileTab}`;

  return (
    <div className={shellClass}>
      <header className="topbar">
        <div>
          <p className="eyebrow">CueMate Engine</p>
          <h1>Performance Control</h1>
        </div>
        <div className="status-row">
          <StatusDot label="API" ok={health.data?.status === "ok"} loading={health.isLoading} />
          <StatusDot label="Scorer" ok={readiness.data?.status === "ready"} loading={readiness.isLoading} />
          <span className={metadata.data?.breaker_open ? "pill hot" : "pill cyan"}>
            {metadata.data?.breaker_open ? "breaker open" : "breaker calm"}
          </span>
        </div>
      </header>

      <aside className="library-pane panel mobile-library">
        <PaneTitle icon={<Library />} title="Library" action={`${playlists.data?.items.length ?? 0} playlists`} />
        <PlaylistList
          playlists={playlists.data?.items ?? []}
          selectedId={selectedPlaylistId}
          onSelect={(id) => {
            setSelectedPlaylistId(id);
            localStorage.setItem("cuemate.playlist", id);
            setCurrentTrackId("");
          }}
        />
        <div className="searchbox">
          <Search size={16} />
          <input value={trackQuery} onChange={(event) => setTrackQuery(event.target.value)} placeholder="Search tracks" />
        </div>
        <TrackList
          tracks={tracks.data?.items ?? []}
          currentTrackId={currentTrackId}
          onSelect={(track) => {
            setCurrentTrackId(track.track_id);
            localStorage.setItem("cuemate.current", track.track_id);
          }}
        />
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
          onPlay={(trackId) => playMutation.mutate(trackId)}
          playing={playMutation.isPending}
        />
      </main>

      <aside className="detail-pane panel mobile-feedback">
        <CandidateDetail
          candidate={selectedCandidate}
          response={recommendations.data}
          onConfirm={(trackId) => playMutation.mutate(trackId)}
          confirming={playMutation.isPending}
        />
        <FeedbackPanel feedback={feedback.data} />
      </aside>

      <aside className="admin-pane panel mobile-admin">
        <PaneTitle icon={<SlidersHorizontal />} title="Ops" action={selectedPlaylist?.is_stale ? "stale" : "current"} />
        <OpsPanel
          playlist={selectedPlaylist}
          jobs={jobs.data?.items ?? []}
          onQueue={() => enqueueMutation.mutate()}
          queueBusy={enqueueMutation.isPending}
          queueResult={enqueueMutation.data?.queued_count}
          onSnapshot={() => snapshotMutation.mutate()}
          snapshotBusy={snapshotMutation.isPending}
          onCorrection={(field, value) => correctionMutation.mutate({ field, new_value: value })}
          correctionBusy={correctionMutation.isPending}
        />
      </aside>

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
        <button className={mobileTab === "admin" ? "active" : ""} onClick={() => setMobileTab("admin")}>
          <Settings size={18} /> Ops
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
  onPlay,
  playing,
}: {
  response?: RecommendationResponse;
  loading: boolean;
  error: Error | null;
  selected: LaneItem | null;
  onSelect: (item: LaneItem) => void;
  onPlay: (trackId: string) => void;
  playing: boolean;
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
                    <span className="candidate-stats">{item.tempo_key.tempo_text || item.tempo_key.key_text || item.move}</span>
                  </TooltipAnchor>
                  <TooltipAnchor text={metricDescriptions.Risk} className="candidate-title-tip" focusable={false}>
                    <span className={pillClass(item.risk)}>{item.risk}</span>
                  </TooltipAnchor>
                </span>
                <button
                  className="icon-action"
                  disabled={playing}
                  aria-label={`Mark ${item.title || item.track_id} as played`}
                  onClick={(event) => { event.stopPropagation(); onPlay(item.track_id); }}
                >
                  <Check size={16} />
                </button>
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
}: {
  candidate: LaneItem | null;
  response?: RecommendationResponse;
  onConfirm: (trackId: string) => void;
  confirming: boolean;
}) {
  return (
    <section className="detail-block">
      <PaneTitle icon={<Sparkles />} title="Candidate" action={response?.meta.recommendation_event_id ? "event armed" : "preview"} />
      {!candidate ? (
        <p className="muted">Select a recommendation to inspect why it works, what to watch, and how to hand it off.</p>
      ) : (
        <>
          <h2>{candidate.artist ? `${candidate.artist} - ${candidate.title}` : candidate.title}</h2>
          <div className="metric-grid">
            <Metric label="Fit score" value={`${Math.round(candidate.score * 100)}/100`} hint={metricDescriptions["Fit score"]} />
            <Metric label="Move" value={candidate.move} hint={metricDescriptions.Move} />
            <Metric label="Risk" value={candidate.risk} hint={metricDescriptions.Risk} />
            <Metric label="Confidence" value={pct(candidate.move_confidence)} hint={metricDescriptions.Confidence} />
          </div>
          <button className="wide-action confirm-next" disabled={confirming || !response?.meta.recommendation_event_id} onClick={() => onConfirm(candidate.track_id)}>
            <Check size={16} />
            {confirming ? "Setting next song..." : "Set as next song and make current"}
          </button>
          <p className="action-note">Records this recommendation as played, adds the previous base to history, and rescans from this track.</p>
          <NoteList title="Reasons" notes={candidate.reasons} />
          <NoteList title="Watchouts" notes={candidate.watchouts} />
          <NoteList title="Handoff" notes={candidate.explanation.handoff?.notes ?? []} />
        </>
      )}
    </section>
  );
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
      <PaneTitle icon={<BarChart3 />} title="Feedback" action={feedback?.weights.source ?? "static"} />
      <div className="metric-grid">
        <Metric label="Events" value={(feedback?.metrics.total_events ?? 0).toString()} hint={metricDescriptions.Events} />
        <Metric label="Top 1" value={pct(feedback?.metrics.chosen_top1_rate)} hint={metricDescriptions["Top 1"]} />
        <Metric label="Top 3" value={pct(feedback?.metrics.chosen_top3_rate)} hint={metricDescriptions["Top 3"]} />
        <Metric label="Pairs" value={(feedback?.metrics.pairwise_comparison_count ?? 0).toString()} hint={metricDescriptions.Pairs} />
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={chartData}>
            <XAxis dataKey="shortName" tick={{ fill: "#8b98aa", fontSize: 10 }} />
            <YAxis hide domain={[0, 100]} />
            <Tooltip formatter={(value: number, _name, item) => [`${value}%`, item.payload.name]} contentStyle={{ background: "#111720", border: "1px solid #273447" }} />
            <Bar dataKey="value" fill="#5eead4" radius={[4, 4, 0, 0]}>
              <LabelList dataKey="value" position="top" formatter={(value: number) => `${value}%`} fill="#eef5ff" fontSize={11} fontWeight={800} />
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

function OpsPanel({
  playlist,
  jobs,
  onQueue,
  queueBusy,
  queueResult,
  onSnapshot,
  snapshotBusy,
  onCorrection,
  correctionBusy,
}: {
  playlist?: Playlist;
  jobs: { id: number; status: string; track_id: string | null; created_at: string; error_message: string | null }[];
  onQueue: () => void;
  queueBusy: boolean;
  queueResult?: number;
  onSnapshot: () => void;
  snapshotBusy: boolean;
  onCorrection: (field: "bpm" | "key", value: string | number) => void;
  correctionBusy: boolean;
}) {
  const [bpm, setBpm] = useState("");
  const [keyValue, setKeyValue] = useState("");
  return (
    <div className="ops">
      <div className="metric-grid">
        <Metric label="Ready" value={`${playlist?.track_count_analyzed ?? 0}/${playlist?.track_count ?? 0}`} />
        <Metric label="Feedback" value={(playlist?.feedback_event_count ?? 0).toString()} />
      </div>
      <button className="wide-action" disabled={!playlist || queueBusy} onClick={onQueue}>
        <RefreshCw size={16} /> {queueBusy ? "Queueing..." : "Queue staged analysis"}
      </button>
      {queueResult != null ? <p className="muted">{queueResult} analysis jobs queued.</p> : null}
      <button className="wide-action secondary" disabled={!playlist || snapshotBusy} onClick={onSnapshot}>
        <Waves size={16} /> {snapshotBusy ? "Generating..." : "Generate mobile snapshot"}
      </button>
      <div className="correction-box">
        <p className="eyebrow">Manual correction</p>
        <div className="inline-inputs">
          <input value={bpm} onChange={(event) => setBpm(event.target.value)} placeholder="BPM" inputMode="decimal" />
          <button disabled={!bpm || correctionBusy} onClick={() => onCorrection("bpm", Number(bpm))}>Save</button>
        </div>
        <div className="inline-inputs">
          <input value={keyValue} onChange={(event) => setKeyValue(event.target.value)} placeholder="Key e.g. 8A" />
          <button disabled={!keyValue || correctionBusy} onClick={() => onCorrection("key", keyValue)}>Save</button>
        </div>
      </div>
      <div className="job-list">
        {jobs.slice(0, 8).map((job) => (
          <div key={job.id} className="job">
            <span className={pillClass(job.status)}>{job.status}</span>
            <small>{job.track_id ?? job.created_at}</small>
          </div>
        ))}
      </div>
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
