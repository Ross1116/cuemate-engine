export type Playlist = {
  playlist_id: string;
  name: string;
  track_count: number;
  track_count_analyzed: number;
  eligible_track_count: number;
  is_stale: boolean;
  stale_reason: string | null;
  feedback_event_count: number;
  feedback_last_tuned_at: string | null;
};

export type Track = {
  track_id: string;
  title: string;
  artist: string;
  position: number;
  bpm: number | null;
  key: string | null;
  intensity_band: string | null;
  role_hints: string[];
  analysis_state: string;
};

export type LaneItem = {
  track_id: string;
  title: string;
  artist: string;
  score: number;
  raw_score: number;
  penalty_multiplier: number;
  ranking_strength: string;
  move: string;
  move_confidence: number;
  move_note: string | null;
  risk: string;
  risk_score: number;
  primary_lane: string | null;
  secondary_lane: boolean;
  component_scores: Record<string, number>;
  component_confidences: Record<string, number>;
  weights_used: Record<string, number>;
  transition_features: Record<string, number | string | null>;
  candidate_features: Record<string, number | string | string[] | null>;
  tempo_key: { tempo_text: string; key_text: string; key_state: string };
  advisory_hints: string[];
  reasons: string[];
  watchouts: string[];
  explanation: {
    summary: string[];
    why: string[];
    watch: string[];
    character_shift: string[];
    handoff: { level: string; notes: string[] } | null;
    tempo_key: { tempo_text: string; key_text: string; key_state: string };
  };
  windows: Record<string, unknown>;
};

export type TrackFeatureDetail = {
  track_id: string;
  title: string;
  artist: string;
  basic: Record<string, number | string | null>;
  absolute: Record<string, number | string | null>;
  semantic: Record<string, number | string | null>;
  relative: Record<string, number | string | string[] | Record<string, number> | null>;
  analysis: Record<string, number | string | null>;
};

export type RecommendationResponse = {
  mode: string;
  recommendations_status: string;
  current_track: {
    track_id: string;
    title: string;
    artist: string;
    bpm: number | null;
    key: string | null;
    intensity_band: string | null;
    role_hints: string[];
  };
  target: string;
  set_context: {
    trend: { label: string; direction: string };
    session_notes: string[];
    history_length: number;
    has_gaps: boolean;
  };
  recommendation_confidence: number;
  lane_order: string[];
  lanes: Record<string, { availability: string; items: LaneItem[]; empty_reason: string | null }>;
  meta: {
    analysis_coverage: string;
    weight_adaptation: Record<string, unknown>;
    scoring_contract_id: string;
    status_note: string | null;
    fallback_note: string | null;
    recommendation_event_id: string | null;
    best_alternative_lanes: string[];
  };
};

export type FeedbackSummary = {
  playlist_id: string;
  playlist_name: string;
  metrics: {
    total_events: number;
    contributory_events: number;
    ranked_events: number;
    pairwise_comparison_count: number;
    chosen_top1_rate: number;
    chosen_top3_rate: number;
    chosen_top5_rate: number;
    mean_chosen_rank: number | null;
    lane_acceptance_counts: Record<string, number>;
    higher_scored_lane_skip_counts: Record<string, number>;
  };
  weights: {
    source: string;
    static: Record<string, number>;
    base: Record<string, number>;
    tuned: Record<string, number> | null;
    effective: Record<string, number>;
  };
  tuning: {
    last_tuned_at: string | null;
    feedback_event_count: number;
    notes: string[];
    metrics: Record<string, unknown>;
  };
};

export type AnalysisJob = {
  id: number;
  playlist_id: string | null;
  track_id: string | null;
  track_path: string;
  status: string;
  priority: number;
  analysis_mode: string;
  job_kind: string;
  error_message: string | null;
  duration_seconds: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
};

export type ToolCommandRequest = {
  action: string;
  name?: string;
  paths?: string[];
  source?: "rekordbox" | "traktor" | "serato";
  library?: string;
  playlist?: string;
  analysis_mode?: "fast_pass" | "staged" | "full";
  force?: boolean;
  limit?: number;
  path?: string;
  print_backend_diagnostics?: boolean;
};

export type ToolCommandResult = {
  status: string;
  mode: "foreground" | "background";
  command: string[];
  exit_code?: number;
  output?: string;
  pid?: number;
  log_path?: string;
};

export type PickPathRequest = {
  kind: "folder" | "audio_files" | "dj_library_file";
};

export type PickPathResult = {
  paths: string[];
};

const apiBase = import.meta.env.DEV ? "/api" : "";

export class ApiError extends Error {
  status: number;
  payload?: unknown;

  constructor(message: string, status: number, payload?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function hasToolDiagnostics(payload: unknown) {
  return isRecord(payload) && ("status" in payload || "command" in payload || "output" in payload);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let payload: unknown;
    try {
      payload = await response.json();
      if (isRecord(payload) && typeof payload.error === "string") message = payload.error;
      if (hasToolDiagnostics(payload)) message = JSON.stringify(payload);
    } catch {
      // Keep HTTP status message.
    }
    throw new ApiError(message, response.status, payload);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/healthz"),
  readiness: () => request<{ status: string; error?: string }>("/readyz"),
  metadata: () =>
    request<{
      metadata: { capability_flags?: Record<string, boolean>; active_signatures?: Record<string, string> };
      breaker_open: boolean;
      failure_count: number;
      metadata_fresh: boolean;
      metadata_error: string | null;
    }>("/scoring/metadata"),
  playlists: () => request<{ items: Playlist[] }>("/playlists"),
  playlistTracks: (playlistId: string, query = "") =>
    request<{ items: Track[] }>(
      `/playlists/${encodeURIComponent(playlistId)}/tracks?limit=500${query ? `&query=${encodeURIComponent(query)}` : ""}`,
    ),
  trackFeatures: (playlistId: string, trackId: string) =>
    request<TrackFeatureDetail>(`/playlists/${encodeURIComponent(playlistId)}/tracks/${encodeURIComponent(trackId)}/features`),
  trackSearch: (playlistId: string, query: string) =>
    request<{ items: Track[] }>(
      `/tracks/search?playlist_id=${encodeURIComponent(playlistId)}&query=${encodeURIComponent(query)}&limit=50`,
    ),
  recommendations: (body: {
    playlist_id: string;
    current_track_id: string;
    target: string;
    history_track_ids: string[];
    max_per_lane: number;
  }) =>
    request<RecommendationResponse>("/recommendations", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  played: (body: { recommendation_event_id: string; chosen_track_id: string }) =>
    request<{ chosen_was_recommended: boolean; skipped_over: string[] }>("/events/played", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  feedback: (playlistId: string) =>
    request<FeedbackSummary>("/feedback/summary", {
      method: "POST",
      body: JSON.stringify({ playlist_id: playlistId }),
    }),
  jobs: (playlistId?: string) =>
    request<{ items: AnalysisJob[] }>(
      `/analysis/jobs?limit=25${playlistId ? `&playlist_id=${encodeURIComponent(playlistId)}` : ""}`,
    ),
  correction: (body: { track_id: string; field: "bpm" | "key"; new_value: number | string }) =>
    request<{
      changed: boolean;
      requires_reanalysis: boolean;
      correction_id: string | null;
      queued_job_id: number | null;
      affected_playlist_ids: string[];
    }>("/corrections", { method: "POST", body: JSON.stringify(body) }),
  enqueueAnalysis: (playlistId: string, force = false) =>
    request<{ queued_count: number }>(`/playlists/${encodeURIComponent(playlistId)}/analysis/enqueue`, {
      method: "POST",
      body: JSON.stringify({ analysis_mode: "staged", force }),
    }),
  snapshot: (playlistId: string) =>
    request<Record<string, unknown>>("/sync/playlists/snapshot", {
      method: "POST",
      body: JSON.stringify({ playlist_id: playlistId }),
    }),
  toolCommand: (body: ToolCommandRequest) =>
    request<ToolCommandResult>("/tools/cli", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  pickPath: (body: PickPathRequest) =>
    request<PickPathResult>("/tools/pick-path", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
