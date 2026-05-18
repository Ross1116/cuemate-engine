package recommendationsrepo

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

var (
	ErrPlaylistNotFound            = errors.New("playlist not found")
	ErrTrackNotFound               = errors.New("track not found in playlist")
	ErrRecommendationEventNotFound = errors.New("recommendation event not found")
	ErrRelativeRefreshNeeded       = errors.New("playlist relative artifacts require refresh")
	ErrSnapshotNotFound            = errors.New("snapshot not found")
)

type PlaylistRef struct {
	ID   string
	Name string
}

type PlaylistSummary struct {
	ID                  string
	Name                string
	TrackCount          int
	CreatedAt           string
	UpdatedAt           string
	TrackCountAnalyzed  int
	EligibleTrackCount  int
	IsStale             bool
	StaleReason         string
	FeedbackEventCount  int
	FeedbackLastTunedAt *string
}

type AnalysisJobCounts struct {
	Pending   int
	Running   int
	Completed int
	Failed    int
}

type PlaylistAnalysisStatus struct {
	PlaylistID      string
	PlaylistName    string
	TotalTracks     int
	ReadyTracks     int
	Counts          AnalysisJobCounts
	IsStale         bool
	StaleReason     string
	LatestError     *string
	PercentComplete int
	NextAction      string
}

type TrackContextRecord struct {
	TrackID                   string
	FilePath                  string
	FileHash                  *string
	Title                     string
	Artist                    string
	Position                  int
	BPM                       *float64
	Key                       *string
	KeyConfidence             *float64
	KeySource                 *string
	KeyAgreement              *int32
	EnergyRel                 *float64
	BassRel                   *float64
	DrumsRel                  *float64
	VocalsRel                 *float64
	GrooveRel                 *float64
	IntensityBand             *string
	RoleHints                 []string
	AnalysisSignature         *string
	ConfigSignature           *string
	ScoringContractAtAnalysis *string
}

func (r TrackContextRecord) Scoreable() bool {
	return r.BPM != nil && r.Key != nil && r.AnalysisSignature != nil && r.ConfigSignature != nil && r.ScoringContractAtAnalysis != nil
}

type PlaylistStats struct {
	PlaylistID            string
	TrackCountTotal       int
	TrackCountAnalyzed    int
	EligibleTrackCount    int
	EnergySpread          *float64
	RelativeSignature     string
	IsStale               bool
	StaleReason           string
	AdaptedWeights        map[string]float64
	FeedbackTunedWeights  map[string]float64
	FeedbackTuningNotes   []string
	FeedbackEventCount    int
	FeedbackLastTunedAt   *string
	FeedbackTuningMetrics map[string]any
}

type RecommendationEventRecord struct {
	ID                       string
	PlaylistID               string
	CurrentTrackID           string
	Target                   string
	CandidateCount           int
	RecommendationConfidence *float64
	RecommendationsStatus    string
	LanesReturnedJSON        string
	TrackChosen              *string
	ChosenWasRecommended     *bool
	SkippedOverJSON          *string
	AdaptedWeightsJSON       *string
	ScoringContractID        string
	Timestamp                string
	PlayedAt                 *string
}

type RecommendationEventItemRecord struct {
	EventID                string
	LaneID                 string
	LaneRank               int
	CandidateTrackID       string
	FinalScore             float64
	RawScore               float64
	PenaltyMultiplier      float64
	Move                   string
	MoveConfidence         float64
	Risk                   string
	RiskScore              float64
	PrimaryLane            *string
	SecondaryLane          bool
	ComponentScoresJSON    string
	ConfidencesJSON        string
	WeightsUsedJSON        string
	TransitionFeaturesJSON string
}

type ManualCorrectionRecord struct {
	ID          string
	TrackID     string
	Field       string
	OldValue    string
	NewValue    string
	CorrectedAt string
}

type HydratedRecommendations struct {
	Playlist   PlaylistRef
	Current    TrackContextRecord
	Candidates []TrackContextRecord
	History    []TrackContextRecord
	HasGaps    bool
	Stats      *PlaylistStats
}

type PlaylistTrackSnapshot struct {
	TrackID       string
	Title         string
	Artist        string
	Position      int
	BPM           *float64
	Key           *string
	IntensityBand *string
	RoleHints     []string
	AnalysisState string
}

type TrackFeatureDetail struct {
	TrackID                    string
	Title                      string
	Artist                     string
	BPM                        *float64
	Key                        *string
	KeyConfidence              *float64
	KeySource                  *string
	KeyAgreement               *int32
	EnergyAbs                  *float64
	EnergyHeuristicAbs         *float64
	EnergySustained            *float64
	EnergyPeak                 *float64
	LoudnessNorm               *float64
	BassAbs                    *float64
	DrumsAbs                   *float64
	HarmonicAbs                *float64
	GrooveAbs                  *float64
	VocalsAbs                  *float64
	VocalsConfidence           *float64
	DanceabilityAbs            *float64
	ArousalAbs                 *float64
	ValenceAbs                 *float64
	MoodAggressiveAbs          *float64
	MoodPartyAbs               *float64
	MoodRelaxedAbs             *float64
	EnergyEssentiaFused        *float64
	EnergyEssentiaBucket       *string
	EssentiaSemanticSource     *string
	EssentiaSemanticInferredAt *string
	AnalysisMode               *string
	AnalyzedAt                 *string
	AnalysisSignature          *string
	ConfigSignature            *string
	ScoringContractAtAnalysis  *string
	EnergyRel                  *float64
	BassRel                    *float64
	DrumsRel                   *float64
	VocalsRel                  *float64
	GrooveRel                  *float64
	EnergySpread               *float64
	BassSpread                 *float64
	DrumsSpread                *float64
	VocalsSpread               *float64
	GrooveSpread               *float64
	IntensityBand              *string
	IntensityMembership        map[string]float64
	RoleHints                  []string
}

type PlaylistSyncState struct {
	PlaylistID              string
	LastSnapshotID          string
	LastSnapshotGeneratedAt string
	LastSnapshotAckedAt     *string
	UpdatedAt               string
}

type execContexter interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
}

type prepareContexter interface {
	PrepareContext(ctx context.Context, query string) (*sql.Stmt, error)
}

type queryRowContexter interface {
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

type SyncOutboxItem struct {
	ID          int64
	EntityType  string
	EntityID    string
	Action      string
	PayloadJSON string
	CreatedAt   string
	SyncedAt    *string
}

type RemotePairingToken struct {
	TokenHash   string
	CreatedAt   string
	ExpiresAt   string
	UsedAt      *string
	DeviceLabel *string
}

type RemoteSession struct {
	SessionHash string
	DeviceLabel *string
	CreatedAt   string
	LastSeenAt  string
	ExpiresAt   string
	RevokedAt   *string
}

type FeedbackTuningJobRecord struct {
	ID             int64
	PlaylistID     string
	Status         string
	TriggerEventID *string
	CreatedAt      string
	StartedAt      *string
	FinishedAt     *string
	ErrorMessage   *string
}

type AnalysisJobRecord struct {
	ID              int64
	PlaylistID      *string
	TrackID         *string
	TrackPath       string
	Status          string
	Priority        int
	AnalysisMode    string
	JobKind         string
	ErrorMessage    *string
	DurationSeconds *float64
	CreatedAt       string
	StartedAt       *string
	CompletedAt     *string
}

type Repository struct {
	db *sql.DB
}

func Open(path string) (*Repository, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite database: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	if _, err := db.Exec("PRAGMA busy_timeout = 5000"); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("configure sqlite busy timeout: %w", err)
	}
	return &Repository{db: db}, nil
}

func (r *Repository) Close() error {
	return r.db.Close()
}

func (r *Repository) DB() *sql.DB {
	return r.db
}

func (r *Repository) RunInTx(ctx context.Context, fn func(*sql.Tx) error) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() {
		_ = tx.Rollback()
	}()
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit()
}

func (r *Repository) ResolvePlaylist(ctx context.Context, playlistID, playlistName string) (PlaylistRef, error) {
	switch {
	case strings.TrimSpace(playlistID) != "" && strings.TrimSpace(playlistName) != "":
		return PlaylistRef{}, errors.New("provide only one of playlist_id or playlist_name")
	case strings.TrimSpace(playlistID) == "" && strings.TrimSpace(playlistName) == "":
		return PlaylistRef{}, errors.New("playlist_id or playlist_name is required")
	}

	var row *sql.Row
	if strings.TrimSpace(playlistID) != "" {
		row = r.db.QueryRowContext(ctx, "SELECT id, name FROM playlists WHERE id = ?", playlistID)
	} else {
		row = r.db.QueryRowContext(ctx, "SELECT id, name FROM playlists WHERE name = ?", playlistName)
	}

	var playlist PlaylistRef
	if err := row.Scan(&playlist.ID, &playlist.Name); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return PlaylistRef{}, ErrPlaylistNotFound
		}
		return PlaylistRef{}, err
	}
	return playlist, nil
}

func (r *Repository) ListPlaylists(ctx context.Context) ([]PlaylistSummary, error) {
	rows, err := r.db.QueryContext(ctx, `
		SELECT
		  p.id,
		  p.name,
		  p.track_count,
		  p.created_at,
		  p.updated_at,
		  COALESCE(ps.track_count_analyzed, 0),
		  COALESCE(ps.eligible_track_count, 0),
		  COALESCE(ps.is_stale, 0),
		  COALESCE(ps.stale_reason, ''),
		  COALESCE(ps.feedback_event_count, 0),
		  ps.feedback_last_tuned_at
		FROM playlists p
		LEFT JOIN playlist_stats ps ON ps.playlist_id = p.id
		ORDER BY p.updated_at DESC, p.name ASC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []PlaylistSummary
	for rows.Next() {
		var item PlaylistSummary
		var staleInt int64
		var tunedAt sql.NullString
		if err := rows.Scan(
			&item.ID,
			&item.Name,
			&item.TrackCount,
			&item.CreatedAt,
			&item.UpdatedAt,
			&item.TrackCountAnalyzed,
			&item.EligibleTrackCount,
			&staleInt,
			&item.StaleReason,
			&item.FeedbackEventCount,
			&tunedAt,
		); err != nil {
			return nil, err
		}
		item.IsStale = staleInt != 0
		if tunedAt.Valid {
			item.FeedbackLastTunedAt = &tunedAt.String
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (r *Repository) GetPlaylistAnalysisStatus(ctx context.Context, playlistID string) (PlaylistAnalysisStatus, error) {
	playlist, err := r.ResolvePlaylist(ctx, playlistID, "")
	if err != nil {
		return PlaylistAnalysisStatus{}, err
	}
	tracks, err := r.GetPlaylistSnapshotTracks(ctx, playlist.ID)
	if err != nil {
		return PlaylistAnalysisStatus{}, err
	}
	status := PlaylistAnalysisStatus{
		PlaylistID:   playlist.ID,
		PlaylistName: playlist.Name,
		TotalTracks:  len(tracks),
		NextAction:   "none",
	}
	for _, track := range tracks {
		if track.AnalysisState == "ready" {
			status.ReadyTracks++
		}
	}
	stats, err := r.GetPlaylistStats(ctx, playlist.ID)
	if err != nil {
		return PlaylistAnalysisStatus{}, err
	}
	if stats != nil {
		status.IsStale = stats.IsStale
		status.StaleReason = stats.StaleReason
	}
	var latestError sql.NullString
	rows, err := r.db.QueryContext(ctx, `
		SELECT status, COUNT(*)
		FROM analysis_jobs
		WHERE playlist_id = ?
		GROUP BY status
	`, playlist.ID)
	if err != nil {
		return PlaylistAnalysisStatus{}, err
	}
	for rows.Next() {
		var jobStatus string
		var count int
		if err := rows.Scan(&jobStatus, &count); err != nil {
			_ = rows.Close()
			return PlaylistAnalysisStatus{}, err
		}
		switch jobStatus {
		case "pending":
			status.Counts.Pending = count
		case "running":
			status.Counts.Running = count
		case "completed":
			status.Counts.Completed = count
		case "failed":
			status.Counts.Failed = count
		}
	}
	if err := rows.Close(); err != nil {
		return PlaylistAnalysisStatus{}, err
	}
	if err := r.db.QueryRowContext(ctx, `
		SELECT error_message
		FROM analysis_jobs
		WHERE playlist_id = ? AND status = 'failed' AND error_message IS NOT NULL
		ORDER BY completed_at DESC, started_at DESC, created_at DESC, id DESC
		LIMIT 1
	`, playlist.ID).Scan(&latestError); err != nil && !errors.Is(err, sql.ErrNoRows) {
		return PlaylistAnalysisStatus{}, err
	}
	if latestError.Valid {
		status.LatestError = &latestError.String
	}
	if status.TotalTracks > 0 {
		status.PercentComplete = int(float64(status.ReadyTracks) / float64(status.TotalTracks) * 100)
	}
	switch {
	case status.Counts.Failed > 0:
		status.NextAction = "inspect_failures"
	case status.Counts.Running > 0 || status.Counts.Pending > 0:
		status.NextAction = "run_worker"
	case status.IsStale || status.ReadyTracks < status.TotalTracks:
		status.NextAction = "smart_refresh"
	default:
		status.NextAction = "none"
	}
	return status, nil
}

func (r *Repository) DeletePlaylist(ctx context.Context, playlistID string) error {
	playlist, err := r.ResolvePlaylist(ctx, playlistID, "")
	if err != nil {
		return err
	}
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() {
		if tx != nil {
			_ = tx.Rollback()
		}
	}()
	for _, statement := range []string{
		`DELETE FROM recommendation_event_items WHERE event_id IN (SELECT id FROM recommendation_events WHERE playlist_id = ?)`,
		`DELETE FROM feedback_tuning_jobs WHERE playlist_id = ?`,
		`DELETE FROM recommendation_events WHERE playlist_id = ?`,
		`DELETE FROM playlist_sync_state WHERE playlist_id = ?`,
		`DELETE FROM track_features_rel WHERE playlist_id = ?`,
		`DELETE FROM playlist_stats WHERE playlist_id = ?`,
		`DELETE FROM analysis_jobs WHERE playlist_id = ?`,
		`DELETE FROM playlist_tracks WHERE playlist_id = ?`,
		`DELETE FROM playlists WHERE id = ?`,
	} {
		if _, err := tx.ExecContext(ctx, statement, playlist.ID); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	tx = nil
	return nil
}

func (r *Repository) HydrateRecommendations(
	ctx context.Context,
	playlist PlaylistRef,
	currentTrackID string,
	historyTrackIDs []string,
) (*HydratedRecommendations, error) {
	current, err := r.getTrackContext(ctx, playlist.ID, currentTrackID)
	if err != nil {
		return nil, err
	}

	stats, err := r.getPlaylistStats(ctx, playlist.ID)
	if err != nil {
		return nil, err
	}

	candidates, err := r.getScoringCandidates(ctx, playlist.ID)
	if err != nil {
		return nil, err
	}

	history := make([]TrackContextRecord, 0, len(historyTrackIDs))
	hasGaps := false
	for _, trackID := range historyTrackIDs {
		item, err := r.getTrackContext(ctx, playlist.ID, trackID)
		if err != nil {
			if errors.Is(err, ErrTrackNotFound) {
				hasGaps = true
				continue
			}
			return nil, err
		}
		if item.EnergyRel == nil {
			hasGaps = true
		}
		history = append(history, item)
	}

	return &HydratedRecommendations{
		Playlist:   playlist,
		Current:    current,
		Candidates: candidates,
		History:    history,
		HasGaps:    hasGaps,
		Stats:      stats,
	}, nil
}

func (r *Repository) GetPlaylistSnapshotTracks(ctx context.Context, playlistID string) ([]PlaylistTrackSnapshot, error) {
	rows, err := r.db.QueryContext(
		ctx,
		`
		SELECT
		  t.id,
		  COALESCE(t.title, ''),
		  COALESCE(t.artist, ''),
		  pt.position,
		  f.bpm,
		  f.key,
		  r.intensity_band,
		  r.role_hints,
		  r.track_id,
		  f.analysis_signature,
		  f.config_signature,
		  f.scoring_contract_id_at_analysis
		FROM playlist_tracks pt
		JOIN tracks t ON t.id = pt.track_id
		LEFT JOIN track_features_abs f ON f.track_id = t.id
		LEFT JOIN track_features_rel r ON r.track_id = t.id AND r.playlist_id = pt.playlist_id
		WHERE pt.playlist_id = ?
		ORDER BY pt.position ASC
		`,
		playlistID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []PlaylistTrackSnapshot
	for rows.Next() {
		var track PlaylistTrackSnapshot
		var bpm sql.NullFloat64
		var musicalKey sql.NullString
		var intensityBand sql.NullString
		var roleHints sql.NullString
		var relTrackID sql.NullString
		var analysisSig sql.NullString
		var configSig sql.NullString
		var scoringContract sql.NullString
		if err := rows.Scan(
			&track.TrackID,
			&track.Title,
			&track.Artist,
			&track.Position,
			&bpm,
			&musicalKey,
			&intensityBand,
			&roleHints,
			&relTrackID,
			&analysisSig,
			&configSig,
			&scoringContract,
		); err != nil {
			return nil, err
		}
		if bpm.Valid {
			track.BPM = &bpm.Float64
		}
		if musicalKey.Valid {
			track.Key = &musicalKey.String
		}
		if intensityBand.Valid {
			track.IntensityBand = &intensityBand.String
		}
		if roleHints.Valid && strings.TrimSpace(roleHints.String) != "" {
			if err := json.Unmarshal([]byte(roleHints.String), &track.RoleHints); err != nil {
				log.Printf("warning: failed to decode playlist snapshot role_hints %q: %v", roleHints.String, err)
			}
		}
		switch {
		case !analysisSig.Valid || !configSig.Valid || !scoringContract.Valid || !relTrackID.Valid:
			track.AnalysisState = "unanalysed"
		default:
			track.AnalysisState = "ready"
		}
		out = append(out, track)
	}
	return out, rows.Err()
}

func (r *Repository) ListPlaylistTracks(ctx context.Context, playlistID, query, analysisState string, limit, offset int) ([]PlaylistTrackSnapshot, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}
	tracks, err := r.GetPlaylistSnapshotTracks(ctx, playlistID)
	if err != nil {
		return nil, err
	}
	query = strings.ToLower(strings.TrimSpace(query))
	analysisState = strings.TrimSpace(analysisState)
	filtered := make([]PlaylistTrackSnapshot, 0, len(tracks))
	for _, track := range tracks {
		if query != "" {
			haystack := strings.ToLower(track.Title + " " + track.Artist + " " + track.TrackID)
			if !strings.Contains(haystack, query) {
				continue
			}
		}
		if analysisState != "" && track.AnalysisState != analysisState {
			continue
		}
		filtered = append(filtered, track)
	}
	if offset >= len(filtered) {
		return []PlaylistTrackSnapshot{}, nil
	}
	end := offset + limit
	if end > len(filtered) {
		end = len(filtered)
	}
	return filtered[offset:end], nil
}

func (r *Repository) GetTrackFeatureDetail(ctx context.Context, playlistID, trackID string) (TrackFeatureDetail, error) {
	row := r.db.QueryRowContext(ctx, `
		SELECT
		  t.id,
		  COALESCE(t.title, ''),
		  COALESCE(t.artist, ''),
		  f.bpm,
		  f.key,
		  f.key_confidence,
		  f.key_source,
		  f.key_agreement,
		  f.energy_abs,
		  f.energy_heuristic_abs,
		  f.energy_sustained,
		  f.energy_peak,
		  f.loudness_norm,
		  f.bass_abs,
		  f.drums_abs,
		  f.harmonic_abs,
		  f.groove_abs,
		  f.vocals_abs,
		  f.vocals_confidence,
		  f.danceability_abs,
		  f.arousal_abs,
		  f.valence_abs,
		  f.mood_aggressive_abs,
		  f.mood_party_abs,
		  f.mood_relaxed_abs,
		  f.energy_essentia_fused,
		  f.energy_essentia_bucket,
		  f.essentia_semantic_source,
		  f.essentia_semantic_inferred_at,
		  f.analysis_mode,
		  f.analyzed_at,
		  f.analysis_signature,
		  f.config_signature,
		  f.scoring_contract_id_at_analysis,
		  rel.energy_rel,
		  rel.bass_rel,
		  rel.drums_rel,
		  rel.vocals_rel,
		  rel.groove_rel,
		  rel.energy_spread,
		  rel.bass_spread,
		  rel.drums_spread,
		  rel.vocals_spread,
		  rel.groove_spread,
		  rel.intensity_band,
		  rel.intensity_membership,
		  rel.role_hints
		FROM playlist_tracks pt
		JOIN tracks t ON t.id = pt.track_id
		LEFT JOIN track_features_abs f ON f.track_id = t.id
		LEFT JOIN track_features_rel rel ON rel.track_id = t.id AND rel.playlist_id = pt.playlist_id
		WHERE pt.playlist_id = ? AND t.id = ?
	`, playlistID, trackID)

	var detail TrackFeatureDetail
	var bpm, keyConfidence, energyAbs, energyHeuristicAbs, energySustained, energyPeak sql.NullFloat64
	var loudnessNorm, bassAbs, drumsAbs, harmonicAbs, grooveAbs, vocalsAbs, vocalsConfidence sql.NullFloat64
	var danceabilityAbs, arousalAbs, valenceAbs, moodAggressiveAbs, moodPartyAbs, moodRelaxedAbs sql.NullFloat64
	var energyEssentiaFused sql.NullFloat64
	var energyRel, bassRel, drumsRel, vocalsRel, grooveRel sql.NullFloat64
	var energySpread, bassSpread, drumsSpread, vocalsSpread, grooveSpread sql.NullFloat64
	var keyAgreement sql.NullInt64
	var key, keySource, energyEssentiaBucket, semanticSource, semanticInferredAt sql.NullString
	var analysisMode, analyzedAt, analysisSignature, configSignature, scoringContract sql.NullString
	var intensityBand, intensityMembership, roleHints sql.NullString
	if err := row.Scan(
		&detail.TrackID,
		&detail.Title,
		&detail.Artist,
		&bpm,
		&key,
		&keyConfidence,
		&keySource,
		&keyAgreement,
		&energyAbs,
		&energyHeuristicAbs,
		&energySustained,
		&energyPeak,
		&loudnessNorm,
		&bassAbs,
		&drumsAbs,
		&harmonicAbs,
		&grooveAbs,
		&vocalsAbs,
		&vocalsConfidence,
		&danceabilityAbs,
		&arousalAbs,
		&valenceAbs,
		&moodAggressiveAbs,
		&moodPartyAbs,
		&moodRelaxedAbs,
		&energyEssentiaFused,
		&energyEssentiaBucket,
		&semanticSource,
		&semanticInferredAt,
		&analysisMode,
		&analyzedAt,
		&analysisSignature,
		&configSignature,
		&scoringContract,
		&energyRel,
		&bassRel,
		&drumsRel,
		&vocalsRel,
		&grooveRel,
		&energySpread,
		&bassSpread,
		&drumsSpread,
		&vocalsSpread,
		&grooveSpread,
		&intensityBand,
		&intensityMembership,
		&roleHints,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return TrackFeatureDetail{}, ErrTrackNotFound
		}
		return TrackFeatureDetail{}, err
	}

	detail.BPM = nullableFloatPtr(bpm)
	detail.Key = nullableStringPtr(key)
	detail.KeyConfidence = nullableFloatPtr(keyConfidence)
	detail.KeySource = nullableStringPtr(keySource)
	if keyAgreement.Valid {
		value := int32(keyAgreement.Int64)
		detail.KeyAgreement = &value
	}
	detail.EnergyAbs = nullableFloatPtr(energyAbs)
	detail.EnergyHeuristicAbs = nullableFloatPtr(energyHeuristicAbs)
	detail.EnergySustained = nullableFloatPtr(energySustained)
	detail.EnergyPeak = nullableFloatPtr(energyPeak)
	detail.LoudnessNorm = nullableFloatPtr(loudnessNorm)
	detail.BassAbs = nullableFloatPtr(bassAbs)
	detail.DrumsAbs = nullableFloatPtr(drumsAbs)
	detail.HarmonicAbs = nullableFloatPtr(harmonicAbs)
	detail.GrooveAbs = nullableFloatPtr(grooveAbs)
	detail.VocalsAbs = nullableFloatPtr(vocalsAbs)
	detail.VocalsConfidence = nullableFloatPtr(vocalsConfidence)
	detail.DanceabilityAbs = nullableFloatPtr(danceabilityAbs)
	detail.ArousalAbs = nullableFloatPtr(arousalAbs)
	detail.ValenceAbs = nullableFloatPtr(valenceAbs)
	detail.MoodAggressiveAbs = nullableFloatPtr(moodAggressiveAbs)
	detail.MoodPartyAbs = nullableFloatPtr(moodPartyAbs)
	detail.MoodRelaxedAbs = nullableFloatPtr(moodRelaxedAbs)
	detail.EnergyEssentiaFused = nullableFloatPtr(energyEssentiaFused)
	detail.EnergyEssentiaBucket = nullableStringPtr(energyEssentiaBucket)
	detail.EssentiaSemanticSource = nullableStringPtr(semanticSource)
	detail.EssentiaSemanticInferredAt = nullableStringPtr(semanticInferredAt)
	detail.AnalysisMode = nullableStringPtr(analysisMode)
	detail.AnalyzedAt = nullableStringPtr(analyzedAt)
	detail.AnalysisSignature = nullableStringPtr(analysisSignature)
	detail.ConfigSignature = nullableStringPtr(configSignature)
	detail.ScoringContractAtAnalysis = nullableStringPtr(scoringContract)
	detail.EnergyRel = nullableFloatPtr(energyRel)
	detail.BassRel = nullableFloatPtr(bassRel)
	detail.DrumsRel = nullableFloatPtr(drumsRel)
	detail.VocalsRel = nullableFloatPtr(vocalsRel)
	detail.GrooveRel = nullableFloatPtr(grooveRel)
	detail.EnergySpread = nullableFloatPtr(energySpread)
	detail.BassSpread = nullableFloatPtr(bassSpread)
	detail.DrumsSpread = nullableFloatPtr(drumsSpread)
	detail.VocalsSpread = nullableFloatPtr(vocalsSpread)
	detail.GrooveSpread = nullableFloatPtr(grooveSpread)
	detail.IntensityBand = nullableStringPtr(intensityBand)
	if intensityMembership.Valid && strings.TrimSpace(intensityMembership.String) != "" {
		if err := json.Unmarshal([]byte(intensityMembership.String), &detail.IntensityMembership); err != nil {
			log.Printf("warning: failed to decode intensity_membership %q: %v", intensityMembership.String, err)
		}
	}
	if roleHints.Valid && strings.TrimSpace(roleHints.String) != "" {
		if err := json.Unmarshal([]byte(roleHints.String), &detail.RoleHints); err != nil {
			log.Printf("warning: failed to decode role_hints %q: %v", roleHints.String, err)
		}
	}
	return detail, nil
}

func nullableFloatPtr(value sql.NullFloat64) *float64 {
	if !value.Valid {
		return nil
	}
	return &value.Float64
}

func nullableStringPtr(value sql.NullString) *string {
	if !value.Valid {
		return nil
	}
	return &value.String
}

func (r *Repository) SearchTracks(ctx context.Context, playlistID, query string, limit int) ([]PlaylistTrackSnapshot, error) {
	if strings.TrimSpace(playlistID) != "" {
		return r.ListPlaylistTracks(ctx, playlistID, query, "", limit, 0)
	}
	if limit <= 0 || limit > 100 {
		limit = 25
	}
	like := "%" + strings.ToLower(strings.TrimSpace(query)) + "%"
	rows, err := r.db.QueryContext(ctx, `
		SELECT
		  t.id,
		  COALESCE(t.title, ''),
		  COALESCE(t.artist, ''),
		  0,
		  f.bpm,
		  f.key,
		  NULL,
		  NULL,
		  f.analysis_signature,
		  f.config_signature,
		  f.scoring_contract_id_at_analysis
		FROM tracks t
		LEFT JOIN track_features_abs f ON f.track_id = t.id
		WHERE ? = '%%' OR lower(t.title) LIKE ? OR lower(t.artist) LIKE ? OR lower(t.id) LIKE ?
		ORDER BY COALESCE(t.artist, ''), COALESCE(t.title, ''), t.id
		LIMIT ?
	`, like, like, like, like, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []PlaylistTrackSnapshot
	for rows.Next() {
		var track PlaylistTrackSnapshot
		var bpm sql.NullFloat64
		var musicalKey sql.NullString
		var intensityBand sql.NullString
		var roleHints sql.NullString
		var analysisSig sql.NullString
		var configSig sql.NullString
		var scoringContract sql.NullString
		if err := rows.Scan(
			&track.TrackID,
			&track.Title,
			&track.Artist,
			&track.Position,
			&bpm,
			&musicalKey,
			&intensityBand,
			&roleHints,
			&analysisSig,
			&configSig,
			&scoringContract,
		); err != nil {
			return nil, err
		}
		if bpm.Valid {
			track.BPM = &bpm.Float64
		}
		if musicalKey.Valid {
			track.Key = &musicalKey.String
		}
		if intensityBand.Valid {
			track.IntensityBand = &intensityBand.String
		}
		if roleHints.Valid && strings.TrimSpace(roleHints.String) != "" {
			_ = json.Unmarshal([]byte(roleHints.String), &track.RoleHints)
		}
		if !analysisSig.Valid || !configSig.Valid || !scoringContract.Valid {
			track.AnalysisState = "unanalysed"
		} else {
			track.AnalysisState = "ready"
		}
		out = append(out, track)
	}
	return out, rows.Err()
}

func (r *Repository) GetTrackForCorrection(ctx context.Context, trackID string) (TrackContextRecord, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT
		  t.id,
		  t.file_path,
		  t.file_hash,
		  COALESCE(t.title, ''),
		  COALESCE(t.artist, ''),
		  0,
		  f.bpm,
		  f.key,
		  f.key_confidence,
		  f.key_source,
		  f.key_agreement,
		  NULL,
		  NULL,
		  NULL,
		  NULL,
		  NULL,
		  NULL,
		  NULL,
		  f.analysis_signature,
		  f.config_signature,
		  f.scoring_contract_id_at_analysis
		FROM tracks t
		LEFT JOIN track_features_abs f ON f.track_id = t.id
		WHERE t.id = ?
		`,
		trackID,
	)
	record, err := scanTrackContext(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return TrackContextRecord{}, ErrTrackNotFound
		}
		return TrackContextRecord{}, err
	}
	return record, nil
}

func (r *Repository) QueuePlaylistAnalysis(
	ctx context.Context,
	playlistID string,
	analysisMode string,
	force bool,
	analysisSignature string,
	configSignature string,
) (int, error) {
	if strings.TrimSpace(analysisMode) == "" {
		analysisMode = "staged"
	}
	rows, err := r.db.QueryContext(ctx, `
		SELECT
		  t.id,
		  t.file_path,
		  t.file_hash,
		  f.track_id,
		  f.analysis_signature,
		  f.config_signature
		FROM playlist_tracks pt
		JOIN tracks t ON t.id = pt.track_id
		LEFT JOIN track_features_abs f ON f.track_id = t.id
		WHERE pt.playlist_id = ?
		ORDER BY pt.position ASC
	`, playlistID)
	if err != nil {
		return 0, err
	}
	defer rows.Close()

	type pendingTrack struct {
		id                string
		path              string
		fileHash          *string
		analyzed          bool
		storedAnalysisSig *string
		storedConfigSig   *string
	}
	var tracks []pendingTrack
	for rows.Next() {
		var item pendingTrack
		var fileHash sql.NullString
		var analyzedID sql.NullString
		var storedAnalysisSig sql.NullString
		var storedConfigSig sql.NullString
		if err := rows.Scan(&item.id, &item.path, &fileHash, &analyzedID, &storedAnalysisSig, &storedConfigSig); err != nil {
			return 0, err
		}
		if fileHash.Valid {
			item.fileHash = &fileHash.String
		}
		if storedAnalysisSig.Valid {
			item.storedAnalysisSig = &storedAnalysisSig.String
		}
		if storedConfigSig.Valid {
			item.storedConfigSig = &storedConfigSig.String
		}
		item.analyzed = analyzedID.Valid
		tracks = append(tracks, item)
	}
	if err := rows.Err(); err != nil {
		return 0, err
	}

	count := 0
	for _, track := range tracks {
		if track.analyzed && !force && stringPtrEqual(track.storedAnalysisSig, analysisSignature) && stringPtrEqual(track.storedConfigSig, configSignature) {
			continue
		}
		if _, err := r.CreateAnalysisJobWithKind(
			ctx,
			&playlistID,
			track.id,
			track.path,
			"enrichment",
			analysisMode,
			analysisSignature,
			configSignature,
			track.fileHash,
			50,
			NowUTC(),
		); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

func stringPtrEqual(value *string, expected string) bool {
	return value != nil && *value == expected
}

func (r *Repository) ListAnalysisJobs(ctx context.Context, playlistID, status string, limit int) ([]AnalysisJobRecord, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	where := []string{"1=1"}
	args := []any{}
	if strings.TrimSpace(playlistID) != "" {
		where = append(where, "playlist_id = ?")
		args = append(args, playlistID)
	}
	if strings.TrimSpace(status) != "" {
		where = append(where, "status = ?")
		args = append(args, status)
	}
	args = append(args, limit)
	rows, err := r.db.QueryContext(ctx, fmt.Sprintf(`
		SELECT
		  id, playlist_id, track_id, track_path, status, priority, analysis_mode,
		  job_kind, error_message, duration_seconds, created_at, started_at, completed_at
		FROM analysis_jobs
		WHERE %s
		ORDER BY created_at DESC, id DESC
		LIMIT ?
	`, strings.Join(where, " AND ")), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []AnalysisJobRecord
	for rows.Next() {
		var item AnalysisJobRecord
		var playlistIDValue sql.NullString
		var trackID sql.NullString
		var errorMessage sql.NullString
		var duration sql.NullFloat64
		var startedAt sql.NullString
		var completedAt sql.NullString
		if err := rows.Scan(
			&item.ID,
			&playlistIDValue,
			&trackID,
			&item.TrackPath,
			&item.Status,
			&item.Priority,
			&item.AnalysisMode,
			&item.JobKind,
			&errorMessage,
			&duration,
			&item.CreatedAt,
			&startedAt,
			&completedAt,
		); err != nil {
			return nil, err
		}
		if playlistIDValue.Valid {
			item.PlaylistID = &playlistIDValue.String
		}
		if trackID.Valid {
			item.TrackID = &trackID.String
		}
		if errorMessage.Valid {
			item.ErrorMessage = &errorMessage.String
		}
		if duration.Valid {
			item.DurationSeconds = &duration.Float64
		}
		if startedAt.Valid {
			item.StartedAt = &startedAt.String
		}
		if completedAt.Valid {
			item.CompletedAt = &completedAt.String
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (r *Repository) GetTrackImportedValues(ctx context.Context, trackID string) (*float64, *string, error) {
	row := r.db.QueryRowContext(ctx, "SELECT imported_bpm, imported_key FROM tracks WHERE id = ?", trackID)
	var importedBPM sql.NullFloat64
	var importedKey sql.NullString
	if err := row.Scan(&importedBPM, &importedKey); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil, ErrTrackNotFound
		}
		return nil, nil, err
	}
	var bpm *float64
	var key *string
	if importedBPM.Valid {
		bpm = &importedBPM.Float64
	}
	if importedKey.Valid {
		key = &importedKey.String
	}
	return bpm, key, nil
}

func (r *Repository) UpdateTrackImportedBPM(ctx context.Context, trackID string, bpm float64, updatedAt string) error {
	return r.updateTrackImportedBPM(ctx, r.db, trackID, bpm, updatedAt)
}

func (r *Repository) UpdateTrackImportedBPMTx(ctx context.Context, tx *sql.Tx, trackID string, bpm float64, updatedAt string) error {
	return r.updateTrackImportedBPM(ctx, tx, trackID, bpm, updatedAt)
}

func (r *Repository) UpdateTrackImportedKey(ctx context.Context, trackID string, key string, updatedAt string) error {
	return r.updateTrackImportedKey(ctx, r.db, trackID, key, updatedAt)
}

func (r *Repository) UpdateTrackImportedKeyTx(ctx context.Context, tx *sql.Tx, trackID string, key string, updatedAt string) error {
	return r.updateTrackImportedKey(ctx, tx, trackID, key, updatedAt)
}

func (r *Repository) updateTrackImportedBPM(ctx context.Context, exec execContexter, trackID string, bpm float64, updatedAt string) error {
	result, err := exec.ExecContext(
		ctx,
		"UPDATE tracks SET imported_bpm = ?, updated_at = ? WHERE id = ?",
		bpm,
		updatedAt,
		trackID,
	)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return ErrTrackNotFound
	}
	return nil
}

func (r *Repository) updateTrackImportedKey(ctx context.Context, exec execContexter, trackID string, key string, updatedAt string) error {
	result, err := exec.ExecContext(
		ctx,
		"UPDATE tracks SET imported_key = ?, updated_at = ? WHERE id = ?",
		key,
		updatedAt,
		trackID,
	)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return ErrTrackNotFound
	}
	return nil
}

func (r *Repository) GetPlaylistsContainingTrack(ctx context.Context, trackID string) ([]PlaylistRef, error) {
	rows, err := r.db.QueryContext(
		ctx,
		`
		SELECT p.id, p.name
		FROM playlist_tracks pt
		JOIN playlists p ON p.id = pt.playlist_id
		WHERE pt.track_id = ?
		ORDER BY p.name ASC
		`,
		trackID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []PlaylistRef
	for rows.Next() {
		var item PlaylistRef
		if err := rows.Scan(&item.ID, &item.Name); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (r *Repository) MarkPlaylistsStale(ctx context.Context, playlistIDs []string, reason, markedAt string) error {
	return r.markPlaylistsStale(ctx, r.db, playlistIDs, reason, markedAt)
}

func (r *Repository) MarkPlaylistsStaleTx(ctx context.Context, tx *sql.Tx, playlistIDs []string, reason, markedAt string) error {
	return r.markPlaylistsStale(ctx, tx, playlistIDs, reason, markedAt)
}

func (r *Repository) markPlaylistsStale(ctx context.Context, exec execContexter, playlistIDs []string, reason, markedAt string) error {
	if len(playlistIDs) == 0 {
		return nil
	}
	placeholders := make([]string, len(playlistIDs))
	args := make([]any, 0, len(playlistIDs)+2)
	args = append(args, reason, markedAt)
	for i, playlistID := range playlistIDs {
		placeholders[i] = "?"
		args = append(args, playlistID)
	}
	query := fmt.Sprintf(
		"UPDATE playlist_stats SET is_stale = 1, stale_reason = ?, stale_marked_at = ? WHERE playlist_id IN (%s)",
		strings.Join(placeholders, ", "),
	)
	_, err := exec.ExecContext(ctx, query, args...)
	return err
}

func (r *Repository) CreateAnalysisJobWithKind(
	ctx context.Context,
	playlistID *string,
	trackID string,
	trackPath string,
	jobKind string,
	analysisMode string,
	analysisSignature string,
	configSignature string,
	sourceFileHash *string,
	priority int,
	createdAt string,
) (int64, error) {
	var jobID int64
	err := r.RunInTx(ctx, func(tx *sql.Tx) error {
		var err error
		jobID, err = r.createAnalysisJobWithKindTx(
			ctx,
			tx,
			playlistID,
			trackID,
			trackPath,
			jobKind,
			analysisMode,
			analysisSignature,
			configSignature,
			sourceFileHash,
			priority,
			createdAt,
		)
		return err
	})
	if err != nil {
		return 0, err
	}
	return jobID, nil
}

func (r *Repository) CreateAnalysisJobWithKindTx(
	ctx context.Context,
	tx *sql.Tx,
	playlistID *string,
	trackID string,
	trackPath string,
	jobKind string,
	analysisMode string,
	analysisSignature string,
	configSignature string,
	sourceFileHash *string,
	priority int,
	createdAt string,
) (int64, error) {
	return r.createAnalysisJobWithKindTx(
		ctx,
		tx,
		playlistID,
		trackID,
		trackPath,
		jobKind,
		analysisMode,
		analysisSignature,
		configSignature,
		sourceFileHash,
		priority,
		createdAt,
	)
}

func (r *Repository) createAnalysisJobWithKindTx(
	ctx context.Context,
	exec execContexter,
	playlistID *string,
	trackID string,
	trackPath string,
	jobKind string,
	analysisMode string,
	analysisSignature string,
	configSignature string,
	sourceFileHash *string,
	priority int,
	createdAt string,
) (int64, error) {
	_, err := exec.ExecContext(
		ctx,
		`
		DELETE FROM analysis_jobs
		WHERE track_id = ?
		  AND job_kind = ?
		  AND status = 'pending'
		  AND analysis_signature = ?
		  AND config_signature = ?
		  AND playlist_id IS ?
		`,
		trackID,
		jobKind,
		analysisSignature,
		configSignature,
		nullString(playlistID),
	)
	if err != nil {
		return 0, err
	}
	result, err := exec.ExecContext(
		ctx,
		`
		INSERT INTO analysis_jobs (
		  playlist_id, track_id, track_path, job_kind, status, priority, analysis_mode,
		  analysis_signature, config_signature, source_file_hash, created_at
		) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
		`,
		nullString(playlistID),
		trackID,
		trackPath,
		jobKind,
		priority,
		analysisMode,
		analysisSignature,
		configSignature,
		nullString(sourceFileHash),
		createdAt,
	)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func (r *Repository) InsertManualCorrection(ctx context.Context, record ManualCorrectionRecord) error {
	return r.insertManualCorrection(ctx, r.db, record)
}

func (r *Repository) InsertManualCorrectionTx(ctx context.Context, tx *sql.Tx, record ManualCorrectionRecord) error {
	return r.insertManualCorrection(ctx, tx, record)
}

func (r *Repository) insertManualCorrection(ctx context.Context, exec execContexter, record ManualCorrectionRecord) error {
	_, err := exec.ExecContext(
		ctx,
		`
		INSERT INTO manual_corrections (
		  id, user_id, track_id, field, old_value, new_value, corrected_at
		) VALUES (?, 'local', ?, ?, ?, ?, ?)
		`,
		record.ID,
		record.TrackID,
		record.Field,
		record.OldValue,
		record.NewValue,
		record.CorrectedAt,
	)
	return err
}

func (r *Repository) InsertRecommendationEvent(ctx context.Context, record RecommendationEventRecord) error {
	return r.insertRecommendationEvent(ctx, r.db, record)
}

func (r *Repository) InsertRecommendationEventTx(ctx context.Context, tx *sql.Tx, record RecommendationEventRecord) error {
	return r.insertRecommendationEvent(ctx, tx, record)
}

func (r *Repository) insertRecommendationEvent(ctx context.Context, exec execContexter, record RecommendationEventRecord) error {
	_, err := exec.ExecContext(
		ctx,
		`
		INSERT INTO recommendation_events (
		  id, user_id, playlist_id, current_track_id, target, candidate_count,
		  recommendation_confidence, recommendations_status, lanes_returned, track_chosen,
		  chosen_was_recommended, skipped_over, adapted_weights, scoring_contract_id, timestamp, played_at
		) VALUES (?, 'local', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`,
		record.ID,
		record.PlaylistID,
		record.CurrentTrackID,
		record.Target,
		record.CandidateCount,
		nullFloat64(record.RecommendationConfidence),
		record.RecommendationsStatus,
		record.LanesReturnedJSON,
		nullString(record.TrackChosen),
		nullBool(record.ChosenWasRecommended),
		nullString(record.SkippedOverJSON),
		nullString(record.AdaptedWeightsJSON),
		record.ScoringContractID,
		record.Timestamp,
		nullString(record.PlayedAt),
	)
	return err
}

func (r *Repository) InsertRecommendationEventItems(ctx context.Context, items []RecommendationEventItemRecord) error {
	return r.RunInTx(ctx, func(tx *sql.Tx) error {
		return r.insertRecommendationEventItems(ctx, tx, items)
	})
}

func (r *Repository) InsertRecommendationEventItemsTx(ctx context.Context, tx *sql.Tx, items []RecommendationEventItemRecord) error {
	return r.insertRecommendationEventItems(ctx, tx, items)
}

func (r *Repository) insertRecommendationEventItems(ctx context.Context, exec execContexter, items []RecommendationEventItemRecord) error {
	if len(items) == 0 {
		return nil
	}
	preparer, ok := exec.(prepareContexter)
	if !ok {
		return errors.New("exec context does not support prepared statements")
	}
	stmt, err := preparer.PrepareContext(
		ctx,
		`
		INSERT INTO recommendation_event_items (
		  event_id, lane_id, lane_rank, candidate_track_id, final_score, raw_score,
		  penalty_multiplier, move, move_confidence, risk, risk_score, primary_lane,
		  secondary_lane, component_scores_json, confidences_json, weights_used_json,
		  transition_features_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`,
	)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, item := range items {
		secondaryLane := int64(0)
		if item.SecondaryLane {
			secondaryLane = 1
		}
		if _, err := stmt.ExecContext(
			ctx,
			item.EventID,
			item.LaneID,
			item.LaneRank,
			item.CandidateTrackID,
			item.FinalScore,
			item.RawScore,
			item.PenaltyMultiplier,
			item.Move,
			item.MoveConfidence,
			item.Risk,
			item.RiskScore,
			nullString(item.PrimaryLane),
			secondaryLane,
			item.ComponentScoresJSON,
			item.ConfidencesJSON,
			item.WeightsUsedJSON,
			item.TransitionFeaturesJSON,
		); err != nil {
			return err
		}
	}
	return nil
}

func (r *Repository) GetRecommendationEvent(ctx context.Context, eventID string) (*RecommendationEventRecord, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT
		  id, playlist_id, current_track_id, target, candidate_count,
		  recommendation_confidence, recommendations_status, lanes_returned, track_chosen,
		  chosen_was_recommended, skipped_over, adapted_weights, scoring_contract_id, timestamp, played_at
		FROM recommendation_events
		WHERE id = ?
		`,
		eventID,
	)
	var record RecommendationEventRecord
	var recommendationConfidence sql.NullFloat64
	var trackChosen sql.NullString
	var chosenWasRecommended sql.NullInt64
	var skippedOver sql.NullString
	var adaptedWeights sql.NullString
	var playedAt sql.NullString
	if err := row.Scan(
		&record.ID,
		&record.PlaylistID,
		&record.CurrentTrackID,
		&record.Target,
		&record.CandidateCount,
		&recommendationConfidence,
		&record.RecommendationsStatus,
		&record.LanesReturnedJSON,
		&trackChosen,
		&chosenWasRecommended,
		&skippedOver,
		&adaptedWeights,
		&record.ScoringContractID,
		&record.Timestamp,
		&playedAt,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrRecommendationEventNotFound
		}
		return nil, err
	}
	if recommendationConfidence.Valid {
		record.RecommendationConfidence = &recommendationConfidence.Float64
	}
	if trackChosen.Valid {
		record.TrackChosen = &trackChosen.String
	}
	if chosenWasRecommended.Valid {
		value := chosenWasRecommended.Int64 != 0
		record.ChosenWasRecommended = &value
	}
	if skippedOver.Valid {
		record.SkippedOverJSON = &skippedOver.String
	}
	if adaptedWeights.Valid {
		record.AdaptedWeightsJSON = &adaptedWeights.String
	}
	if playedAt.Valid {
		record.PlayedAt = &playedAt.String
	}
	return &record, nil
}

func (r *Repository) GetRecommendationEventItems(ctx context.Context, eventID string) ([]RecommendationEventItemRecord, error) {
	itemsByEvent, err := r.ListRecommendationEventItemsByEventIDs(ctx, []string{eventID})
	if err != nil {
		return nil, err
	}
	return itemsByEvent[eventID], nil
}

func (r *Repository) ListRecommendationEventItemsByEventIDs(ctx context.Context, eventIDs []string) (map[string][]RecommendationEventItemRecord, error) {
	itemsByEvent := make(map[string][]RecommendationEventItemRecord, len(eventIDs))
	if len(eventIDs) == 0 {
		return itemsByEvent, nil
	}
	placeholders := strings.TrimSuffix(strings.Repeat("?,", len(eventIDs)), ",")
	rows, err := r.db.QueryContext(
		ctx,
		fmt.Sprintf(`
		SELECT
		  event_id, lane_id, lane_rank, candidate_track_id, final_score, raw_score,
		  penalty_multiplier, move, move_confidence, risk, risk_score, primary_lane,
		  secondary_lane, component_scores_json, confidences_json, weights_used_json,
		  transition_features_json
		FROM recommendation_event_items
		WHERE event_id IN (%s)
		ORDER BY event_id ASC, lane_id ASC, lane_rank ASC, candidate_track_id ASC
		`, placeholders),
		stringSliceArgs(eventIDs)...,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var item RecommendationEventItemRecord
		var primaryLane sql.NullString
		var secondaryLane int64
		if err := rows.Scan(
			&item.EventID,
			&item.LaneID,
			&item.LaneRank,
			&item.CandidateTrackID,
			&item.FinalScore,
			&item.RawScore,
			&item.PenaltyMultiplier,
			&item.Move,
			&item.MoveConfidence,
			&item.Risk,
			&item.RiskScore,
			&primaryLane,
			&secondaryLane,
			&item.ComponentScoresJSON,
			&item.ConfidencesJSON,
			&item.WeightsUsedJSON,
			&item.TransitionFeaturesJSON,
		); err != nil {
			return nil, err
		}
		if primaryLane.Valid {
			item.PrimaryLane = &primaryLane.String
		}
		item.SecondaryLane = secondaryLane != 0
		itemsByEvent[item.EventID] = append(itemsByEvent[item.EventID], item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return itemsByEvent, nil
}

func (r *Repository) ListRecommendationEventsByPlaylist(ctx context.Context, playlistID string) ([]RecommendationEventRecord, error) {
	return r.ListRecommendationEventsByPlaylistWindow(ctx, playlistID, "", "")
}

func (r *Repository) ListRecommendationEventsByPlaylistWindow(ctx context.Context, playlistID, since, until string) ([]RecommendationEventRecord, error) {
	where := []string{"playlist_id = ?"}
	args := []any{playlistID}
	if strings.TrimSpace(since) != "" {
		where = append(where, "played_at >= ?")
		args = append(args, since)
	}
	if strings.TrimSpace(until) != "" {
		where = append(where, "played_at <= ?")
		args = append(args, until)
	}
	rows, err := r.db.QueryContext(
		ctx,
		fmt.Sprintf(`
		SELECT
		  id, playlist_id, current_track_id, target, candidate_count,
		  recommendation_confidence, recommendations_status, lanes_returned, track_chosen,
		  chosen_was_recommended, skipped_over, adapted_weights, scoring_contract_id, timestamp, played_at
		FROM recommendation_events
		WHERE %s
		ORDER BY COALESCE(played_at, timestamp) ASC, id ASC
		`, strings.Join(where, " AND ")),
		args...,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var items []RecommendationEventRecord
	for rows.Next() {
		var record RecommendationEventRecord
		var recommendationConfidence sql.NullFloat64
		var trackChosen sql.NullString
		var chosenWasRecommended sql.NullInt64
		var skippedOver sql.NullString
		var adaptedWeights sql.NullString
		var playedAt sql.NullString
		if err := rows.Scan(
			&record.ID,
			&record.PlaylistID,
			&record.CurrentTrackID,
			&record.Target,
			&record.CandidateCount,
			&recommendationConfidence,
			&record.RecommendationsStatus,
			&record.LanesReturnedJSON,
			&trackChosen,
			&chosenWasRecommended,
			&skippedOver,
			&adaptedWeights,
			&record.ScoringContractID,
			&record.Timestamp,
			&playedAt,
		); err != nil {
			return nil, err
		}
		if recommendationConfidence.Valid {
			record.RecommendationConfidence = &recommendationConfidence.Float64
		}
		if trackChosen.Valid {
			record.TrackChosen = &trackChosen.String
		}
		if chosenWasRecommended.Valid {
			value := chosenWasRecommended.Int64 != 0
			record.ChosenWasRecommended = &value
		}
		if skippedOver.Valid {
			record.SkippedOverJSON = &skippedOver.String
		}
		if adaptedWeights.Valid {
			record.AdaptedWeightsJSON = &adaptedWeights.String
		}
		if playedAt.Valid {
			record.PlayedAt = &playedAt.String
		}
		items = append(items, record)
	}
	return items, rows.Err()
}

func stringSliceArgs(values []string) []any {
	args := make([]any, 0, len(values))
	for _, value := range values {
		args = append(args, value)
	}
	return args
}

func (r *Repository) UpdateRecommendationEventChoice(
	ctx context.Context,
	eventID string,
	chosenTrackID string,
	chosenWasRecommended bool,
	skippedOverJSON string,
	playedAt string,
) error {
	return r.updateRecommendationEventChoice(ctx, r.db, eventID, chosenTrackID, chosenWasRecommended, skippedOverJSON, playedAt)
}

func (r *Repository) UpdateRecommendationEventChoiceTx(
	ctx context.Context,
	tx *sql.Tx,
	eventID string,
	chosenTrackID string,
	chosenWasRecommended bool,
	skippedOverJSON string,
	playedAt string,
) error {
	return r.updateRecommendationEventChoice(ctx, tx, eventID, chosenTrackID, chosenWasRecommended, skippedOverJSON, playedAt)
}

func (r *Repository) updateRecommendationEventChoice(
	ctx context.Context,
	exec execContexter,
	eventID string,
	chosenTrackID string,
	chosenWasRecommended bool,
	skippedOverJSON string,
	playedAt string,
) error {
	value := int64(0)
	if chosenWasRecommended {
		value = 1
	}
	result, err := exec.ExecContext(
		ctx,
		`
		UPDATE recommendation_events
		SET track_chosen = ?, chosen_was_recommended = ?, skipped_over = ?, played_at = ?
		WHERE id = ?
		`,
		chosenTrackID,
		value,
		skippedOverJSON,
		playedAt,
		eventID,
	)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return ErrRecommendationEventNotFound
	}
	return nil
}

func (r *Repository) UpsertFeedbackTuningJob(
	ctx context.Context,
	playlistID string,
	triggerEventID *string,
	createdAt string,
) (int64, error) {
	var jobID int64
	err := r.RunInTx(ctx, func(tx *sql.Tx) error {
		var err error
		jobID, err = r.upsertFeedbackTuningJob(ctx, tx, playlistID, triggerEventID, createdAt)
		return err
	})
	if err != nil {
		return 0, err
	}
	return jobID, nil
}

func (r *Repository) UpsertFeedbackTuningJobTx(
	ctx context.Context,
	tx *sql.Tx,
	playlistID string,
	triggerEventID *string,
	createdAt string,
) (int64, error) {
	return r.upsertFeedbackTuningJob(ctx, tx, playlistID, triggerEventID, createdAt)
}

func (r *Repository) upsertFeedbackTuningJob(
	ctx context.Context,
	exec execContexter,
	playlistID string,
	triggerEventID *string,
	createdAt string,
) (int64, error) {
	_, err := exec.ExecContext(
		ctx,
		`
		UPDATE feedback_tuning_jobs
		SET trigger_event_id = COALESCE(?, trigger_event_id)
		WHERE playlist_id = ?
		  AND status = 'pending'
		`,
		nullString(triggerEventID),
		playlistID,
	)
	if err != nil {
		return 0, err
	}
	queryer, ok := exec.(queryRowContexter)
	if !ok {
		return 0, fmt.Errorf("feedback tuning job upsert requires query-capable exec context")
	}
	row := queryer.QueryRowContext(
		ctx,
		`
		SELECT id
		FROM feedback_tuning_jobs
		WHERE playlist_id = ?
		  AND status = 'pending'
		ORDER BY id DESC
		LIMIT 1
		`,
		playlistID,
	)
	var existingID int64
	if err := row.Scan(&existingID); err == nil {
		return existingID, nil
	} else if !errors.Is(err, sql.ErrNoRows) {
		return 0, err
	}
	result, err := exec.ExecContext(
		ctx,
		`
		INSERT INTO feedback_tuning_jobs (
		  playlist_id, status, trigger_event_id, created_at
		) VALUES (?, 'pending', ?, ?)
		`,
		playlistID,
		nullString(triggerEventID),
		createdAt,
	)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func (r *Repository) InsertSyncOutbox(
	ctx context.Context,
	entityType string,
	entityID string,
	action string,
	payloadJSON string,
	createdAt string,
) (int64, error) {
	return r.insertSyncOutbox(ctx, r.db, entityType, entityID, action, payloadJSON, createdAt)
}

func (r *Repository) InsertSyncOutboxTx(
	ctx context.Context,
	tx *sql.Tx,
	entityType string,
	entityID string,
	action string,
	payloadJSON string,
	createdAt string,
) (int64, error) {
	return r.insertSyncOutbox(ctx, tx, entityType, entityID, action, payloadJSON, createdAt)
}

func (r *Repository) insertSyncOutbox(
	ctx context.Context,
	exec execContexter,
	entityType string,
	entityID string,
	action string,
	payloadJSON string,
	createdAt string,
) (int64, error) {
	result, err := exec.ExecContext(
		ctx,
		`
		INSERT INTO sync_outbox (entity_type, entity_id, action, payload_json, created_at)
		VALUES (?, ?, ?, ?, ?)
		`,
		entityType,
		entityID,
		action,
		payloadJSON,
		createdAt,
	)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func (r *Repository) UpsertPlaylistSyncState(
	ctx context.Context,
	playlistID string,
	snapshotID string,
	generatedAt string,
	updatedAt string,
) error {
	_, err := r.db.ExecContext(
		ctx,
		`
		INSERT INTO playlist_sync_state (
		  playlist_id, last_snapshot_id, last_snapshot_generated_at, last_snapshot_acked_at, updated_at
		) VALUES (?, ?, ?, NULL, ?)
		ON CONFLICT(playlist_id) DO UPDATE SET
		  last_snapshot_id = excluded.last_snapshot_id,
		  last_snapshot_generated_at = excluded.last_snapshot_generated_at,
		  last_snapshot_acked_at = NULL,
		  updated_at = excluded.updated_at
		`,
		playlistID,
		snapshotID,
		generatedAt,
		updatedAt,
	)
	return err
}

func (r *Repository) GetPlaylistSyncState(ctx context.Context, playlistID string) (*PlaylistSyncState, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT playlist_id, last_snapshot_id, last_snapshot_generated_at, last_snapshot_acked_at, updated_at
		FROM playlist_sync_state
		WHERE playlist_id = ?
		`,
		playlistID,
	)
	return scanPlaylistSyncState(row)
}

func (r *Repository) GetPlaylistSyncStateBySnapshotID(ctx context.Context, snapshotID string) (*PlaylistSyncState, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT playlist_id, last_snapshot_id, last_snapshot_generated_at, last_snapshot_acked_at, updated_at
		FROM playlist_sync_state
		WHERE last_snapshot_id = ?
		`,
		snapshotID,
	)
	return scanPlaylistSyncState(row)
}

func (r *Repository) AckPlaylistSnapshot(ctx context.Context, snapshotID string, ackedAt string) (*PlaylistSyncState, error) {
	state, err := r.GetPlaylistSyncStateBySnapshotID(ctx, snapshotID)
	if err != nil {
		return nil, err
	}
	if state == nil {
		return nil, ErrSnapshotNotFound
	}
	result, err := r.db.ExecContext(
		ctx,
		`
		UPDATE playlist_sync_state
		SET last_snapshot_acked_at = ?, updated_at = ?
		WHERE playlist_id = ? AND last_snapshot_id = ?
		`,
		ackedAt,
		ackedAt,
		state.PlaylistID,
		snapshotID,
	)
	if err != nil {
		return nil, err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return nil, err
	}
	if rowsAffected == 0 {
		return nil, ErrSnapshotNotFound
	}
	return r.GetPlaylistSyncState(ctx, state.PlaylistID)
}

func (r *Repository) PullUnsyncedOutbox(ctx context.Context, limit int) ([]SyncOutboxItem, bool, error) {
	rows, err := r.db.QueryContext(
		ctx,
		`
		SELECT id, entity_type, entity_id, action, payload_json, created_at, synced_at
		FROM sync_outbox
		WHERE synced_at IS NULL
		ORDER BY id ASC
		LIMIT ?
		`,
		limit+1,
	)
	if err != nil {
		return nil, false, err
	}
	defer rows.Close()

	items := make([]SyncOutboxItem, 0, limit+1)
	for rows.Next() {
		var item SyncOutboxItem
		var syncedAt sql.NullString
		if err := rows.Scan(
			&item.ID,
			&item.EntityType,
			&item.EntityID,
			&item.Action,
			&item.PayloadJSON,
			&item.CreatedAt,
			&syncedAt,
		); err != nil {
			return nil, false, err
		}
		if syncedAt.Valid {
			item.SyncedAt = &syncedAt.String
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}
	hasMore := len(items) > limit
	if hasMore {
		items = items[:limit]
	}
	return items, hasMore, nil
}

func (r *Repository) AckOutboxThroughID(ctx context.Context, ackThroughID int64, ackedAt string) (int64, error) {
	result, err := r.db.ExecContext(
		ctx,
		`
		UPDATE sync_outbox
		SET synced_at = ?
		WHERE id <= ? AND synced_at IS NULL
		`,
		ackedAt,
		ackThroughID,
	)
	if err != nil {
		return 0, err
	}
	return result.RowsAffected()
}

func (r *Repository) CreateRemotePairingToken(ctx context.Context, tokenHash string, deviceLabel *string, createdAt string, expiresAt string) error {
	_, err := r.db.ExecContext(
		ctx,
		`
		INSERT INTO remote_pairing_tokens (token_hash, created_at, expires_at, device_label)
		VALUES (?, ?, ?, ?)
		`,
		tokenHash,
		createdAt,
		expiresAt,
		nullString(deviceLabel),
	)
	return err
}

func (r *Repository) ConsumeRemotePairingToken(ctx context.Context, tokenHash string, usedAt string) (*RemotePairingToken, error) {
	var token *RemotePairingToken
	err := r.RunInTx(ctx, func(tx *sql.Tx) error {
		row := tx.QueryRowContext(
			ctx,
			`
			SELECT token_hash, created_at, expires_at, used_at, device_label
			FROM remote_pairing_tokens
			WHERE token_hash = ?
			`,
			tokenHash,
		)
		record, err := scanRemotePairingToken(row)
		if err != nil {
			return err
		}
		if record == nil || record.UsedAt != nil || record.ExpiresAt < usedAt {
			token = nil
			return nil
		}
		_, err = tx.ExecContext(ctx, "UPDATE remote_pairing_tokens SET used_at = ? WHERE token_hash = ?", usedAt, tokenHash)
		if err != nil {
			return err
		}
		record.UsedAt = &usedAt
		token = record
		return nil
	})
	return token, err
}

func (r *Repository) CreateRemoteSession(ctx context.Context, sessionHash string, deviceLabel *string, createdAt string, expiresAt string) error {
	_, err := r.db.ExecContext(
		ctx,
		`
		INSERT INTO remote_sessions (session_hash, device_label, created_at, last_seen_at, expires_at)
		VALUES (?, ?, ?, ?, ?)
		`,
		sessionHash,
		nullString(deviceLabel),
		createdAt,
		createdAt,
		expiresAt,
	)
	return err
}

func (r *Repository) GetValidRemoteSession(ctx context.Context, sessionHash string, now string) (*RemoteSession, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT session_hash, device_label, created_at, last_seen_at, expires_at, revoked_at
		FROM remote_sessions
		WHERE session_hash = ?
		  AND revoked_at IS NULL
		  AND expires_at >= ?
		`,
		sessionHash,
		now,
	)
	session, err := scanRemoteSession(row)
	if err != nil {
		return nil, err
	}
	if session != nil {
		_, err = r.db.ExecContext(ctx, "UPDATE remote_sessions SET last_seen_at = ? WHERE session_hash = ?", now, sessionHash)
		if err != nil {
			return nil, err
		}
		session.LastSeenAt = now
	}
	return session, nil
}

func (r *Repository) RevokeRemoteSession(ctx context.Context, sessionHash string, revokedAt string) error {
	_, err := r.db.ExecContext(
		ctx,
		`
		UPDATE remote_sessions
		SET revoked_at = ?
		WHERE session_hash = ?
		`,
		revokedAt,
		sessionHash,
	)
	return err
}

func scanRemotePairingToken(row *sql.Row) (*RemotePairingToken, error) {
	var token RemotePairingToken
	var usedAt sql.NullString
	var deviceLabel sql.NullString
	if err := row.Scan(&token.TokenHash, &token.CreatedAt, &token.ExpiresAt, &usedAt, &deviceLabel); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	if usedAt.Valid {
		token.UsedAt = &usedAt.String
	}
	if deviceLabel.Valid {
		token.DeviceLabel = &deviceLabel.String
	}
	return &token, nil
}

func scanRemoteSession(row *sql.Row) (*RemoteSession, error) {
	var session RemoteSession
	var deviceLabel sql.NullString
	var revokedAt sql.NullString
	if err := row.Scan(&session.SessionHash, &deviceLabel, &session.CreatedAt, &session.LastSeenAt, &session.ExpiresAt, &revokedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	if deviceLabel.Valid {
		session.DeviceLabel = &deviceLabel.String
	}
	if revokedAt.Valid {
		session.RevokedAt = &revokedAt.String
	}
	return &session, nil
}

func NowUTC() string {
	return time.Now().UTC().Format(time.RFC3339)
}

func (r *Repository) ListFeedbackTuningJobsByPlaylist(ctx context.Context, playlistID string) ([]FeedbackTuningJobRecord, error) {
	rows, err := r.db.QueryContext(
		ctx,
		`
		SELECT id, playlist_id, status, trigger_event_id, created_at, started_at, finished_at, error_message
		FROM feedback_tuning_jobs
		WHERE playlist_id = ?
		ORDER BY id ASC
		`,
		playlistID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var jobs []FeedbackTuningJobRecord
	for rows.Next() {
		var item FeedbackTuningJobRecord
		var triggerEventID sql.NullString
		var startedAt sql.NullString
		var finishedAt sql.NullString
		var errorMessage sql.NullString
		if err := rows.Scan(
			&item.ID,
			&item.PlaylistID,
			&item.Status,
			&triggerEventID,
			&item.CreatedAt,
			&startedAt,
			&finishedAt,
			&errorMessage,
		); err != nil {
			return nil, err
		}
		if triggerEventID.Valid {
			item.TriggerEventID = &triggerEventID.String
		}
		if startedAt.Valid {
			item.StartedAt = &startedAt.String
		}
		if finishedAt.Valid {
			item.FinishedAt = &finishedAt.String
		}
		if errorMessage.Valid {
			item.ErrorMessage = &errorMessage.String
		}
		jobs = append(jobs, item)
	}
	return jobs, rows.Err()
}

func scanPlaylistSyncState(scanner interface{ Scan(...any) error }) (*PlaylistSyncState, error) {
	var state PlaylistSyncState
	var ackedAt sql.NullString
	if err := scanner.Scan(
		&state.PlaylistID,
		&state.LastSnapshotID,
		&state.LastSnapshotGeneratedAt,
		&ackedAt,
		&state.UpdatedAt,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	if ackedAt.Valid {
		state.LastSnapshotAckedAt = &ackedAt.String
	}
	return &state, nil
}

func (r *Repository) getTrackContext(ctx context.Context, playlistID, trackID string) (TrackContextRecord, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT
		  t.id,
		  t.file_path,
		  t.file_hash,
		  COALESCE(t.title, ''),
		  COALESCE(t.artist, ''),
		  pt.position,
		  f.bpm,
		  f.key,
		  f.key_confidence,
		  f.key_source,
		  f.key_agreement,
		  r.energy_rel,
		  r.bass_rel,
		  r.drums_rel,
		  r.vocals_rel,
		  r.groove_rel,
		  r.intensity_band,
		  r.role_hints,
		  f.analysis_signature,
		  f.config_signature,
		  f.scoring_contract_id_at_analysis
		FROM playlist_tracks pt
		JOIN tracks t ON t.id = pt.track_id
		LEFT JOIN track_features_abs f ON f.track_id = t.id
		LEFT JOIN track_features_rel r ON r.track_id = t.id AND r.playlist_id = pt.playlist_id
		WHERE pt.playlist_id = ? AND t.id = ?
		`,
		playlistID,
		trackID,
	)
	record, err := scanTrackContext(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return TrackContextRecord{}, ErrTrackNotFound
		}
		return TrackContextRecord{}, err
	}
	return record, nil
}

func (r *Repository) getScoringCandidates(ctx context.Context, playlistID string) ([]TrackContextRecord, error) {
	rows, err := r.db.QueryContext(
		ctx,
		`
		SELECT
		  t.id,
		  t.file_path,
		  t.file_hash,
		  COALESCE(t.title, ''),
		  COALESCE(t.artist, ''),
		  pt.position,
		  f.bpm,
		  f.key,
		  f.key_confidence,
		  f.key_source,
		  f.key_agreement,
		  r.energy_rel,
		  r.bass_rel,
		  r.drums_rel,
		  r.vocals_rel,
		  r.groove_rel,
		  r.intensity_band,
		  r.role_hints,
		  f.analysis_signature,
		  f.config_signature,
		  f.scoring_contract_id_at_analysis
		FROM playlist_tracks pt
		JOIN tracks t ON t.id = pt.track_id
		JOIN track_features_abs f ON f.track_id = t.id
		LEFT JOIN track_features_rel r ON r.track_id = t.id AND r.playlist_id = pt.playlist_id
		WHERE pt.playlist_id = ?
		ORDER BY pt.position ASC
		`,
		playlistID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []TrackContextRecord
	for rows.Next() {
		record, err := scanTrackContext(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, record)
	}
	return out, rows.Err()
}

func (r *Repository) getPlaylistStats(ctx context.Context, playlistID string) (*PlaylistStats, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT
		  playlist_id,
		  track_count_total,
		  track_count_analyzed,
		  eligible_track_count,
		  energy_spread,
		  relative_signature,
		  is_stale,
		  COALESCE(stale_reason, ''),
		  adapted_weights,
		  feedback_tuned_weights,
		  feedback_tuning_notes,
		  feedback_event_count,
		  feedback_last_tuned_at,
		  feedback_tuning_metrics
		FROM playlist_stats
		WHERE playlist_id = ?
		`,
		playlistID,
	)

	var stats PlaylistStats
	var energy sql.NullFloat64
	var adapted sql.NullString
	var feedbackTuned sql.NullString
	var feedbackNotes sql.NullString
	var staleInt int64
	var feedbackEventCount sql.NullInt64
	var feedbackLastTunedAt sql.NullString
	var feedbackMetrics sql.NullString
	if err := row.Scan(
		&stats.PlaylistID,
		&stats.TrackCountTotal,
		&stats.TrackCountAnalyzed,
		&stats.EligibleTrackCount,
		&energy,
		&stats.RelativeSignature,
		&staleInt,
		&stats.StaleReason,
		&adapted,
		&feedbackTuned,
		&feedbackNotes,
		&feedbackEventCount,
		&feedbackLastTunedAt,
		&feedbackMetrics,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	if energy.Valid {
		stats.EnergySpread = &energy.Float64
	}
	stats.IsStale = staleInt != 0
	if adapted.Valid && strings.TrimSpace(adapted.String) != "" {
		if err := json.Unmarshal([]byte(adapted.String), &stats.AdaptedWeights); err != nil {
			return nil, fmt.Errorf("decode adapted_weights: %w", err)
		}
	}
	if feedbackTuned.Valid && strings.TrimSpace(feedbackTuned.String) != "" {
		if err := json.Unmarshal([]byte(feedbackTuned.String), &stats.FeedbackTunedWeights); err != nil {
			return nil, fmt.Errorf("decode feedback_tuned_weights: %w", err)
		}
	}
	if feedbackNotes.Valid && strings.TrimSpace(feedbackNotes.String) != "" {
		if err := json.Unmarshal([]byte(feedbackNotes.String), &stats.FeedbackTuningNotes); err != nil {
			return nil, fmt.Errorf("decode feedback_tuning_notes: %w", err)
		}
	}
	if feedbackEventCount.Valid {
		stats.FeedbackEventCount = int(feedbackEventCount.Int64)
	}
	if feedbackLastTunedAt.Valid {
		stats.FeedbackLastTunedAt = &feedbackLastTunedAt.String
	}
	if feedbackMetrics.Valid && strings.TrimSpace(feedbackMetrics.String) != "" {
		if err := json.Unmarshal([]byte(feedbackMetrics.String), &stats.FeedbackTuningMetrics); err != nil {
			return nil, fmt.Errorf("decode feedback_tuning_metrics: %w", err)
		}
	}
	return &stats, nil
}

func (r *Repository) GetPlaylistStats(ctx context.Context, playlistID string) (*PlaylistStats, error) {
	return r.getPlaylistStats(ctx, playlistID)
}

func scanTrackContext(scanner interface{ Scan(...any) error }) (TrackContextRecord, error) {
	var record TrackContextRecord
	var bpm sql.NullFloat64
	var fileHash sql.NullString
	var musicalKey sql.NullString
	var keyConfidence sql.NullFloat64
	var keySource sql.NullString
	var keyAgreement sql.NullInt64
	var energyRel sql.NullFloat64
	var bassRel sql.NullFloat64
	var drumsRel sql.NullFloat64
	var vocalsRel sql.NullFloat64
	var grooveRel sql.NullFloat64
	var intensityBand sql.NullString
	var roleHints sql.NullString
	var analysisSignature sql.NullString
	var configSignature sql.NullString
	var scoringContract sql.NullString

	if err := scanner.Scan(
		&record.TrackID,
		&record.FilePath,
		&fileHash,
		&record.Title,
		&record.Artist,
		&record.Position,
		&bpm,
		&musicalKey,
		&keyConfidence,
		&keySource,
		&keyAgreement,
		&energyRel,
		&bassRel,
		&drumsRel,
		&vocalsRel,
		&grooveRel,
		&intensityBand,
		&roleHints,
		&analysisSignature,
		&configSignature,
		&scoringContract,
	); err != nil {
		return TrackContextRecord{}, err
	}

	if bpm.Valid {
		record.BPM = &bpm.Float64
	}
	if fileHash.Valid {
		record.FileHash = &fileHash.String
	}
	if musicalKey.Valid {
		record.Key = &musicalKey.String
	}
	if keyConfidence.Valid {
		record.KeyConfidence = &keyConfidence.Float64
	}
	if keySource.Valid {
		record.KeySource = &keySource.String
	}
	if keyAgreement.Valid {
		value := int32(keyAgreement.Int64)
		record.KeyAgreement = &value
	}
	if energyRel.Valid {
		record.EnergyRel = &energyRel.Float64
	}
	if bassRel.Valid {
		record.BassRel = &bassRel.Float64
	}
	if drumsRel.Valid {
		record.DrumsRel = &drumsRel.Float64
	}
	if vocalsRel.Valid {
		record.VocalsRel = &vocalsRel.Float64
	}
	if grooveRel.Valid {
		record.GrooveRel = &grooveRel.Float64
	}
	if intensityBand.Valid {
		record.IntensityBand = &intensityBand.String
	}
	if roleHints.Valid && strings.TrimSpace(roleHints.String) != "" {
		if err := json.Unmarshal([]byte(roleHints.String), &record.RoleHints); err != nil {
			log.Printf("warning: failed to decode role_hints %q for track %s: %v", roleHints.String, record.TrackID, err)
		}
	}
	if analysisSignature.Valid {
		record.AnalysisSignature = &analysisSignature.String
	}
	if configSignature.Valid {
		record.ConfigSignature = &configSignature.String
	}
	if scoringContract.Valid {
		record.ScoringContractAtAnalysis = &scoringContract.String
	}
	return record, nil
}

func nullString(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func nullFloat64(value *float64) any {
	if value == nil {
		return nil
	}
	return *value
}

func nullBool(value *bool) any {
	if value == nil {
		return nil
	}
	if *value {
		return 1
	}
	return 0
}
