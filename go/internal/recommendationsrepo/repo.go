package recommendationsrepo

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

var (
	ErrPlaylistNotFound            = errors.New("playlist not found")
	ErrTrackNotFound               = errors.New("track not found in playlist")
	ErrRecommendationEventNotFound = errors.New("recommendation event not found")
	ErrRelativeRefreshNeeded       = errors.New("playlist relative artifacts require refresh")
)

type PlaylistRef struct {
	ID   string
	Name string
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
	PlaylistID         string
	TrackCountTotal    int
	TrackCountAnalyzed int
	EligibleTrackCount int
	EnergySpread       *float64
	RelativeSignature  string
	IsStale            bool
	StaleReason        string
	AdaptedWeights     map[string]float64
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

type PlaylistSyncState struct {
	PlaylistID              string
	LastSnapshotID          string
	LastSnapshotGeneratedAt string
	LastSnapshotAckedAt     *string
	UpdatedAt               string
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

type Repository struct {
	db *sql.DB
}

func Open(path string) (*Repository, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite database: %w", err)
	}
	return &Repository{db: db}, nil
}

func (r *Repository) Close() error {
	return r.db.Close()
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
		switch {
		case !analysisSig.Valid || !configSig.Valid || !scoringContract.Valid:
			track.AnalysisState = "unanalysed"
		default:
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
	_, err := r.db.ExecContext(
		ctx,
		"UPDATE tracks SET imported_bpm = ?, updated_at = ? WHERE id = ?",
		bpm,
		updatedAt,
		trackID,
	)
	return err
}

func (r *Repository) UpdateTrackImportedKey(ctx context.Context, trackID string, key string, updatedAt string) error {
	_, err := r.db.ExecContext(
		ctx,
		"UPDATE tracks SET imported_key = ?, updated_at = ? WHERE id = ?",
		key,
		updatedAt,
		trackID,
	)
	return err
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
	if len(playlistIDs) == 0 {
		return nil
	}
	for _, playlistID := range playlistIDs {
		if _, err := r.db.ExecContext(
			ctx,
			"UPDATE playlist_stats SET is_stale = 1, stale_reason = ?, stale_marked_at = ? WHERE playlist_id = ?",
			reason,
			markedAt,
			playlistID,
		); err != nil {
			return err
		}
	}
	return nil
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
	_, err := r.db.ExecContext(
		ctx,
		`
		DELETE FROM analysis_jobs
		WHERE track_id = ?
		  AND job_kind = ?
		  AND status = 'pending'
		  AND analysis_signature = ?
		  AND config_signature = ?
		`,
		trackID,
		jobKind,
		analysisSignature,
		configSignature,
	)
	if err != nil {
		return 0, err
	}
	result, err := r.db.ExecContext(
		ctx,
		`
		INSERT INTO analysis_jobs (
		  playlist_id, track_id, track_path, job_kind, status, priority, analysis_mode,
		  analysis_signature, config_signature, source_file_hash, created_at
		) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
		`,
		nullStringPtr(playlistID),
		trackID,
		trackPath,
		jobKind,
		priority,
		analysisMode,
		analysisSignature,
		configSignature,
		nullStringPtr(sourceFileHash),
		createdAt,
	)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func (r *Repository) InsertManualCorrection(ctx context.Context, record ManualCorrectionRecord) error {
	_, err := r.db.ExecContext(
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
	_, err := r.db.ExecContext(
		ctx,
		`
		INSERT INTO recommendation_events (
		  id, user_id, playlist_id, current_track_id, target, candidate_count,
		  recommendation_confidence, recommendations_status, lanes_returned, track_chosen,
		  chosen_was_recommended, skipped_over, adapted_weights, scoring_contract_id, timestamp
		) VALUES (?, 'local', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
	)
	return err
}

func (r *Repository) GetRecommendationEvent(ctx context.Context, eventID string) (*RecommendationEventRecord, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT
		  id, playlist_id, current_track_id, target, candidate_count,
		  recommendation_confidence, recommendations_status, lanes_returned, track_chosen,
		  chosen_was_recommended, skipped_over, adapted_weights, scoring_contract_id, timestamp
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
	return &record, nil
}

func (r *Repository) UpdateRecommendationEventChoice(
	ctx context.Context,
	eventID string,
	chosenTrackID string,
	chosenWasRecommended bool,
	skippedOverJSON string,
) error {
	value := int64(0)
	if chosenWasRecommended {
		value = 1
	}
	_, err := r.db.ExecContext(
		ctx,
		`
		UPDATE recommendation_events
		SET track_chosen = ?, chosen_was_recommended = ?, skipped_over = ?
		WHERE id = ?
		`,
		chosenTrackID,
		value,
		skippedOverJSON,
		eventID,
	)
	return err
}

func (r *Repository) InsertSyncOutbox(
	ctx context.Context,
	entityType string,
	entityID string,
	action string,
	payloadJSON string,
	createdAt string,
) (int64, error) {
	result, err := r.db.ExecContext(
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
		return nil, nil
	}
	_, err = r.db.ExecContext(
		ctx,
		`
		UPDATE playlist_sync_state
		SET last_snapshot_acked_at = ?, updated_at = ?
		WHERE playlist_id = ?
		`,
		ackedAt,
		ackedAt,
		state.PlaylistID,
	)
	if err != nil {
		return nil, err
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

func NowUTC() string {
	return time.Now().UTC().Format(time.RFC3339)
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
		  adapted_weights
		FROM playlist_stats
		WHERE playlist_id = ?
		`,
		playlistID,
	)

	var stats PlaylistStats
	var energy sql.NullFloat64
	var adapted sql.NullString
	var staleInt int64
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
		_ = json.Unmarshal([]byte(roleHints.String), &record.RoleHints)
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

func nullStringPtr(value *string) any {
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
