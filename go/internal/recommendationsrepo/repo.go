package recommendationsrepo

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	_ "modernc.org/sqlite"
)

var (
	ErrPlaylistNotFound      = errors.New("playlist not found")
	ErrTrackNotFound         = errors.New("track not found in playlist")
	ErrRelativeRefreshNeeded = errors.New("playlist relative artifacts require refresh")
)

type PlaylistRef struct {
	ID   string
	Name string
}

type TrackContextRecord struct {
	TrackID                   string
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

type HydratedRecommendations struct {
	Playlist   PlaylistRef
	Current    TrackContextRecord
	Candidates []TrackContextRecord
	History    []TrackContextRecord
	HasGaps    bool
	Stats      *PlaylistStats
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

func (r *Repository) getTrackContext(ctx context.Context, playlistID, trackID string) (TrackContextRecord, error) {
	row := r.db.QueryRowContext(
		ctx,
		`
		SELECT
		  t.id,
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

func scanTrackContext(scanner interface{ Scan(...any) error }) (TrackContextRecord, error) {
	var record TrackContextRecord
	var bpm sql.NullFloat64
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
