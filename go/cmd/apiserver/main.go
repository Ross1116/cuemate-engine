package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"github.com/Ross1116/cuemate-engine/go/internal/recommendationsrepo"
	"github.com/Ross1116/cuemate-engine/go/internal/scoringruntime"
	"github.com/google/uuid"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	defaultAPIAddr = "127.0.0.1:8080"
)

var liveLaneOrder = []string{"maintain", "build", "reset", "jump", "contrast"}

type appConfig struct {
	Addr     string
	Database string
}

type recommendationRequest struct {
	PlaylistID      string   `json:"playlist_id"`
	PlaylistName    string   `json:"playlist_name"`
	CurrentTrackID  string   `json:"current_track_id"`
	HistoryTrackIDs []string `json:"history_track_ids"`
	Target          string   `json:"target"`
	MaxPerLane      int32    `json:"max_per_lane"`
}

type playedEventRequest struct {
	RecommendationEventID string `json:"recommendation_event_id"`
	ChosenTrackID         string `json:"chosen_track_id"`
	PlayedAt              string `json:"played_at"`
}

type correctionRequest struct {
	TrackID     string      `json:"track_id"`
	Field       string      `json:"field"`
	NewValue    interface{} `json:"new_value"`
	CorrectedAt string      `json:"corrected_at"`
}

type snapshotRequest struct {
	PlaylistID   string `json:"playlist_id"`
	PlaylistName string `json:"playlist_name"`
}

type feedbackSummaryRequest struct {
	PlaylistID   string `json:"playlist_id"`
	PlaylistName string `json:"playlist_name"`
	Since        string `json:"since"`
	Until        string `json:"until"`
}

type snapshotAckRequest struct {
	SnapshotID string `json:"snapshot_id"`
	AckedAt    string `json:"acked_at"`
}

type outboxPullRequest struct {
	Limit int `json:"limit"`
}

type outboxAckRequest struct {
	AckThroughID int64  `json:"ack_through_id"`
	AckedAt      string `json:"acked_at"`
}

type server struct {
	repo    *recommendationsrepo.Repository
	runtime *scoringruntime.Runtime
}

func main() {
	os.Exit(run())
}

func run() int {
	cfg := loadConfig()
	repo, err := recommendationsrepo.Open(cfg.Database)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer repo.Close()

	runtime, err := scoringruntime.NewDefault()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer runtime.Close()

	srv := &server{repo: repo, runtime: runtime}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", srv.handleHealthz)
	mux.HandleFunc("/readyz", srv.handleReadyz)
	mux.HandleFunc("/scoring/metadata", srv.handleMetadata)
	mux.HandleFunc("/recommendations", srv.handleRecommendations)
	mux.HandleFunc("/events/played", srv.handlePlayedEvent)
	mux.HandleFunc("/feedback/summary", srv.handleFeedbackSummary)
	mux.HandleFunc("/corrections", srv.handleCorrections)
	mux.HandleFunc("/sync/playlists/snapshot", srv.handleSnapshot)
	mux.HandleFunc("/sync/playlists/snapshot/ack", srv.handleSnapshotAck)
	mux.HandleFunc("/sync/outbox/pull", srv.handleOutboxPull)
	mux.HandleFunc("/sync/outbox/ack", srv.handleOutboxAck)

	httpServer := &http.Server{
		Addr:         cfg.Addr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	serverErr := make(chan error, 1)
	go func() {
		log.Printf("Go API server listening on %s", cfg.Addr)
		serverErr <- httpServer.ListenAndServe()
	}()

	sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case err := <-serverErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		return 0
	case <-sigCtx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			log.Printf("Go API server shutdown failed for %s: %v", cfg.Addr, err)
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		if err := <-serverErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		log.Printf("Go API server shut down cleanly on %s", cfg.Addr)
		return 0
	}
}

func loadConfig() appConfig {
	addr := strings.TrimSpace(os.Getenv("GO_API_ADDR"))
	if addr == "" {
		addr = defaultAPIAddr
	}
	databaseURL := strings.TrimSpace(os.Getenv("DATABASE_URL"))
	if databaseURL == "" {
		databaseURL = "sqlite:data/cuemate.db"
	}
	return appConfig{
		Addr:     addr,
		Database: strings.TrimPrefix(databaseURL, "sqlite:"),
	}
}

func (s *server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (s *server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	_, err := s.runtime.RefreshMetadata(r.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "not_ready", "error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready"})
}

func (s *server) handleMetadata(w http.ResponseWriter, r *http.Request) {
	metadata, err := s.runtime.RefreshMetadata(r.Context())
	if err != nil {
		metadata = s.runtime.CachedMetadata()
	}
	breakerOpen, failures := s.runtime.State()
	writeJSON(w, http.StatusOK, map[string]any{
		"metadata":       metadata,
		"breaker_open":   breakerOpen,
		"failure_count":  failures,
		"metadata_fresh": err == nil,
		"metadata_error": errorString(err),
	})
}

func (s *server) handleRecommendations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	var req recommendationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if strings.TrimSpace(req.CurrentTrackID) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "current_track_id is required"})
		return
	}
	target := req.Target
	if target == "" {
		target = "maintain"
	}

	playlist, err := s.repo.ResolvePlaylist(r.Context(), req.PlaylistID, req.PlaylistName)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}

	hydrated, err := s.repo.HydrateRecommendations(r.Context(), playlist, req.CurrentTrackID, req.HistoryTrackIDs)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, recommendationsrepo.ErrTrackNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}

	metadata, err := s.runtime.RefreshMetadata(r.Context())
	if err != nil {
		writeJSON(w, http.StatusOK, degradedRecommendationsPayload(hydrated, target, "temporarily_unavailable", scoringruntime.DescribeUnavailable(err), nil))
		return
	}

	if reason := relativeRefreshReason(hydrated.Stats, metadata.GetExpectedRelativeSignature()); reason != "" {
		writeJSON(w, http.StatusOK, degradedRecommendationsPayload(hydrated, target, "requires_reanalysis", reason, metadata))
		return
	}
	if !hydrated.Current.Scoreable() {
		writeJSON(w, http.StatusOK, degradedRecommendationsPayload(hydrated, target, "requires_reanalysis", "Current track is not fully analyzed for live recommendations.", metadata))
		return
	}

	request := buildRecommendationsRequest(hydrated, target, req.MaxPerLane)
	response, err := s.runtime.GetRecommendations(r.Context(), request)
	if err != nil {
		switch {
		case scoringruntime.IsCompatibilityError(err):
			writeJSON(w, http.StatusOK, degradedRecommendationsPayload(hydrated, target, "requires_reanalysis", err.Error(), metadata))
		case scoringruntime.IsRequestError(err):
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		default:
			writeJSON(w, http.StatusOK, degradedRecommendationsPayload(hydrated, target, "temporarily_unavailable", scoringruntime.DescribeUnavailable(err), metadata))
		}
		return
	}

	payload := translateRecommendationsResponse(response)
	applyPlaylistWeightSource(payload, hydrated.Stats)
	eventID, err := s.recordRecommendationEvent(r.Context(), hydrated, response, payload)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	meta := payload["meta"].(map[string]any)
	meta["recommendation_event_id"] = eventID
	writeJSON(w, http.StatusOK, payload)
}

func (s *server) handlePlayedEvent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req playedEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if strings.TrimSpace(req.RecommendationEventID) == "" || strings.TrimSpace(req.ChosenTrackID) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "recommendation_event_id and chosen_track_id are required"})
		return
	}

	event, err := s.repo.GetRecommendationEvent(r.Context(), req.RecommendationEventID)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, recommendationsrepo.ErrRecommendationEventNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}

	wasRecommended, skippedOver, err := derivePlayedOutcome(event.LanesReturnedJSON, req.ChosenTrackID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	skippedJSONBytes, err := json.Marshal(skippedOver)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	playedAt := req.PlayedAt
	if strings.TrimSpace(playedAt) == "" {
		playedAt = recommendationsrepo.NowUTC()
	}
	err = s.repo.RunInTx(r.Context(), func(tx *sql.Tx) error {
		if err := s.repo.UpdateRecommendationEventChoiceTx(
			r.Context(),
			tx,
			event.ID,
			req.ChosenTrackID,
			wasRecommended,
			string(skippedJSONBytes),
		); err != nil {
			return err
		}
		outboxPayload, err := json.Marshal(map[string]any{
			"recommendation_event_id": event.ID,
			"playlist_id":             event.PlaylistID,
			"current_track_id":        event.CurrentTrackID,
			"chosen_track_id":         req.ChosenTrackID,
			"chosen_was_recommended":  wasRecommended,
			"skipped_over":            skippedOver,
			"played_at":               playedAt,
		})
		if err != nil {
			log.Printf("failed to marshal recommendation event outbox payload for %s: %v", event.ID, err)
			return fmt.Errorf("failed to encode sync payload: %w", err)
		}
		if _, err = s.repo.InsertSyncOutboxTx(
			r.Context(),
			tx,
			"recommendation_event",
			event.ID,
			"played",
			string(outboxPayload),
			playedAt,
		); err != nil {
			return err
		}
		_, err = s.repo.UpsertFeedbackTuningJobTx(
			r.Context(),
			tx,
			event.PlaylistID,
			&event.ID,
			playedAt,
		)
		return err
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"recommendation_event_id": event.ID,
		"chosen_track_id":         req.ChosenTrackID,
		"chosen_was_recommended":  wasRecommended,
		"skipped_over":            skippedOver,
	})
}

func (s *server) handleCorrections(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req correctionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if strings.TrimSpace(req.TrackID) == "" || strings.TrimSpace(req.Field) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "track_id and field are required"})
		return
	}
	track, err := s.repo.GetTrackForCorrection(r.Context(), req.TrackID)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, recommendationsrepo.ErrTrackNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	metadata, err := s.runtime.RefreshMetadata(r.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": scoringruntime.DescribeUnavailable(err)})
		return
	}

	correctedAt := req.CorrectedAt
	if strings.TrimSpace(correctedAt) == "" {
		correctedAt = recommendationsrepo.NowUTC()
	}
	importedBPM, importedKey, err := s.repo.GetTrackImportedValues(r.Context(), req.TrackID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	var correction recommendationsrepo.ManualCorrectionRecord
	changed := false
	switch req.Field {
	case "bpm":
		newValue, ok := parseCorrectionBPM(req.NewValue)
		if !ok {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "new_value must be a positive number for bpm"})
			return
		}
		oldString := ""
		if importedBPM != nil {
			oldString = fmt.Sprintf("%.6f", *importedBPM)
			changed = *importedBPM != newValue
		} else {
			changed = true
		}
		if changed {
			correction = recommendationsrepo.ManualCorrectionRecord{
				ID:          uuid.NewString(),
				TrackID:     req.TrackID,
				Field:       "bpm",
				OldValue:    oldString,
				NewValue:    fmt.Sprintf("%.6f", newValue),
				CorrectedAt: correctedAt,
			}
		}
	case "key":
		newValue, ok := parseCorrectionKey(req.NewValue)
		if !ok {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "new_value must be a non-empty string for key"})
			return
		}
		oldString := ""
		if importedKey != nil {
			oldString = *importedKey
			changed = !strings.EqualFold(*importedKey, newValue)
		} else {
			changed = true
		}
		if changed {
			correction = recommendationsrepo.ManualCorrectionRecord{
				ID:          uuid.NewString(),
				TrackID:     req.TrackID,
				Field:       "key",
				OldValue:    oldString,
				NewValue:    newValue,
				CorrectedAt: correctedAt,
			}
		}
	default:
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "field must be one of: bpm, key"})
		return
	}

	if !changed {
		writeJSON(w, http.StatusOK, map[string]any{
			"changed":               false,
			"requires_reanalysis":   false,
			"correction_id":         nil,
			"queued_job_id":         nil,
			"affected_playlist_ids": []string{},
		})
		return
	}
	playlists, err := s.repo.GetPlaylistsContainingTrack(r.Context(), req.TrackID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	playlistIDs := make([]string, 0, len(playlists))
	for _, playlist := range playlists {
		playlistIDs = append(playlistIDs, playlist.ID)
	}
	active := metadata.GetActiveSignatures()
	outboxPayload, err := json.Marshal(map[string]any{
		"correction_id":             correction.ID,
		"track_id":                  req.TrackID,
		"field":                     correction.Field,
		"new_value":                 correction.NewValue,
		"corrected_at":              correctedAt,
		"affected_playlist_ids":     playlistIDs,
		"requires_snapshot_refresh": true,
	})
	if err != nil {
		log.Printf("failed to marshal manual correction outbox payload for %s: %v", correction.ID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to encode sync payload"})
		return
	}
	var jobID int64
	err = s.repo.RunInTx(r.Context(), func(tx *sql.Tx) error {
		switch correction.Field {
		case "bpm":
			newValue, _ := parseCorrectionBPM(req.NewValue)
			if err := s.repo.UpdateTrackImportedBPMTx(r.Context(), tx, req.TrackID, newValue, correctedAt); err != nil {
				return err
			}
		case "key":
			newValue, _ := parseCorrectionKey(req.NewValue)
			if err := s.repo.UpdateTrackImportedKeyTx(r.Context(), tx, req.TrackID, newValue, correctedAt); err != nil {
				return err
			}
		}
		if err := s.repo.InsertManualCorrectionTx(r.Context(), tx, correction); err != nil {
			return err
		}
		if err := s.repo.MarkPlaylistsStaleTx(r.Context(), tx, playlistIDs, "manual_correction", correctedAt); err != nil {
			return err
		}
		var err error
		jobID, err = s.repo.CreateAnalysisJobWithKindTx(
			r.Context(),
			tx,
			nil,
			req.TrackID,
			track.FilePath,
			"full",
			"full",
			active.GetAnalysisSignature(),
			active.GetConfigSignature(),
			track.FileHash,
			100,
			correctedAt,
		)
		if err != nil {
			return err
		}
		_, err = s.repo.InsertSyncOutboxTx(r.Context(), tx, "manual_correction", correction.ID, "upsert", string(outboxPayload), correctedAt)
		return err
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"changed":               true,
		"requires_reanalysis":   true,
		"correction_id":         correction.ID,
		"queued_job_id":         jobID,
		"affected_playlist_ids": playlistIDs,
	})
}

func (s *server) handleSnapshot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req snapshotRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	playlist, err := s.repo.ResolvePlaylist(r.Context(), req.PlaylistID, req.PlaylistName)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	stats, err := s.repo.GetPlaylistStats(r.Context(), playlist.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	metadata, err := s.runtime.RefreshMetadata(r.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": scoringruntime.DescribeUnavailable(err)})
		return
	}
	tracks, err := s.repo.GetPlaylistSnapshotTracks(r.Context(), playlist.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	previousSyncState, err := s.repo.GetPlaylistSyncState(r.Context(), playlist.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	generatedAt := recommendationsrepo.NowUTC()
	snapshotID := uuid.NewString()
	if err := s.repo.UpsertPlaylistSyncState(r.Context(), playlist.ID, snapshotID, generatedAt, generatedAt); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	syncState, err := s.repo.GetPlaylistSyncState(r.Context(), playlist.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	snapshotTracks := make([]any, 0, len(tracks))
	readyCount := 0
	for _, track := range tracks {
		analysisState := track.AnalysisState
		if analysisState == "ready" {
			if reason := relativeRefreshReason(stats, metadata.GetExpectedRelativeSignature()); reason != "" {
				analysisState = "incompatible"
			}
		}
		if analysisState == "ready" {
			readyCount++
		}
		snapshotTracks = append(snapshotTracks, map[string]any{
			"track_id":       track.TrackID,
			"title":          track.Title,
			"artist":         track.Artist,
			"analysis_state": analysisState,
			"summary": map[string]any{
				"bpm":            track.BPM,
				"key":            stringOrNil(track.Key),
				"intensity_band": stringOrNil(track.IntensityBand),
				"outro":          nil,
			},
		})
	}
	totalTracks := len(tracks)
	coverageRatio := 0.0
	if totalTracks > 0 {
		coverageRatio = float64(readyCount) / float64(totalTracks)
	}
	freshnessReason := ""
	compatibilityState := "exact"
	requiresReanalysis := false
	lastSyncedAt := any(nil)
	deliveryState := "never_synced"
	if syncState != nil {
		switch {
		case syncState.LastSnapshotAckedAt == nil:
			if previousSyncState != nil && previousSyncState.LastSnapshotAckedAt != nil {
				deliveryState = "pending_ack"
				lastSyncedAt = *previousSyncState.LastSnapshotAckedAt
			}
		default:
			deliveryState = "acked"
			lastSyncedAt = *syncState.LastSnapshotAckedAt
		}
	}
	if stats != nil {
		if reason := relativeRefreshReason(stats, metadata.GetExpectedRelativeSignature()); reason != "" {
			requiresReanalysis = true
			compatibilityState = "incompatible"
			freshnessReason = reason
		}
	}
	payload := map[string]any{
		"snapshot_id":         snapshotID,
		"playlist_id":         playlist.ID,
		"snapshot_source":     "pc",
		"generated_at":        generatedAt,
		"analysis_signature":  metadata.GetActiveSignatures().GetAnalysisSignature(),
		"config_signature":    metadata.GetActiveSignatures().GetConfigSignature(),
		"scoring_contract_id": metadata.GetActiveSignatures().GetScoringContractId(),
		"analysis_coverage": map[string]any{
			"analyzed_tracks":  readyCount,
			"total_tracks":     totalTracks,
			"coverage_ratio":   coverageRatio,
			"vocals_available": metadata.GetCapabilityFlags()["vocals_available"],
		},
		"freshness": map[string]any{
			"last_synced_at":      lastSyncedAt,
			"delivery_state":      deliveryState,
			"requires_reanalysis": requiresReanalysis,
			"compatibility_state": compatibilityState,
			"note":                nullIfEmpty(freshnessReason),
		},
		"tracks": snapshotTracks,
		"precomputed": map[string]any{
			"live_recommendations": []any{},
			"playlist_analysis":    nil,
		},
	}
	writeJSON(w, http.StatusOK, payload)
}

func (s *server) handleSnapshotAck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req snapshotAckRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if strings.TrimSpace(req.SnapshotID) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "snapshot_id is required"})
		return
	}
	ackedAt := req.AckedAt
	if strings.TrimSpace(ackedAt) == "" {
		ackedAt = recommendationsrepo.NowUTC()
	}
	state, err := s.repo.AckPlaylistSnapshot(r.Context(), req.SnapshotID, ackedAt)
	if err != nil {
		if errors.Is(err, recommendationsrepo.ErrSnapshotNotFound) {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "snapshot not found"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"snapshot_id": state.LastSnapshotID,
		"playlist_id": state.PlaylistID,
		"acked_at":    stringOrNil(state.LastSnapshotAckedAt),
	})
}

func (s *server) handleOutboxPull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req outboxPullRequest
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	limit := req.Limit
	switch {
	case limit <= 0:
		limit = 100
	case limit > 500:
		limit = 500
	}
	items, hasMore, err := s.repo.PullUnsyncedOutbox(r.Context(), limit)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	responseItems := make([]any, 0, len(items))
	var nextAckID any = nil
	for _, item := range items {
		var payload any
		if err := json.Unmarshal([]byte(item.PayloadJSON), &payload); err != nil {
			payload = item.PayloadJSON
		}
		responseItems = append(responseItems, map[string]any{
			"id":          item.ID,
			"entity_type": item.EntityType,
			"entity_id":   item.EntityID,
			"action":      item.Action,
			"payload":     payload,
			"created_at":  item.CreatedAt,
		})
		nextAckID = item.ID
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"items":       responseItems,
		"has_more":    hasMore,
		"next_ack_id": nextAckID,
	})
}

func (s *server) handleOutboxAck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req outboxAckRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if req.AckThroughID <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "ack_through_id must be a positive integer"})
		return
	}
	ackedAt := req.AckedAt
	if strings.TrimSpace(ackedAt) == "" {
		ackedAt = recommendationsrepo.NowUTC()
	}
	ackedCount, err := s.repo.AckOutboxThroughID(r.Context(), req.AckThroughID, ackedAt)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ack_through_id": req.AckThroughID,
		"acked_count":    ackedCount,
		"acked_at":       ackedAt,
	})
}

func relativeRefreshReason(stats *recommendationsrepo.PlaylistStats, expectedRelativeSignature string) string {
	if stats == nil {
		return "Playlist relative features are missing; refresh relative analysis first."
	}
	if stats.IsStale {
		reason := stats.StaleReason
		if reason == "" {
			reason = "stale"
		}
		return fmt.Sprintf("Playlist relative features are stale (%s); refresh relative analysis first.", reason)
	}
	if expectedRelativeSignature != "" && stats.RelativeSignature != expectedRelativeSignature {
		return "Playlist relative features are out of date for the active scorer; refresh relative analysis first."
	}
	return ""
}

func buildRecommendationsRequest(
	hydrated *recommendationsrepo.HydratedRecommendations,
	target string,
	maxPerLane int32,
) *scoringv1.GetRecommendationsRequest {
	current := trackToProto(hydrated.Current)

	candidates := make([]*scoringv1.TrackContext, 0, len(hydrated.Candidates))
	for _, item := range hydrated.Candidates {
		if item.TrackID == hydrated.Current.TrackID || !item.Scoreable() {
			continue
		}
		candidates = append(candidates, trackToProto(item))
	}

	history := make([]*scoringv1.HistoryEntry, 0, len(hydrated.History))
	for idx, item := range hydrated.History {
		playsAgo := int32(len(hydrated.History) - idx)
		entry := &scoringv1.HistoryEntry{
			TrackId:    item.TrackID,
			MusicalKey: stringValue(item.Key),
			Relation:   "played",
			PlaysAgo:   &playsAgo,
		}
		if item.EnergyRel != nil {
			entry.EnergyRel = item.EnergyRel
		}
		history = append(history, entry)
	}

	req := &scoringv1.GetRecommendationsRequest{
		CurrentTrack:  current,
		Candidates:    candidates,
		History:       history,
		PlaylistStats: &scoringv1.PlaylistStatsContext{},
		TargetLane:    target,
		MaxPerLane:    maxPerLane,
	}
	if hydrated.Stats != nil {
		if hydrated.Stats.EnergySpread != nil {
			req.PlaylistStats.EnergySpread = hydrated.Stats.EnergySpread
		}
		weights, source := effectivePlaylistWeights(hydrated.Stats)
		if len(weights) > 0 {
			req.PlaylistStats.AdaptedWeights = weights
		}
		req.PlaylistStats.WeightSourceEnum = playlistWeightSourceEnum(source)
	} else {
		req.PlaylistStats.WeightSourceEnum = scoringv1.WeightSource_WEIGHT_SOURCE_STATIC
	}
	return req
}

func trackToProto(item recommendationsrepo.TrackContextRecord) *scoringv1.TrackContext {
	msg := &scoringv1.TrackContext{
		TrackId:       item.TrackID,
		Signatures:    signaturePayloadFromRecord(item),
		Bpm:           floatValue(item.BPM),
		MusicalKey:    stringValue(item.Key),
		KeySource:     stringValue(item.KeySource),
		IntensityBand: stringValue(item.IntensityBand),
		RoleHints:     item.RoleHints,
		Title:         item.Title,
		Artist:        item.Artist,
	}
	if item.KeyConfidence != nil {
		msg.KeyConfidence = item.KeyConfidence
	}
	if item.KeyAgreement != nil {
		value := *item.KeyAgreement
		msg.KeyAgreement = &value
	}
	if item.EnergyRel != nil {
		msg.EnergyRel = item.EnergyRel
	}
	if item.BassRel != nil {
		msg.BassRel = item.BassRel
	}
	if item.DrumsRel != nil {
		msg.DrumsRel = item.DrumsRel
	}
	if item.VocalsRel != nil {
		msg.VocalsRel = item.VocalsRel
	}
	if item.GrooveRel != nil {
		msg.GrooveRel = item.GrooveRel
	}
	return msg
}

func signaturePayloadFromRecord(item recommendationsrepo.TrackContextRecord) *scoringv1.SignatureMetadata {
	return &scoringv1.SignatureMetadata{
		AnalysisSignature: stringValue(item.AnalysisSignature),
		ConfigSignature:   stringValue(item.ConfigSignature),
		ScoringContractId: stringValue(item.ScoringContractAtAnalysis),
	}
}

func degradedRecommendationsPayload(
	hydrated *recommendationsrepo.HydratedRecommendations,
	target, statusName, note string,
	metadata *scoringv1.GetScoringMetadataResponse,
) map[string]any {
	capabilities := map[string]bool{}
	scoringContractID := ""
	if metadata != nil {
		capabilities = metadata.GetCapabilityFlags()
		scoringContractID = metadata.GetActiveSignatures().GetScoringContractId()
	}
	return map[string]any{
		"mode":                   "live",
		"recommendations_status": statusName,
		"current_track":          currentTrackPayload(hydrated.Current),
		"target":                 target,
		"set_context": map[string]any{
			"trend": map[string]any{
				"label":     "unavailable",
				"direction": "unknown",
			},
			"session_notes":  []string{},
			"history_length": len(hydrated.History),
			"has_gaps":       hydrated.HasGaps,
		},
		"recommendation_confidence": 0.0,
		"capabilities": map[string]any{
			"flags": capabilities,
		},
		"lane_order": liveLaneOrderForTarget(target),
		"lanes":      emptyLaneMap(note),
		"meta": map[string]any{
			"analysis_coverage":       analysisCoverage(hydrated),
			"filters_applied":         []string{},
			"strict_mode":             false,
			"assistant_mode":          true,
			"weight_adaptation":       map[string]any{"mode": "unavailable"},
			"scoring_contract_id":     scoringContractID,
			"status_note":             note,
			"recommendation_event_id": nil,
		},
	}
}

func translateRecommendationsResponse(resp *scoringv1.GetRecommendationsResponse) map[string]any {
	lanes := make(map[string]any, len(resp.GetLanes()))
	for _, lane := range resp.GetLanes() {
		items := make([]any, 0, len(lane.GetItems()))
		for _, item := range lane.GetItems() {
			items = append(items, map[string]any{
				"track_id":         item.GetCandidate().GetTrackId(),
				"title":            item.GetCandidate().GetTitle(),
				"artist":           item.GetCandidate().GetArtist(),
				"score":            item.GetFinalScore(),
				"ranking_strength": item.GetRankingStrength(),
				"move":             item.GetMove(),
				"move_confidence":  item.GetMoveConfidence(),
				"move_note":        item.GetMoveNote(),
				"risk":             item.GetRisk(),
				"risk_score":       item.GetRiskScore(),
				"tempo_key": map[string]any{
					"tempo_text": item.GetTempoKey().GetTempoText(),
					"key_text":   item.GetTempoKey().GetKeyText(),
					"key_state":  item.GetTempoKey().GetKeyState(),
				},
				"advisory_hints": append([]string{}, item.GetAdvisoryHints()...),
				"reasons":        append([]string{}, item.GetReasons()...),
				"watchouts":      append([]string{}, item.GetWatchouts()...),
				"explanation": map[string]any{
					"summary": append([]string{}, item.GetExplanation().GetSummary()...),
					"why":     append([]string{}, item.GetExplanation().GetWhy()...),
					"watch":   append([]string{}, item.GetExplanation().GetWatch()...),
					"handoff": advisoryPayload(item.GetExplanation().GetHandoff()),
					"tempo_key": map[string]any{
						"tempo_text": item.GetExplanation().GetTempoKey().GetTempoText(),
						"key_text":   item.GetExplanation().GetTempoKey().GetKeyText(),
						"key_state":  item.GetExplanation().GetTempoKey().GetKeyState(),
					},
					"character_shift": append([]string{}, item.GetExplanation().GetCharacterShift()...),
				},
				"windows": map[string]any{
					"candidate_intro_32": advisoryPayload(item.GetWindows().GetCandidateIntro_32()),
				},
			})
		}
		lanes[lane.GetLaneGroup().GetLaneId()] = map[string]any{
			"availability": lane.GetAvailability(),
			"items":        items,
			"empty_reason": nullIfEmpty(lane.GetEmptyReason()),
		}
	}
	for _, laneName := range liveLaneOrderForTarget(resp.GetMeta().GetTarget()) {
		if _, ok := lanes[laneName]; !ok {
			lanes[laneName] = map[string]any{
				"availability": "empty",
				"items":        []any{},
				"empty_reason": fmt.Sprintf("No viable %s options after current scoring filters.", laneName),
			}
		}
	}

	weights := resp.GetAppliedWeightAdaptation().GetComponentWeights()
	return map[string]any{
		"mode":                   "live",
		"recommendations_status": resp.GetRecommendationsStatus(),
		"current_track":          currentTrackPayloadFromProto(resp.GetCurrentTrack(), resp.GetCurrentOutroSummary()),
		"target":                 resp.GetMeta().GetTarget(),
		"set_context": map[string]any{
			"trend": map[string]any{
				"label":     resp.GetSetContext().GetTrend().GetLabel(),
				"direction": resp.GetSetContext().GetTrend().GetDirection(),
			},
			"session_notes":  append([]string{}, resp.GetSetContext().GetSessionNotes()...),
			"history_length": resp.GetSetContext().GetHistoryLength(),
			"has_gaps":       resp.GetSetContext().GetHasGaps(),
		},
		"recommendation_confidence": resp.GetRecommendationConfidence(),
		"capabilities": map[string]any{
			"flags": resp.GetCapabilities().GetFlags(),
		},
		"lane_order": append([]string{}, resp.GetLaneOrder()...),
		"lanes":      lanes,
		"meta": map[string]any{
			"analysis_coverage": fmt.Sprintf("%d/%d", resp.GetMeta().GetScoredCandidates(), resp.GetMeta().GetTotalCandidates()),
			"filters_applied":   []string{},
			"strict_mode":       false,
			"assistant_mode":    true,
			"weight_adaptation": map[string]any{
				"mode":                 resp.GetAppliedWeightAdaptation().GetAdaptationId(),
				"harmonic_effective":   weights["harmonic"],
				"bass_effective":       weights["bass_transition"],
				"tempo_effective":      nullIfZero(weights["tempo"]),
				"energy_effective":     nullIfZero(weights["target_energy"]),
				"vocal_effective":      nullIfZero(weights["vocal_transition"]),
				"similarity_effective": nullIfZero(weights["transition_support"]),
			},
			"scoring_contract_id":      resp.GetActiveSignatures().GetScoringContractId(),
			"status_note":              nullIfEmpty(resp.GetStatusNote()),
			"current_track_id":         resp.GetMeta().GetCurrentTrackId(),
			"requested_lane_available": resp.GetMeta().GetRequestedLaneAvailable(),
			"best_alternative_lanes":   append([]string{}, resp.GetMeta().GetBestAlternativeLanes()...),
			"fallback_note":            nullIfEmpty(resp.GetMeta().GetFallbackNote()),
			"recommendation_event_id":  nil,
		},
	}
}

func (s *server) recordRecommendationEvent(
	ctx context.Context,
	hydrated *recommendationsrepo.HydratedRecommendations,
	resp *scoringv1.GetRecommendationsResponse,
	payload map[string]any,
) (*string, error) {
	if resp.GetRecommendationsStatus() != "available" {
		return nil, nil
	}

	lanesSnapshot := make(map[string][]map[string]any)
	eventItems := make([]recommendationsrepo.RecommendationEventItemRecord, 0)
	eventID := uuid.NewString()
	for _, lane := range resp.GetLanes() {
		items := make([]map[string]any, 0, len(lane.GetItems()))
		for idx, item := range lane.GetItems() {
			items = append(items, map[string]any{
				"track_id":       item.GetCandidate().GetTrackId(),
				"score":          item.GetFinalScore(),
				"primary_lane":   item.GetPrimaryLane(),
				"secondary_lane": item.GetSecondaryLane(),
			})
			componentScoresJSON, err := json.Marshal(item.GetComponentScores())
			if err != nil {
				return nil, err
			}
			confidencesJSON, err := json.Marshal(item.GetComponentConfidences())
			if err != nil {
				return nil, err
			}
			weightsUsedJSON, err := json.Marshal(item.GetWeightsUsed())
			if err != nil {
				return nil, err
			}
			transitionFeatures := item.GetTransitionFeatures()
			transitionFeaturesJSON, err := json.Marshal(map[string]any{
				"effective_bpm_distance":   transitionFeatures.GetEffectiveBpmDistance(),
				"raw_bpm_distance":         transitionFeatures.GetRawBpmDistance(),
				"bpm_relationship":         transitionFeatures.GetBpmRelationship(),
				"key_distance":             transitionFeatures.GetKeyDistance(),
				"key_compat_label":         transitionFeatures.GetKeyCompatLabel(),
				"key_confidence_current":   transitionFeatures.GetKeyConfidenceCurrent(),
				"key_confidence_candidate": transitionFeatures.GetKeyConfidenceCandidate(),
				"delta_energy_rel":         transitionFeatures.GetDeltaEnergyRel(),
				"delta_bass_rel":           transitionFeatures.GetDeltaBassRel(),
				"current_vocals_rel":       transitionVocalValue(transitionFeatures, true),
				"candidate_vocals_rel":     transitionVocalValue(transitionFeatures, false),
				"current_outro_low_end":    transitionFeatures.GetCurrentOutroLowEnd(),
				"candidate_intro_low_end":  transitionFeatures.GetCandidateIntroLowEnd(),
			})
			if err != nil {
				return nil, err
			}
			var primaryLane *string
			if laneValue := firstNonEmpty(item.GetPrimaryLane(), lane.GetLaneGroup().GetLaneId()); laneValue != "" {
				primaryLane = &laneValue
			}
			eventItems = append(eventItems, recommendationsrepo.RecommendationEventItemRecord{
				EventID:                eventID,
				LaneID:                 lane.GetLaneGroup().GetLaneId(),
				LaneRank:               idx + 1,
				CandidateTrackID:       item.GetCandidate().GetTrackId(),
				FinalScore:             item.GetFinalScore(),
				RawScore:               item.GetRawScore(),
				PenaltyMultiplier:      item.GetPenaltyMultiplier(),
				Move:                   item.GetMove(),
				MoveConfidence:         item.GetMoveConfidence(),
				Risk:                   item.GetRisk(),
				RiskScore:              item.GetRiskScore(),
				PrimaryLane:            primaryLane,
				SecondaryLane:          item.GetSecondaryLane(),
				ComponentScoresJSON:    string(componentScoresJSON),
				ConfidencesJSON:        string(confidencesJSON),
				WeightsUsedJSON:        string(weightsUsedJSON),
				TransitionFeaturesJSON: string(transitionFeaturesJSON),
			})
		}
		lanesSnapshot[lane.GetLaneGroup().GetLaneId()] = items
	}
	compactLanes := map[string]any{
		"lane_order": payload["lane_order"],
		"lanes":      lanesSnapshot,
	}
	lanesJSON, err := json.Marshal(compactLanes)
	if err != nil {
		return nil, err
	}
	var adaptedWeightsJSON *string
	if weights := resp.GetAppliedWeightAdaptation().GetComponentWeights(); len(weights) > 0 {
		raw, err := json.Marshal(weights)
		if err != nil {
			return nil, err
		}
		value := string(raw)
		adaptedWeightsJSON = &value
	}
	record := recommendationsrepo.RecommendationEventRecord{
		ID:                    eventID,
		PlaylistID:            hydrated.Playlist.ID,
		CurrentTrackID:        hydrated.Current.TrackID,
		Target:                resp.GetMeta().GetTarget(),
		CandidateCount:        int(resp.GetMeta().GetScoredCandidates()),
		RecommendationsStatus: resp.GetRecommendationsStatus(),
		LanesReturnedJSON:     string(lanesJSON),
		AdaptedWeightsJSON:    adaptedWeightsJSON,
		ScoringContractID:     resp.GetActiveSignatures().GetScoringContractId(),
		Timestamp:             recommendationsrepo.NowUTC(),
	}
	if confidence := resp.GetRecommendationConfidence(); confidence > 0 {
		record.RecommendationConfidence = &confidence
	}
	if err := s.repo.RunInTx(ctx, func(tx *sql.Tx) error {
		if err := s.repo.InsertRecommendationEventTx(ctx, tx, record); err != nil {
			return err
		}
		return s.repo.InsertRecommendationEventItemsTx(ctx, tx, eventItems)
	}); err != nil {
		return nil, err
	}
	return &eventID, nil
}

func derivePlayedOutcome(lanesJSON string, chosenTrackID string) (bool, []string, error) {
	var payload struct {
		LaneOrder []string                    `json:"lane_order"`
		Lanes     map[string][]map[string]any `json:"lanes"`
	}
	if err := json.Unmarshal([]byte(lanesJSON), &payload); err != nil {
		return false, nil, err
	}
	chosenWasRecommended := false
	chosenScore := 0.0
	chosenLane := ""
	for _, laneName := range payload.LaneOrder {
		items := payload.Lanes[laneName]
		for _, item := range items {
			if asString(item["track_id"]) == chosenTrackID {
				chosenWasRecommended = true
				chosenScore = asFloat(item["score"])
				chosenLane = firstNonEmpty(asString(item["primary_lane"]), laneName)
				break
			}
		}
		if chosenWasRecommended {
			break
		}
	}
	var skipped []string
	if chosenWasRecommended {
		for _, laneName := range payload.LaneOrder {
			items := payload.Lanes[laneName]
			if laneName == chosenLane || len(items) == 0 {
				continue
			}
			if asFloat(items[0]["score"]) > chosenScore {
				skipped = append(skipped, laneName)
			}
		}
	}
	return chosenWasRecommended, skipped, nil
}

func (s *server) handleFeedbackSummary(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req feedbackSummaryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	since, until, err := validateFeedbackSummaryWindow(req.Since, req.Until)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	playlist, err := s.repo.ResolvePlaylist(r.Context(), req.PlaylistID, req.PlaylistName)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	events, err := s.repo.ListRecommendationEventsByPlaylistWindow(r.Context(), playlist.ID, since, until)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	eventIDs := make([]string, 0, len(events))
	for _, event := range events {
		eventIDs = append(eventIDs, event.ID)
	}
	itemsByEvent, err := s.repo.ListRecommendationEventItemsByEventIDs(r.Context(), eventIDs)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	stats, err := s.repo.GetPlaylistStats(r.Context(), playlist.ID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	rpcReq, err := buildFeedbackSummaryRPCRequest(playlist, events, itemsByEvent, stats, since, until)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	rpcResp, err := s.runtime.GetFeedbackSummary(r.Context(), rpcReq)
	if err == nil {
		writeJSON(w, http.StatusOK, translateFeedbackSummaryResponse(rpcResp))
		return
	}
	if scoringruntime.IsUnavailable(err) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": scoringruntime.DescribeUnavailable(err)})
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
}

func validateFeedbackSummaryWindow(rawSince, rawUntil string) (string, string, error) {
	since := strings.TrimSpace(rawSince)
	until := strings.TrimSpace(rawUntil)
	var parsedSince time.Time
	var parsedUntil time.Time
	var hasSince bool
	var hasUntil bool
	if since != "" {
		value, err := time.Parse(time.RFC3339, since)
		if err != nil {
			return "", "", fmt.Errorf("since must be a valid RFC3339 timestamp")
		}
		parsedSince = value
		hasSince = true
		since = value.Format(time.RFC3339)
	}
	if until != "" {
		value, err := time.Parse(time.RFC3339, until)
		if err != nil {
			return "", "", fmt.Errorf("until must be a valid RFC3339 timestamp")
		}
		parsedUntil = value
		hasUntil = true
		until = value.Format(time.RFC3339)
	}
	if hasSince && hasUntil && parsedSince.After(parsedUntil) {
		return "", "", fmt.Errorf("since must be before or equal to until")
	}
	return since, until, nil
}

func buildFeedbackSummaryRPCRequest(
	playlist recommendationsrepo.PlaylistRef,
	events []recommendationsrepo.RecommendationEventRecord,
	itemsByEvent map[string][]recommendationsrepo.RecommendationEventItemRecord,
	stats *recommendationsrepo.PlaylistStats,
	since, until string,
) (*scoringv1.GetFeedbackSummaryRequest, error) {
	req := &scoringv1.GetFeedbackSummaryRequest{
		PlaylistId:   playlist.ID,
		PlaylistName: playlist.Name,
		Window: &scoringv1.FeedbackSummaryWindow{
			Since: since,
			Until: until,
		},
		PlaylistStats: &scoringv1.FeedbackSummaryPlaylistStats{
			AdaptedWeights:       map[string]float64{},
			FeedbackTunedWeights: map[string]float64{},
		},
	}
	if stats != nil {
		req.PlaylistStats.AdaptedWeights = copyWeightMap(stats.AdaptedWeights)
		req.PlaylistStats.FeedbackTunedWeights = copyWeightMap(stats.FeedbackTunedWeights)
		req.PlaylistStats.FeedbackTuningNotes = append([]string{}, stats.FeedbackTuningNotes...)
		req.PlaylistStats.FeedbackEventCount = int32(stats.FeedbackEventCount)
		if stats.FeedbackLastTunedAt != nil {
			req.PlaylistStats.FeedbackLastTunedAt = *stats.FeedbackLastTunedAt
		}
		if len(stats.FeedbackTuningMetrics) > 0 {
			metrics, err := structpb.NewStruct(stats.FeedbackTuningMetrics)
			if err != nil {
				return nil, err
			}
			req.PlaylistStats.FeedbackTuningMetrics = metrics
		}
	}
	req.Events = make([]*scoringv1.FeedbackSummaryEvent, 0, len(events))
	for _, event := range events {
		pbEvent := &scoringv1.FeedbackSummaryEvent{
			EventId:              event.ID,
			Timestamp:            event.Timestamp,
			ChosenWasRecommended: event.ChosenWasRecommended != nil && *event.ChosenWasRecommended,
			Items:                make([]*scoringv1.FeedbackSummaryEventItem, 0, len(itemsByEvent[event.ID])),
		}
		if event.TrackChosen != nil {
			pbEvent.TrackChosen = *event.TrackChosen
		}
		for _, item := range itemsByEvent[event.ID] {
			pbEvent.Items = append(pbEvent.Items, &scoringv1.FeedbackSummaryEventItem{
				CandidateTrackId: item.CandidateTrackID,
				FinalScore:       item.FinalScore,
				LaneId:           item.LaneID,
				PrimaryLane:      firstNonEmpty(stringValue(item.PrimaryLane), item.LaneID),
			})
		}
		req.Events = append(req.Events, pbEvent)
	}
	return req, nil
}

func translateFeedbackSummaryResponse(resp *scoringv1.GetFeedbackSummaryResponse) map[string]any {
	metrics := resp.GetMetrics()
	weights := resp.GetWeights()
	tuning := resp.GetTuning()
	var meanChosenRank any
	if metrics != nil && metrics.MeanChosenRank != nil {
		meanChosenRank = metrics.GetMeanChosenRank()
	}
	tuningMetrics := map[string]any{}
	if tuning != nil && tuning.GetMetrics() != nil {
		tuningMetrics = tuning.GetMetrics().AsMap()
	}
	return map[string]any{
		"playlist_id":   resp.GetPlaylistId(),
		"playlist_name": resp.GetPlaylistName(),
		"window": map[string]any{
			"since": nullIfEmpty(resp.GetWindow().GetSince()),
			"until": nullIfEmpty(resp.GetWindow().GetUntil()),
		},
		"metrics": map[string]any{
			"total_events":                   metrics.GetTotalEvents(),
			"contributory_events":            metrics.GetContributoryEvents(),
			"ranked_events":                  metrics.GetRankedEvents(),
			"pairwise_comparison_count":      metrics.GetPairwiseComparisonCount(),
			"chosen_top1_rate":               metrics.GetChosenTop1Rate(),
			"chosen_top3_rate":               metrics.GetChosenTop3Rate(),
			"chosen_top5_rate":               metrics.GetChosenTop5Rate(),
			"mean_chosen_rank":               meanChosenRank,
			"lane_acceptance_counts":         int32MapToAnyMap(metrics.GetLaneAcceptanceCounts()),
			"higher_scored_lane_skip_counts": int32MapToAnyMap(metrics.GetHigherScoredLaneSkipCounts()),
		},
		"weights": map[string]any{
			"source":    weights.GetSource(),
			"static":    float64MapToAnyMap(weights.GetStaticWeights()),
			"base":      float64MapToAnyMap(weights.GetBaseWeights()),
			"tuned":     nilIfEmptyWeightMap(weights.GetTunedWeights()),
			"effective": float64MapToAnyMap(weights.GetEffectiveWeights()),
		},
		"tuning": map[string]any{
			"last_tuned_at":        nullIfEmpty(tuning.GetLastTunedAt()),
			"feedback_event_count": tuning.GetFeedbackEventCount(),
			"notes":                append([]string{}, tuning.GetNotes()...),
			"metrics":              tuningMetrics,
		},
	}
}

func parseCorrectionBPM(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, typed > 0
	case int:
		return float64(typed), typed > 0
	default:
		return 0, false
	}
}

func playlistWeightSource(stats *recommendationsrepo.PlaylistStats) string {
	if stats != nil && len(stats.FeedbackTunedWeights) > 0 {
		return "feedback_tuned_weights"
	}
	if stats != nil && len(stats.AdaptedWeights) > 0 {
		return "adapted_weights"
	}
	return "static"
}

func playlistWeightSourceEnum(source string) scoringv1.WeightSource {
	switch strings.TrimSpace(source) {
	case "feedback_tuned_weights":
		return scoringv1.WeightSource_WEIGHT_SOURCE_FEEDBACK_TUNED
	case "adapted_weights":
		return scoringv1.WeightSource_WEIGHT_SOURCE_ADAPTED
	case "static":
		return scoringv1.WeightSource_WEIGHT_SOURCE_STATIC
	default:
		return scoringv1.WeightSource_WEIGHT_SOURCE_UNSPECIFIED
	}
}

func effectivePlaylistWeights(stats *recommendationsrepo.PlaylistStats) (map[string]float64, string) {
	if stats != nil && len(stats.FeedbackTunedWeights) > 0 {
		return copyWeightMap(stats.FeedbackTunedWeights), "feedback_tuned_weights"
	}
	if stats != nil && len(stats.AdaptedWeights) > 0 {
		return copyWeightMap(stats.AdaptedWeights), "adapted_weights"
	}
	return nil, "static"
}

func applyPlaylistWeightSource(payload map[string]any, stats *recommendationsrepo.PlaylistStats) {
	meta, ok := payload["meta"].(map[string]any)
	if !ok {
		return
	}
	weightAdaptation, ok := meta["weight_adaptation"].(map[string]any)
	if !ok {
		return
	}
	source := playlistWeightSource(stats)
	weightAdaptation["mode"] = source
	weightAdaptation["source"] = source
}

func copyWeightMap(source map[string]float64) map[string]float64 {
	if len(source) == 0 {
		return map[string]float64{}
	}
	out := make(map[string]float64, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

func float64MapToAnyMap(source map[string]float64) map[string]any {
	if len(source) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

func int32MapToAnyMap(source map[string]int32) map[string]any {
	if len(source) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

func nilIfEmptyWeightMap(source map[string]float64) any {
	if len(source) == 0 {
		return nil
	}
	return float64MapToAnyMap(source)
}

func valueOrNil(value *float64) any {
	if value == nil {
		return nil
	}
	return *value
}

func transitionVocalValue(features *scoringv1.TransitionFeatures, current bool) any {
	if features == nil {
		return nil
	}
	if current {
		return valueOrNil(features.CurrentVocalsRel)
	}
	return valueOrNil(features.CandidateVocalsRel)
}

func parseCorrectionKey(value any) (string, bool) {
	typed, ok := value.(string)
	if !ok {
		return "", false
	}
	typed = strings.TrimSpace(typed)
	return typed, typed != ""
}

func asString(value any) string {
	typed, _ := value.(string)
	return typed
}

func asFloat(value any) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	default:
		return 0
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func currentTrackPayload(record recommendationsrepo.TrackContextRecord) map[string]any {
	return map[string]any{
		"track_id":       record.TrackID,
		"title":          record.Title,
		"artist":         record.Artist,
		"bpm":            record.BPM,
		"key":            stringOrNil(record.Key),
		"intensity_band": stringOrNil(record.IntensityBand),
		"role_hints":     record.RoleHints,
		"outro_summary":  nil,
	}
}

func currentTrackPayloadFromProto(track *scoringv1.TrackContext, outro *scoringv1.OutroSummary) map[string]any {
	var outroPayload any
	if outro != nil && outro.GetText() != "" {
		outroPayload = map[string]any{
			"text":            outro.GetText(),
			"cleanliness_abs": outro.GetCleanlinessAbs(),
		}
	}
	return map[string]any{
		"track_id":       track.GetTrackId(),
		"title":          track.GetTitle(),
		"artist":         track.GetArtist(),
		"bpm":            track.GetBpm(),
		"key":            nullIfEmpty(track.GetMusicalKey()),
		"intensity_band": nullIfEmpty(track.GetIntensityBand()),
		"role_hints":     append([]string{}, track.GetRoleHints()...),
		"outro_summary":  outroPayload,
	}
}

func advisoryPayload(advisory *scoringv1.AdvisoryText) any {
	if advisory == nil || (advisory.GetLevel() == "" && len(advisory.GetNotes()) == 0) {
		return nil
	}
	return map[string]any{
		"level": advisory.GetLevel(),
		"notes": append([]string{}, advisory.GetNotes()...),
	}
}

func liveLaneOrderForTarget(target string) []string {
	ordered := make([]string, 0, len(liveLaneOrder))
	if target != "" {
		ordered = append(ordered, target)
	}
	for _, lane := range liveLaneOrder {
		if lane != target {
			ordered = append(ordered, lane)
		}
	}
	return ordered
}

func emptyLaneMap(note string) map[string]any {
	lanes := make(map[string]any, len(liveLaneOrder))
	for _, lane := range liveLaneOrder {
		lanes[lane] = map[string]any{
			"availability": "empty",
			"items":        []any{},
			"empty_reason": note,
		}
	}
	return lanes
}

func analysisCoverage(hydrated *recommendationsrepo.HydratedRecommendations) string {
	total := len(hydrated.Candidates)
	if total == 0 {
		return "0/0"
	}
	analyzed := 0
	for _, candidate := range hydrated.Candidates {
		if candidate.EnergyRel != nil {
			analyzed++
		}
	}
	return fmt.Sprintf("%d/%d", analyzed, total)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func errorString(err error) any {
	if err == nil {
		return nil
	}
	return err.Error()
}

func nullIfEmpty(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func nullIfZero(value float64) any {
	if value == 0 {
		return nil
	}
	return value
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func stringOrNil(value *string) any {
	if value == nil || strings.TrimSpace(*value) == "" {
		return nil
	}
	return *value
}

func floatValue(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}
