package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
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
	defaultAPIAddr                  = "127.0.0.1:8080"
	pythonModule                    = "cuemate_analysis"
	maxRecommendationEventsResponse = 200
)

var liveLaneOrder = []string{"maintain", "build", "reset", "jump", "contrast"}

type appConfig struct {
	Addr     string
	Database string
	WebDist  string
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

type remotePairingTokenRequest struct {
	DeviceLabel string `json:"device_label"`
}

type remotePairRequest struct {
	Token       string `json:"token"`
	DeviceLabel string `json:"device_label"`
}

type enqueueAnalysisRequest struct {
	AnalysisMode string `json:"analysis_mode"`
	Force        bool   `json:"force"`
}

type toolCommandRequest struct {
	Action                  string   `json:"action"`
	Name                    string   `json:"name"`
	Paths                   []string `json:"paths"`
	Source                  string   `json:"source"`
	Library                 string   `json:"library"`
	Playlist                string   `json:"playlist"`
	AnalysisMode            string   `json:"analysis_mode"`
	Force                   bool     `json:"force"`
	Limit                   int      `json:"limit"`
	Path                    string   `json:"path"`
	PrintBackendDiagnostics bool     `json:"print_backend_diagnostics"`
}

type pickPathRequest struct {
	Kind string `json:"kind"`
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
	mux.HandleFunc("/setup/status", srv.handleSetupStatus)
	mux.HandleFunc("/scoring/metadata", srv.handleMetadata)
	mux.HandleFunc("/playlists", srv.handlePlaylists)
	mux.HandleFunc("/playlists/", srv.handlePlaylistRoutes)
	mux.HandleFunc("/tracks/search", srv.handleTrackSearch)
	mux.HandleFunc("/analysis/jobs", srv.handleAnalysisJobs)
	mux.HandleFunc("/recommendation-events", srv.handleRecommendationEvents)
	mux.HandleFunc("/recommendations", srv.handleRecommendations)
	mux.HandleFunc("/events/played", srv.handlePlayedEvent)
	mux.HandleFunc("/feedback/summary", srv.handleFeedbackSummary)
	mux.HandleFunc("/corrections", srv.handleCorrections)
	mux.HandleFunc("/sync/playlists/snapshot", srv.handleSnapshot)
	mux.HandleFunc("/sync/playlists/snapshot/ack", srv.handleSnapshotAck)
	mux.HandleFunc("/sync/outbox/pull", srv.handleOutboxPull)
	mux.HandleFunc("/sync/outbox/ack", srv.handleOutboxAck)
	mux.HandleFunc("/remote/status", srv.handleRemoteStatus)
	mux.HandleFunc("/remote/pairing-token", srv.handleRemotePairingToken)
	mux.HandleFunc("/remote/pair", srv.handleRemotePair)
	mux.HandleFunc("/remote/logout", srv.handleRemoteLogout)
	mux.HandleFunc("/tools/cli", srv.handleToolCommand)
	mux.HandleFunc("/tools/runs/", srv.handleToolRun)
	mux.HandleFunc("/tools/pick-path", srv.handlePickPath)
	mux.HandleFunc("/", handleWebApp(cfg.WebDist))

	httpServer := &http.Server{
		Addr:         cfg.Addr,
		Handler:      srv.remoteAccessMiddleware(mux),
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
		WebDist:  firstNonEmpty(strings.TrimSpace(os.Getenv("WEB_DIST_DIR")), "web/dist"),
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

func (s *server) handleSetupStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	statePath := strings.TrimSpace(os.Getenv("CUEMATE_SETUP_STATE_PATH"))
	logDir := strings.TrimSpace(os.Getenv("CUEMATE_LOG_DIR"))
	payload := map[string]any{
		"available":    false,
		"status":       "unknown",
		"step":         "",
		"message":      "",
		"core_ready":   true,
		"docker_ready": false,
		"model_ready":  false,
		"mobile_ready": remoteBaseURL() != "",
		"log_dir":      nullIfEmpty(logDir),
	}
	if statePath == "" {
		writeJSON(w, http.StatusOK, payload)
		return
	}
	data, err := os.ReadFile(statePath)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			payload["message"] = err.Error()
		}
		writeJSON(w, http.StatusOK, payload)
		return
	}
	var state map[string]any
	if err := json.Unmarshal(data, &state); err != nil {
		payload["message"] = "setup state could not be read"
		writeJSON(w, http.StatusOK, payload)
		return
	}
	for key, value := range state {
		payload[key] = value
	}
	payload["available"] = true
	if _, ok := payload["log_dir"]; !ok || payload["log_dir"] == "" {
		payload["log_dir"] = nullIfEmpty(logDir)
	}
	writeJSON(w, http.StatusOK, payload)
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

func handleWebApp(webDist string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}
		webDistAbs, err := filepath.Abs(webDist)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		webDistAbs = filepath.Clean(webDistAbs)
		cleanPath := filepath.Clean(strings.TrimPrefix(r.URL.Path, "/"))
		if cleanPath == "." {
			cleanPath = "index.html"
		}
		if filepath.IsAbs(cleanPath) || cleanPath == ".." || strings.HasPrefix(cleanPath, ".."+string(filepath.Separator)) {
			http.NotFound(w, r)
			return
		}
		fullPath, err := filepath.Abs(filepath.Join(webDistAbs, cleanPath))
		if err != nil {
			http.NotFound(w, r)
			return
		}
		fullPath = filepath.Clean(fullPath)
		rel, err := filepath.Rel(webDistAbs, fullPath)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			http.NotFound(w, r)
			return
		}
		if info, err := os.Stat(fullPath); err == nil && !info.IsDir() {
			http.ServeFile(w, r, fullPath)
			return
		}
		http.ServeFile(w, r, filepath.Join(webDistAbs, "index.html"))
	}
}

func (s *server) remoteAccessMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isLocalRequest(r) || isRemotePublicRequest(r) {
			next.ServeHTTP(w, r)
			return
		}
		if isRemoteBlockedPath(r.URL.Path) {
			writeJSON(w, http.StatusForbidden, map[string]string{"error": "remote sessions cannot access this endpoint"})
			return
		}
		session, ok := s.remoteSession(r)
		if !ok {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "remote pairing required"})
			return
		}
		_ = session
		next.ServeHTTP(w, r)
	})
}

func isLocalRequest(r *http.Request) bool {
	host, _, err := net.SplitHostPort(strings.TrimSpace(r.RemoteAddr))
	if err != nil {
		host = strings.Trim(strings.TrimSpace(r.RemoteAddr), "[]")
	}
	if host == "" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && (ip.IsLoopback() || ip.IsUnspecified())
}

func isRemotePublicRequest(r *http.Request) bool {
	switch {
	case r.URL.Path == "/remote/status":
		return true
	case r.URL.Path == "/remote/pair" && r.Method == http.MethodPost:
		return true
	case r.Method == http.MethodGet || r.Method == http.MethodHead:
		return !isAPIPath(r.URL.Path)
	default:
		return false
	}
}

func isAPIPath(path string) bool {
	for _, prefix := range []string{
		"/healthz",
		"/readyz",
		"/scoring/",
		"/playlists",
		"/tracks/",
		"/analysis/",
		"/recommendation-events",
		"/recommendations",
		"/events/",
		"/feedback/",
		"/corrections",
		"/sync/",
		"/setup/",
		"/tools/",
		"/remote/",
	} {
		if path == strings.TrimSuffix(prefix, "/") || strings.HasPrefix(path, prefix) {
			return true
		}
	}
	return false
}

func isRemoteBlockedPath(path string) bool {
	for _, prefix := range []string{"/tools/", "/analysis/", "/corrections", "/sync/outbox", "/remote/pairing-token"} {
		if path == strings.TrimSuffix(prefix, "/") || strings.HasPrefix(path, prefix) {
			return true
		}
	}
	return false
}

func (s *server) remoteSession(r *http.Request) (*recommendationsrepo.RemoteSession, bool) {
	cookie, err := r.Cookie("cuemate_remote_session")
	if err != nil || strings.TrimSpace(cookie.Value) == "" {
		return nil, false
	}
	session, err := s.repo.GetValidRemoteSession(r.Context(), hashSecret(cookie.Value), recommendationsrepo.NowUTC())
	if err != nil {
		log.Printf("remote session lookup failed: %v", err)
		return nil, false
	}
	return session, session != nil
}

func (s *server) handlePlaylists(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	playlists, err := s.repo.ListPlaylists(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	items := make([]any, 0, len(playlists))
	for _, playlist := range playlists {
		items = append(items, playlistSummaryPayload(playlist))
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items})
}

func (s *server) handlePlaylistRoutes(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/playlists/"), "/")
	if rest == "" {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "playlist route not found"})
		return
	}
	parts := strings.Split(rest, "/")
	playlistID := parts[0]
	switch {
	case len(parts) == 1 && (r.Method == http.MethodGet || r.Method == http.MethodDelete):
		s.handlePlaylistDetail(w, r, playlistID)
	case len(parts) == 2 && parts[1] == "tracks":
		s.handlePlaylistTracks(w, r, playlistID)
	case len(parts) == 4 && parts[1] == "tracks" && parts[3] == "features":
		trackID, err := url.PathUnescape(parts[2])
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid track id"})
			return
		}
		s.handleTrackFeatureDetail(w, r, playlistID, trackID)
	case len(parts) == 3 && parts[1] == "analysis" && parts[2] == "enqueue":
		s.handlePlaylistAnalysisEnqueue(w, r, playlistID)
	case len(parts) == 3 && parts[1] == "analysis" && parts[2] == "refresh":
		s.handlePlaylistAnalysisRefresh(w, r, playlistID)
	case len(parts) == 3 && parts[1] == "analysis" && parts[2] == "status":
		s.handlePlaylistAnalysisStatus(w, r, playlistID)
	default:
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "playlist route not found"})
	}
}

func (s *server) handlePlaylistDetail(w http.ResponseWriter, r *http.Request, playlistID string) {
	if r.Method == http.MethodDelete {
		if err := s.repo.DeletePlaylist(r.Context(), playlistID); err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
				status = http.StatusNotFound
			}
			writeJSON(w, status, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"removed": true, "playlist_id": playlistID})
		return
	}
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	playlist, err := s.repo.ResolvePlaylist(r.Context(), playlistID, "")
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
	writeJSON(w, http.StatusOK, map[string]any{
		"playlist_id": playlist.ID,
		"name":        playlist.Name,
		"stats":       playlistStatsPayload(stats),
	})
}

func (s *server) handlePlaylistTracks(w http.ResponseWriter, r *http.Request, playlistID string) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	if _, err := s.repo.ResolvePlaylist(r.Context(), playlistID, ""); err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	query := r.URL.Query().Get("query")
	analysisState := r.URL.Query().Get("analysis_state")
	limit := queryInt(r, "limit", 100)
	offset := queryInt(r, "offset", 0)
	metadata, metadataErr := s.runtime.RefreshMetadata(r.Context())
	var analysisSignature, configSignature, scoringContractID string
	if metadataErr == nil {
		analysisSignature, configSignature, scoringContractID = activeSignatureValues(metadata.GetActiveSignatures())
	}
	tracks, err := s.repo.ListPlaylistTracks(r.Context(), playlistID, query, analysisState, limit, offset, analysisSignature, configSignature, scoringContractID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	items := make([]any, 0, len(tracks))
	for _, track := range tracks {
		items = append(items, playlistTrackPayload(track))
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items, "limit": limit, "offset": offset})
}

func (s *server) handleTrackFeatureDetail(w http.ResponseWriter, r *http.Request, playlistID string, trackID string) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	if _, err := s.repo.ResolvePlaylist(r.Context(), playlistID, ""); err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	detail, err := s.repo.GetTrackFeatureDetail(r.Context(), playlistID, trackID)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, recommendationsrepo.ErrTrackNotFound) {
			status = http.StatusNotFound
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, trackFeatureDetailPayload(detail))
}

func (s *server) handleTrackSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	playlistID := r.URL.Query().Get("playlist_id")
	tracks, err := s.repo.SearchTracks(r.Context(), playlistID, r.URL.Query().Get("query"), queryInt(r, "limit", 25))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	items := make([]any, 0, len(tracks))
	for _, track := range tracks {
		items = append(items, playlistTrackPayload(track))
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items})
}

func (s *server) handleAnalysisJobs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	jobs, err := s.repo.ListAnalysisJobs(r.Context(), r.URL.Query().Get("playlist_id"), r.URL.Query().Get("status"), queryInt(r, "limit", 50))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	items := make([]any, 0, len(jobs))
	for _, job := range jobs {
		items = append(items, analysisJobPayload(job))
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items})
}

func (s *server) handleRecommendationEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	playlistID := strings.TrimSpace(r.URL.Query().Get("playlist_id"))
	if playlistID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "playlist_id is required"})
		return
	}
	events, err := s.repo.ListRecommendationEventsByPlaylist(r.Context(), playlistID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	selectedEvents := boundedRecommendationEvents(events, queryInt(r, "limit", 25))
	items := make([]any, 0, len(selectedEvents))
	for _, event := range selectedEvents {
		items = append(items, recommendationEventPayload(event))
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items})
}

func boundedRecommendationEvents(events []recommendationsrepo.RecommendationEventRecord, requestedLimit int) []recommendationsrepo.RecommendationEventRecord {
	if requestedLimit <= 0 {
		return events[:0]
	}
	if requestedLimit > maxRecommendationEventsResponse {
		requestedLimit = maxRecommendationEventsResponse
	}
	if requestedLimit > len(events) {
		requestedLimit = len(events)
	}
	return events[:requestedLimit]
}

func (s *server) handleToolCommand(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req toolCommandRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	cliArgs, background, err := buildToolCommand(req)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	pythonExe, err := pythonExecutable()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	command := append([]string{pythonExe}, cliArgs...)
	if background {
		result, err := startBackgroundToolCommand(pythonExe, cliArgs)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}
		result["command"] = command
		writeJSON(w, http.StatusAccepted, result)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 25*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, pythonExe, cliArgs...)
	output, err := cmd.CombinedOutput()
	exitCode := 0
	if err != nil {
		exitCode = 1
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			exitCode = exitErr.ExitCode()
		}
	}
	status := "completed"
	if ctx.Err() == context.DeadlineExceeded {
		status = "timeout"
	}
	httpStatus := http.StatusOK
	if exitCode != 0 || status == "timeout" {
		httpStatus = http.StatusBadRequest
	}
	writeJSON(w, httpStatus, map[string]any{
		"status":    status,
		"mode":      "foreground",
		"command":   command,
		"exit_code": exitCode,
		"output":    string(output),
	})
}

func (s *server) handlePickPath(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req pickPathRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	script, err := pickerScript(req.Kind)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-STA", "-Command", script)
	output, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		writeJSON(w, http.StatusRequestTimeout, map[string]string{"error": "file picker timed out"})
		return
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": strings.TrimSpace(string(output))})
		return
	}
	paths, err := parsePickerPaths(output)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("parse picker output: %v", err)})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"paths": paths})
}

func parsePickerPaths(output []byte) ([]string, error) {
	payload := bytes.TrimSpace(output)
	var paths []string
	if err := json.Unmarshal(payload, &paths); err == nil {
		return paths, nil
	}
	var singlePath string
	if err := json.Unmarshal(payload, &singlePath); err != nil {
		return nil, err
	}
	if strings.TrimSpace(singlePath) == "" {
		return []string{}, nil
	}
	return []string{singlePath}, nil
}

func (s *server) handleToolRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	runID := strings.Trim(strings.TrimPrefix(r.URL.Path, "/tools/runs/"), "/")
	metaPath, err := toolRunMetaPath(runID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "tool run not found"})
		return
	}
	data, err := os.ReadFile(metaPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "tool run not found"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "tool run metadata is invalid"})
		return
	}
	logPath, _ := payload["log_path"].(string)
	tail, err := readToolRunTail(logPath, 16*1024)
	if err != nil {
		tail = ""
	}
	payload["output_tail"] = tail
	writeJSON(w, http.StatusOK, payload)
}

func validRunID(runID string) bool {
	if len(runID) != 36 {
		return false
	}
	_, err := uuid.Parse(runID)
	return err == nil
}

func pickerScript(kind string) (string, error) {
	switch strings.TrimSpace(kind) {
	case "folder":
		return `
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose a folder to import into CueMate'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  ConvertTo-Json -InputObject @($dialog.SelectedPath) -Compress
} else {
  ConvertTo-Json -InputObject @() -Compress
}
`, nil
	case "audio_files":
		return `
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose audio files to import into CueMate'
$dialog.Filter = 'Audio files|*.mp3;*.wav;*.aiff;*.aif;*.flac;*.m4a;*.ogg|All files|*.*'
$dialog.Multiselect = $true
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  ConvertTo-Json -InputObject @($dialog.FileNames) -Compress
} else {
  ConvertTo-Json -InputObject @() -Compress
}
`, nil
	case "dj_library_file":
		return `
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose Rekordbox XML or Traktor NML export'
$dialog.Filter = 'DJ library exports|*.xml;*.nml|Rekordbox XML|*.xml|Traktor NML|*.nml|All files|*.*'
$dialog.Multiselect = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  ConvertTo-Json -InputObject @($dialog.FileName) -Compress
} else {
  ConvertTo-Json -InputObject @() -Compress
}
`, nil
	default:
		return "", errors.New("kind must be folder, audio_files, or dj_library_file")
	}
}

func buildToolCommand(req toolCommandRequest) ([]string, bool, error) {
	action := strings.TrimSpace(req.Action)
	args := []string{"-m", pythonModule}
	switch action {
	case "import_playlist":
		name, err := safeCLIValue(req.Name, "name")
		if err != nil {
			return nil, false, err
		}
		if name == "" {
			return nil, false, errors.New("name is required")
		}
		paths, err := safeCLIPaths(req.Paths)
		if err != nil {
			return nil, false, err
		}
		if len(paths) == 0 {
			return nil, false, errors.New("at least one path is required")
		}
		args = append(args, "import-playlist", "--name", name)
		args = append(args, "--")
		args = append(args, paths...)
		return args, true, nil
	case "list_dj_playlists":
		source, err := normalizeDJSource(req.Source)
		if err != nil {
			return nil, false, err
		}
		library, err := safeCLIPath(req.Library, "library")
		if err != nil {
			return nil, false, err
		}
		if library == "" {
			return nil, false, errors.New("library is required")
		}
		return append(args, "list-dj-playlists", "--source", source, "--library", library), false, nil
	case "import_dj_playlist":
		source, err := normalizeDJSource(req.Source)
		if err != nil {
			return nil, false, err
		}
		library, err := safeCLIPath(req.Library, "library")
		if err != nil {
			return nil, false, err
		}
		playlist, err := safeCLIValue(req.Playlist, "playlist")
		if err != nil {
			return nil, false, err
		}
		if library == "" || playlist == "" {
			return nil, false, errors.New("library and playlist are required")
		}
		args = append(args, "import-dj-playlist", "--source", source, "--library", library, "--playlist", playlist)
		name, err := safeCLIValue(req.Name, "name")
		if err != nil {
			return nil, false, err
		}
		if name != "" {
			args = append(args, "--name", name)
		}
		return args, true, nil
	case "analyze_playlist":
		playlist, err := safeCLIValue(req.Playlist, "playlist")
		if err != nil {
			return nil, false, err
		}
		if playlist == "" {
			return nil, false, errors.New("playlist is required")
		}
		mode := strings.TrimSpace(req.AnalysisMode)
		if mode == "" {
			mode = "staged"
		}
		if mode != "fast_pass" && mode != "staged" && mode != "full" {
			return nil, false, errors.New("analysis_mode must be fast_pass, staged, or full")
		}
		args = append(args, "analyze-playlist", "--playlist", playlist, "--analysis-mode", mode)
		if req.Force {
			args = append(args, "--force")
		}
		return args, true, nil
	case "run_analysis_worker":
		limit := boundedPositive(req.Limit, 100, 1000)
		args = append(args, "run-analysis-worker", "--limit", strconv.Itoa(limit))
		if req.PrintBackendDiagnostics {
			args = append(args, "--print-backend-diagnostics")
		}
		return args, true, nil
	case "run_feedback_worker":
		limit := boundedPositive(req.Limit, 50, 1000)
		return append(args, "run-feedback-worker", "--limit", strconv.Itoa(limit)), true, nil
	case "prewarm_model_services":
		args = append(args, "prewarm-model-services")
		path, err := safeCLIPath(req.Path, "path")
		if err != nil {
			return nil, false, err
		}
		if path != "" {
			args = append(args, "--path", path)
		}
		return args, true, nil
	case "download_essentia_models":
		return append(args, "download-essentia-semantic-models"), true, nil
	default:
		return nil, false, fmt.Errorf("unsupported action %q", action)
	}
}

func startBackgroundToolCommand(pythonExe string, cliArgs []string) (map[string]any, error) {
	runID := uuid.NewString()
	logDir := toolRunDir()
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, err
	}
	startedAt := time.Now().UTC().Format(time.RFC3339Nano)
	logPath := filepath.Join(logDir, runID+".log")
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(pythonExe, cliArgs...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return nil, err
	}
	meta := map[string]any{
		"run_id":     runID,
		"status":     "running",
		"mode":       "background",
		"pid":        cmd.Process.Pid,
		"command":    append([]string{pythonExe}, cliArgs...),
		"log_path":   logPath,
		"started_at": startedAt,
	}
	if err := writeToolRunMeta(runID, meta); err != nil {
		_ = logFile.Close()
		return nil, err
	}
	go func() {
		err := cmd.Wait()
		finishedAt := time.Now().UTC().Format(time.RFC3339Nano)
		status := "completed"
		if err != nil {
			status = "failed"
			_, _ = fmt.Fprintf(logFile, "\n[cuemate] command failed: %v\n", err)
		}
		_ = logFile.Close()
		meta["status"] = status
		meta["finished_at"] = finishedAt
		if err != nil {
			meta["error"] = err.Error()
		}
		_ = writeToolRunMeta(runID, meta)
	}()
	return map[string]any{
		"run_id":   runID,
		"status":   "started",
		"mode":     "background",
		"pid":      cmd.Process.Pid,
		"log_path": logPath,
	}, nil
}

func toolRunDir() string {
	if logDir := strings.TrimSpace(os.Getenv("CUEMATE_LOG_DIR")); logDir != "" {
		return filepath.Join(logDir, "tool-runs")
	}
	return filepath.Join("tmp", "tool-runs")
}

func toolRunMetaPath(runID string) (string, error) {
	parsed, err := uuid.Parse(runID)
	if err != nil || parsed.String() != runID {
		return "", errors.New("invalid tool run id")
	}
	logDir, err := filepath.Abs(toolRunDir())
	if err != nil {
		return "", err
	}
	metaPath, err := filepath.Abs(filepath.Join(logDir, parsed.String()+".json"))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(logDir, metaPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", errors.New("invalid tool run path")
	}
	return metaPath, nil
}

func writeToolRunMeta(runID string, payload map[string]any) error {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	metaPath, err := toolRunMetaPath(runID)
	if err != nil {
		return err
	}
	return os.WriteFile(metaPath, data, 0o644)
}

func readToolRunTail(logPath string, maxBytes int64) (string, error) {
	if strings.TrimSpace(logPath) == "" {
		return "", nil
	}
	logDir, err := filepath.Abs(toolRunDir())
	if err != nil {
		return "", err
	}
	cleanLogPath, err := filepath.Abs(filepath.Clean(logPath))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(logDir, cleanLogPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", errors.New("invalid tool log path")
	}
	file, err := os.Open(cleanLogPath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", err
	}
	offset := int64(0)
	if info.Size() > maxBytes {
		offset = info.Size() - maxBytes
	}
	if _, err := file.Seek(offset, 0); err != nil {
		return "", err
	}
	data, err := io.ReadAll(file)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func pythonExecutable() (string, error) {
	configured := strings.TrimSpace(os.Getenv("CUEMATE_PYTHON"))
	if configured != "" {
		if !filepath.IsAbs(configured) {
			return "", errors.New("CUEMATE_PYTHON must be an absolute path")
		}
		return filepath.Clean(configured), nil
	}
	pythonExe, err := exec.LookPath("python")
	if err != nil {
		return "", err
	}
	return pythonExe, nil
}

func normalizeDJSource(source string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "rekordbox", "traktor", "serato":
		return strings.ToLower(strings.TrimSpace(source)), nil
	default:
		return "", errors.New("source must be rekordbox, traktor, or serato")
	}
}

func safeCLIValue(value string, field string) (string, error) {
	trimmed := strings.TrimSpace(value)
	if strings.ContainsRune(trimmed, '\x00') {
		return "", fmt.Errorf("%s contains an invalid character", field)
	}
	if strings.HasPrefix(trimmed, "-") {
		return "", fmt.Errorf("%s must not start with '-'", field)
	}
	return trimmed, nil
}

func safeCLIPath(value string, field string) (string, error) {
	trimmed, err := safeCLIValue(value, field)
	if err != nil {
		return "", err
	}
	if trimmed == "" {
		return "", nil
	}
	return filepath.Clean(trimmed), nil
}

func safeCLIPaths(values []string) ([]string, error) {
	out := make([]string, 0, len(values))
	for _, value := range values {
		trimmed, err := safeCLIPath(value, "path")
		if err != nil {
			return nil, err
		}
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out, nil
}

func boundedPositive(value, fallback, max int) int {
	if value <= 0 {
		return fallback
	}
	if value > max {
		return max
	}
	return value
}

func validAnalysisMode(mode string) bool {
	switch strings.TrimSpace(mode) {
	case "", "fast_pass", "staged", "full":
		return true
	default:
		return false
	}
}

func (s *server) handlePlaylistAnalysisEnqueue(w http.ResponseWriter, r *http.Request, playlistID string) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req enqueueAnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if !validAnalysisMode(req.AnalysisMode) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "analysis_mode must be fast_pass, staged, or full"})
		return
	}
	if _, err := s.repo.ResolvePlaylist(r.Context(), playlistID, ""); err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
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
	active := metadata.GetActiveSignatures()
	queued, err := s.repo.QueuePlaylistAnalysis(
		r.Context(),
		playlistID,
		req.AnalysisMode,
		req.Force,
		active.GetAnalysisSignature(),
		active.GetConfigSignature(),
	)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"playlist_id": playlistID, "queued_count": queued})
}

func (s *server) handlePlaylistAnalysisStatus(w http.ResponseWriter, r *http.Request, playlistID string) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	metadata, metadataErr := s.runtime.RefreshMetadata(r.Context())
	var analysisSignature, configSignature, scoringContractID string
	if metadataErr == nil {
		analysisSignature, configSignature, scoringContractID = activeSignatureValues(metadata.GetActiveSignatures())
	}
	status, err := s.repo.GetPlaylistAnalysisStatus(r.Context(), playlistID, analysisSignature, configSignature, scoringContractID)
	if err != nil {
		httpStatus := http.StatusInternalServerError
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
			httpStatus = http.StatusNotFound
		}
		writeJSON(w, httpStatus, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, playlistAnalysisStatusPayload(status))
}

func (s *server) handlePlaylistAnalysisRefresh(w http.ResponseWriter, r *http.Request, playlistID string) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req enqueueAnalysisRequest
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
			return
		}
	}
	if !validAnalysisMode(req.AnalysisMode) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "analysis_mode must be fast_pass, staged, or full"})
		return
	}
	if _, err := s.repo.ResolvePlaylist(r.Context(), playlistID, ""); err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, recommendationsrepo.ErrPlaylistNotFound) {
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
	active := metadata.GetActiveSignatures()
	queued, err := s.repo.QueuePlaylistAnalysis(
		r.Context(),
		playlistID,
		req.AnalysisMode,
		req.Force,
		active.GetAnalysisSignature(),
		active.GetConfigSignature(),
	)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	status, err := s.repo.GetPlaylistAnalysisStatus(
		r.Context(),
		playlistID,
		active.GetAnalysisSignature(),
		active.GetConfigSignature(),
		active.GetScoringContractId(),
	)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"playlist_id":  playlistID,
		"queued_count": queued,
		"status":       playlistAnalysisStatusPayload(status),
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
	} else {
		parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(playedAt))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "played_at must be an RFC3339 timestamp"})
			return
		}
		playedAt = parsed.UTC().Format(time.RFC3339)
	}
	err = s.repo.RunInTx(r.Context(), func(tx *sql.Tx) error {
		if err := s.repo.UpdateRecommendationEventChoiceTx(
			r.Context(),
			tx,
			event.ID,
			req.ChosenTrackID,
			wasRecommended,
			string(skippedJSONBytes),
			playedAt,
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
	tracks, err := s.repo.GetPlaylistSnapshotTracks(
		r.Context(),
		playlist.ID,
		metadata.GetActiveSignatures().GetAnalysisSignature(),
		metadata.GetActiveSignatures().GetConfigSignature(),
		metadata.GetActiveSignatures().GetScoringContractId(),
	)
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

func (s *server) handleRemoteStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	_, paired := s.remoteSession(r)
	writeJSON(w, http.StatusOK, map[string]any{
		"enabled":       remoteBaseURL() != "",
		"mode":          "tailscale",
		"remote_url":    nullIfEmpty(remoteBaseURL()),
		"request_local": isLocalRequest(r),
		"paired":        paired,
	})
}

func (s *server) handleRemotePairingToken(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	if !isLocalRequest(r) {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "pairing tokens can only be created locally"})
		return
	}
	var req remotePairingTokenRequest
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
			return
		}
	}
	token, err := randomURLToken(32)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	now := time.Now().UTC()
	expiresAt := now.Add(5 * time.Minute)
	if err := s.repo.CreateRemotePairingToken(r.Context(), hashSecret(token), optionalString(req.DeviceLabel), now.Format(time.RFC3339), expiresAt.Format(time.RFC3339)); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	baseURL := remoteBaseURL()
	if baseURL == "" {
		baseURL = "http://" + r.Host
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"token":      token,
		"expires_at": expiresAt.Format(time.RFC3339),
		"pair_url":   strings.TrimRight(baseURL, "/") + "/?pair_token=" + url.QueryEscape(token),
	})
}

func (s *server) handleRemotePair(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req remotePairRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON request"})
		return
	}
	if strings.TrimSpace(req.Token) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "token is required"})
		return
	}
	now := time.Now().UTC()
	token, err := s.repo.ConsumeRemotePairingToken(r.Context(), hashSecret(req.Token), now.Format(time.RFC3339))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	if token == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "pairing token is invalid or expired"})
		return
	}
	sessionSecret, err := randomURLToken(32)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	expiresAt := now.Add(30 * 24 * time.Hour)
	deviceLabel := optionalString(firstNonEmpty(req.DeviceLabel, stringValue(token.DeviceLabel), "Mobile device"))
	if err := s.repo.CreateRemoteSession(r.Context(), hashSecret(sessionSecret), deviceLabel, now.Format(time.RFC3339), expiresAt.Format(time.RFC3339)); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     "cuemate_remote_session",
		Value:    sessionSecret,
		Path:     "/",
		Expires:  expiresAt,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   !isLocalRequest(r),
	})
	writeJSON(w, http.StatusOK, map[string]any{"status": "paired", "expires_at": expiresAt.Format(time.RFC3339)})
}

func (s *server) handleRemoteLogout(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	if cookie, err := r.Cookie("cuemate_remote_session"); err == nil {
		_ = s.repo.RevokeRemoteSession(r.Context(), hashSecret(cookie.Value), recommendationsrepo.NowUTC())
	}
	http.SetCookie(w, &http.Cookie{
		Name:     "cuemate_remote_session",
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   !isLocalRequest(r),
	})
	writeJSON(w, http.StatusOK, map[string]string{"status": "logged_out"})
}

func remoteBaseURL() string {
	return strings.TrimRight(strings.TrimSpace(os.Getenv("CUEMATE_REMOTE_URL")), "/")
}

func randomURLToken(size int) (string, error) {
	buf := make([]byte, size)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

func hashSecret(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(sum[:])
}

func optionalString(value string) *string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return nil
	}
	return &trimmed
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
		_, source := effectivePlaylistWeights(hydrated.Stats)
		if len(hydrated.Stats.AdaptedWeights) > 0 {
			req.PlaylistStats.AdaptedWeights = copyWeightMap(hydrated.Stats.AdaptedWeights)
		}
		if len(hydrated.Stats.FeedbackTunedWeights) > 0 {
			req.PlaylistStats.FeedbackTunedWeights = copyWeightMap(hydrated.Stats.FeedbackTunedWeights)
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
				"track_id":              item.GetCandidate().GetTrackId(),
				"title":                 item.GetCandidate().GetTitle(),
				"artist":                item.GetCandidate().GetArtist(),
				"score":                 item.GetFinalScore(),
				"ranking_strength":      item.GetRankingStrength(),
				"move":                  item.GetMove(),
				"move_confidence":       item.GetMoveConfidence(),
				"move_note":             item.GetMoveNote(),
				"risk":                  item.GetRisk(),
				"risk_score":            item.GetRiskScore(),
				"raw_score":             item.GetRawScore(),
				"penalty_multiplier":    item.GetPenaltyMultiplier(),
				"primary_lane":          nullIfEmpty(item.GetPrimaryLane()),
				"secondary_lane":        item.GetSecondaryLane(),
				"component_scores":      float64MapToAnyMap(item.GetComponentScores()),
				"component_confidences": float64MapToAnyMap(item.GetComponentConfidences()),
				"weights_used":          float64MapToAnyMap(item.GetWeightsUsed()),
				"transition_features":   transitionFeaturesPayload(item.GetTransitionFeatures()),
				"candidate_features":    trackContextFeaturePayload(item.GetCandidate()),
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
	playedEvents := make([]recommendationsrepo.RecommendationEventRecord, 0, len(events))
	eventIDs := make([]string, 0, len(events))
	for _, event := range events {
		if event.PlayedAt == nil {
			continue
		}
		playedEvents = append(playedEvents, event)
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
	rpcReq, err := buildFeedbackSummaryRPCRequest(playlist, playedEvents, itemsByEvent, stats, since, until)
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
		parsedSince = value.UTC()
		hasSince = true
		since = parsedSince.Format(time.RFC3339)
	}
	if until != "" {
		value, err := time.Parse(time.RFC3339, until)
		if err != nil {
			return "", "", fmt.Errorf("until must be a valid RFC3339 timestamp")
		}
		parsedUntil = value.UTC()
		hasUntil = true
		until = parsedUntil.Format(time.RFC3339)
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
			Timestamp:            firstNonEmpty(stringValue(event.PlayedAt), event.Timestamp),
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
			"source":    feedbackSummaryWeightSource(weights.GetSource()),
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

func feedbackSummaryWeightSource(source scoringv1.WeightSource) string {
	switch source {
	case scoringv1.WeightSource_WEIGHT_SOURCE_FEEDBACK_TUNED:
		return "feedback_tuned_weights"
	case scoringv1.WeightSource_WEIGHT_SOURCE_ADAPTED:
		return "adapted_weights"
	case scoringv1.WeightSource_WEIGHT_SOURCE_STATIC:
		return "static"
	default:
		return ""
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

func int32OrNil(value *int32) any {
	if value == nil {
		return nil
	}
	return *value
}

func trackContextFeaturePayload(track *scoringv1.TrackContext) map[string]any {
	if track == nil {
		return map[string]any{}
	}
	return map[string]any{
		"bpm":            track.GetBpm(),
		"key":            nullIfEmpty(track.GetMusicalKey()),
		"key_confidence": valueOrNil(track.KeyConfidence),
		"key_source":     nullIfEmpty(track.GetKeySource()),
		"key_agreement":  int32OrNil(track.KeyAgreement),
		"energy_rel":     valueOrNil(track.EnergyRel),
		"bass_rel":       valueOrNil(track.BassRel),
		"drums_rel":      valueOrNil(track.DrumsRel),
		"vocals_rel":     valueOrNil(track.VocalsRel),
		"groove_rel":     valueOrNil(track.GrooveRel),
		"intensity_band": nullIfEmpty(track.GetIntensityBand()),
		"role_hints":     append([]string{}, track.GetRoleHints()...),
	}
}

func transitionFeaturesPayload(features *scoringv1.TransitionFeatures) map[string]any {
	if features == nil {
		return map[string]any{}
	}
	return map[string]any{
		"effective_bpm_distance":   features.GetEffectiveBpmDistance(),
		"raw_bpm_distance":         features.GetRawBpmDistance(),
		"bpm_relationship":         features.GetBpmRelationship(),
		"key_distance":             features.GetKeyDistance(),
		"key_compat_label":         features.GetKeyCompatLabel(),
		"key_confidence_current":   features.GetKeyConfidenceCurrent(),
		"key_confidence_candidate": features.GetKeyConfidenceCandidate(),
		"delta_energy_rel":         features.GetDeltaEnergyRel(),
		"delta_bass_rel":           features.GetDeltaBassRel(),
		"current_vocals_rel":       transitionVocalValue(features, true),
		"candidate_vocals_rel":     transitionVocalValue(features, false),
		"current_outro_low_end":    features.GetCurrentOutroLowEnd(),
		"candidate_intro_low_end":  features.GetCandidateIntroLowEnd(),
	}
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

func activeSignatureValues(signatures *scoringv1.SignatureMetadata) (string, string, string) {
	if signatures == nil {
		return "", "", ""
	}
	return signatures.GetAnalysisSignature(), signatures.GetConfigSignature(), signatures.GetScoringContractId()
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

func playlistSummaryPayload(playlist recommendationsrepo.PlaylistSummary) map[string]any {
	return map[string]any{
		"playlist_id":            playlist.ID,
		"name":                   playlist.Name,
		"track_count":            playlist.TrackCount,
		"created_at":             playlist.CreatedAt,
		"updated_at":             playlist.UpdatedAt,
		"track_count_analyzed":   playlist.TrackCountAnalyzed,
		"eligible_track_count":   playlist.EligibleTrackCount,
		"is_stale":               playlist.IsStale,
		"stale_reason":           nullIfEmpty(playlist.StaleReason),
		"feedback_event_count":   playlist.FeedbackEventCount,
		"feedback_last_tuned_at": stringOrNil(playlist.FeedbackLastTunedAt),
	}
}

func playlistStatsPayload(stats *recommendationsrepo.PlaylistStats) any {
	if stats == nil {
		return nil
	}
	return map[string]any{
		"track_count_total":       stats.TrackCountTotal,
		"track_count_analyzed":    stats.TrackCountAnalyzed,
		"eligible_track_count":    stats.EligibleTrackCount,
		"energy_spread":           valueOrNil(stats.EnergySpread),
		"relative_signature":      stats.RelativeSignature,
		"is_stale":                stats.IsStale,
		"stale_reason":            nullIfEmpty(stats.StaleReason),
		"adapted_weights":         float64MapToAnyMap(stats.AdaptedWeights),
		"feedback_tuned_weights":  nilIfEmptyWeightMap(stats.FeedbackTunedWeights),
		"feedback_tuning_notes":   append([]string{}, stats.FeedbackTuningNotes...),
		"feedback_event_count":    stats.FeedbackEventCount,
		"feedback_last_tuned_at":  stringOrNil(stats.FeedbackLastTunedAt),
		"feedback_tuning_metrics": stats.FeedbackTuningMetrics,
	}
}

func playlistTrackPayload(track recommendationsrepo.PlaylistTrackSnapshot) map[string]any {
	return map[string]any{
		"track_id":       track.TrackID,
		"title":          track.Title,
		"artist":         track.Artist,
		"position":       track.Position,
		"bpm":            valueOrNil(track.BPM),
		"key":            stringOrNil(track.Key),
		"intensity_band": stringOrNil(track.IntensityBand),
		"role_hints":     append([]string{}, track.RoleHints...),
		"analysis_state": track.AnalysisState,
	}
}

func playlistAnalysisStatusPayload(status recommendationsrepo.PlaylistAnalysisStatus) map[string]any {
	return map[string]any{
		"playlist_id":      status.PlaylistID,
		"playlist_name":    status.PlaylistName,
		"total_tracks":     status.TotalTracks,
		"ready_tracks":     status.ReadyTracks,
		"outdated_tracks":  status.OutdatedTracks,
		"percent_complete": status.PercentComplete,
		"is_stale":         status.IsStale,
		"stale_reason":     nullIfEmpty(status.StaleReason),
		"latest_error":     stringOrNil(status.LatestError),
		"next_action":      status.NextAction,
		"jobs": map[string]any{
			"pending":   status.Counts.Pending,
			"running":   status.Counts.Running,
			"completed": status.Counts.Completed,
			"failed":    status.Counts.Failed,
		},
	}
}

func trackFeatureDetailPayload(detail recommendationsrepo.TrackFeatureDetail) map[string]any {
	return map[string]any{
		"track_id": detail.TrackID,
		"title":    detail.Title,
		"artist":   detail.Artist,
		"basic": map[string]any{
			"bpm":            valueOrNil(detail.BPM),
			"key":            stringOrNil(detail.Key),
			"key_confidence": valueOrNil(detail.KeyConfidence),
			"key_source":     stringOrNil(detail.KeySource),
			"key_agreement":  int32OrNil(detail.KeyAgreement),
		},
		"absolute": map[string]any{
			"energy_abs":           valueOrNil(detail.EnergyAbs),
			"energy_heuristic_abs": valueOrNil(detail.EnergyHeuristicAbs),
			"energy_sustained":     valueOrNil(detail.EnergySustained),
			"energy_peak":          valueOrNil(detail.EnergyPeak),
			"loudness_norm":        valueOrNil(detail.LoudnessNorm),
			"bass_abs":             valueOrNil(detail.BassAbs),
			"drums_abs":            valueOrNil(detail.DrumsAbs),
			"harmonic_abs":         valueOrNil(detail.HarmonicAbs),
			"groove_abs":           valueOrNil(detail.GrooveAbs),
			"vocals_abs":           valueOrNil(detail.VocalsAbs),
			"vocals_confidence":    valueOrNil(detail.VocalsConfidence),
		},
		"semantic": map[string]any{
			"danceability_abs":       valueOrNil(detail.DanceabilityAbs),
			"arousal_abs":            valueOrNil(detail.ArousalAbs),
			"valence_abs":            valueOrNil(detail.ValenceAbs),
			"mood_aggressive_abs":    valueOrNil(detail.MoodAggressiveAbs),
			"mood_party_abs":         valueOrNil(detail.MoodPartyAbs),
			"mood_relaxed_abs":       valueOrNil(detail.MoodRelaxedAbs),
			"energy_essentia_fused":  valueOrNil(detail.EnergyEssentiaFused),
			"energy_essentia_bucket": stringOrNil(detail.EnergyEssentiaBucket),
			"source":                 stringOrNil(detail.EssentiaSemanticSource),
			"inferred_at":            stringOrNil(detail.EssentiaSemanticInferredAt),
		},
		"relative": map[string]any{
			"energy_rel":           valueOrNil(detail.EnergyRel),
			"bass_rel":             valueOrNil(detail.BassRel),
			"drums_rel":            valueOrNil(detail.DrumsRel),
			"vocals_rel":           valueOrNil(detail.VocalsRel),
			"groove_rel":           valueOrNil(detail.GrooveRel),
			"energy_spread":        valueOrNil(detail.EnergySpread),
			"bass_spread":          valueOrNil(detail.BassSpread),
			"drums_spread":         valueOrNil(detail.DrumsSpread),
			"vocals_spread":        valueOrNil(detail.VocalsSpread),
			"groove_spread":        valueOrNil(detail.GrooveSpread),
			"intensity_band":       stringOrNil(detail.IntensityBand),
			"intensity_membership": float64MapToAnyMap(detail.IntensityMembership),
			"role_hints":           append([]string{}, detail.RoleHints...),
		},
		"analysis": map[string]any{
			"analysis_mode":                   stringOrNil(detail.AnalysisMode),
			"analyzed_at":                     stringOrNil(detail.AnalyzedAt),
			"analysis_signature":              stringOrNil(detail.AnalysisSignature),
			"config_signature":                stringOrNil(detail.ConfigSignature),
			"scoring_contract_id_at_analysis": stringOrNil(detail.ScoringContractAtAnalysis),
		},
	}
}

func analysisJobPayload(job recommendationsrepo.AnalysisJobRecord) map[string]any {
	return map[string]any{
		"id":               job.ID,
		"playlist_id":      stringOrNil(job.PlaylistID),
		"track_id":         stringOrNil(job.TrackID),
		"track_path":       job.TrackPath,
		"status":           job.Status,
		"priority":         job.Priority,
		"analysis_mode":    job.AnalysisMode,
		"job_kind":         job.JobKind,
		"error_message":    stringOrNil(job.ErrorMessage),
		"duration_seconds": valueOrNil(job.DurationSeconds),
		"created_at":       job.CreatedAt,
		"started_at":       stringOrNil(job.StartedAt),
		"completed_at":     stringOrNil(job.CompletedAt),
	}
}

func recommendationEventPayload(event recommendationsrepo.RecommendationEventRecord) map[string]any {
	return map[string]any{
		"event_id":                  event.ID,
		"playlist_id":               event.PlaylistID,
		"current_track_id":          event.CurrentTrackID,
		"target":                    event.Target,
		"candidate_count":           event.CandidateCount,
		"recommendation_confidence": valueOrNil(event.RecommendationConfidence),
		"recommendations_status":    event.RecommendationsStatus,
		"track_chosen":              stringOrNil(event.TrackChosen),
		"chosen_was_recommended":    boolOrNil(event.ChosenWasRecommended),
		"scoring_contract_id":       event.ScoringContractID,
		"timestamp":                 event.Timestamp,
		"played_at":                 stringOrNil(event.PlayedAt),
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

func queryInt(r *http.Request, key string, fallback int) int {
	raw := strings.TrimSpace(r.URL.Query().Get(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func boolOrNil(value *bool) any {
	if value == nil {
		return nil
	}
	return *value
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
