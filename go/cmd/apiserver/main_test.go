package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"github.com/Ross1116/cuemate-engine/go/internal/recommendationsrepo"
	"github.com/Ross1116/cuemate-engine/go/internal/scoringruntime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	_ "modernc.org/sqlite"
)

type fakeRuntimeClient struct {
	metadataResp *scoringv1.GetScoringMetadataResponse
	metadataErr  error
	recResp      *scoringv1.GetRecommendationsResponse
	recErr       error
}

func (f *fakeRuntimeClient) GetScoringMetadata(context.Context, *scoringv1.GetScoringMetadataRequest, ...grpc.CallOption) (*scoringv1.GetScoringMetadataResponse, error) {
	if f.metadataErr != nil {
		return nil, f.metadataErr
	}
	return f.metadataResp, nil
}

func (f *fakeRuntimeClient) GetRecommendations(context.Context, *scoringv1.GetRecommendationsRequest, ...grpc.CallOption) (*scoringv1.GetRecommendationsResponse, error) {
	if f.recErr != nil {
		return nil, f.recErr
	}
	return f.recResp, nil
}

func (f *fakeRuntimeClient) Close() error { return nil }

func TestRecommendationsHappyPath(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		recResp: &scoringv1.GetRecommendationsResponse{
			RecommendationsStatus:    "available",
			RecommendationConfidence: 0.81,
			CurrentTrack: &scoringv1.TrackContext{
				TrackId:       "trk_current",
				Title:         "Current",
				Artist:        "Tester",
				Bpm:           128,
				MusicalKey:    "8A",
				IntensityBand: "Drive",
				RoleHints:     []string{"groove"},
			},
			SetContext: &scoringv1.SetContext{
				Trend:         &scoringv1.SetTrend{Label: "building", Direction: "up"},
				HistoryLength: 1,
				HasGaps:       false,
			},
			LaneOrder: []string{"maintain", "build", "reset", "jump", "contrast"},
			Lanes: []*scoringv1.RecommendationLane{
				{
					LaneGroup:    &scoringv1.LaneGroup{LaneId: "maintain", DisplayName: "Maintain"},
					Availability: "available",
					Items: []*scoringv1.ScoredCandidate{
						{
							Candidate:       &scoringv1.TrackContext{TrackId: "trk_candidate", Title: "Candidate", Artist: "Tester"},
							FinalScore:      0.91,
							RankingStrength: 1.0,
							Move:            "maintain",
							Risk:            "low",
							RiskScore:       0.1,
							TempoKey:        &scoringv1.TempoKeySummary{TempoText: "0 BPM (match)", KeyText: "8A→8A (same)", KeyState: "normal"},
							AdvisoryHints:   []string{"Keeps pressure steady"},
							Reasons:         []string{"Keeps pressure steady"},
							Watchouts:       []string{},
							Explanation:     &scoringv1.ExplanationBlock{Summary: []string{"Keeps pressure steady"}},
						},
					},
				},
			},
			Meta: &scoringv1.RecommendationMeta{
				Target:                 "maintain",
				TotalCandidates:        2,
				FilteredCandidates:     2,
				ScoredCandidates:       2,
				CurrentTrackId:         "trk_current",
				RequestedLaneAvailable: true,
			},
			Capabilities: &scoringv1.CapabilityMetadata{Flags: map[string]bool{"explanations_available": true}},
			AppliedWeightAdaptation: &scoringv1.WeightAdaptation{
				AdaptationId: "static_weights",
				ComponentWeights: map[string]float64{
					"harmonic":        0.12,
					"bass_transition": 0.15,
					"tempo":           0.10,
					"target_energy":   0.22,
				},
			},
			ActiveSignatures: &scoringv1.SignatureMetadata{ScoringContractId: "m3-v1"},
		},
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/recommendations", bytes.NewBufferString(`{"playlist_name":"Test Playlist","current_track_id":"trk_current","target":"maintain","history_track_ids":["trk_history"]}`))
	srv.handleRecommendations(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	if payload["recommendations_status"] != "available" {
		t.Fatalf("recommendations_status = %#v", payload["recommendations_status"])
	}
	lanes := payload["lanes"].(map[string]any)
	if _, ok := lanes["maintain"]; !ok {
		t.Fatalf("lanes missing maintain: %#v", lanes)
	}
}

func TestRecommendationsRequiresReanalysisWhenRelativeStatsAreStale(t *testing.T) {
	srv := newTestServer(t, true, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/recommendations", bytes.NewBufferString(`{"playlist_name":"Test Playlist","current_track_id":"trk_current"}`))
	srv.handleRecommendations(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var payload map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &payload)
	if payload["recommendations_status"] != "requires_reanalysis" {
		t.Fatalf("recommendations_status = %#v", payload["recommendations_status"])
	}
}

func TestRecommendationsTemporarilyUnavailableWhenScorerFails(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		recErr:       status.Error(codes.Unavailable, "scorer down"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/recommendations", bytes.NewBufferString(`{"playlist_name":"Test Playlist","current_track_id":"trk_current"}`))
	srv.handleRecommendations(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var payload map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &payload)
	if payload["recommendations_status"] != "temporarily_unavailable" {
		t.Fatalf("recommendations_status = %#v", payload["recommendations_status"])
	}
}

func newTestServer(t *testing.T, stale bool, relativeSignature string, client *fakeRuntimeClient) *server {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	defer db.Close()

	for _, stmt := range []string{
		`CREATE TABLE playlists (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE);`,
		`CREATE TABLE tracks (id TEXT PRIMARY KEY, file_path TEXT NOT NULL UNIQUE, title TEXT, artist TEXT);`,
		`CREATE TABLE playlist_tracks (playlist_id TEXT NOT NULL, track_id TEXT NOT NULL, position INTEGER NOT NULL);`,
		`CREATE TABLE track_features_abs (
			track_id TEXT PRIMARY KEY,
			bpm REAL NOT NULL,
			key TEXT NOT NULL,
			key_confidence REAL NOT NULL,
			key_source TEXT NOT NULL,
			key_agreement INTEGER,
			analysis_signature TEXT NOT NULL,
			config_signature TEXT NOT NULL,
			scoring_contract_id_at_analysis TEXT
		);`,
		`CREATE TABLE track_features_rel (
			playlist_id TEXT NOT NULL,
			track_id TEXT NOT NULL,
			position INTEGER NOT NULL,
			energy_rel REAL NOT NULL,
			bass_rel REAL NOT NULL,
			drums_rel REAL NOT NULL,
			vocals_rel REAL,
			groove_rel REAL NOT NULL,
			intensity_band TEXT NOT NULL,
			role_hints TEXT NOT NULL,
			relative_signature TEXT NOT NULL
		);`,
		`CREATE TABLE playlist_stats (
			playlist_id TEXT PRIMARY KEY,
			track_count_total INTEGER NOT NULL,
			track_count_analyzed INTEGER NOT NULL,
			eligible_track_count INTEGER NOT NULL,
			energy_spread REAL,
			adapted_weights TEXT,
			relative_signature TEXT NOT NULL,
			is_stale INTEGER NOT NULL DEFAULT 0,
			stale_reason TEXT
		);`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("exec schema: %v", err)
		}
	}

	if _, err := db.Exec(`INSERT INTO playlists (id, name) VALUES ('pl_1', 'Test Playlist')`); err != nil {
		t.Fatalf("insert playlist: %v", err)
	}
	for _, insert := range []string{
		`INSERT INTO tracks (id, file_path, title, artist) VALUES ('trk_current', '/music/current.flac', 'Current', 'Tester')`,
		`INSERT INTO tracks (id, file_path, title, artist) VALUES ('trk_candidate', '/music/candidate.flac', 'Candidate', 'Tester')`,
		`INSERT INTO tracks (id, file_path, title, artist) VALUES ('trk_history', '/music/history.flac', 'History', 'Tester')`,
		`INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES ('pl_1', 'trk_current', 1)`,
		`INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES ('pl_1', 'trk_candidate', 2)`,
		`INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES ('pl_1', 'trk_history', 3)`,
		`INSERT INTO track_features_abs (track_id, bpm, key, key_confidence, key_source, key_agreement, analysis_signature, config_signature, scoring_contract_id_at_analysis) VALUES ('trk_current', 128.0, '8A', 0.9, 'musicalkeycnn', 1, 'm1-ebd25381ebad', 'default', 'm3-v1')`,
		`INSERT INTO track_features_abs (track_id, bpm, key, key_confidence, key_source, key_agreement, analysis_signature, config_signature, scoring_contract_id_at_analysis) VALUES ('trk_candidate', 128.0, '8A', 0.9, 'musicalkeycnn', 1, 'm1-ebd25381ebad', 'default', 'm3-v1')`,
		`INSERT INTO track_features_abs (track_id, bpm, key, key_confidence, key_source, key_agreement, analysis_signature, config_signature, scoring_contract_id_at_analysis) VALUES ('trk_history', 127.0, '8A', 0.9, 'musicalkeycnn', 1, 'm1-ebd25381ebad', 'default', 'm3-v1')`,
		`INSERT INTO track_features_rel (playlist_id, track_id, position, energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel, intensity_band, role_hints, relative_signature) VALUES ('pl_1', 'trk_current', 1, 0.5, 0.5, 0.5, 0.1, 0.5, 'Drive', '["groove"]', '` + relativeSignature + `')`,
		`INSERT INTO track_features_rel (playlist_id, track_id, position, energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel, intensity_band, role_hints, relative_signature) VALUES ('pl_1', 'trk_candidate', 2, 0.6, 0.6, 0.5, 0.2, 0.5, 'Drive', '["groove"]', '` + relativeSignature + `')`,
		`INSERT INTO track_features_rel (playlist_id, track_id, position, energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel, intensity_band, role_hints, relative_signature) VALUES ('pl_1', 'trk_history', 3, 0.45, 0.4, 0.5, 0.1, 0.5, 'Drive', '["groove"]', '` + relativeSignature + `')`,
	} {
		if _, err := db.Exec(insert); err != nil {
			t.Fatalf("seed insert: %v", err)
		}
	}
	staleInt := 0
	staleReason := ""
	if stale {
		staleInt = 1
		staleReason = "absolute_track_changed"
	}
	if _, err := db.Exec(`INSERT INTO playlist_stats (playlist_id, track_count_total, track_count_analyzed, eligible_track_count, energy_spread, adapted_weights, relative_signature, is_stale, stale_reason) VALUES ('pl_1', 3, 3, 3, 0.2, '{"harmonic":0.12}', ?, ?, ?)`, relativeSignature, staleInt, staleReason); err != nil {
		t.Fatalf("insert playlist_stats: %v", err)
	}

	repo, err := recommendationsrepo.Open(dbPath)
	if err != nil {
		t.Fatalf("repo open: %v", err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	return &server{
		repo:    repo,
		runtime: scoringruntime.New(client, 2),
	}
}

func fakeMetadata(relativeSignature string) *scoringv1.GetScoringMetadataResponse {
	return &scoringv1.GetScoringMetadataResponse{
		ActiveSignatures: &scoringv1.SignatureMetadata{
			AnalysisSignature: "m1-ebd25381ebad",
			ConfigSignature:   "default",
			ScoringContractId: "m3-v1",
		},
		CapabilityFlags:           map[string]bool{"explanations_available": true},
		ExpectedRelativeSignature: relativeSignature,
	}
}
