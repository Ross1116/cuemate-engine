package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
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

func TestRecommendationsCreatesRecommendationEvent(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		recResp: &scoringv1.GetRecommendationsResponse{
			RecommendationsStatus: "available",
			CurrentTrack:          &scoringv1.TrackContext{TrackId: "trk_current"},
			Meta: &scoringv1.RecommendationMeta{
				Target:           "maintain",
				TotalCandidates:  2,
				ScoredCandidates: 1,
				CurrentTrackId:   "trk_current",
			},
			LaneOrder: []string{"maintain", "build", "reset", "jump", "contrast"},
			Lanes: []*scoringv1.RecommendationLane{
				{
					LaneGroup:    &scoringv1.LaneGroup{LaneId: "maintain"},
					Availability: "available",
					Items: []*scoringv1.ScoredCandidate{
						{Candidate: &scoringv1.TrackContext{TrackId: "trk_candidate"}, FinalScore: 0.9},
					},
				},
			},
			ActiveSignatures: &scoringv1.SignatureMetadata{ScoringContractId: "m3-v1"},
		},
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/recommendations", bytes.NewBufferString(`{"playlist_name":"Test Playlist","current_track_id":"trk_current"}`))
	srv.handleRecommendations(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &payload)
	meta := payload["meta"].(map[string]any)
	if meta["recommendation_event_id"] == nil {
		t.Fatalf("expected recommendation_event_id in response meta: %#v", meta)
	}
	eventID := meta["recommendation_event_id"].(string)
	items, err := srv.repo.GetRecommendationEventItems(context.Background(), eventID)
	if err != nil {
		t.Fatalf("GetRecommendationEventItems() error = %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("event items = %#v", items)
	}
	if items[0].CandidateTrackID != "trk_candidate" || items[0].LaneID != "maintain" || items[0].LaneRank != 1 {
		t.Fatalf("unexpected event item = %#v", items[0])
	}
}

func TestPlayedEventUpdatesRecommendationEvent(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	ctx := context.Background()
	eventID := "evt_1"
	lanesJSON := `{"lane_order":["maintain","build"],"lanes":{"maintain":[{"track_id":"trk_candidate","score":0.9}],"build":[{"track_id":"trk_history","score":0.8}]}}`
	err := srv.repo.InsertRecommendationEvent(ctx, recommendationsrepo.RecommendationEventRecord{
		ID:                    eventID,
		PlaylistID:            "pl_1",
		CurrentTrackID:        "trk_current",
		Target:                "maintain",
		CandidateCount:        2,
		RecommendationsStatus: "available",
		LanesReturnedJSON:     lanesJSON,
		ScoringContractID:     "m3-v1",
		Timestamp:             "2026-04-09T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("InsertRecommendationEvent() error = %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/events/played", bytes.NewBufferString(`{"recommendation_event_id":"evt_1","chosen_track_id":"trk_candidate"}`))
	srv.handlePlayedEvent(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &payload)
	if payload["chosen_was_recommended"] != true {
		t.Fatalf("chosen_was_recommended = %#v", payload["chosen_was_recommended"])
	}
	jobs, err := srv.repo.ListFeedbackTuningJobsByPlaylist(context.Background(), "pl_1")
	if err != nil {
		t.Fatalf("ListFeedbackTuningJobsByPlaylist() error = %v", err)
	}
	if len(jobs) != 1 || jobs[0].Status != "pending" || jobs[0].TriggerEventID == nil || *jobs[0].TriggerEventID != eventID {
		t.Fatalf("feedback tuning jobs = %#v", jobs)
	}
}

func TestFeedbackSummaryReturnsPlaylistMetrics(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	ctx := context.Background()
	eventID := "evt_feedback_1"
	err := srv.repo.InsertRecommendationEvent(ctx, recommendationsrepo.RecommendationEventRecord{
		ID:                    eventID,
		PlaylistID:            "pl_1",
		CurrentTrackID:        "trk_current",
		Target:                "reset",
		CandidateCount:        2,
		RecommendationsStatus: "available",
		LanesReturnedJSON:     `{"lane_order":["reset","build"],"lanes":{"reset":[{"track_id":"trk_candidate","score":0.9}]}}`,
		TrackChosen:           stringPtr("trk_candidate"),
		ChosenWasRecommended:  boolPtr(true),
		ScoringContractID:     "m3-v1",
		Timestamp:             "2026-04-10T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("InsertRecommendationEvent() error = %v", err)
	}
	err = srv.repo.InsertRecommendationEventItems(ctx, []recommendationsrepo.RecommendationEventItemRecord{
		{
			EventID:                eventID,
			LaneID:                 "reset",
			LaneRank:               1,
			CandidateTrackID:       "trk_candidate",
			FinalScore:             0.9,
			RawScore:               0.9,
			PenaltyMultiplier:      1.0,
			Move:                   "reset",
			MoveConfidence:         0.9,
			Risk:                   "low",
			RiskScore:              0.1,
			PrimaryLane:            stringPtr("reset"),
			SecondaryLane:          false,
			ComponentScoresJSON:    `{"harmonic":0.85}`,
			ConfidencesJSON:        `{"harmonic":1.0}`,
			WeightsUsedJSON:        `{"harmonic":0.12}`,
			TransitionFeaturesJSON: `{"effective_bpm_distance":1.0}`,
		},
	})
	if err != nil {
		t.Fatalf("InsertRecommendationEventItems() error = %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleFeedbackSummary(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	metrics := payload["metrics"].(map[string]any)
	if metrics["total_events"] != float64(1) || metrics["contributory_events"] != float64(1) {
		t.Fatalf("metrics = %#v", metrics)
	}
	weights := payload["weights"].(map[string]any)
	if weights["source"] != "adapted_weights" {
		t.Fatalf("weights source = %#v", weights["source"])
	}
}

func TestCorrectionsQueueReanalysisAndOutbox(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/corrections", bytes.NewBufferString(`{"track_id":"trk_current","field":"bpm","new_value":129.0}`))
	srv.handleCorrections(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &payload)
	if payload["changed"] != true {
		t.Fatalf("changed = %#v", payload["changed"])
	}
	if payload["queued_job_id"] == nil {
		t.Fatalf("expected queued_job_id")
	}

	outboxRec := httptest.NewRecorder()
	outboxReq := httptest.NewRequest(http.MethodPost, "/sync/outbox/pull", bytes.NewBufferString(`{}`))
	srv.handleOutboxPull(outboxRec, outboxReq)
	if outboxRec.Code != http.StatusOK {
		t.Fatalf("outbox pull status = %d body=%s", outboxRec.Code, outboxRec.Body.String())
	}
	var outboxPayload map[string]any
	_ = json.Unmarshal(outboxRec.Body.Bytes(), &outboxPayload)
	items := outboxPayload["items"].([]any)
	if len(items) != 1 {
		t.Fatalf("outbox items = %#v", items)
	}
	itemPayload := items[0].(map[string]any)["payload"].(map[string]any)
	if itemPayload["requires_snapshot_refresh"] != true {
		t.Fatalf("requires_snapshot_refresh = %#v", itemPayload["requires_snapshot_refresh"])
	}
	if _, ok := itemPayload["affected_playlist_ids"]; !ok {
		t.Fatalf("affected_playlist_ids missing from payload: %#v", itemPayload)
	}
}

func TestSnapshotExportsPlaylistScopedPayload(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/sync/playlists/snapshot", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleSnapshot(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &payload)
	if payload["playlist_id"] != "pl_1" {
		t.Fatalf("playlist_id = %#v", payload["playlist_id"])
	}
	freshness := payload["freshness"].(map[string]any)
	if freshness["delivery_state"] != "never_synced" {
		t.Fatalf("delivery_state = %#v", freshness["delivery_state"])
	}
	if freshness["last_synced_at"] != nil {
		t.Fatalf("last_synced_at = %#v", freshness["last_synced_at"])
	}
	precomputed := payload["precomputed"].(map[string]any)
	if precomputed["playlist_analysis"] != nil {
		t.Fatalf("playlist_analysis = %#v", precomputed["playlist_analysis"])
	}
}

func TestSnapshotAckUpdatesDeliveryState(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	firstRec := httptest.NewRecorder()
	firstReq := httptest.NewRequest(http.MethodPost, "/sync/playlists/snapshot", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleSnapshot(firstRec, firstReq)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("snapshot status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}
	var firstPayload map[string]any
	_ = json.Unmarshal(firstRec.Body.Bytes(), &firstPayload)
	snapshotID := firstPayload["snapshot_id"].(string)

	ackRec := httptest.NewRecorder()
	ackReq := httptest.NewRequest(http.MethodPost, "/sync/playlists/snapshot/ack", bytes.NewBufferString(`{"snapshot_id":"`+snapshotID+`","acked_at":"2026-04-09T01:00:00Z"}`))
	srv.handleSnapshotAck(ackRec, ackReq)
	if ackRec.Code != http.StatusOK {
		t.Fatalf("snapshot ack status = %d body=%s", ackRec.Code, ackRec.Body.String())
	}

	secondRec := httptest.NewRecorder()
	secondReq := httptest.NewRequest(http.MethodPost, "/sync/playlists/snapshot", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleSnapshot(secondRec, secondReq)
	if secondRec.Code != http.StatusOK {
		t.Fatalf("snapshot status = %d body=%s", secondRec.Code, secondRec.Body.String())
	}
	var secondPayload map[string]any
	_ = json.Unmarshal(secondRec.Body.Bytes(), &secondPayload)
	freshness := secondPayload["freshness"].(map[string]any)
	if freshness["delivery_state"] != "pending_ack" {
		t.Fatalf("delivery_state = %#v", freshness["delivery_state"])
	}
	if freshness["last_synced_at"] != "2026-04-09T01:00:00Z" {
		t.Fatalf("last_synced_at = %#v", freshness["last_synced_at"])
	}

	thirdRec := httptest.NewRecorder()
	thirdReq := httptest.NewRequest(http.MethodPost, "/sync/playlists/snapshot/ack", bytes.NewBufferString(`{"snapshot_id":"`+secondPayload["snapshot_id"].(string)+`","acked_at":"2026-04-09T02:00:00Z"}`))
	srv.handleSnapshotAck(thirdRec, thirdReq)
	if thirdRec.Code != http.StatusOK {
		t.Fatalf("snapshot ack status = %d body=%s", thirdRec.Code, thirdRec.Body.String())
	}

	state, err := srv.repo.GetPlaylistSyncState(context.Background(), "pl_1")
	if err != nil {
		t.Fatalf("GetPlaylistSyncState() error = %v", err)
	}
	if state == nil || state.LastSnapshotAckedAt == nil || *state.LastSnapshotAckedAt != "2026-04-09T02:00:00Z" {
		t.Fatalf("sync state = %#v", state)
	}
}

func TestSnapshotAckUnknownIDReturns404(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/sync/playlists/snapshot/ack", bytes.NewBufferString(`{"snapshot_id":"missing"}`))
	srv.handleSnapshotAck(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestOutboxPullAndAckLifecycle(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	ctx := context.Background()
	if _, err := srv.repo.InsertSyncOutbox(ctx, "recommendation_event", "evt_1", "played", `{"foo":"bar1"}`, "2026-04-09T00:00:00Z"); err != nil {
		t.Fatalf("InsertSyncOutbox() error = %v", err)
	}
	if _, err := srv.repo.InsertSyncOutbox(ctx, "manual_correction", "corr_1", "upsert", `{"foo":"bar2"}`, "2026-04-09T00:01:00Z"); err != nil {
		t.Fatalf("InsertSyncOutbox() error = %v", err)
	}

	pullRec := httptest.NewRecorder()
	pullReq := httptest.NewRequest(http.MethodPost, "/sync/outbox/pull", bytes.NewBufferString(`{"limit":1}`))
	srv.handleOutboxPull(pullRec, pullReq)
	if pullRec.Code != http.StatusOK {
		t.Fatalf("pull status = %d body=%s", pullRec.Code, pullRec.Body.String())
	}
	var pullPayload map[string]any
	_ = json.Unmarshal(pullRec.Body.Bytes(), &pullPayload)
	if pullPayload["has_more"] != true {
		t.Fatalf("has_more = %#v", pullPayload["has_more"])
	}
	items := pullPayload["items"].([]any)
	if len(items) != 1 {
		t.Fatalf("items = %#v", items)
	}
	nextAckID := int64(pullPayload["next_ack_id"].(float64))

	ackRec := httptest.NewRecorder()
	ackReq := httptest.NewRequest(http.MethodPost, "/sync/outbox/ack", bytes.NewBufferString(`{"ack_through_id":`+jsonNumber(nextAckID)+`,"acked_at":"2026-04-09T03:00:00Z"}`))
	srv.handleOutboxAck(ackRec, ackReq)
	if ackRec.Code != http.StatusOK {
		t.Fatalf("ack status = %d body=%s", ackRec.Code, ackRec.Body.String())
	}

	secondPullRec := httptest.NewRecorder()
	secondPullReq := httptest.NewRequest(http.MethodPost, "/sync/outbox/pull", bytes.NewBufferString(`{"limit":10}`))
	srv.handleOutboxPull(secondPullRec, secondPullReq)
	if secondPullRec.Code != http.StatusOK {
		t.Fatalf("pull status = %d body=%s", secondPullRec.Code, secondPullRec.Body.String())
	}
	var secondPullPayload map[string]any
	_ = json.Unmarshal(secondPullRec.Body.Bytes(), &secondPullPayload)
	secondItems := secondPullPayload["items"].([]any)
	if len(secondItems) != 1 {
		t.Fatalf("items after ack = %#v", secondItems)
	}
	if secondItems[0].(map[string]any)["entity_id"] != "corr_1" {
		t.Fatalf("remaining item = %#v", secondItems[0])
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

	schemaPath := filepath.Join("..", "..", "..", "db", "schema.sql")
	schemaSQL, err := os.ReadFile(schemaPath)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", schemaPath, err)
	}
	if _, err := db.Exec(string(schemaSQL)); err != nil {
		t.Fatalf("exec schema: %v", err)
	}

	if _, err := db.Exec(`INSERT INTO playlists (id, name, track_count, created_at, updated_at) VALUES ('pl_1', 'Test Playlist', 3, '2026-04-09T00:00:00Z', '2026-04-09T00:00:00Z')`); err != nil {
		t.Fatalf("insert playlist: %v", err)
	}
	for _, insert := range []string{
		`INSERT INTO tracks (id, file_path, file_hash, title, artist, imported_at, updated_at, imported_bpm, imported_key) VALUES ('trk_current', '/music/current.flac', 'hash_current', 'Current', 'Tester', '2026-04-09T00:00:00Z', '2026-04-09T00:00:00Z', 128.0, '8A')`,
		`INSERT INTO tracks (id, file_path, file_hash, title, artist, imported_at, updated_at, imported_bpm, imported_key) VALUES ('trk_candidate', '/music/candidate.flac', 'hash_candidate', 'Candidate', 'Tester', '2026-04-09T00:00:00Z', '2026-04-09T00:00:00Z', 128.0, '8A')`,
		`INSERT INTO tracks (id, file_path, file_hash, title, artist, imported_at, updated_at, imported_bpm, imported_key) VALUES ('trk_history', '/music/history.flac', 'hash_history', 'History', 'Tester', '2026-04-09T00:00:00Z', '2026-04-09T00:00:00Z', 127.0, '8A')`,
		`INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at) VALUES ('pl_1', 'trk_current', 1, '2026-04-09T00:00:00Z')`,
		`INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at) VALUES ('pl_1', 'trk_candidate', 2, '2026-04-09T00:00:00Z')`,
		`INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at) VALUES ('pl_1', 'trk_history', 3, '2026-04-09T00:00:00Z')`,
		`INSERT INTO track_features_abs (track_id, source_file_hash, bpm, bpm_confidence, bpm_source, time_signature, time_signature_confidence, key, key_number, key_letter, key_confidence, key_source, key_agreement, energy_abs, loudness_lufs, loudness_norm, bass_abs, analyzed_at, analysis_signature, config_signature, scoring_contract_id_at_analysis) VALUES ('trk_current', 'hash_current', 128.0, 0.98, 'tempocnn', '4/4', 0.6, '8A', 8, 'A', 0.9, 'musicalkeycnn', 1, 0.60, -10.0, 0.70, 0.50, '2026-04-09T00:00:00Z', 'm1-ebd25381ebad', 'default', 'm3-v1')`,
		`INSERT INTO track_features_abs (track_id, source_file_hash, bpm, bpm_confidence, bpm_source, time_signature, time_signature_confidence, key, key_number, key_letter, key_confidence, key_source, key_agreement, energy_abs, loudness_lufs, loudness_norm, bass_abs, analyzed_at, analysis_signature, config_signature, scoring_contract_id_at_analysis) VALUES ('trk_candidate', 'hash_candidate', 128.0, 0.98, 'tempocnn', '4/4', 0.6, '8A', 8, 'A', 0.9, 'musicalkeycnn', 1, 0.62, -10.0, 0.70, 0.60, '2026-04-09T00:00:00Z', 'm1-ebd25381ebad', 'default', 'm3-v1')`,
		`INSERT INTO track_features_abs (track_id, source_file_hash, bpm, bpm_confidence, bpm_source, time_signature, time_signature_confidence, key, key_number, key_letter, key_confidence, key_source, key_agreement, energy_abs, loudness_lufs, loudness_norm, bass_abs, analyzed_at, analysis_signature, config_signature, scoring_contract_id_at_analysis) VALUES ('trk_history', 'hash_history', 127.0, 0.98, 'tempocnn', '4/4', 0.6, '8A', 8, 'A', 0.9, 'musicalkeycnn', 1, 0.50, -10.0, 0.70, 0.40, '2026-04-09T00:00:00Z', 'm1-ebd25381ebad', 'default', 'm3-v1')`,
		`INSERT INTO track_features_rel (playlist_id, track_id, position, energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel, energy_spread, bass_spread, drums_spread, vocals_spread, groove_spread, intensity_band, intensity_membership, role_hints, valid_as_of_track_count, relative_signature, analysis_signature, config_signature, refreshed_at) VALUES ('pl_1', 'trk_current', 1, 0.5, 0.5, 0.5, 0.1, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 'Drive', '{"drive":1.0}', '["groove"]', 3, '` + relativeSignature + `', 'm1-ebd25381ebad', 'default', '2026-04-09T00:00:00Z')`,
		`INSERT INTO track_features_rel (playlist_id, track_id, position, energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel, energy_spread, bass_spread, drums_spread, vocals_spread, groove_spread, intensity_band, intensity_membership, role_hints, valid_as_of_track_count, relative_signature, analysis_signature, config_signature, refreshed_at) VALUES ('pl_1', 'trk_candidate', 2, 0.6, 0.6, 0.5, 0.2, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 'Drive', '{"drive":1.0}', '["groove"]', 3, '` + relativeSignature + `', 'm1-ebd25381ebad', 'default', '2026-04-09T00:00:00Z')`,
		`INSERT INTO track_features_rel (playlist_id, track_id, position, energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel, energy_spread, bass_spread, drums_spread, vocals_spread, groove_spread, intensity_band, intensity_membership, role_hints, valid_as_of_track_count, relative_signature, analysis_signature, config_signature, refreshed_at) VALUES ('pl_1', 'trk_history', 3, 0.45, 0.4, 0.5, 0.1, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 'Drive', '{"drive":1.0}', '["groove"]', 3, '` + relativeSignature + `', 'm1-ebd25381ebad', 'default', '2026-04-09T00:00:00Z')`,
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
	if _, err := db.Exec(`INSERT INTO playlist_stats (playlist_id, track_count_total, track_count_analyzed, eligible_track_count, energy_spread, bass_spread, drums_spread, vocals_spread, harmonic_spread, groove_spread, avg_harmonic, key_diversity, bpm_range, adapted_weights, adaptation_strength, weight_adaptation_notes, status, energy_source_used, relative_signature, refreshed_at, is_stale, stale_reason, stale_marked_at) VALUES ('pl_1', 3, 3, 3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.3, 2.0, '{"harmonic":0.12}', 0.7, '[]', 'ok', 'canonical', ?, '2026-04-09T00:00:00Z', ?, ?, NULL)`, relativeSignature, staleInt, staleReason); err != nil {
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

func jsonNumber(value int64) string {
	return fmt.Sprintf("%d", value)
}

func stringPtr(value string) *string {
	return &value
}

func boolPtr(value bool) *bool {
	return &value
}
