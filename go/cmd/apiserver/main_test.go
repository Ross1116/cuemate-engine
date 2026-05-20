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
	"strings"
	"testing"
	"time"

	scoringv1 "github.com/Ross1116/cuemate-engine/go/gen/djengine/scoring/v1"
	"github.com/Ross1116/cuemate-engine/go/internal/recommendationsrepo"
	"github.com/Ross1116/cuemate-engine/go/internal/scoringruntime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	_ "modernc.org/sqlite"
)

type fakeRuntimeClient struct {
	metadataResp    *scoringv1.GetScoringMetadataResponse
	metadataErr     error
	recResp         *scoringv1.GetRecommendationsResponse
	recErr          error
	feedbackResp    *scoringv1.GetFeedbackSummaryResponse
	feedbackErr     error
	lastRecReq      *scoringv1.GetRecommendationsRequest
	lastFeedbackReq *scoringv1.GetFeedbackSummaryRequest
}

func (f *fakeRuntimeClient) GetScoringMetadata(context.Context, *scoringv1.GetScoringMetadataRequest, ...grpc.CallOption) (*scoringv1.GetScoringMetadataResponse, error) {
	if f.metadataErr != nil {
		return nil, f.metadataErr
	}
	return f.metadataResp, nil
}

func (f *fakeRuntimeClient) GetRecommendations(ctx context.Context, req *scoringv1.GetRecommendationsRequest, _ ...grpc.CallOption) (*scoringv1.GetRecommendationsResponse, error) {
	f.lastRecReq = req
	if f.recErr != nil {
		return nil, f.recErr
	}
	return f.recResp, nil
}

func (f *fakeRuntimeClient) GetFeedbackSummary(ctx context.Context, req *scoringv1.GetFeedbackSummaryRequest, _ ...grpc.CallOption) (*scoringv1.GetFeedbackSummaryResponse, error) {
	f.lastFeedbackReq = req
	if f.feedbackErr != nil {
		return nil, f.feedbackErr
	}
	if f.feedbackResp == nil {
		return nil, status.Error(codes.Unimplemented, "feedback summary rpc not configured in fake runtime")
	}
	return f.feedbackResp, nil
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

func TestShowcaseRecommendationsDoNotCreateRecommendationEvent(t *testing.T) {
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
	srv.showcase = true

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/recommendations", bytes.NewBufferString(`{"playlist_name":"Test Playlist","current_track_id":"trk_current"}`))
	srv.handleRecommendations(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	meta := payload["meta"].(map[string]any)
	if meta["recommendation_event_id"] != nil {
		t.Fatalf("recommendation_event_id = %#v, want nil", meta["recommendation_event_id"])
	}
	var count int
	if err := srv.repo.DB().QueryRow(`SELECT COUNT(*) FROM recommendation_events`).Scan(&count); err != nil {
		t.Fatalf("count recommendation_events: %v", err)
	}
	if count != 0 {
		t.Fatalf("recommendation_events count = %d, want 0", count)
	}
}

func TestRecommendationEventsClampsNegativeLimit(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	err := srv.repo.InsertRecommendationEvent(context.Background(), recommendationsrepo.RecommendationEventRecord{
		ID:                    "evt_limit",
		PlaylistID:            "pl_1",
		CurrentTrackID:        "trk_current",
		Target:                "maintain",
		CandidateCount:        1,
		RecommendationsStatus: "available",
		LanesReturnedJSON:     `{"lane_order":["maintain"],"lanes":{"maintain":[]}}`,
		ScoringContractID:     "m3-v1",
		Timestamp:             "2026-04-09T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("InsertRecommendationEvent() error = %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/recommendation-events?playlist_id=pl_1&limit=-1", nil)
	srv.handleRecommendationEvents(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	if got := len(payload["items"].([]any)); got != 0 {
		t.Fatalf("items len = %d, want 0", got)
	}
}

func TestRecommendationEventsCapsExcessiveLimit(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/recommendation-events?playlist_id=pl_1&limit=999999", nil)
	srv.handleRecommendationEvents(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	if got := len(payload["items"].([]any)); got > maxRecommendationEventsResponse {
		t.Fatalf("items len = %d, want <= %d", got, maxRecommendationEventsResponse)
	}
}

func TestBoundedRecommendationEventsCapsLimit(t *testing.T) {
	events := make([]recommendationsrepo.RecommendationEventRecord, maxRecommendationEventsResponse+1)
	selected := boundedRecommendationEvents(events, 999999)
	if len(selected) != maxRecommendationEventsResponse {
		t.Fatalf("selected len = %d, want %d", len(selected), maxRecommendationEventsResponse)
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

func TestPlayedEventRejectsInvalidPlayedAt(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	err := srv.repo.InsertRecommendationEvent(context.Background(), recommendationsrepo.RecommendationEventRecord{
		ID:                    "evt_bad_played_at",
		PlaylistID:            "pl_1",
		CurrentTrackID:        "trk_current",
		Target:                "maintain",
		CandidateCount:        1,
		RecommendationsStatus: "available",
		LanesReturnedJSON:     `{"lane_order":["maintain"],"lanes":{"maintain":[{"track_id":"trk_candidate","score":0.9}]}}`,
		ScoringContractID:     "m3-v1",
		Timestamp:             "2026-04-09T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("InsertRecommendationEvent() error = %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/events/played", bytes.NewBufferString(`{"recommendation_event_id":"evt_bad_played_at","chosen_track_id":"trk_candidate","played_at":"not-a-time"}`))
	srv.handlePlayedEvent(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestPlayedEventKeepsSinglePendingFeedbackJobPerPlaylist(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	ctx := context.Background()
	for _, eventID := range []string{"evt_1", "evt_2"} {
		err := srv.repo.InsertRecommendationEvent(ctx, recommendationsrepo.RecommendationEventRecord{
			ID:                    eventID,
			PlaylistID:            "pl_1",
			CurrentTrackID:        "trk_current",
			Target:                "maintain",
			CandidateCount:        2,
			RecommendationsStatus: "available",
			LanesReturnedJSON:     `{"lane_order":["maintain"],"lanes":{"maintain":[{"track_id":"trk_candidate","score":0.9}]}}`,
			ScoringContractID:     "m3-v1",
			Timestamp:             "2026-04-09T00:00:00Z",
		})
		if err != nil {
			t.Fatalf("InsertRecommendationEvent(%s) error = %v", eventID, err)
		}
	}

	for _, eventID := range []string{"evt_1", "evt_2"} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/events/played", bytes.NewBufferString(`{"recommendation_event_id":"`+eventID+`","chosen_track_id":"trk_candidate"}`))
		srv.handlePlayedEvent(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
		}
	}

	jobs, err := srv.repo.ListFeedbackTuningJobsByPlaylist(context.Background(), "pl_1")
	if err != nil {
		t.Fatalf("ListFeedbackTuningJobsByPlaylist() error = %v", err)
	}
	if len(jobs) != 1 || jobs[0].Status != "pending" || jobs[0].TriggerEventID == nil || *jobs[0].TriggerEventID != "evt_2" {
		t.Fatalf("feedback tuning jobs = %#v", jobs)
	}
}

func TestFeedbackSummaryReturnsPlaylistMetrics(t *testing.T) {
	client := &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		feedbackResp: &scoringv1.GetFeedbackSummaryResponse{
			PlaylistId:   "pl_1",
			PlaylistName: "Test Playlist",
			Window:       &scoringv1.FeedbackSummaryWindow{},
			Metrics: &scoringv1.FeedbackSummaryMetrics{
				TotalEvents:             1,
				ContributoryEvents:      1,
				RankedEvents:            1,
				PairwiseComparisonCount: 1,
				ChosenTop1Rate:          1.0,
				ChosenTop3Rate:          1.0,
				ChosenTop5Rate:          1.0,
				MeanChosenRank:          float64Ptr(1.0),
				LaneAcceptanceCounts:    map[string]int32{"reset": 1},
				HigherScoredLaneSkipCounts: map[string]int32{
					"build": 1,
				},
			},
			Weights: &scoringv1.FeedbackSummaryWeights{
				Source:           scoringv1.WeightSource_WEIGHT_SOURCE_ADAPTED,
				StaticWeights:    map[string]float64{"harmonic": 0.12},
				BaseWeights:      map[string]float64{"harmonic": 0.14},
				EffectiveWeights: map[string]float64{"harmonic": 0.14},
			},
			Tuning: &scoringv1.FeedbackSummaryTuning{
				FeedbackEventCount: 1,
				Notes:              []string{"Warm start"},
			},
		},
	}
	srv := newTestServer(t, false, "rel_sig_current", client)
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
		PlayedAt:              stringPtr("2026-04-10T00:00:00Z"),
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
	if client.lastFeedbackReq == nil {
		t.Fatalf("expected feedback summary RPC request to be sent")
	}
	if client.lastFeedbackReq.PlaylistId != "pl_1" || len(client.lastFeedbackReq.Events) != 1 {
		t.Fatalf("feedback request = %#v", client.lastFeedbackReq)
	}
}

func TestFeedbackSummaryHonorsWindowFilters(t *testing.T) {
	client := &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		feedbackResp: &scoringv1.GetFeedbackSummaryResponse{
			PlaylistId:   "pl_1",
			PlaylistName: "Test Playlist",
			Window: &scoringv1.FeedbackSummaryWindow{
				Since: "2026-04-10T12:00:00Z",
			},
			Metrics: &scoringv1.FeedbackSummaryMetrics{
				TotalEvents:             1,
				ContributoryEvents:      1,
				PairwiseComparisonCount: 1,
			},
			Weights: &scoringv1.FeedbackSummaryWeights{
				Source:           scoringv1.WeightSource_WEIGHT_SOURCE_ADAPTED,
				StaticWeights:    map[string]float64{"harmonic": 0.12},
				BaseWeights:      map[string]float64{"harmonic": 0.14},
				EffectiveWeights: map[string]float64{"harmonic": 0.14},
			},
			Tuning: &scoringv1.FeedbackSummaryTuning{},
		},
	}
	srv := newTestServer(t, false, "rel_sig_current", client)
	ctx := context.Background()
	events := []struct {
		id        string
		timestamp string
	}{
		{id: "evt_old", timestamp: "2026-04-10T00:00:00Z"},
		{id: "evt_new", timestamp: "2026-04-11T00:00:00Z"},
	}
	for _, event := range events {
		err := srv.repo.InsertRecommendationEvent(ctx, recommendationsrepo.RecommendationEventRecord{
			ID:                    event.id,
			PlaylistID:            "pl_1",
			CurrentTrackID:        "trk_current",
			Target:                "reset",
			CandidateCount:        2,
			RecommendationsStatus: "available",
			LanesReturnedJSON:     `{"lane_order":["reset","build"],"lanes":{"reset":[{"track_id":"trk_candidate","score":0.8}],"build":[{"track_id":"trk_history","score":0.9}]}}`,
			TrackChosen:           stringPtr("trk_candidate"),
			ChosenWasRecommended:  boolPtr(true),
			ScoringContractID:     "m3-v1",
			Timestamp:             event.timestamp,
			PlayedAt:              stringPtr(event.timestamp),
		})
		if err != nil {
			t.Fatalf("InsertRecommendationEvent(%s) error = %v", event.id, err)
		}
		err = srv.repo.InsertRecommendationEventItems(ctx, []recommendationsrepo.RecommendationEventItemRecord{
			{
				EventID:                event.id,
				LaneID:                 "reset",
				LaneRank:               1,
				CandidateTrackID:       "trk_candidate",
				FinalScore:             0.8,
				RawScore:               0.8,
				PenaltyMultiplier:      1.0,
				Move:                   "reset",
				MoveConfidence:         0.9,
				Risk:                   "low",
				RiskScore:              0.1,
				PrimaryLane:            stringPtr("reset"),
				ComponentScoresJSON:    `{"harmonic":0.9}`,
				ConfidencesJSON:        `{"harmonic":1.0}`,
				WeightsUsedJSON:        `{"harmonic":0.12}`,
				TransitionFeaturesJSON: `{"effective_bpm_distance":1.0}`,
			},
			{
				EventID:                event.id,
				LaneID:                 "build",
				LaneRank:               1,
				CandidateTrackID:       "trk_history",
				FinalScore:             0.9,
				RawScore:               0.9,
				PenaltyMultiplier:      1.0,
				Move:                   "build",
				MoveConfidence:         0.9,
				Risk:                   "low",
				RiskScore:              0.1,
				PrimaryLane:            stringPtr("build"),
				ComponentScoresJSON:    `{"harmonic":0.4}`,
				ConfidencesJSON:        `{"harmonic":1.0}`,
				WeightsUsedJSON:        `{"harmonic":0.12}`,
				TransitionFeaturesJSON: `{"effective_bpm_distance":1.0}`,
			},
		})
		if err != nil {
			t.Fatalf("InsertRecommendationEventItems(%s) error = %v", event.id, err)
		}
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist","since":"2026-04-10T12:00:00Z"}`))
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
	if metrics["pairwise_comparison_count"] != float64(1) {
		t.Fatalf("pairwise_comparison_count = %#v", metrics["pairwise_comparison_count"])
	}
	if client.lastFeedbackReq == nil || len(client.lastFeedbackReq.Events) != 1 {
		t.Fatalf("feedback request events = %#v", client.lastFeedbackReq)
	}
	if client.lastFeedbackReq.Window.GetSince() != "2026-04-10T12:00:00Z" {
		t.Fatalf("feedback request window = %#v", client.lastFeedbackReq.Window)
	}
	if client.lastFeedbackReq.Events[0].EventId != "evt_new" {
		t.Fatalf("filtered event = %#v", client.lastFeedbackReq.Events[0])
	}
}

func TestFeedbackSummaryNormalizesWindowToUTC(t *testing.T) {
	since, until, err := validateFeedbackSummaryWindow("2026-04-10T22:00:00+10:00", "2026-04-11T01:30:00+10:00")
	if err != nil {
		t.Fatalf("validateFeedbackSummaryWindow() error = %v", err)
	}
	if since != "2026-04-10T12:00:00Z" {
		t.Fatalf("since = %q", since)
	}
	if until != "2026-04-10T15:30:00Z" {
		t.Fatalf("until = %q", until)
	}
}

func TestFeedbackSummaryExcludesUnplayedEvents(t *testing.T) {
	client := &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		feedbackResp: &scoringv1.GetFeedbackSummaryResponse{
			PlaylistId:   "pl_1",
			PlaylistName: "Test Playlist",
			Window:       &scoringv1.FeedbackSummaryWindow{},
			Metrics:      &scoringv1.FeedbackSummaryMetrics{},
			Weights: &scoringv1.FeedbackSummaryWeights{
				StaticWeights:    map[string]float64{},
				BaseWeights:      map[string]float64{},
				EffectiveWeights: map[string]float64{},
			},
			Tuning: &scoringv1.FeedbackSummaryTuning{},
		},
	}
	srv := newTestServer(t, false, "rel_sig_current", client)
	ctx := context.Background()
	for _, event := range []struct {
		id       string
		playedAt *string
	}{
		{id: "evt_played", playedAt: stringPtr("2026-04-10T00:00:00Z")},
		{id: "evt_unplayed", playedAt: nil},
	} {
		err := srv.repo.InsertRecommendationEvent(ctx, recommendationsrepo.RecommendationEventRecord{
			ID:                    event.id,
			PlaylistID:            "pl_1",
			CurrentTrackID:        "trk_current",
			Target:                "maintain",
			CandidateCount:        1,
			RecommendationsStatus: "available",
			LanesReturnedJSON:     `{"lane_order":["maintain"],"lanes":{"maintain":[{"track_id":"trk_candidate","score":0.9}]}}`,
			TrackChosen:           stringPtr("trk_candidate"),
			ChosenWasRecommended:  boolPtr(true),
			ScoringContractID:     "m3-v1",
			Timestamp:             "2026-04-10T00:00:00Z",
			PlayedAt:              event.playedAt,
		})
		if err != nil {
			t.Fatalf("InsertRecommendationEvent(%s) error = %v", event.id, err)
		}
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleFeedbackSummary(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if client.lastFeedbackReq == nil {
		t.Fatalf("expected feedback summary RPC request")
	}
	if len(client.lastFeedbackReq.Events) != 1 || client.lastFeedbackReq.Events[0].EventId != "evt_played" {
		t.Fatalf("feedback request events = %#v", client.lastFeedbackReq.Events)
	}
}

func TestFeedbackSummaryRejectsInvalidWindow(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist","since":"not-a-timestamp"}`))
	srv.handleFeedbackSummary(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestFeedbackSummaryReturns503OnTransientScorerFailure(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		feedbackErr:  status.Error(codes.Unavailable, "scorer down"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleFeedbackSummary(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestFeedbackSummaryReturns500OnUnimplemented(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
		feedbackErr:  status.Error(codes.Unimplemented, "not rolled out"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist"}`))
	srv.handleFeedbackSummary(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestFeedbackSummaryRejectsInvertedWindow(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/feedback/summary", bytes.NewBufferString(`{"playlist_name":"Test Playlist","since":"2026-04-11T00:00:00Z","until":"2026-04-10T00:00:00Z"}`))
	srv.handleFeedbackSummary(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestBuildRecommendationsRequestSetsWeightSourceStringAndEnum(t *testing.T) {
	energySpread := 0.18
	bpm := 128.0
	key := "8A"
	analysisSig := "m1-ebd25381ebad"
	configSig := "default"
	contractID := "m3-v1"
	hydrated := &recommendationsrepo.HydratedRecommendations{
		Current: recommendationsrepo.TrackContextRecord{
			TrackID:                   "trk_current",
			BPM:                       &bpm,
			Key:                       &key,
			EnergyRel:                 &energySpread,
			AnalysisSignature:         &analysisSig,
			ConfigSignature:           &configSig,
			ScoringContractAtAnalysis: &contractID,
		},
		Stats: &recommendationsrepo.PlaylistStats{
			EnergySpread:         &energySpread,
			FeedbackTunedWeights: map[string]float64{"harmonic": 0.18},
		},
	}

	req := buildRecommendationsRequest(hydrated, "maintain", 3)
	if req.PlaylistStats.GetWeightSourceEnum() != scoringv1.WeightSource_WEIGHT_SOURCE_FEEDBACK_TUNED {
		t.Fatalf("weight_source_enum = %v", req.PlaylistStats.GetWeightSourceEnum())
	}
}

func TestRecommendationsReportFeedbackTunedWeightSource(t *testing.T) {
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
			LaneOrder: []string{"maintain"},
			Lanes: []*scoringv1.RecommendationLane{
				{
					LaneGroup:    &scoringv1.LaneGroup{LaneId: "maintain"},
					Availability: "available",
					Items: []*scoringv1.ScoredCandidate{
						{Candidate: &scoringv1.TrackContext{TrackId: "trk_candidate"}, FinalScore: 0.9},
					},
				},
			},
			AppliedWeightAdaptation: &scoringv1.WeightAdaptation{
				AdaptationId: "adapted_weights",
				ComponentWeights: map[string]float64{
					"harmonic":      0.12,
					"target_energy": 0.22,
				},
			},
			ActiveSignatures: &scoringv1.SignatureMetadata{ScoringContractId: "m3-v1"},
		},
	})
	if err := srv.repo.RunInTx(context.Background(), func(tx *sql.Tx) error {
		_, err := tx.ExecContext(
			context.Background(),
			`UPDATE playlist_stats SET feedback_tuned_weights = ? WHERE playlist_id = 'pl_1'`,
			`{"harmonic":0.20,"target_energy":0.18}`,
		)
		return err
	}); err != nil {
		t.Fatalf("set feedback_tuned_weights: %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/recommendations", bytes.NewBufferString(`{"playlist_name":"Test Playlist","current_track_id":"trk_current"}`))
	srv.handleRecommendations(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	meta := payload["meta"].(map[string]any)
	weights := meta["weight_adaptation"].(map[string]any)
	if weights["mode"] != "feedback_tuned_weights" || weights["source"] != "feedback_tuned_weights" {
		t.Fatalf("weight_adaptation = %#v", weights)
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

func TestRemotePairingTokenIsSingleUse(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	tokenRec := httptest.NewRecorder()
	tokenReq := httptest.NewRequest(http.MethodPost, "/remote/pairing-token", bytes.NewBufferString(`{"device_label":"Phone"}`))
	tokenReq.Host = "127.0.0.1:8080"
	tokenReq.RemoteAddr = "127.0.0.1:49152"
	srv.handleRemotePairingToken(tokenRec, tokenReq)
	if tokenRec.Code != http.StatusOK {
		t.Fatalf("token status = %d body=%s", tokenRec.Code, tokenRec.Body.String())
	}
	var tokenPayload map[string]any
	if err := json.Unmarshal(tokenRec.Body.Bytes(), &tokenPayload); err != nil {
		t.Fatalf("json: %v", err)
	}
	token := tokenPayload["token"].(string)

	pairBody := `{"token":"` + token + `","device_label":"Phone"}`
	pairRec := httptest.NewRecorder()
	pairReq := httptest.NewRequest(http.MethodPost, "/remote/pair", bytes.NewBufferString(pairBody))
	pairReq.Host = "cue.example"
	srv.handleRemotePair(pairRec, pairReq)
	if pairRec.Code != http.StatusOK {
		t.Fatalf("pair status = %d body=%s", pairRec.Code, pairRec.Body.String())
	}
	if len(pairRec.Result().Cookies()) == 0 {
		t.Fatalf("expected session cookie")
	}

	secondPairRec := httptest.NewRecorder()
	secondPairReq := httptest.NewRequest(http.MethodPost, "/remote/pair", bytes.NewBufferString(pairBody))
	secondPairReq.Host = "cue.example"
	srv.handleRemotePair(secondPairRec, secondPairReq)
	if secondPairRec.Code != http.StatusUnauthorized {
		t.Fatalf("second pair status = %d body=%s", secondPairRec.Code, secondPairRec.Body.String())
	}
}

func TestRemoteAccessRequiresSessionForAPI(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/playlists", nil)
	req.Host = "cue.example"
	srv.remoteAccessMiddleware(http.HandlerFunc(srv.handlePlaylists)).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestRemoteAccessAllowsSessionForSafeAPI(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	sessionSecret := "session-secret"
	now := time.Now().UTC()
	if err := srv.repo.CreateRemoteSession(context.Background(), hashSecret(sessionSecret), stringPtr("Phone"), now.Format(time.RFC3339), now.Add(time.Hour).Format(time.RFC3339)); err != nil {
		t.Fatalf("CreateRemoteSession() error = %v", err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/playlists", nil)
	req.Host = "cue.example"
	req.AddCookie(&http.Cookie{Name: "cuemate_remote_session", Value: sessionSecret})
	srv.remoteAccessMiddleware(http.HandlerFunc(srv.handlePlaylists)).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestRemoteAccessBlocksAdminRoutesEvenWithSession(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	sessionSecret := "session-secret"
	now := time.Now().UTC()
	if err := srv.repo.CreateRemoteSession(context.Background(), hashSecret(sessionSecret), stringPtr("Phone"), now.Format(time.RFC3339), now.Add(time.Hour).Format(time.RFC3339)); err != nil {
		t.Fatalf("CreateRemoteSession() error = %v", err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tools/cli", bytes.NewBufferString(`{}`))
	req.Host = "cue.example"
	req.AddCookie(&http.Cookie{Name: "cuemate_remote_session", Value: sessionSecret})
	srv.remoteAccessMiddleware(http.HandlerFunc(srv.handleToolCommand)).ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/app/shutdown", bytes.NewBufferString(`{}`))
	req.Host = "cue.example"
	req.AddCookie(&http.Cookie{Name: "cuemate_remote_session", Value: sessionSecret})
	srv.remoteAccessMiddleware(http.HandlerFunc(srv.handleAppShutdown)).ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("shutdown status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestRemoteAccessDoesNotTrustSpoofedHostHeader(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/playlists", nil)
	req.Host = "127.0.0.1:8080"
	req.RemoteAddr = "203.0.113.10:49152"
	srv.remoteAccessMiddleware(http.HandlerFunc(srv.handlePlaylists)).ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestShowcaseModeBypassesRemotePairing(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	srv.showcase = true
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/setup/status", nil)
	req.RemoteAddr = "203.0.113.10:49152"
	srv.remoteAccessMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})).ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestSetupStatusIsNotRemotePublic(t *testing.T) {
	if isRemotePublicRequest(httptest.NewRequest(http.MethodGet, "/setup/status", nil)) {
		t.Fatalf("/setup/status should not be remote-public")
	}
}

func TestRemotePairingTokenRejectsMalformedJSON(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/remote/pairing-token", bytes.NewBufferString(`{"device_label":`))
	req.RemoteAddr = "127.0.0.1:49152"
	srv.handleRemotePairingToken(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestParsePickerPathsAcceptsArrayAndSingleString(t *testing.T) {
	paths, err := parsePickerPaths([]byte(`["C:\\Music","D:\\Tracks"]`))
	if err != nil {
		t.Fatalf("parse array: %v", err)
	}
	if len(paths) != 2 || paths[0] != `C:\Music` || paths[1] != `D:\Tracks` {
		t.Fatalf("array paths = %#v", paths)
	}

	paths, err = parsePickerPaths([]byte(`"C:\\Music"`))
	if err != nil {
		t.Fatalf("parse string: %v", err)
	}
	if len(paths) != 1 || paths[0] != `C:\Music` {
		t.Fatalf("single path = %#v", paths)
	}
}

func TestPlaylistAnalysisStatusSummarizesJobs(t *testing.T) {
	srv := newTestServer(t, true, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	if _, err := srv.repo.DB().Exec(`INSERT INTO analysis_jobs (playlist_id, track_id, track_path, status, priority, analysis_mode, analysis_signature, config_signature, created_at, error_message, job_kind) VALUES
		('pl_1', 'trk_current', '/music/current.flac', 'pending', 1, 'staged', 'm1-ebd25381ebad', 'default', '2026-04-09T00:00:00Z', NULL, 'full'),
		('pl_1', 'trk_candidate', '/music/candidate.flac', 'running', 1, 'staged', 'm1-ebd25381ebad', 'default', '2026-04-09T00:00:01Z', NULL, 'full'),
		('pl_1', 'trk_history', '/music/history.flac', 'failed', 1, 'staged', 'm1-ebd25381ebad', 'default', '2026-04-09T00:00:02Z', 'model unavailable', 'full')`); err != nil {
		t.Fatalf("insert jobs: %v", err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/playlists/pl_1/analysis/status", nil)
	srv.handlePlaylistAnalysisStatus(rec, req, "pl_1")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	jobs := payload["jobs"].(map[string]any)
	if payload["next_action"] != "inspect_failures" || jobs["pending"].(float64) != 1 || jobs["running"].(float64) != 1 || jobs["failed"].(float64) != 1 {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestPlaylistAnalysisRefreshQueuesSmartAndForce(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/playlists/pl_1/analysis/refresh", bytes.NewBufferString(`{"analysis_mode":"staged"}`))
	srv.handlePlaylistAnalysisRefresh(rec, req, "pl_1")
	if rec.Code != http.StatusOK {
		t.Fatalf("smart status = %d body=%s", rec.Code, rec.Body.String())
	}
	var smart map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &smart); err != nil {
		t.Fatalf("json: %v", err)
	}
	if smart["queued_count"].(float64) != 0 {
		t.Fatalf("smart queued = %#v", smart)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/playlists/pl_1/analysis/refresh", bytes.NewBufferString(`{"analysis_mode":"staged","force":true}`))
	srv.handlePlaylistAnalysisRefresh(rec, req, "pl_1")
	if rec.Code != http.StatusOK {
		t.Fatalf("force status = %d body=%s", rec.Code, rec.Body.String())
	}
	var force map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &force); err != nil {
		t.Fatalf("json: %v", err)
	}
	if force["queued_count"].(float64) != 3 {
		t.Fatalf("force queued = %#v", force)
	}
}

func TestDeletePlaylistRemovesAppStateButKeepsTracks(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodDelete, "/playlists/pl_1", nil)
	srv.handlePlaylistRoutes(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete status = %d body=%s", rec.Code, rec.Body.String())
	}
	var playlistCount, trackCount int
	if err := srv.repo.DB().QueryRow(`SELECT COUNT(*) FROM playlists WHERE id = 'pl_1'`).Scan(&playlistCount); err != nil {
		t.Fatalf("playlist count: %v", err)
	}
	if err := srv.repo.DB().QueryRow(`SELECT COUNT(*) FROM tracks`).Scan(&trackCount); err != nil {
		t.Fatalf("track count: %v", err)
	}
	if playlistCount != 0 || trackCount != 3 {
		t.Fatalf("playlistCount=%d trackCount=%d", playlistCount, trackCount)
	}
}

func TestToolRunRejectsUnknownAndUnsafeIDs(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	for _, path := range []string{
		"/tools/runs/missing",
		"/tools/runs/..%2Fsecret",
		"/tools/runs/5512103",
		"/tools/runs/AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
	} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, path, nil)
		srv.handleToolRun(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("%s status = %d body=%s", path, rec.Code, rec.Body.String())
		}
	}
}

func TestSetupStatusReadsInstallerState(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	statePath := filepath.Join(t.TempDir(), "setup-state.json")
	if err := os.WriteFile(statePath, []byte(`{"step":"prepare-docker","status":"blocked","message":"Docker login required","core_ready":true,"docker_ready":false,"model_ready":false,"mobile_ready":false,"log_dir":"C:\\CueMate\\logs"}`), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	t.Setenv("CUEMATE_SETUP_STATE_PATH", statePath)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/setup/status", nil)
	srv.handleSetupStatus(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	if payload["status"] != "blocked" || payload["core_ready"] != true || payload["docker_ready"] != false {
		t.Fatalf("setup payload = %#v", payload)
	}
}

func TestSetupStatusReportsShowcaseMode(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	srv.showcase = true
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/setup/status", nil)
	srv.handleSetupStatus(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	if payload["mode"] != "showcase" || payload["read_only"] != true || payload["status"] != "complete" {
		t.Fatalf("setup payload = %#v", payload)
	}
}

func TestShowcaseModeRejectsMutatingEndpoints(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})
	srv.showcase = true
	cases := []struct {
		name   string
		path   string
		method string
		body   string
		call   func(http.ResponseWriter, *http.Request)
	}{
		{
			name:   "playlist delete",
			path:   "/playlists/pl_1",
			method: http.MethodDelete,
			call:   srv.handlePlaylistRoutes,
		},
		{
			name:   "analysis enqueue",
			path:   "/playlists/pl_1/analysis/enqueue",
			method: http.MethodPost,
			body:   `{"analysis_mode":"staged"}`,
			call:   srv.handlePlaylistRoutes,
		},
		{
			name:   "played event",
			path:   "/events/played",
			method: http.MethodPost,
			body:   `{"recommendation_event_id":"evt_1","chosen_track_id":"trk_candidate"}`,
			call:   srv.handlePlayedEvent,
		},
		{
			name:   "tool command",
			path:   "/tools/cli",
			method: http.MethodPost,
			body:   `{}`,
			call:   srv.handleToolCommand,
		},
		{
			name:   "app shutdown",
			path:   "/app/shutdown",
			method: http.MethodPost,
			body:   `{}`,
			call:   srv.handleAppShutdown,
		},
		{
			name:   "analysis workers stop",
			path:   "/analysis/workers/stop",
			method: http.MethodPost,
			body:   `{}`,
			call:   srv.handleStopAnalysisWorkers,
		},
		{
			name:   "remote token",
			path:   "/remote/pairing-token",
			method: http.MethodPost,
			body:   `{"device_label":"Phone"}`,
			call:   srv.handleRemotePairingToken,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(tc.method, tc.path, bytes.NewBufferString(tc.body))
			tc.call(rec, req)
			if rec.Code != http.StatusForbidden {
				t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
			}
			var payload map[string]string
			if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
				t.Fatalf("json: %v", err)
			}
			if payload["error"] != "CueMate showcase mode is read-only." {
				t.Fatalf("payload = %#v", payload)
			}
		})
	}
}

func TestLoadConfigRecognizesShowcaseMode(t *testing.T) {
	t.Setenv("CUEMATE_SHOWCASE_MODE", "1")
	cfg := loadConfig()
	if !cfg.Showcase {
		t.Fatalf("Showcase = false, want true")
	}
}

func TestClientPlaylistAndTrackBrowseEndpoints(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	req := httptest.NewRequest(http.MethodGet, "/playlists", nil)
	rec := httptest.NewRecorder()
	srv.handlePlaylists(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var playlists map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &playlists); err != nil {
		t.Fatalf("json: %v", err)
	}
	items := playlists["items"].([]any)
	if len(items) != 1 || items[0].(map[string]any)["playlist_id"] != "pl_1" {
		t.Fatalf("playlists = %#v", playlists)
	}

	req = httptest.NewRequest(http.MethodGet, "/playlists/pl_1/tracks?query=Candidate", nil)
	rec = httptest.NewRecorder()
	srv.handlePlaylistRoutes(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var tracks map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &tracks); err != nil {
		t.Fatalf("json: %v", err)
	}
	trackItems := tracks["items"].([]any)
	if len(trackItems) != 1 || trackItems[0].(map[string]any)["track_id"] != "trk_candidate" {
		t.Fatalf("tracks = %#v", tracks)
	}
}

func TestClientPlaylistTracksDoesNotRequireScorerMetadata(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataErr: status.Error(codes.Unavailable, "scorer down"),
	})

	req := httptest.NewRequest(http.MethodGet, "/playlists/pl_1/tracks?query=Candidate", nil)
	rec := httptest.NewRecorder()
	srv.handlePlaylistRoutes(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var tracks map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &tracks); err != nil {
		t.Fatalf("json: %v", err)
	}
	trackItems := tracks["items"].([]any)
	if len(trackItems) != 1 || trackItems[0].(map[string]any)["track_id"] != "trk_candidate" {
		t.Fatalf("tracks = %#v", tracks)
	}
}

func TestClientAnalysisEnqueueEndpoint(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	req := httptest.NewRequest(http.MethodPost, "/playlists/pl_1/analysis/enqueue", bytes.NewBufferString(`{"analysis_mode":"staged","force":true}`))
	rec := httptest.NewRecorder()
	srv.handlePlaylistRoutes(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json: %v", err)
	}
	if payload["queued_count"] != float64(3) {
		t.Fatalf("payload = %#v", payload)
	}
	jobs, err := srv.repo.ListAnalysisJobs(context.Background(), "pl_1", "pending", 10)
	if err != nil {
		t.Fatalf("ListAnalysisJobs: %v", err)
	}
	if len(jobs) != 3 {
		t.Fatalf("jobs = %#v", jobs)
	}
}

func TestClientAnalysisEnqueueRejectsInvalidMode(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	req := httptest.NewRequest(http.MethodPost, "/playlists/pl_1/analysis/enqueue", bytes.NewBufferString(`{"analysis_mode":"surprise"}`))
	rec := httptest.NewRecorder()
	srv.handlePlaylistRoutes(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestQueuePlaylistAnalysisRequeuesStaleSignatures(t *testing.T) {
	srv := newTestServer(t, false, "rel_sig_current", &fakeRuntimeClient{
		metadataResp: fakeMetadata("rel_sig_current"),
	})

	queued, err := srv.repo.QueuePlaylistAnalysis(context.Background(), "pl_1", "staged", false, "m1-ebd25381ebad", "default")
	if err != nil {
		t.Fatalf("QueuePlaylistAnalysis(current signatures) error = %v", err)
	}
	if queued != 0 {
		t.Fatalf("queued current signatures = %d, want 0", queued)
	}

	queued, err = srv.repo.QueuePlaylistAnalysis(context.Background(), "pl_1", "staged", false, "m1-new", "default")
	if err != nil {
		t.Fatalf("QueuePlaylistAnalysis(stale signatures) error = %v", err)
	}
	if queued != 3 {
		t.Fatalf("queued stale signatures = %d, want 3", queued)
	}
}

func TestBuildToolCommandSeparatesImportPlaylistPaths(t *testing.T) {
	args, background, err := buildToolCommand(toolCommandRequest{
		Action: "import_playlist",
		Name:   "Test Playlist",
		Paths:  []string{`C:\Music\track.flac`},
	})
	if err != nil {
		t.Fatalf("buildToolCommand() error = %v", err)
	}
	if !background {
		t.Fatalf("background = false")
	}
	want := []string{"-m", pythonModule, "import-playlist", "--name", "Test Playlist", "--", `C:\Music\track.flac`}
	if fmt.Sprint(args) != fmt.Sprint(want) {
		t.Fatalf("args = %#v, want %#v", args, want)
	}
}

func TestBuildToolCommandRejectsOptionLikeUserValues(t *testing.T) {
	_, _, err := buildToolCommand(toolCommandRequest{
		Action:   "analyze_playlist",
		Playlist: "--help",
	})
	if err == nil {
		t.Fatalf("expected error for option-like playlist")
	}
	if !strings.Contains(err.Error(), "must not start with '-'") {
		t.Fatalf("error = %v", err)
	}
}

func TestBuildToolCommandCapsAnalysisWorkerBatch(t *testing.T) {
	args, background, err := buildToolCommand(toolCommandRequest{
		Action: "run_analysis_worker",
		Limit:  1000,
	})
	if err != nil {
		t.Fatalf("buildToolCommand() error = %v", err)
	}
	if !background {
		t.Fatalf("background = false")
	}
	want := []string{"-m", pythonModule, "run-analysis-worker", "--limit", "5"}
	if fmt.Sprint(args) != fmt.Sprint(want) {
		t.Fatalf("args = %#v, want %#v", args, want)
	}
}

func TestWebAppFallbackServesIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<main>CueMate</main>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	req := httptest.NewRequest(http.MethodGet, "/performance/deck", nil)
	rec := httptest.NewRecorder()
	handleWebApp(dir)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "CueMate") {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestWebAppRejectsPathTraversal(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<main>CueMate</main>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	req := httptest.NewRequest(http.MethodGet, "/../secret.txt", nil)
	rec := httptest.NewRecorder()
	handleWebApp(dir)(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
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

func float64Ptr(value float64) *float64 {
	return &value
}
