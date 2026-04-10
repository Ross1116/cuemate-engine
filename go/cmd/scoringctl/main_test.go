package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadScoreFixture(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "score_candidate.json")
	request, err := loadScoreFixture(path)
	if err != nil {
		t.Fatalf("loadScoreFixture() error = %v", err)
	}

	if got := request.GetCurrentTrack().GetTrackId(); got != "trk_current" {
		t.Fatalf("current track: expected %q, got %q", "trk_current", got)
	}
	if got := request.GetCandidate().GetTrackId(); got != "trk_candidate" {
		t.Fatalf("candidate track: expected %q, got %q", "trk_candidate", got)
	}
}

func TestLoadScoreFixtureRejectsMissingTrackIDs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte(`{"target_lane":"reset"}`), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	_, err := loadScoreFixture(path)
	if err == nil {
		t.Fatal("expected error")
	}
}
